# monkey patch: Python 3.12ではpkgutil.ImpImporterが削除されているため、zipimporterを代用
import pkgutil
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter

import streamlit as st
import requests
import json
import io
import faiss
import numpy as np
import PyPDF2
import concurrent.futures
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------------------
# PDFからのテキスト抽出と前処理
# -------------------------------
def extract_text_from_pdf(pdf_path):
    """PDFから全ページのテキストを抽出"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text_into_chunks(text, max_length=500):
    """テキストをmax_length文字ごとにチャンクに分割"""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def create_vector_index(chunks, model):
    """各チャンクをバッチ処理(batch_size=32)でエンコードし、FAISSインデックスを作成"""
    embeddings = model.encode(chunks, convert_to_numpy=True, batch_size=32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

@st.cache_resource(show_spinner=False)
def load_pdf_index():
    """
    Structure_Base.pdfからテキスト抽出、チャンク分割、フィルタリング、ベクトル化、インデックス作成
    ※Structure_Base.pdfはプロジェクトルートに配置してください
    """
    pdf_path = "Structure_Base.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    # 空文字、空白のみ、または非常に短いチャンクを除外
    chunks = [chunk for chunk in chunks if chunk.strip() and len(chunk.strip()) > 5]
    st.write(f"PDFから抽出した有効なテキストチャンク数: {len(chunks)}")
    vec_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = create_vector_index(chunks, vec_model)
    return index, chunks, vec_model

def search_relevant_chunks(query, model, index, chunks, top_k=3):
    """ユーザーのクエリに基づき、関連チャンクをFAISSから検索"""
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# -------------------------------
# 画像キャプション生成
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_caption_model():
    """BLIPの画像キャプションモデルとプロセッサのロード"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image, processor, model):
    """
    1枚の画像からキャプション生成
    ・画像は最大幅800pxにリサイズし、RGBに変換
    ・BLIPには単一画像の場合もリストとして渡し、padding=Trueを指定
    """
    max_width = 800
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    inputs = processor([image], return_tensors="pt", padding=True)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# -------------------------------
# Gemini API 呼び出し
# -------------------------------
def generate_report_with_gemini(prompt_text):
    """Gemini API (gemini-2.0-flash) を呼び出してレポート生成"""
    try:
        api_key = st.secrets["gemini"]["API_KEY"]
    except KeyError:
        return {"error": "Gemini API Keyが .streamlit/secrets.toml に設定されていません。"}
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Streamlit アプリメイン
# -------------------------------
def main():
    st.set_page_config(page_title="多角的レポート生成システム", layout="wide")
    st.title("多角的レポート生成システム (Python 3.12.9 対応)")
    st.markdown(
        """
        このアプリは、**Structure_Base.pdf** に含まれる国土交通省の基準情報と、  
        ユーザーがアップロードまたはカメラで撮影した1枚の画像から抽出されたキャプションを組み合わせ、  
        国土交通省の基準に基づいた外壁・構造物の状態分析レポートを生成します。  
        
        ※ AIには「非破壊検査」「建築業」「材料学」の専門知識を与え、不明な部分はインターネット検索で補完し、ハルシネーションは絶対に避けるよう指示しています。  
        また、劣化度をA～Dで定量化し、総合的に判断した推定寿命も表示します。  
        """
    )
    
    # ユーザーの質問入力
    user_query = st.text_input("質問を入力してください（例: 外壁のひび割れ基準について）")
    
    # 画像入力：ファイルアップロードとカメラ撮影（どちらか1枚を採用）
    st.markdown("### 画像入力")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ファイルアップロード")
        image_file = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="upload")
    with col2:
        st.subheader("カメラ撮影")
        image_capture = st.camera_input("カメラで写真を撮影してください", key="camera")
    
    # 撮影画像があれば優先して採用、なければアップロード画像
    image = None
    if image_capture is not None:
        image = Image.open(image_capture)
    elif image_file is not None:
        image = Image.open(image_file)
    
    if image is not None:
        st.image(image, caption="選択された画像", use_container_width=True)
        processor, cap_model = load_caption_model()
        try:
            caption = generate_image_caption(image, processor, cap_model)
        except Exception as e:
            caption = f"エラー: {e}"
        st.write("生成された画像キャプション:")
        st.write(caption)
    else:
        caption = ""
    
    if st.button("レポート生成"):
        if not user_query:
            st.error("質問を入力してください。")
            return
        
        # PDFからの情報読み込み
        index, chunks, vec_model = load_pdf_index()
        relevant_chunks = search_relevant_chunks(user_query, vec_model, index, chunks)
        context = "\n\n".join(relevant_chunks)
        
        # プロンプト作成（ユーザーの質問、PDF情報、画像キャプションの統合）
        prompt = f"ユーザーの質問: {user_query}\n\n"
        prompt += "以下は Structure_Base.pdf から抽出された関連情報です:\n" + context + "\n\n"
        if caption:
            prompt += "さらに、アップロードまたは撮影された画像のキャプションは以下の通りです:\n" + caption + "\n\n"
        prompt += (
            "上記情報を基に、非破壊検査、建築業、材料学の専門知識を活用し、"
            "不明な部分はインターネット検索で補完し、ハルシネーションは絶対に避け、"
            "国土交通省の基準に沿った外壁・構造物の状態分析レポートを作成してください。\n"
            "また、劣化度はA～Dで定量化し、総合的に判断した推定寿命も明示してください。"
        )
        
        st.info("Gemini API にプロンプトを送信中です…")
        result = generate_report_with_gemini(prompt)
        if "error" in result:
            st.error(f"API呼び出しエラー: {result['error']}")
        else:
            try:
                report_text = result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                st.error("レポート抽出中にエラーが発生しました: " + str(e))
                return
            
            # レポートを行ごとに分割し、最初の数行を概要（総合評価）として表示、詳細はExpanderに隠す
            report_lines = report_text.splitlines()
            if len(report_lines) > 5:
                summary = "\n".join(report_lines[:5])
                details = "\n".join(report_lines[5:])
            else:
                summary = report_text
                details = ""
            
            st.markdown("## 総合評価 (概要)")
            st.markdown(summary)
            if details:
                with st.expander("詳細を表示"):
                    st.markdown(details)
            
            # レポート共有機能：ダウンロードボタン
            st.download_button("レポートをダウンロード", report_text, file_name="report.txt", mime="text/plain")
    
    st.markdown("---")
    st.info(
        "このシステムは、Structure_Base.pdf の情報と、アップロードまたはカメラで撮影された1枚の画像から抽出されたキャプションを組み合わせ、\n"
        "ユーザーの質問に基づいた外壁・構造物の状態分析レポートを生成します。"
    )

if __name__ == "__main__":
    main()
