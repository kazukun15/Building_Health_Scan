# monkey patch: Python 3.12 では pkgutil.ImpImporter が削除されているため、代わりに zipimporter を設定
import pkgutil
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter

import sys
import streamlit as st

# Python バージョンチェックは削除（Python 3.12.9 で実行する前提）

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

# --------------- PDF からのテキスト抽出と処理 ---------------
def extract_text_from_pdf(pdf_path):
    """PDF から全ページのテキストを抽出"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text_into_chunks(text, max_length=500):
    """テキストを max_length 文字ごとにチャンクに分割"""
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def create_vector_index(chunks, model):
    """各チャンクをベクトル化して FAISS インデックスを作成"""
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

@st.cache_resource(show_spinner=False)
def load_pdf_index():
    """
    Structure_Base.pdf からテキスト抽出、チャンク分割、ベクトル化、インデックス作成
    ※ Structure_Base.pdf はプロジェクトルートに配置してください
    """
    pdf_path = "Structure_Base.pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    st.write(f"PDF から抽出したテキストチャンク数: {len(chunks)}")
    vec_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = create_vector_index(chunks, vec_model)
    return index, chunks, vec_model

def search_relevant_chunks(query, model, index, chunks, top_k=3):
    """ユーザーのクエリに基づき、関連チャンクを FAISS から検索"""
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# --------------- 画像キャプション生成 ---------------
@st.cache_resource(show_spinner=False)
def load_caption_model():
    """BLIP の画像キャプションモデルとプロセッサのロード"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image, processor, model):
    """
    1 枚の画像からキャプション生成
    ・画像は最大幅 800px にリサイズし、RGB に変換
    """
    max_width = 800
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# --------------- Gemini API 呼び出し ---------------
def generate_report_with_gemini(prompt_text):
    """Gemini API (gemini-2.0-flash) を呼び出してレポート生成"""
    try:
        api_key = st.secrets["gemini"]["API_KEY"]
    except KeyError:
        return {"error": "Gemini API Key が .streamlit/secrets.toml に設定されていません。"}
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# --------------- Streamlit アプリメイン ---------------
def main():
    st.title("多角的レポート生成システム (Python 3.12.9 対応)")
    st.write(
        "このアプリは、Structure_Base.pdf に含まれる国土交通省の基準情報と、"
        "ユーザーがアップロードした複数枚の画像から抽出されたキャプションを組み合わせ、"
        "総合的な外壁・構造物の状態分析レポートを生成します。"
    )
    
    # ユーザーからの質問入力
    user_query = st.text_input("質問を入力してください（例: 外壁のひび割れ基準について）")
    
    # サイドバー：画像アップロード（複数枚対応）
    st.sidebar.header("画像アップロード (複数枚対応)")
    input_method = st.sidebar.selectbox("画像入力方法", ["なし", "アップロード", "カメラ撮影"])
    
    images = []
    if input_method == "アップロード":
        uploaded_files = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            images = [Image.open(file) for file in uploaded_files]
    elif input_method == "カメラ撮影":
        captured_image = st.camera_input("カメラで画像を撮影してください")
        if captured_image is not None:
            images = [Image.open(captured_image)]
    
    if images:
        st.subheader("入力画像一覧")
        for idx, img in enumerate(images):
            st.image(img, caption=f"画像 {idx+1}", use_container_width=True)
    
    if st.button("レポート生成"):
        if not user_query:
            st.error("質問を入力してください。")
            return
        
        # 複数画像のキャプション生成を並列処理で実施
        captions = []
        if images:
            st.write("画像キャプション生成中です...")
            processor, cap_model = load_caption_model()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                captions = list(executor.map(lambda img: generate_image_caption(img, processor, cap_model), images))
            st.write("生成された各画像のキャプション:")
            for i, cap in enumerate(captions):
                st.write(f"画像 {i+1}: {cap}")
        
        # PDF からの情報読み込み
        index, chunks, vec_model = load_pdf_index()
        relevant_chunks = search_relevant_chunks(user_query, vec_model, index, chunks)
        context = "\n\n".join(relevant_chunks)
        
        # プロンプト作成（ユーザーの質問、PDF 情報、画像キャプションの統合）
        prompt = f"ユーザーの質問: {user_query}\n\n"
        prompt += "以下は Structure_Base.pdf から抽出された関連情報です:\n" + context + "\n\n"
        if captions:
            combined_captions = "\n".join([f"画像 {i+1}: {cap}" for i, cap in enumerate(captions)])
            prompt += "さらに、アップロードされた画像のキャプションは以下の通りです:\n" + combined_captions + "\n\n"
        prompt += (
            "上記情報を基に、国土交通省の基準に沿った外壁・構造物の状態分析レポートを、"
            "項目ごとに整理された一般的なレポート形式で作成してください。"
        )
        
        st.write("Gemini API にプロンプトを送信中です…")
        result = generate_report_with_gemini(prompt)
        if "error" in result:
            st.error(f"API 呼び出しエラー: {result['error']}")
        else:
            try:
                report_text = result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                st.error("レポート抽出中にエラーが発生しました: " + str(e))
                return
            st.subheader("生成されたレポート")
            st.markdown(report_text)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "このシステムは、Structure_Base.pdf の情報と複数の画像から抽出されたキャプションを組み合わせ、"
        "ユーザーの質問に基づいた多角的なレポートを生成します。\n"
        "本コードは Python 3.12.9 環境で動作するように調整されています。"
    )

if __name__ == "__main__":
    main()
