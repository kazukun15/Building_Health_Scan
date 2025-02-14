import os
import streamlit as st
import requests
import json
import io
import faiss
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ---------------------------
# PDFからテキスト抽出と前処理
# ---------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text_into_chunks(text, max_length=500):
    # 単純な分割例：max_length文字ごとに分割（必要に応じて改良してください）
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# ---------------------------
# ベクトル化とインデックス作成
# ---------------------------
def create_vector_index(chunks, model):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# ---------------------------
# キャプション生成（オプション）
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ---------------------------
# Gemini API 呼び出し
# ---------------------------
def generate_report_with_gemini(prompt_text):
    try:
        api_key = st.secrets["gemini"]["API_KEY"]
    except KeyError:
        return {"error": "Gemini API Key is missing. Please add it to your .streamlit/secrets.toml file."}
    
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt_text}]}
        ]
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# RAGパイプライン：PDF情報の読み込みとクエリ検索
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_pdf_index():
    pdf_path = "Structure_Base.pdf"  # GitHubに配置したPDF
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    st.write(f"PDFから抽出したテキストチャンク数: {len(chunks)}")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, embeddings = create_vector_index(chunks, model)
    return index, chunks, model

def search_relevant_chunks(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ---------------------------
# Streamlitアプリメイン部分
# ---------------------------
def main():
    st.title("建物分析君")
    st.write("このアプリは、リポジトリに含まれる Structure_Base.pdf の内容を元に、ユーザーのクエリに対して国土交通省基準に沿ったレポートを生成します。")
    
    # ユーザークエリ入力
    user_query = st.text_input("質問を入力してください（例: 外壁のひび割れ基準について）")
    
    # オプション：画像キャプション生成（画像アップロード）
    st.sidebar.header("オプション: 画像からキャプション生成")
    input_method = st.sidebar.selectbox("画像入力方法", ["なし", "アップロード", "カメラ撮影"])
    image = None
    caption = ""
    if input_method == "アップロード":
        uploaded_file = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    elif input_method == "カメラ撮影":
        captured_image = st.camera_input("カメラで画像を撮影してください")
        if captured_image is not None:
            image = Image.open(captured_image)
    
    if image is not None:
        st.subheader("入力画像")
        st.image(image, caption="選択された画像", use_container_width=True)
        processor, cap_model = load_caption_model()
        caption = generate_image_caption(image, processor, cap_model)
        st.write("生成された画像キャプション:")
        st.write(caption)
    
    if st.button("レポート生成"):
        if not user_query:
            st.error("質問を入力してください。")
            return
        
        # PDFからRAGに利用するインデックスを読み込む
        index, chunks, vec_model = load_pdf_index()
        relevant_chunks = search_relevant_chunks(user_query, vec_model, index, chunks)
        context = "\n\n".join(relevant_chunks)
        
        # プロンプト作成（ユーザーの質問、PDFからの関連情報、オプションの画像キャプションを組み込む）
        prompt = (
            f"ユーザーの質問: {user_query}\n\n"
            "以下は、Structure_Base.pdf から抽出された関連情報です:\n"
            f"{context}\n\n"
        )
        if caption:
            prompt += f"さらに、アップロードされた画像のキャプションは次の通りです: '{caption}'\n\n"
        prompt += (
            "上記情報を参考にして、国土交通省の基準に沿った外壁・構造物の状態分析レポートを、"
            "一般的なレポート形式（項目ごとに整理された文章）で作成してください。"
        )
        
        st.write("Gemini API にプロンプトを送信中です…")
        result = generate_report_with_gemini(prompt)
        
        if "error" in result:
            st.error(f"API呼び出しエラー: {result['error']}")
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
        "このシステムは、Structure_Base.pdf の内容と（任意の）画像キャプションをRAGシステムに取り込み、"
        "ユーザーの質問に対して国土交通省基準に準拠したレポートを生成します。"
    )

if __name__ == "__main__":
    main()
