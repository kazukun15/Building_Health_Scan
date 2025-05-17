import pkgutil
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter

import streamlit as st
import requests
import json
import faiss
import numpy as np
import PyPDF2
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

# --------------- 設定 ----------------
st.set_page_config(
    page_title="スマホで建物分析RAG",
    layout="centered",  # スマホではcentered推奨
    initial_sidebar_state="collapsed"
)

# --------------- ユーザーガイド付きイメージ図 ----------------
with st.expander("📱 スマホUIイメージ図（クリックで開閉）", expanded=True):
    st.image("https://raw.githubusercontent.com/streamlit/streamlit-example-apps/main/assets/mobile_mockup_simple.png", caption="モバイル利用イメージ図", use_column_width=True)

# --------------- 各種関数 ----------------
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
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def create_vector_index(chunks, model):
    embeddings = model.encode(chunks, convert_to_numpy=True, batch_size=32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

@st.cache_resource(show_spinner=False)
def load_pdf_index():
    pdf_paths = [
        "Structure_Base.pdf",
        "kamijimachou_Public_facility_management_plan.pdf",
        "minatoku_Public_facility_management_plan.pdf"
    ]
    combined_text = ""
    for path in pdf_paths:
        combined_text += extract_text_from_pdf(path) + "\n"
    chunks = split_text_into_chunks(combined_text)
    chunks = [chunk for chunk in chunks if chunk.strip() and len(chunk.strip()) > 5]
    vec_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = vec_model.tokenizer
    valid_chunks = [chunk for chunk in chunks if len(tokenizer.encode(chunk, add_special_tokens=True)) > 0]
    index = create_vector_index(valid_chunks, vec_model)
    return index, valid_chunks, vec_model

def search_relevant_chunks(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

@st.cache_resource(show_spinner=False)
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image, processor, model):
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

def generate_report_with_gemini(prompt_text):
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

# ---------------- UIメイン ------------------
st.title("スマホ特化：ファシリティ分析RAG")
st.markdown(
    """
    1. **建物や外壁の画像**をアップロード・撮影  
    2. **質問を入力**  
    3. **「レポート生成」ボタンをタップ**  
    ---  
    画像＋専門資料を活用した分析AIが、**非破壊検査・建築分野の専門知識**で回答します！
    """
)

with st.form(key="analysis_form"):
    # スマホではカメラ入力を推奨し、一度に一つの画像
    st.markdown("### 1. 画像の入力")
    upload_col, camera_col = st.columns(2)
    image = None
    with upload_col:
        image_file = st.file_uploader("画像アップロード", type=["jpg", "jpeg", "png"], key="up_file")
        if image_file:
            image = Image.open(image_file)
    with camera_col:
        image_camera = st.camera_input("撮影", key="camera_in")
        if image_camera:
            image = Image.open(image_camera)
    if image:
        st.image(image, caption="選択中の画像", use_column_width=True)
        processor, cap_model = load_caption_model()
        with st.spinner("画像解析中..."):
            try:
                caption = generate_image_caption(image, processor, cap_model)
            except Exception as e:
                caption = f"キャプション生成エラー: {e}"
        st.info(f"画像キャプション：{caption}")
    else:
        caption = ""
        st.warning("画像を選択してください。")

    st.markdown("### 2. 質問を入力")
    user_query = st.text_area("質問を入力（例：外壁のひび割れの基準は？）", max_chars=150)

    st.form_submit_button("レポート生成", use_container_width=True)

if image and user_query:
    # ステップ3: レポート生成
    with st.spinner("PDFと画像情報を統合してAIレポートを作成中..."):
        index, chunks, vec_model = load_pdf_index()
        relevant_chunks = search_relevant_chunks(user_query, vec_model, index, chunks)
        context = "\n\n".join(relevant_chunks)
        prompt = (
            f"ユーザーの質問: {user_query}\n\n"
            f"以下はPDF（Structure_Base.pdf等）からの関連情報:\n{context}\n\n"
        )
        if caption:
            prompt += f"画像キャプション:\n{caption}\n\n"
        prompt += (
            "これらを元に、非破壊検査・建築・材料学の知識を活用し、国土交通省基準で分析レポートを日本語で作成せよ。"
            " 劣化度A～D・推定寿命も明示。誤情報は排除。"
        )
        result = generate_report_with_gemini(prompt)
        if "error" in result:
            st.error(f"API呼び出しエラー: {result['error']}")
        else:
            try:
                report_text = result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                st.error("レポート抽出エラー: " + str(e))
                report_text = None
            if report_text:
                st.markdown("## 生成レポート")
                st.markdown(report_text)
                st.download_button("レポートをダウンロード", report_text, file_name="report.txt", mime="text/plain")
else:
    st.info("画像と質問の両方を入力してください。")

st.markdown("---")
st.caption("スマートフォン利用時は画面を縦にスクロールし、各項目を順番に入力してください。")
