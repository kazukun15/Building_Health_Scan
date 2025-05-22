import streamlit as st
import requests
import faiss
import numpy as np
import PyPDF2
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------------------
# PDF処理関連関数
# -------------------------------

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                st.warning(f"PDFファイル '{pdf_path}' は暗号化されています。")
                return ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except FileNotFoundError:
        st.error(f"PDFファイル '{pdf_path}' が見つかりません。")
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"PDF読み取りエラー: {e}")
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
    chunks = [chunk for chunk in chunks if chunk.strip()]
    vec_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = create_vector_index(chunks, vec_model)
    return index, chunks, vec_model

def search_relevant_chunks(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# -------------------------------
# 画像キャプション生成関連関数
# -------------------------------

@st.cache_resource(show_spinner=False)
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image, processor, model):
    if image.width > 800:
        image = image.resize((800, int(image.height * 800 / image.width)))
    if image.mode != 'RGB':
        image = image.convert('RGB')

    prompt = (
        "画像内の壁と床の状態を詳しく日本語で説明してください。"
        "ひび割れ、汚れ、膨らみ、剥がれ、変色などの損傷を含め、建物検査官の視点で記述してください。"
    )
    inputs = processor([image], text=prompt, return_tensors="pt", padding=True)
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# -------------------------------
# Gemini API 呼び出し関連関数
# -------------------------------

def generate_report_with_gemini(prompt_text):
    try:
        api_key = st.secrets["gemini"]["API_KEY"]
    except KeyError:
        st.error("Gemini APIキーが設定されていません。")
        return None

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-1.5-flash-latest:generateContent?key=" + api_key
    )

    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    try:
        response = requests.post(endpoint, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"APIリクエストエラー: {e}")
        return None

# -------------------------------
# Streamlit UI部分
# -------------------------------

st.set_page_config(page_title="壁・床状態分析ツール", layout="wide")
st.title("🧱 壁・床状態分析レポート生成ツール")

st.markdown("""
建物の壁や床の状態をAIが解析し、詳細レポートを生成します。  
画像をアップロードするか、カメラで撮影してください。
""")

# 画像入力セクション
image = None
with st.expander("📸 画像の入力方法を選択"):
    image_file = st.file_uploader("画像ファイルをアップロード", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("またはカメラで撮影")

    if image_file:
        image = Image.open(image_file)
    elif camera_image:
        image = Image.open(camera_image)

    if image:
        st.image(image, caption="使用する画像", use_column_width=True)

# 補足情報入力
with st.expander("📝 補足情報を入力（任意）"):
    wall_note = st.text_area("壁に関する補足情報（ひび割れ、変色など）")
    floor_note = st.text_area("床に関する補足情報（きしみ、沈みなど）")
    user_query = st.text_input("特に質問したい内容")

# レポート生成
if st.button("📊 レポート生成開始"):
    if not image:
        st.error("画像を入力してください。")
        st.stop()

    processor, cap_model = load_caption_model()

    with st.spinner("画像を解析中…"):
        caption = generate_image_caption(image, processor, cap_model)
    st.success("画像キャプションを生成しました。")
    st.write(caption)

    index, chunks, vec_model = load_pdf_index()
    relevant_chunks = search_relevant_chunks(user_query, vec_model, index, chunks)
    context = "\n\n".join(relevant_chunks)

    prompt = (
        f"ユーザーの質問: {user_query}\n\n"
        f"関連情報:\n{context}\n\n"
        f"画像キャプション:\n{caption}\n\n"
        f"壁の補足情報:\n{wall_note}\n\n"
        f"床の補足情報:\n{floor_note}\n\n"
        "上記を基に、壁と床の状態を分析し、以下のMarkdown形式でレポート作成:\n"
        "## 1. 壁の状態分析\n"
        "### 1.1 現状\n(壁の現状)\n"
        "### 1.2 劣化度 (A～Dで評価)\n"
        "### 1.3 推定寿命\n"
        "### 1.4 必要な対策\n"
        "### 1.5 注意点\n"
        "## 2. 床の状態分析\n"
        "### 2.1 現状\n(床の現状)\n"
        "### 2.2 劣化度 (A～Dで評価)\n"
        "### 2.3 推定寿命\n"
        "### 2.4 必要な対策\n"
        "### 2.5 注意点\n"
        "情報不足時は専門的な知識やインターネット検索も併用し、素人にも理解しやすく簡潔に回答してください。"
    )

    with st.spinner("レポートを生成中…"):
        result = generate_report_with_gemini(prompt)
        if result:
            report_text = result["candidates"][0]["content"]["parts"][0]["text"]
            st.success("レポート生成完了！")

            for section in report_text.split("## "):
                if section.strip():
                    header, content = section.split("\n", 1)
                    with st.expander(header.strip(), expanded=True):
                        st.markdown(content.strip())

            st.download_button("📥 レポートをダウンロード", report_text, "report.md")
        else:
            st.error("レポート生成に失敗しました。")
