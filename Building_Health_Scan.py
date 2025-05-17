import streamlit as st
import requests
import faiss
import numpy as np
import PyPDF2
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------------------
# 定数定義
# -------------------------------
PDF_PATHS = [
    "Structure_Base.pdf",
    "kamijimachou_Public_facility_management_plan.pdf",
    "minatoku_Public_facility_management_plan.pdf"
]
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

# -------------------------------
# PDFからのテキスト抽出と前処理
# -------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if reader.is_encrypted:
                st.warning(f"PDFファイル '{pdf_path}' は暗号化されており、読み取れない可能性があります。")
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except FileNotFoundError:
        st.error(f"PDFファイルが見つかりません: {pdf_path}")
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"PDFファイルの読み取りエラー ({pdf_path}): {e}")
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
    combined_text = ""
    for path in PDF_PATHS:
        combined_text += extract_text_from_pdf(path) + "\n"
    chunks = split_text_into_chunks(combined_text)
    chunks = [chunk for chunk in chunks if chunk.strip()]
    vec_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = create_vector_index(chunks, vec_model)
    return index, chunks, vec_model

def search_relevant_chunks(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# -------------------------------
# 画像キャプション生成
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_caption_model():
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
    return processor, model

def generate_image_caption(image, processor, model):
    max_width = 800
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    prompt = (
        "画像内の壁と床の現在の状態を詳細に説明してください。"
        "ひび割れ、汚れ、膨らみ、剥がれ、変色、その他の目立つ損傷などを含めて、"
        "日本の経験豊富な建物検査官のように記述してください。"
        "出力は日本語でお願いします。"
    )
    inputs = processor([image], text=prompt, return_tensors="pt", padding=True)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# -------------------------------
# Gemini API 呼び出し
# -------------------------------
def generate_report_with_gemini(prompt_text):
    try:
        api_key = st.secrets["gemini"]["API_KEY"]
    except KeyError:
        st.error("Gemini API Keyが .streamlit/secrets.toml に設定されていません。")
        return None
    endpoint = f"{GEMINI_API_ENDPOINT}?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"APIリクエストエラー: {e}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")
        return {"error": str(e)}

# -------------------------------
# Streamlit アプリメイン
# -------------------------------
st.set_page_config(page_title="壁・床状態分析ツール", layout="wide")
st.title("壁・床の状態分析レポート生成ツール")

st.markdown(
    """
    このアプリは、建物の壁や床の状態をAIが解析し、詳細なレポートを生成します。
    画像をアップロードし、必要に応じて壁や床に関する補足情報を入力してください。
    """
)

# 画像入力
st.markdown("### 画像入力")
image_file = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
image = None
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

# 補足情報入力
st.markdown("### 補足情報入力")
wall_note = st.text_area("壁に関する補足情報（例：ひび割れ、変色、剥がれなど）")
floor_note = st.text_area("床に関する補足情報（例：きしみ、沈み、汚れなど）")
user_query = st.text_input("特に知りたいことや質問を入力してください")

# レポート生成
if st.button("レポート生成"):
    if image is None:
        st.error("画像をアップロードしてください。")
    else:
        processor, cap_model = load_caption_model()
        with st.spinner("画像キャプションを生成中..."):
            try:
                caption = generate_image_caption(image, processor, cap_model)
            except Exception as e:
                caption = f"キャプション生成エラー: {e}"
        st.write("生成された画像キャプション:")
        st.write(caption)

        with st.spinner("PDFから関連情報を抽出中..."):
            index, chunks, vec_model = load_pdf_index()
            relevant_chunks = search_relevant_chunks(user_query, vec_model, index, chunks)
            context = "\n\n".join(relevant_chunks)

        # プロンプト作成
        prompt = (
            f"ユーザーの質問: {user_query}\n\n"
            f"以下はPDFからの関連情報:\n{context}\n\n"
            f"画像キャプション:\n{caption}\n\n"
            f"壁の補足情報:\n{wall_note}\n\n"
            f"床の補足情報:\n{floor_note}\n\n"
            "上記の情報を基に、壁と床の状態を詳細に分析し、以下のMarkdown形式でレポートを作成してください:\n"
            "## 1. 壁の状態分析\n"
            "### 1.1. 現状\n"
            "(ここに壁の現状を記述)\n"
            "### 1.2. 劣化度\n"
            "評価: (A～Dで評価)\n"
            "### 1.3. 推定寿命\n"
            "(ここに推定寿命を記述)\n"
            "### 1.4. 必要な対策\n"
            "(ここに具体的な対策を記述)\n"
            "### 1.5. 注意点\n"
            "(ここに注意点を記述)\n"
            "## 2. 床の状態分析\n"
            "### 2.1. 現状\n"
            "(ここに床の現状を記述)\n"
            "### 2.2. 劣化度\n"
            "評価:
::contentReference[oaicite:17]{index=17}
 
