import streamlit as st
import requests
import faiss
import numpy as np
import PyPDF2
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------------------
# PDFからのテキスト抽出と前処理
# -------------------------------
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

# -------------------------------
# 画像キャプション生成
# -------------------------------
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
    prompt = (
        "Describe in detail the current state of the wall and floor in the image, "
        "including cracks, stains, bulges, peeling, discoloration, or any notable damage, "
        "as if you are an experienced building inspector in Japan. "
        "Output in Japanese."
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
        return {"error": "Gemini API Keyが .streamlit/secrets.toml に設定されていません。"}
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash-preview-image-generation:generateContent?key=" + api_key
    )
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
            "上記の情報を基に、壁と床の状態を詳細に分析し、劣化度（A～D）、推定寿命、必要な対策、注意点を含むレポートを日本語で作成してください。"
        )

        with st.spinner("Gemini APIでレポートを生成中..."):
            result = generate_report_with_gemini(prompt)
            if "error" in result:
                st.error(f"API呼び出しエラー: {result['error']}")
            else:
                try:
                    report_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    st.markdown("## 生成されたレポート")
                    st.markdown(report_text)
                    st.download_button("レポートをダウンロード", report_text, file_name="report.txt", mime="text/plain")
                except Exception as e:
                    st.error("レポート抽出中にエラーが発生しました: " + str(e))
