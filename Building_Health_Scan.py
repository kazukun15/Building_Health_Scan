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

# ---------------
