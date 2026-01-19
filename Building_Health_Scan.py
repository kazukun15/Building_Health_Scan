# -*- coding: utf-8 -*-
"""
建物診断くん Pro (Gemini 2.5 Native Edition)
Author: Gemini API Specialist
Date: 2026-01-20
Description: Exclusive implementation using gemini-2.5-pro and gemini-2.5-flash
"""

import os
import io
import re
import math
import base64
import statistics
import unicodedata
import time
import pkgutil
from typing import List, Tuple, Dict, Any, Set, Optional

# --- Streamlit & Visualization ---
import streamlit as st
from streamlit_folium import st_folium
import folium

# --- Data & Processing ---
import requests
import PyPDF2
from PIL import Image, ImageOps

# --- Optional Dependencies (Back Camera) ---
HAVE_BACK_CAM = False
try:
    from streamlit_back_camera_input import back_camera_input
    HAVE_BACK_CAM = True
except ImportError:
    pass

# Patch for older environments
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter

# ===========================================================
# 1. Configuration & Constants
# ===========================================================

class Config:
    APP_TITLE = "建物診断くん Pro"
    APP_VERSION = "4.0.0 (Gemini 2.5 Native)"
    
    # UI Colors
    COLOR_PRIMARY = "#0066CC"
    COLOR_BG = "#F4F6F9"
    
    # System Limits
    MAX_IMAGES_TOTAL = 16
    
    # 【重要】Gemini 2.5系のみを使用 (1.5系/exp系は完全排除)
    PROFILES = {
        "🚀 高速モード (2.5 Flash)": {
            "max_img_dim": 1024,      # 2.5は処理能力が高いため解像度アップ
            "rag_top_k": 4,
            "img_quality": 80,
            "gemini_model": "gemini-2.5-flash"  # 最新高速モデル
        },
        "🛡️ 通常モード (2.5 Pro)": {
            "max_img_dim": 2048,      # 2.5 Proは高解像度入力に強い
            "rag_top_k": 8,           # コンテキストウィンドウ拡大に対応
            "img_quality": 95,
            "gemini_model": "gemini-2.5-pro"    # 最新最強モデル
        }
    }

    # PDF Sources (Mock/Placeholder)
    PDF_SOURCES = [
        ("Structure_Base.pdf", "Structure_Base.pdf"),
        ("上島町 公共施設等総合管理計画", "kamijimachou_Public_facility_management_plan.pdf"),
        ("港区 公共施設マネジメント計画", "minatoku_Public_facility_management_plan.pdf"),
    ]
    
    DOC_TYPES = {
        "Structure_Base.pdf": "基準",
        "kamijimachou_Public_facility_management_plan.pdf": "計画",
        "minatoku_Public_facility_management_plan.pdf": "計画",
    }

    # Diagnosis Dictionaries
    TEMPLATES = {
        "外壁（タイル）": {"mode": "strict"},
        "壁（仕上げ・躯体）": {"mode": "strict"},
        "屋上（防水シート）": {"mode": "maintenance"},
    }

    QUERY_SYNONYMS = {
        "ひび割れ": ["ひび割れ", "クラック", "亀裂", "ヘアクラック"],
        "浮き": ["浮き", "うき", "剥離予兆", "層間剥離"],
        "漏水": ["含水", "浸水", "漏水", "雨水侵入", "雨漏り", "エフロ"],
        "白蟻": ["白蟻", "シロアリ", "蟻害", "食害"],
    }

# ===========================================================
# 2. Utilities
# ===========================================================

class Utils:
    @staticmethod
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\r\n]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w一-龥ぁ-んァ-ン％℃\.]", " ", text)
        return text.split()

    @staticmethod
    def expand_query(query: str) -> List[str]:
        tokens = set(Utils.tokenize(query))
        for key, syns in Config.QUERY_SYNONYMS.items():
            if key in query or any(s in query for s in syns):
                for s in syns:
                    tokens.update(Utils.tokenize(s))
        return list(tokens)

    @staticmethod
    def fix_image_orientation(image: Image.Image) -> Image.Image:
        try:
            return ImageOps.exif_transpose(image)
        except Exception:
            return image

class ImageAnalyzer:
    @staticmethod
    def to_gemini_part(image: Image.Image, profile: Dict) -> Dict:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        max_dim = profile["max_img_dim"]
        quality = profile["img_quality"]

        if max(image.width, image.height) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.LANCZOS)
            
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality, optimize=True)
        return {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(buf.getvalue()).decode("utf-8")
            }
        }

# ===========================================================
# 3. RAG Engine
# ===========================================================

class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.doc_freqs = []
        self.idf = {}
        self.N = 0
        self.avgdl = 0

    def fit(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.N = len(docs)
        self.doc_freqs = []
        df_counter = {}
        total_len = 0
        for doc in docs:
            tokens = doc["_tokens"]
            length = len(tokens)
            total_len += length
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append((length, freqs))
            for t in freqs:
                df_counter[t] = df_counter.get(t, 0) + 1
        
        self.avgdl = total_len / self.N if self.N > 0 else 0
        for t, freq in df_counter.items():
            self.idf[t] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def search(self, query_tokens: List[str], top_k: int = 5) -> List[Tuple[float, int]]:
        scores = []
        for idx, (doc_len, freqs) in enumerate(self.doc_freqs):
            score = 0.0
            for t in query_tokens:
                if t in freqs:
                    tf = freqs[t]
                    idf = self.idf.get(t, 0)
                    num = tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avgdl or 1)))
                    score += idf * (num / den)
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

class RAGEngine:
    _instance = None
    def __init__(self):
        self.index = BM25Index()
        self.chunks = []
        self.ready = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance

    def load_documents(self):
        if self.ready: return
        chunks = []
        for filename, path in Config.PDF_SOURCES:
            # Mock behavior for missing files (Standard in 2026 production envs for robustness)
            if not os.path.exists(path):
                chunks.append({"text": f"{filename} (System: Virtual Document)", "doc": filename, "doc_type": "System", "page": 0, "_tokens": ["test"]})
                continue
            try:
                reader = PyPDF2.PdfReader(path)
                doc_type = Config.DOC_TYPES.get(filename, "その他")
                for i, page in enumerate(reader.pages):
                    t = Utils.normalize_text(page.extract_text() or "")
                    if len(t) < 10: continue
                    step, overlap = 800, 100
                    for s in range(0, len(t), step - overlap):
                        sub = t[s:s+step]
                        if len(sub) > 50:
                            chunks.append({
                                "text": sub, "doc": filename, "doc_type": doc_type, "page": i+1,
                                "_tokens": Utils.tokenize(sub),
                                "_has_num": bool(re.search(r"\d", sub))
                            })
            except Exception: pass
        self.chunks = chunks
        if chunks: self.index.fit(chunks)
        self.ready = True

    def query(self, text: str, filters: Dict, top_k: int) -> List[Dict]:
        if not self.ready: self.load_documents()
        tokens = Utils.expand_query(text)
        raw = self.index.search(tokens, top_k=top_k * 3)
        res = []
        for sc, idx in raw:
            doc = self.chunks[idx]
            if doc["doc_type"] == "基準": sc *= filters.get("base_boost", 1.0)
            res.append((sc, doc))
        res.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in res[:top_k]]

# ===========================================================
# 4. API Wrapper (Gemini 2.5 Native)
# ===========================================================

def call_gemini(api_key: str, prompt: str, rag_docs: List[Dict], images: Dict[str, List[Image.Image]], profile: Dict) -> str:
    # Gemini 2.5はコンテキストウィンドウが広いため、より詳細なプロンプト構築が可能
    rag_txt = "\n".join([f"[{d['doc']} p.{d['page']}] {d['text'][:800]}" for d in rag_docs]) if rag_docs else "(参照文献なし - 一般原則適用)"
    
    parts = [{"text": f"""
# SYSTEM ROLE
あなたは最新の「Gemini 2.5」を搭載した、国土交通省基準準拠の建築診断AIです。
ハルシネーション（幻覚）を厳格に排除し、事実と画像根拠に基づいた診断を行います。

# ユーザーの依頼
{prompt}

# 参照データ (RAG)
{rag_txt}

# 診断プロトコル
1. 安全性最優先: リスクの見落としは許されません。
2. 根拠明示: 判断の根拠となるRAGデータまたは画像特徴を具体的に指摘してください。
3. 推論深度: {profile['gemini_model']} の推論能力を最大限活かし、単なる表面観察ではなく、劣化メカニズム（中性化、凍害、疲労等）まで考察してください。
"""}]

    for k in ["vis", "ir", "micro"]:
        if images[k]:
            parts.append({"text": f"\n【{k.upper()}画像】"})
            for img in images[k]:
                parts.append(ImageAnalyzer.to_gemini_part(img, profile))

    # 使用モデルの強制 (2.5系のみ)
    model_name = profile["gemini_model"]
    if "2.5" not in model_name:
        return "System Error: Illegal model version detected. Only Gemini 2.5 is permitted."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    # 2.5系は高速ですが、堅牢性のためにリトライロジックは維持
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json={"contents": [{"parts": parts}]}, headers={"Content-Type": "application/json"}, timeout=120)
            
            if resp.status_code in [503, 429]:
                time.sleep(1 * (attempt + 1))
                continue
            
            if resp.status_code == 404:
                return f"API Error (404): モデル {model_name} が見つかりません。APIキーの権限を確認してください。"
                
            resp.raise_for_status()
            
            data = resp.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                content = data["candidates"][0].get("content")
                if content and "parts" in content:
                    return content["parts"][0]["text"]
            return "Error: Empty response from AI."

        except Exception as e:
            if attempt < max_retries - 1:
                continue
            raise e

    return "Server Error: Connection failed after retries."

# ===========================================================
# 5. UI Components
# ===========================================================

def load_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Noto Sans JP', sans-serif; background-color: {Config.COLOR_BG}; }}
    .app-card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
    .card-title {{ font-size: 1.1rem; font-weight: bold; color: {Config.COLOR_PRIMARY}; border-bottom: 2px solid #eee; padding-bottom: 5px; margin-bottom: 15px; }}
    </style>
    """, unsafe_allow_html=True)

# ===========================================================
# 6. Main App
# ===========================================================

def main():
    st.set_page_config(page_title=Config.APP_TITLE, layout="wide")
    load_css()
    
    if "images" not in st.session_state: st.session_state.images = {"vis": [], "ir": [], "micro": []}
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ 設定 (v4.0)")
        
        # モデル選択: 2.5系のみ表示
        mode_name = st.radio("エンジン選択", list(Config.PROFILES.keys()))
        current_profile = Config.PROFILES[mode_name]
        
        st.code(f"Core: {current_profile['gemini_model']}", language="text")
        
        api_key = st.text_input("Gemini API Key", type="password")
        if not api_key and "gemini" in st.secrets: api_key = st.secrets["gemini"]["API_KEY"]
        
        if api_key: st.success("System Ready")
        else: st.error("Key Required")

    st.markdown(f"### 🏗️ {Config.APP_TITLE} <span style='font-size:0.8em; color:gray'>Powered by Gemini 2.5</span>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📷 撮影・アップロード", "📋 診断実行", "📍 マップ"])

    # 1. Camera & Upload
    with tab1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">アウトカメラ撮影 (現場優先)</div>', unsafe_allow_html=True)
        
        img_cat = st.radio("種別", ["可視画像", "赤外線", "顕微鏡"], horizontal=True)
        target_key = "vis" if "可視" in img_cat else ("ir" if "赤外線" in img_cat else "micro")

        col_cam, col_file = st.columns(2)
        
        with col_cam:
            st.caption("📱 カメラ入力")
            cam_img = None
            if HAVE_BACK_CAM:
                cam_img_buffer = back_camera_input(key=f"back_cam_{target_key}")
                if cam_img_buffer: cam_img = Image.open(cam_img_buffer)
            else:
                cam_buffer = st.camera_input("カメラ起動", key=f"std_cam_{target_key}")
                if cam_buffer: cam_img = Image.open(cam_buffer)
            
            if cam_img:
                if st.button("追加", key="add_cam"):
                    cam_img = Utils.fix_image_orientation(cam_img)
                    st.session_state.images[target_key].append(cam_img)
                    st.toast("画像を追加しました")

        with col_file:
            st.caption("📂 ファイル参照")
            uploaded = st.file_uploader("選択", accept_multiple_files=True, key=f"up_{target_key}")
            if uploaded:
                for u in uploaded:
                    img = Image.open(u)
                    st.session_state.images[target_key].append(Utils.fix_image_orientation(img))
                st.toast(f"{len(uploaded)}枚追加")

        total_imgs = sum(len(v) for v in st.session_state.images.values())
        if total_imgs > 0:
            st.divider()
            if st.button("全削除"):
                st.session_state.images = {"vis": [], "ir": [], "micro": []}
                st.rerun()
            cols = st.columns(6)
            all_imgs = [img for k in ["vis", "ir", "micro"] for img in st.session_state.images[k]]
            for i, img in enumerate(all_imgs):
                cols[i % 6].image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Diagnosis
    with tab2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">🚀 診断開始 ({current_profile["gemini_model"]})</div>', unsafe_allow_html=True)
        
        template = st.selectbox("対象部位", list(Config.TEMPLATES.keys()))
        condition = st.text_area("特記事項・症状", height=80, placeholder="例：3階バルコニー下端にクラックあり。")
        
        if st.button("AI診断を実行", type="primary", use_container_width=True):
            if not api_key:
                st.error("API Keyがありません")
            elif total_imgs == 0 and not condition.strip():
                 st.warning("画像または症状を入力してください。")
            else:
                rag = RAGEngine.get_instance()
                with st.spinner(f"{current_profile['gemini_model']} で解析中..."):
                    rag_res = rag.query(f"{template} {condition}", {"base_boost": 1.5}, top_k=current_profile["rag_top_k"])
                    try:
                        res = call_gemini(api_key, f"部位:{template} 状況:{condition}", rag_res, st.session_state.images, current_profile)
                        st.session_state["result"] = res
                        st.session_state["refs"] = rag_res
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        if "result" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["result"])
            with st.expander("Evidence (RAG Source)"):
                for r in st.session_state.get("refs", []):
                    st.caption(f"[{r['doc']}] {r['text'][:100]}...")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Map
    with tab3:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        m = folium.Map(location=[35.6895, 139.6917], zoom_start=16)
        st_folium(m, height=300, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
