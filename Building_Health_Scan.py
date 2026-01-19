# -*- coding: utf-8 -*-
"""
建物診断くん Pro (Speed & Field Optimized Edition)
Author: World Class Program Designer
Date: 2026-01-20
Description: RAG-based Building Diagnosis System with Speed Modes & Back Camera Support
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
from datetime import date
from typing import List, Tuple, Dict, Any, Set, Optional

# --- Streamlit & Visualization ---
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import folium

# --- Data & Processing ---
import requests
import PyPDF2
from PIL import Image, ImageFilter, ImageOps

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
    APP_VERSION = "3.1.0 Speed & Field"
    
    # UI Colors
    COLOR_PRIMARY = "#0066CC"
    COLOR_BG = "#F4F6F9"
    
    # System Limits
    MAX_IMAGES_TOTAL = 16
    
    # Performance Profiles
    PROFILES = {
        "🚀 高速モード": {
            "max_img_dim": 800,      # 画像を小さくリサイズ
            "rag_top_k": 3,          # 文献検索数を減らす
            "img_quality": 70,       # JPEG圧縮率
            "gemini_model": "gemini-2.5-flash" 
        },
        "🛡️ 通常モード (精密)": {
            "max_img_dim": 1600,     # 詳細な解析用
            "rag_top_k": 6,          # 多くの文献を参照
            "img_quality": 90,
            "gemini_model": "gemini-2.5-flash"
        }
    }

    # PDF Sources (Placeholder names)
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
        "ひび割れ": ["ひび割れ", "クラック", "亀裂"],
        "浮き": ["浮き", "うき", "剥離予兆"],
        "漏水": ["含水", "浸水", "漏水", "雨水侵入", "雨漏り"],
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
        """Convert image to Gemini payload with profile-based optimization"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        max_dim = profile["max_img_dim"]
        quality = profile["img_quality"]

        # Resize logic
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
            if not os.path.exists(path):
                # Dummy data for demo if file missing
                chunks.append({"text": f"{filename} (File missing)", "doc": filename, "doc_type": "System", "page": 0, "_tokens": ["test"]})
                continue
            try:
                reader = PyPDF2.PdfReader(path)
                doc_type = Config.DOC_TYPES.get(filename, "その他")
                for i, page in enumerate(reader.pages):
                    t = Utils.normalize_text(page.extract_text() or "")
                    if len(t) < 10: continue
                    # Chunking
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
        # Get more candidates then filter
        raw = self.index.search(tokens, top_k=top_k * 3)
        res = []
        for sc, idx in raw:
            doc = self.chunks[idx]
            # Boost logic
            if doc["doc_type"] == "基準": sc *= filters.get("base_boost", 1.0)
            if doc["_has_num"]: sc *= filters.get("num_boost", 1.0)
            res.append((sc, doc))
        res.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in res[:top_k]]

# ===========================================================
# 4. API Wrapper
# ===========================================================

def call_gemini(api_key: str, prompt: str, rag_docs: List[Dict], images: Dict[str, List[Image.Image]], profile: Dict) -> str:
    # Build RAG Text
    rag_txt = "\n".join([f"[{d['doc']} p.{d['page']}] {d['text'][:600]}" for d in rag_docs]) if rag_docs else "(参照文献なし)"
    
    parts = [{"text": f"""
# あなたの役割
国土交通省基準に準拠した建築診断AI。現場の診断士を支援する。

# 依頼内容
{prompt}

# 参照データ (RAG)
{rag_txt}

# 指示
- 危険性がある場合は「要詳細調査」等の安全側のアクションを推奨。
- 参照データがある場合は必ず引用する。
- 結論を先に述べ、その後に理由を箇条書きで簡潔に。
"""}]

    # Process Images with profile settings
    for k in ["vis", "ir", "micro"]:
        if images[k]:
            parts.append({"text": f"\n【{k.upper()}画像】"})
            for img in images[k]:
                parts.append(ImageAnalyzer.to_gemini_part(img, profile))

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{profile['gemini_model']}:generateContent?key={api_key}"
    resp = requests.post(url, json={"contents": [{"parts": parts}]}, headers={"Content-Type": "application/json"}, timeout=120)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

# ===========================================================
# 5. UI Components
# ===========================================================

def load_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Noto Sans JP', sans-serif; background-color: {Config.COLOR_BG}; }}
    .app-card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    .card-title {{ font-size: 1.1rem; font-weight: bold; color: {Config.COLOR_PRIMARY}; border-bottom: 2px solid #eee; padding-bottom: 5px; margin-bottom: 15px; }}
    /* Radio Button as Tabs */
    div[role="radiogroup"] > label {{ background: white; padding: 10px 20px; border-radius: 8px; border: 1px solid #ddd; margin-right: 10px; }}
    div[role="radiogroup"] > label[data-checked="true"] {{ background: {Config.COLOR_PRIMARY} !important; color: white !important; border: none; }}
    </style>
    """, unsafe_allow_html=True)

# ===========================================================
# 6. Main App
# ===========================================================

def main():
    st.set_page_config(page_title=Config.APP_TITLE, layout="wide")
    load_css()
    
    if "images" not in st.session_state: st.session_state.images = {"vis": [], "ir": [], "micro": []}
    
    # --- Sidebar (Settings & Mode) ---
    with st.sidebar:
        st.markdown("### ⚙️ 設定")
        
        # Mode Selection (Highlighted)
        st.markdown("##### 動作モード")
        mode_name = st.radio("Mode", list(Config.PROFILES.keys()), label_visibility="collapsed")
        current_profile = Config.PROFILES[mode_name]
        
        st.caption(f"画像サイズ: {current_profile['max_img_dim']}px | 参照数: {current_profile['rag_top_k']}")
        
        api_key = st.text_input("Gemini API Key", type="password")
        if not api_key and "gemini" in st.secrets: api_key = st.secrets["gemini"]["API_KEY"]
        
        if api_key: st.success("API Ready")
        else: st.error("API Key Missing")

    st.markdown(f"### 🏗️ {Config.APP_TITLE} <span style='font-size:0.8em; color:gray'>Ver {Config.APP_VERSION}</span>", unsafe_allow_html=True)

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["📷 撮影・アップロード", "📋 診断実行", "📍 マップ"])

    # 1. Camera & Upload
    with tab1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">アウトカメラ撮影 (現場優先)</div>', unsafe_allow_html=True)
        
        img_cat = st.radio("種別", ["可視画像", "赤外線", "顕微鏡"], horizontal=True)
        target_key = "vis" if "可視" in img_cat else ("ir" if "赤外線" in img_cat else "micro")

        col_cam, col_file = st.columns(2)
        
        # Camera Section - Priority on Back Camera
        with col_cam:
            st.caption("📱 カメラ入力")
            cam_img = None
            
            if HAVE_BACK_CAM:
                # Use standard camera input label but use back_camera_input function
                # This ensures back camera is tried first
                cam_img_buffer = back_camera_input(key=f"back_cam_{target_key}")
                if cam_img_buffer:
                    cam_img = Image.open(cam_img_buffer)
            else:
                st.warning("バックカメラライブラリが見つかりません。標準カメラを使用します。")
                cam_buffer = st.camera_input("カメラ起動", key=f"std_cam_{target_key}")
                if cam_buffer:
                    cam_img = Image.open(cam_buffer)
            
            if cam_img:
                if st.button("この画像を追加", key="add_cam"):
                    cam_img = Utils.fix_image_orientation(cam_img)
                    st.session_state.images[target_key].append(cam_img)
                    st.toast("画像を追加しました")

        with col_file:
            st.caption("📂 ファイル参照")
            uploaded = st.file_uploader("フォルダから選択", accept_multiple_files=True, key=f"up_{target_key}")
            if uploaded:
                for u in uploaded:
                    img = Image.open(u)
                    st.session_state.images[target_key].append(Utils.fix_image_orientation(img))
                st.toast(f"{len(uploaded)}枚追加しました")

        # Gallery
        total_imgs = sum(len(v) for v in st.session_state.images.values())
        if total_imgs > 0:
            st.divider()
            st.caption(f"現在の画像 ({total_imgs}枚)")
            if st.button("全削除", key="clear_all"):
                st.session_state.images = {"vis": [], "ir": [], "micro": []}
                st.rerun()
            
            cols = st.columns(6)
            all_flat = []
            for k in ["vis", "ir", "micro"]:
                for img in st.session_state.images[k]: all_flat.append(img)
            
            for i, img in enumerate(all_flat):
                cols[i % 6].image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Diagnosis
    with tab2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">🚀 診断開始 ({mode_name})</div>', unsafe_allow_html=True)
        
        template = st.selectbox("対象部位", list(Config.TEMPLATES.keys()))
        condition = st.text_area("特記事項・症状", height=80, placeholder="例：3階バルコニー下端にクラックあり。")
        
        if st.button("AI診断を実行", type="primary", use_container_width=True):
            if not api_key:
                st.error("API Keyがありません")
            elif total_imgs == 0:
                st.warning("画像がありません")
            else:
                rag = RAGEngine.get_instance()
                with st.spinner(f"{mode_name}で解析中... 画像圧縮とRAG検索を行っています"):
                    # RAG
                    rag_res = rag.query(f"{template} {condition}", {"base_boost": 1.5}, top_k=current_profile["rag_top_k"])
                    
                    # Generate
                    try:
                        res = call_gemini(api_key, f"部位:{template} 状況:{condition}", rag_res, st.session_state.images, current_profile)
                        st.session_state["result"] = res
                        st.session_state["refs"] = rag_res
                    except Exception as e:
                        st.error(f"Error: {e}")

        if "result" in st.session_state:
            st.markdown("---")
            st.markdown(st.session_state["result"])
            with st.expander("参照文献リスト"):
                for r in st.session_state["refs"]:
                    st.caption(f"[{r['doc']}] {r['text'][:100]}...")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Map (Simple)
    with tab3:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        m = folium.Map(location=[35.6895, 139.6917], zoom_start=16)
        st_folium(m, height=300, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
