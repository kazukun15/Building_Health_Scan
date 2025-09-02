# -*- coding: utf-8 -*-
# ===========================================================
# 建物診断くん（スマホ対応・マテリアルデザイン風UI・日本語フォント適用）
# - タイトルの視認性改善（ヒーローヘッダー＋上部余白/行間/overflow調整）
# - 可視/赤外 画像：アップロード + カメラ
# - EXIFからGPS抽出 → Folium地図に表示（手入力も可）
# - 3PDFをRAG（軽量スコア）→ Gemini 2.0 Flash で詳細分析
# - 結果のみ表示：総合評価カードを先頭に、詳細は展開
# - レポートDL（共有）
# Python 3.12 互換／重依存なし（軽量）
# ===========================================================

# Python 3.12: pkgutil.ImpImporter削除回避（古い依存のための保険）
import pkgutil
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter

import os
import io
import re
import base64
import unicodedata
from typing import List, Tuple, Dict, Optional

import streamlit as st
import requests
import PyPDF2
from PIL import Image

# 位置情報・地図
from exif import Image as ExifImage    # pip install exif
import folium                           # pip install folium
from streamlit_folium import st_folium  # pip install streamlit-folium

# -----------------------------------------------------------
# 定数
# -----------------------------------------------------------
APP_TITLE = "建物診断くん"
PDF_SOURCES = [
    ("Structure_Base.pdf", "Structure_Base.pdf"),
    ("上島町 公共施設等総合管理計画", "kamijimachou_Public_facility_management_plan.pdf"),
    ("港区 公共施設マネジメント計画", "minatoku_Public_facility_management_plan.pdf"),
]

# -----------------------------------------------------------
# ユーティリティ
# -----------------------------------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        return ""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                t = p.extract_text() or ""
                text += t + "\n"
    except Exception as e:
        text += f"\n[PDF読込エラー:{pdf_path}:{e}]"
    return normalize_text(text)

def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    chunks, i, N = [], 0, len(text)
    while i < N:
        end = min(i + max_chars, N)
        ch = text[i:end].strip()
        if ch:
            chunks.append(ch)
        i = end - overlap if end - overlap > i else end
    return chunks

def tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9一-龥ぁ-んァ-ン_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def simple_retrieval_score(query: str, chunk: str) -> float:
    q_tokens = set(tokenize(query))
    c_tokens = tokenize(chunk)
    if not q_tokens or not c_tokens:
        return 0.0
    score = 0.0
    for qt in q_tokens:
        score += c_tokens.count(qt) * 1.0
        if qt in c_tokens:
            score += 0.5
    # 冗長ペナルティ（長すぎる塊を弱める）
    score = score / (1.0 + len(chunk) / 2000.0)
    return score

def topk_chunks(query: str, corpus: List[Tuple[str, str]], k: int = 4) -> List[Tuple[str, str]]:
    scored = [(simple_retrieval_score(query, ch), src, ch) for (src, ch) in corpus]
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [(src, ch) for (s, src, ch) in scored[:k] if s > 0]
    if not out and corpus:
        out = [corpus[0]]  # 保険
    return out

def image_to_inline_part(image: Image.Image, max_width: int = 1400) -> Dict:
    # GeminiにBase64で送る（JPEG固定）
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.width > max_width:
        r = max_width / float(image.width)
        image = image.resize((max_width, int(image.height * r)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/jpeg", "data": b64}}

# -----------------------------------------------------------
# EXIF → GPS 抽出（緯度経度を10進度に）
# -----------------------------------------------------------
def _to_deg(value) -> float:
    # EXIFの度分秒[(num,den)...]→ 10進度に変換
    try:
        d = float(value[0].numerator) / float(value[0].denominator)
        m = float(value[1].numerator) / float(value[1].denominator)
        s = float(value[2].numerator) / float(value[2].denominator)
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        try:
            d, m, s = value
            return float(d) + float(m) / 60.0 + float(s) / 3600.0
        except Exception:
            return None

def extract_gps_from_image(uploaded_bytes: bytes) -> Optional[Tuple[float, float]]:
    try:
        exif = ExifImage(uploaded_bytes)
    except Exception:
        return None
    if not exif.has_exif:
        return None
    if not (hasattr(exif, "gps_latitude") and hasattr(exif, "gps_longitude")
            and hasattr(exif, "gps_latitude_ref") and hasattr(exif, "gps_longitude_ref")):
        return None
    lat = _to_deg(exif.gps_latitude)
    lon = _to_deg(exif.gps_longitude)
    if lat is None or lon is None:
        return None
    if exif.gps_latitude_ref in ["S", "s"]:
        lat = -lat
    if exif.gps_longitude_ref in ["W", "w"]:
        lon = -lon
    return (lat, lon)

# -----------------------------------------------------------
# Gemini 呼び出し
# -----------------------------------------------------------
def call_gemini(api_key: str, prompt_text: str, image_parts: List[Dict]) -> Dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-1.5:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    parts = [{"text": prompt_text}]
    parts.extend(image_parts)
    payload = {"contents": [{"parts": parts}]}
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()

def extract_text_from_gemini(result: Dict) -> str:
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""

# -----------------------------------------------------------
# マスタープロンプト（“結果のみ”・総合評価先頭）
# -----------------------------------------------------------
def build_master_prompt(user_q: str,
                        rag_snippets: List[Tuple[str, str]],
                        ir_meta: Dict) -> str:
    rag_block = [f"[{src}] {txt}" for (src, txt) in rag_snippets]
    rag_text = "\n".join(rag_block)

    ir_note = ""
    if ir_meta.get("has_ir"):
        ir_note = (
            "注: 添付には赤外線画像が含まれます。温度分布・熱パターン・潜在的含水や断熱不良等の可能性に留意し、"
            "IRメタ（ε/反射温度/外気温/湿度/距離/角度/レンジ）不足時は信頼度を下げてコメントしてください。\n"
            f"IRメタ: ε={ir_meta.get('emissivity','不明')}, T_ref={ir_meta.get('t_ref','不明')}℃, "
            f"T_amb={ir_meta.get('t_amb','不明')}℃, RH={ir_meta.get('rh','不明')}%, "
            f"距離={ir_meta.get('dist','不明')}m, 角度={ir_meta.get('angle','不明')}°.\n"
        )

    prompt = f"""
あなたは非破壊検査・建築・材料学の上級診断士。国土交通省（MLIT）基準に整合し、与えられた根拠のみで簡潔かつ正確に日本語レポートを作成せよ。
ハルシネーションは禁止。数値閾値や定義はRAG抜粋にある場合のみ使用。無い場合は「未掲載」と明示する。

# 入力
- ユーザー質問: {user_q}
- RAG抜粋（出所付き、これ以外は根拠にしない）:
{rag_text}

{ir_note}
# 出力仕様（“結果のみ”のMarkdown）
1. **総合評価**
   - グレード（A/B/C/D）と主因（1–2行）
   - 推定残存寿命（幅、例：7–12年）／（任意）補修後の期待寿命
2. **主要根拠（可視/IR/NDT）** # 画像から読み取れる事実に限定
3. **MLIT基準との関係** # RAG内の文書名＋節/見出しで簡潔に
4. **推奨対応（優先度順）**
5. **限界と追加データ要望** # 精度向上に必要な追補情報

注意: 画像から直接測れない値（ひび幅等）は断定しない。IRメタ不足時は信頼度を下げる旨を明記。
""".strip()
    return normalize_text(prompt)

# -----------------------------------------------------------
# RAG（3PDFを統合して軽量Top-K抽出）
# -----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_rag_corpus() -> List[Tuple[str, str]]:
    corpus: List[Tuple[str, str]] = []
    for name, path in PDF_SOURCES:
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text, max_chars=900, overlap=120)
        for ch in chunks:
            if len(ch) >= 40:
                corpus.append((name, ch))
    return corpus

# -----------------------------------------------------------
# UI: マテリアルデザイン風 + 日本語フォント + タイトル改善CSS
# -----------------------------------------------------------
def inject_material_css():
    st.markdown("""
    <style>
    /* =================================================================== */
    /* 建物診断くん: Material Design 3 スタイルシート                  */
    /* - モバイルファースト / ライト・ダーク対応 / Streamlit最適化      */
    /* =================================================================== */

    /* 1. フォント読み込み (Google Fonts) */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* 2. デザイントークン定義 (CSS変数) */
    :root {
      /* Color Palette (Light Theme) */
      --md-sys-color-primary: #005AC1;
      --md-sys-color-on-primary: #FFFFFF;
      --md-sys-color-primary-container: #D8E2FF;
      --md-sys-color-on-primary-container: #001A41;
      --md-sys-color-secondary: #575E71;
      --md-sys-color-on-secondary: #FFFFFF;
      --md-sys-color-secondary-container: #DBE2F9;
      --md-sys-color-on-secondary-container: #141B2C;
      --md-sys-color-background: #F8F9FC;
      --md-sys-color-on-background: #1A1C1E;
      --md-sys-color-surface: #FCFCFF;
      --md-sys-color-on-surface: #1A1C1E;
      --md-sys-color-surface-variant: #E1E2EC;
      --md-sys-color-on-surface-variant: #44474F;
      --md-sys-color-outline: #74777F;
      --md-sys-color-outline-variant: #C4C6D0;
      --md-sys-color-error: #BA1A1A;
      --md-sys-color-on-error: #FFFFFF;

      /* Typography */
      --md-sys-font-family: 'Noto Sans JP', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', 'Meiryo', sans-serif;
      --md-sys-line-height: 1.7;
      --md-sys-letter-spacing: 0.01em;

      /* Shape & Elevation */
      --md-sys-shape-corner-medium: 12px;
      --md-sys-shape-corner-large: 16px;
      --md-sys-elevation-1: 0 1px 2px rgba(0,0,0,.08), 0 1px 3px rgba(0,0,0,.04);
      --md-sys-elevation-2: 0 2px 6px rgba(0,0,0,.08), 0 2px 8px rgba(0,0,0,.04);
    }

    /* Dark Theme Variables */
    @media (prefers-color-scheme: dark) {
      :root {
        --md-sys-color-primary: #ADC6FF;
        --md-sys-color-on-primary: #002E69;
        --md-sys-color-primary-container: #004494;
        --md-sys-color-on-primary-container: #D8E2FF;
        --md-sys-color-secondary: #BFC6DC;
        --md-sys-color-on-secondary: #293041;
        --md-sys-color-secondary-container: #3F4759;
        --md-sys-color-on-secondary-container: #DBE2F9;
        --md-sys-color-background: #1A1C1E;
        --md-sys-color-on-background: #E3E2E6;
        --md-sys-color-surface: #222428;
        --md-sys-color-on-surface: #E3E2E6;
        --md-sys-color-surface-variant: #44474F;
        --md-sys-color-on-surface-variant: #C4C6D0;
        --md-sys-color-outline: #8E9099;
        --md-sys-color-outline-variant: #44474F;
        --md-sys-color-error: #FFB4AB;
        --md-sys-color-on-error: #690005;
      }
    }

    /* 3. グローバル & Streamlit基本要素の調整 */
    body {
      background-color: var(--md-sys-color-background);
      color: var(--md-sys-color-on-background);
      font-family: var(--md-sys-font-family);
      overflow-x: hidden;
    }
    .block-container {
      padding: 1rem 1rem 3rem 1rem !important;
      overflow: visible !important;
    }
    header[data-testid="stHeader"], footer[data-testid="stFooter"] {
      display: none !important;
    }
    h1, h2, h3, h4, h5, h6 {
      font-family: var(--md-sys-font-family);
      font-weight: 700;
      color: var(--md-sys-color-on-surface);
    }
    h4 {
      margin-top: 1.5rem;
      padding-bottom: 0.3rem;
      border-bottom: 1px solid var(--md-sys-color-outline-variant);
    }
    p, li {
       font-family: var(--md-sys-font-family) !important;
       line-height: var(--md-sys-line-height);
       letter-spacing: var(--md-sys-letter-spacing);
    }

    /* 4. ヒーローヘッダー */
    .app-hero {
      background: var(--md-sys-color-primary-container);
      color: var(--md-sys-color-on-primary-container);
      border-radius: var(--md-sys-shape-corner-large);
      padding: 1.5rem 1rem;
      margin: -1rem -1rem 1.5rem -1rem;
      display: flex;
      gap: 12px;
      align-items: flex-start;
    }
    .app-hero-icon {
      font-family: 'Material Symbols Outlined';
      font-size: 2.5rem;
      font-variation-settings: 'FILL' 1;
      margin-top: 2px;
    }
    .app-hero-text { flex: 1; }
    .app-hero-title {
      font-size: 1.6rem;
      font-weight: 700;
      line-height: 1.3;
      margin: 0 0 4px 0;
      word-break: keep-all;
      overflow-wrap: anywhere;
    }
    .app-hero-sub {
      font-size: 0.9rem;
      font-weight: 400;
      line-height: 1.5;
      opacity: 0.9;
      margin: 0;
    }

    /* 5. カードUI (.md-card) */
    .md-card {
      background-color: var(--md-sys-color-surface);
      border: 1px solid var(--md-sys-color-outline-variant);
      border-radius: var(--md-sys-shape-corner-large);
      box-shadow: var(--md-sys-elevation-1);
      padding: 1rem 1.25rem;
      margin-bottom: 1rem;
      transition: all 0.2s ease-in-out;
    }
    .good-shadow { box-shadow: var(--md-sys-elevation-2); }
    .md-title {
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--md-sys-color-on-surface-variant);
      margin: 0.2rem 0 0.8rem 0;
    }
    .md-sub {
      color: var(--md-sys-color-on-surface-variant);
      font-size: 0.9rem;
    }

    /* 6. StreamlitコンポーネントのMaterial化 */
    div[data-testid="stTextInput"] > label, div[data-testid="stFileUploader"] > label {
        display: none; /* Hide default labels, use markdown headers */
    }
    div[data-testid="stTextInput"] input, div[data-testid="stTextInput"] textarea {
      background-color: var(--md-sys-color-surface-variant);
      border: 1px solid var(--md-sys-color-outline-variant);
      min-height: 48px;
      padding: 12px;
      border-radius: var(--md-sys-shape-corner-medium);
      transition: all 0.2s ease;
      color: var(--md-sys-color-on-surface-variant);
    }
    div[data-testid="stTextInput"] input:focus, div[data-testid="stTextInput"] textarea:focus {
      border-color: var(--md-sys-color-primary);
      box-shadow: 0 0 0 2px var(--md-sys-color-primary);
    }
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] a {
      min-height: 48px;
      font-family: var(--md-sys-font-family);
      font-weight: 700;
      border: none !important;
      border-radius: var(--md-sys-shape-corner-large) !important;
      background: var(--md-sys-color-primary) !important;
      color: var(--md-sys-color-on-primary) !important;
      box-shadow: var(--md-sys-elevation-1);
      transition: all 0.2s ease-in-out;
      width: 100%;
    }
    div[data-testid="stButton"] button:hover, div[data-testid="stDownloadButton"] a:hover {
      box-shadow: var(--md-sys-elevation-2);
      filter: brightness(1.1);
    }
    div[data-testid="stButton"] button:focus-visible, div[data-testid="stDownloadButton"] a:focus-visible {
      outline: 2px solid var(--md-sys-color-primary);
      outline-offset: 2px;
    }
    div[data-testid="stButton"] button p::before {
      font-family: 'Material Symbols Outlined';
      content: 'science';
      vertical-align: middle;
      margin-right: 8px;
      font-size: 1.2em;
      font-variation-settings: 'WGHT' 500;
    }

    /* Expander (アコーディオン) */
    div[data-testid="stExpander"] {
      border: 1px solid var(--md-sys-color-outline-variant) !important;
      border-radius: var(--md-sys-shape-corner-large) !important;
      box-shadow: none !important;
      background-color: var(--md-sys-color-surface);
    }

    /* 7. レポート出力 (.jp-report) */
    .jp-report {
      font-family: var(--md-sys-font-family) !important;
      line-height: var(--md-sys-line-height);
      color: var(--md-sys-color-on-surface);
    }
    .jp-report h1, .jp-report h2, .jp-report h3, .jp-report h4, .jp-report h5, .jp-report h6 {
      font-family: var(--md-sys-font-family) !important;
      font-weight: 700;
      border: none;
      padding-bottom: 0.2rem;
      margin-top: 1.5rem;
    }
    .jp-report h2, .jp-report h3 {
      margin-top: 1rem;
      font-size: 1.15rem;
      color: var(--md-sys-color-primary);
    }
    .jp-report .md-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 1.3rem;
      color: var(--md-sys-color-primary);
    }
    .jp-report .md-title::before {
      font-family: 'Material Symbols Outlined';
      content: 'summarize';
      font-size: 1.5em;
      font-variation-settings: 'FILL' 1;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------
# メイン
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_material_css()

    # --- タイトル（ヒーローヘッダー） ---
    st.markdown(
        f"""
        <div class="app-hero">
            <span class="app-hero-icon material-symbols-outlined">construction</span>
            <div class="app-hero-text">
                <div class="app-hero-title">{APP_TITLE}</div>
                <div class="app-hero-sub">スマホ最適化UI | 画像(可視/赤外)と参考資料(RAG)でAIが詳細分析</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- 1) 質問 ---
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown("#### 1) 質問")
    user_q = st.text_input("質問", placeholder="分析したいテーマ・質問を入力", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 2) 画像入力 ---
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown("#### 2) 画像入力（可視/赤外）")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**可視画像**")
        vis_file = st.file_uploader("可視画像を選択", type=["jpg", "jpeg", "png"], key="vis_up", label_visibility="collapsed")
        vis_cam = st.camera_input("カメラで撮影", key="vis_cam", label_visibility="collapsed")
    with colB:
        st.markdown("**赤外線（IR）画像（任意）**")
        ir_file = st.file_uploader("IR画像を選択", type=["jpg", "jpeg", "png"], key="ir_up", label_visibility="collapsed")

    with st.expander("IRメタデータ（任意・精度向上）"):
        ir_emiss = st.text_input("放射率 ε", "0.95")
        ir_tref  = st.text_input("反射温度 T_ref [℃]", "20")
        ir_tamb  = st.text_input("外気温 T_amb [℃]", "22")
        ir_rh    = st.text_input("相対湿度 RH [%]", "65")
        ir_dist  = st.text_input("撮影距離 [m]", "5")
        ir_ang   = st.text_input("撮影角度 [°]", "10")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 画像ハンドリング
    vis_img, vis_src_bytes = None, None
    if vis_cam:
        vis_img, vis_src_bytes = Image.open(vis_cam), vis_cam.getvalue()
    elif vis_file:
        vis_img, vis_src_bytes = Image.open(vis_file), vis_file.getvalue()

    ir_img, ir_src_bytes = None, None
    if ir_file:
        ir_img, ir_src_bytes = Image.open(ir_file), ir_file.getvalue()

    # プレビュー
    if vis_img or ir_img:
        st.markdown('<div class="md-card">', unsafe_allow_html=True)
        pv_col1, pv_col2 = st.columns(2)
        with pv_col1:
            if vis_img: st.image(vis_img, caption="可視画像", use_container_width=True)
        with pv_col2:
            if ir_img: st.image(ir_img, caption="IR画像", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- 3) 位置情報 ---
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown("#### 3) 位置情報（自動 or 手入力）")
    lat, lon = None, None
    if vis_src_bytes:
        gps = extract_gps_from_image(vis_src_bytes)
        if gps: lat, lon = gps
    if lat is None and ir_src_bytes:
        gps = extract_gps_from_image(ir_src_bytes)
        if gps: lat, lon = gps

    c1, c2 = st.columns(2)
    with c1: lat = st.text_input("緯度", value=f"{lat:.6f}" if lat else "35.6804")
    with c2: lon = st.text_input("経度", value=f"{lon:.6f}" if lon else "139.7690")
    
    try:
        lat_f, lon_f = float(lat), float(lon)
        m = folium.Map(location=[lat_f, lon_f], zoom_start=18, tiles="OpenStreetMap")
        folium.Marker([lat_f, lon_f], tooltip="対象地点").add_to(m)
        st_folium(m, height=300, use_container_width=True)
    except (ValueError, TypeError):
        st.info("有効な緯度・経度を入力してください。")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 4) 実行ボタン ---
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown("#### 4) 解析の実行")
    run = st.button("詳細分析を開始", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 解析フロー ---
    if run:
        if not user_q:
            st.error("質問を入力してください。")
            return

        corpus = load_rag_corpus()
        snippets = topk_chunks(user_q, corpus, k=4)
        
        image_parts = []
        if vis_img: image_parts.append(image_to_inline_part(vis_img))
        if ir_img: image_parts.append(image_to_inline_part(ir_img))

        ir_meta = {
            "has_ir": ir_img is not None, "emissivity": (ir_emiss or "不明").strip(),
            "t_ref": (ir_tref or "不明").strip(), "t_amb": (ir_tamb or "不明").strip(),
            "rh": (ir_rh or "不明").strip(), "dist": (ir_dist or "不明").strip(),
            "angle": (ir_ang or "不明").strip(),
        }
        
        prompt = build_master_prompt(user_q, snippets, ir_meta)

        try:
            api_key = st.secrets["gemini"]["API_KEY"]
        except (KeyError, FileNotFoundError):
            st.error("Gemini API Keyが未設定です。.streamlit/secrets.tomlに設定してください。")
            return

        with st.spinner("Geminiが分析中…"):
            try:
                result = call_gemini(api_key, prompt, image_parts)
                report_md = extract_text_from_gemini(result)
            except requests.HTTPError as e:
                st.error(f"APIエラー: {e.response.text[:500]}")
                return
            except Exception as e:
                st.error(f"呼び出しエラー: {e}")
                return

        if not report_md:
            st.warning("レポート出力が空でした。入力内容をご確認ください。")
            return
        
        # --- 5) 解析結果 ---
        st.markdown("---")
        st.markdown("### 5) 解析結果")
        
        summary_block = re.search(r"(?:^|\n)##?\s*総合評価[\s\S]*?(?=\n##?\s|\Z)", report_md, re.IGNORECASE)
        summary_text = summary_block.group(0) if summary_block else ""

        if summary_text:
            st.markdown('<div class="md-card good-shadow jp-report">', unsafe_allow_html=True)
            st.markdown('<div class="md-title">総合評価（要約）</div>', unsafe_allow_html=True)
            st.markdown(summary_text, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("詳細レポートを表示", expanded=(not summary_text)):
            st.markdown(f"<div class='jp-report'>{report_md}</div>", unsafe_allow_html=True)

        st.download_button("📄 レポートをダウンロード", report_md, "building_health_report.md", "text/markdown", use_container_width=True)

        with st.expander("（参考）使用したRAG抜粋"):
            for src, ch in snippets:
                snippet = ch[:600] + ('...' if len(ch) > 600 else '')
                st.markdown(f"<div class='jp-report'><b>{src}</b>：{snippet}</div>", unsafe_allow_html=True)

    # --- フッタ ---
    st.markdown("---")
    st.caption("© 建物診断くん — 可視/赤外 × RAG × Gemini。モバイル最適化UI。")

if __name__ == "__main__":
    main()
