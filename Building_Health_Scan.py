# -*- coding: utf-8 -*-
# ===========================================================
# 建物診断くん（スマホ対応・マテリアルデザイン風UI・日本語フォント適用）
# - 可視/赤外 画像：アップロード + カメラ
# - 端末の現在地（HTML5 Geolocation）→ Folium地図に表示（手入力も可）
#   * streamlit-geolocation があれば利用／無ければJSフォールバック
# - RAG：軽量BM25＋ブースト（IR/閾値/自治体）→ Gemini 2.5 Flash で詳細分析
# - 結果のみ表示：総合評価カードを先頭に、詳細は展開
# - レポートDL（共有）
# - 日本語フォント（Noto Sans JP / Noto Serif JP）
# Python 3.12 互換
# ===========================================================

# Python 3.12: pkgutil.ImpImporter削除回避（古い依存のための保険）
import pkgutil
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter

import os
import io
import re
import math
import base64
import unicodedata
from typing import List, Tuple, Dict, Optional, Any

import streamlit as st
import requests
import PyPDF2
from PIL import Image

# 地図
import folium                              # pip install folium
from streamlit_folium import st_folium     # pip install streamlit-folium
from streamlit import components

# --- オプション: streamlit-geolocation があれば使う ---
HAVE_GEO = False
try:
    from streamlit_geolocation import st_geolocation  # pip install streamlit-geolocation
    HAVE_GEO = True
except Exception:
    HAVE_GEO = False

# -----------------------------------------------------------
# 定数
# -----------------------------------------------------------
APP_TITLE = "建物診断くん"
PDF_SOURCES = [
    ("Structure_Base.pdf", "Structure_Base.pdf"),
    ("上島町 公共施設等総合管理計画", "kamijimachou_Public_facility_management_plan.pdf"),
    ("港区 公共施設マネジメント計画", "minatoku_Public_facility_management_plan.pdf"),
]

MAX_SNIPPETS = 6         # LLMへ渡す抜粋数
MAX_SNIPPET_CHARS = 900  # 各抜粋の最大長

# RAG用シノニム（簡易拡張）※必要に応じて追加
QUERY_SYNONYMS = {
    "ひび割れ": ["ひび割れ", "クラック", "亀裂", "ひび"],
    "浮き": ["浮き", "うき"],
    "剥離": ["剥離", "はく離", "はくり"],
    "含水": ["含水", "含湿", "浸水", "漏水", "雨水侵入"],
    "赤外線": ["赤外線", "IR", "サーモ", "熱画像", "サーモグラフィ"],
    "基準": ["基準", "判定", "評価", "閾値"],
    "タイル": ["タイル", "磁器質タイル", "モザイクタイル"],
    "ALC": ["ALC", "軽量気泡コンクリート"],
    "コンクリート": ["コンクリート", "RC", "中性化"],
    "防水": ["防水", "塗膜", "シーリング", "シーラント"],
    "劣化度": ["劣化度", "グレード", "区分", "A", "B", "C", "D"],
}

# -----------------------------------------------------------
# 文字列ユーティリティ
# -----------------------------------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(s: str) -> List[str]:
    # 形態素解析なしの軽量トークナイズ（数字・記号を扱いつつ日本語対応）
    s = s.lower()
    s = re.sub(r"[^\w一-龥ぁ-んァ-ン％℃．\.]", " ", s)
    s = s.replace("．", ".")
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def query_expand_tokens(q: str) -> List[str]:
    tokens = set(tokenize(q))
    for key, syns in QUERY_SYNONYMS.items():
        if key in q or any(s in q for s in syns):
            for s in syns:
                for t in tokenize(s):
                    tokens.add(t)
    return list(tokens)

# -----------------------------------------------------------
# PDF → チャンク（ページ番号付き）
# -----------------------------------------------------------
def extract_chunks_from_pdf(pdf_path: str, title: str,
                            max_chars: int = 900, overlap: int = 120) -> List[Dict[str, Any]]:
    """各PDFをページ単位で読み、その上で文字長チャンク化。ページ範囲を保持。"""
    if not os.path.exists(pdf_path):
        return []
    chunks: List[Dict[str, Any]] = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            buf = ""
            page_start = 1
            for i, page in enumerate(reader.pages, start=1):
                t = page.extract_text() or ""
                # ページ区切りは軽く保持
                page_text = normalize_text(t)
                if not page_text:
                    continue
                # ページテキストをmax_charsで分割
                pos = 0
                while pos < len(page_text):
                    end = min(pos + max_chars, len(page_text))
                    ch = page_text[pos:end].strip()
                    if ch:
                        chunks.append({
                            "doc": title,
                            "path": pdf_path,
                            "text": ch,
                            "page_start": i,
                            "page_end": i,  # 1ページ内で切っているので等しい
                        })
                    # オーバーラップ
                    pos = end - overlap if (end - overlap) > pos else end
    except Exception as e:
        chunks.append({
            "doc": title, "path": pdf_path,
            "text": f"[PDF読込エラー:{pdf_path}:{e}]",
            "page_start": None, "page_end": None
        })
    return chunks

# -----------------------------------------------------------
# RAGインデックス：BM25（軽量）＋メタブースト
# -----------------------------------------------------------
class BM25Index:
    def __init__(self, k1: float = 1.6, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = 0
        self.avgdl = 0.0
        self.df: Dict[str, int] = {}
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_id, tf), ...]
        self.docs: List[Dict[str, Any]] = []  # メタ含む

    @staticmethod
    def _tf(tokens: List[str]) -> Dict[str, int]:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return tf

    def build(self, docs: List[Dict[str, Any]]):
        """docs: {"text", "doc", "page_start", "page_end", ...}"""
        self.docs = docs
        self.N = len(docs)
        lengths = []
        # 事前に全DF/ポスティング作成
        for doc_id, d in enumerate(docs):
            tokens = tokenize(d["text"])
            d["_tokens"] = tokens
            tf = self._tf(tokens)
            lengths.append(len(tokens))
            for term, c in tf.items():
                # df
                self.df[term] = self.df.get(term, 0) + 1
                # postings
                self.postings.setdefault(term, []).append((doc_id, c))
        self.doc_len = lengths
        self.avgdl = (sum(lengths) / self.N) if self.N else 0.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # Okapi BM25のIDF（+1安定項）
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score_doc(self, q_tokens: List[str], doc_id: int) -> float:
        score = 0.0
        dl = self.doc_len[doc_id] if doc_id < len(self.doc_len) else 0
        if dl == 0:
            return 0.0
        # クエリ内重複は無視（集合でOK）
        for term in set(q_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            # doc_idのtfを探す
            tf = 0
            # postingsは通常docごとに1件
            # （低メモリのため線形探索だが、コーパス小規模を想定）
            for did, c in plist:
                if did == doc_id:
                    tf = c
                    break
            if tf == 0:
                continue
            idf = self.idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            score += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
        return score

# -----------------------------------------------------------
# RAG構築（キャッシュ）
# -----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_rag() -> Dict[str, Any]:
    # 1) PDF → メタ付きチャンク
    all_chunks: List[Dict[str, Any]] = []
    for title, path in PDF_SOURCES:
        all_chunks.extend(extract_chunks_from_pdf(path, title))
    # 2) BM25インデックス
    bm25 = BM25Index()
    if all_chunks:
        bm25.build(all_chunks)
    # 3) 追加メタ：数値閾値・IR関連語の有無
    for d in all_chunks:
        txt = d["text"]
        d["_has_numbers"] = bool(re.search(r"\b\d+(\.\d+)?\s*(mm|㎜|％|%|℃)\b", txt))
        d["_has_ir"] = any(k in txt for k in ["赤外線", "サーモ", "IR", "熱画像", "放射率", "反射温度"])
    return {"index": bm25, "docs": all_chunks}

# -----------------------------------------------------------
# RAG検索（BM25 + ブースト）
# -----------------------------------------------------------
def rag_search(query: str, have_ir: bool, k: int = MAX_SNIPPETS) -> List[Dict[str, Any]]:
    data = build_rag()
    bm25: BM25Index = data["index"]
    docs = data["docs"]
    if not docs:
        return []

    q_tokens = query_expand_tokens(query)

    # ブーストフラグ
    want_threshold = any(w in query for w in ["基準", "閾値", "幅", "mm", "％", "%", "℃", "温度"])
    boost_muni = None
    if "港区" in query:
        boost_muni = "港区"
    elif "上島町" in query or "上嶋町" in query:
        boost_muni = "上島町"

    scored: List[Tuple[float, int]] = []
    for doc_id, d in enumerate(docs):
        base = bm25.score_doc(q_tokens, doc_id)

        # 閾値系を求めている → 数値入り抜粋を微ブースト
        if want_threshold and d.get("_has_numbers"):
            base *= 1.12

        # IR画像がある → IR関連の節を微ブースト
        if have_ir and d.get("_has_ir"):
            base *= 1.10

        # 自治体ブースト（質問に自治体が含まれるとき）
        if boost_muni:
            if boost_muni in d["doc"]:
                base *= 1.15

        if base > 0:
            scored.append((base, doc_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [docs[doc_id] for (_, doc_id) in scored[: k]]
    # 長すぎる抜粋はカット
    for t in top:
        if len(t["text"]) > MAX_SNIPPET_CHARS:
            t["text"] = t["text"][:MAX_SNIPPET_CHARS] + "…"
    return top

# -----------------------------------------------------------
# 画像 → Geminiインライン
# -----------------------------------------------------------
def image_to_inline_part(image: Image.Image, max_width: int = 1400) -> Dict:
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
# Gemini 呼び出し（models/gemini-2.5-flash）
# -----------------------------------------------------------
def call_gemini(api_key: str, prompt_text: str, image_parts: List[Dict]) -> Dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    parts = [{"text": prompt_text}]
    parts.extend(image_parts)
    payload = {"contents": [{"parts": parts}]}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def extract_text_from_gemini(result: Dict) -> str:
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""

# -----------------------------------------------------------
# プロンプト（“結果のみ”＋根拠ページを必須表示）
# -----------------------------------------------------------
def build_master_prompt(user_q: str,
                        rag_snippets: List[Dict[str, Any]],
                        ir_meta: Dict) -> str:
    # RAG抜粋（出典＋ページ）
    rag_lines = []
    for d in rag_snippets:
        pg = ""
        if d.get("page_start") is not None:
            if d["page_end"] and d["page_end"] != d["page_start"]:
                pg = f" p.{d['page_start']}-{d['page_end']}"
            else:
                pg = f" p.{d['page_start']}"
        rag_lines.append(f"[{d['doc']}{pg}] {d['text']}")
    rag_text = "\n".join(rag_lines)

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
あなたは非破壊検査・建築・材料学の上級診断士。国土交通省（MLIT）および自治体原典PDFの根拠のみを用いて、日本語で簡潔に“結果のみ”を作成せよ。
ハルシネーションは禁止。数値閾値や定義はRAG抜粋にある場合のみ使用。無い場合は「未掲載」と明示。
各主張に **[出典: 文書名 §/見出し p.xx]** を併記すること（本文中でOK）。

# 入力
- ユーザー質問: {user_q}
- RAG抜粋（出所付き、これ以外は根拠にしない）:
{rag_text}

{ir_note}
# 出力仕様（Markdown、“結果のみ”）
1. **総合評価**
   - グレード（A/B/C/D）と主因（1–2行）
   - 推定残存寿命（幅、例：7–12年）／（任意）補修後の期待寿命
2. **主要根拠（可視/IR/NDT）** # 画像から読み取れる事実に限定、根拠出典を併記
3. **MLIT/自治体基準との関係** # 文書名＋節/見出し＋ページ
4. **推奨対応（優先度順）**
5. **限界と追加データ要望**

注意: 画像から直接測れない値（ひび幅等）は断定しない。IRメタ不足時は信頼度を下げる旨を明記。
""".strip()
    return normalize_text(prompt)

# -----------------------------------------------------------
# UI: マテリアルCSS/フォントは components.html で非表示挿入（先頭に出ない）
# -----------------------------------------------------------
def inject_material_css():
    components.v1.html(
        """
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;900&family=Noto+Serif+JP:wght@500;700&display=swap" rel="stylesheet">
<style id="app-style">
:root{
  --mdc-primary:#2962ff; --mdc-primary-variant:#0039cb; --mdc-secondary:#00b8d4;
  --mdc-bg:#f7f9fc; --mdc-surface:#ffffff; --mdc-on-primary:#ffffff;
  --mdc-outline:rgba(0,0,0,.08); --radius:16px; --shadow:0 6px 18px rgba(0,0,0,.08);
  --tap-min:44px;
}
@media (prefers-color-scheme: dark){
  :root{ --mdc-bg:#0f1115; --mdc-surface:#171a21; --mdc-on-primary:#ffffff; --mdc-outline:rgba(255,255,255,.08); }
}
</style>
<style id="app-style-2">
.block-container{padding-top:2.6rem !important;padding-bottom:2rem;overflow:visible !important;}
body{background:var(--mdc-bg);}
.jp-sans{font-family:'Noto Sans JP',-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Kaku Gothic ProN','Hiragino Sans','Meiryo',sans-serif!important;line-height:1.7;letter-spacing:.01em;}
.jp-serif{font-family:'Noto Serif JP','Hiragino Mincho ProN','Yu Mincho',serif!important;line-height:1.8;letter-spacing:.01em;}
.jp-report h1,.jp-report h2,.jp-report h3,.jp-report p,.jp-report li,.jp-report table{font-family:'Noto Sans JP',-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Kaku Gothic ProN','Hiragino Sans','Meiryo',sans-serif!important;line-height:1.6;}
.app-hero{background:linear-gradient(135deg,var(--mdc-primary),var(--mdc-secondary));color:#fff;border-radius:20px;box-shadow:var(--shadow);padding:14px 16px;margin:0 0 14px 0;position:relative;overflow:visible;}
.app-hero-title{font-family:'Noto Sans JP',sans-serif;font-weight:900;font-size:1.45rem;line-height:1.25;margin:0 0 4px 0;text-shadow:0 1px 2px rgba(0,0,0,.18);word-break:keep-all;overflow-wrap:anywhere;}
.app-hero-sub{font-family:'Noto Sans JP',sans-serif;font-weight:500;font-size:.95rem;line-height:1.5;opacity:.95;margin:0;}
.md-card{background:var(--mdc-surface);border-radius:var(--radius);box-shadow:var(--shadow);padding:1rem 1.1rem;margin:0 0 1rem 0;border:1px solid var(--mdc-outline);}
.md-title{font-size:1.1rem;font-weight:700;margin:0 0 .6rem 0;}
.md-sub{opacity:.85;font-size:.95rem;}
.eval-chip{display:inline-block;padding:.35rem .7rem;border-radius:999px;color:#fff;font-weight:700;letter-spacing:.02em;background:linear-gradient(135deg,var(--mdc-primary),var(--mdc-secondary));}
.good-shadow{box-shadow:var(--shadow);}
.stButton>button,.stTextInput input,.stFileUploader label,.stCameraInput label,.stNumberInput input{min-height:var(--tap-min);border-radius:12px!important;font-weight:600;}
.stButton>button{background:linear-gradient(135deg,var(--mdc-primary),var(--mdc-secondary))!important;color:#fff!important;border:none!important;box-shadow:var(--shadow);}
:where(button,input,select,textarea):focus-visible{outline:3px solid color-mix(in srgb,var(--mdc-primary) 60%, white);outline-offset:2px;border-radius:12px;}
</style>
        """,
        height=0,
    )

# -----------------------------------------------------------
# 端末位置のフォールバック取得（query params経由）
# -----------------------------------------------------------
def geolocate_fallback_via_query_params(show_widget: bool = True) -> Tuple[Optional[float], Optional[float]]:
    params = st.experimental_get_query_params()
    lat = params.get("lat", [None])[0]
    lon = params.get("lon", [None])[0]
    if lat and lon:
        try:
            return float(lat), float(lon)
        except Exception:
            return None, None

    if show_widget:
        components.v1.html(
            """
<script>
(function(){
  if (!navigator.geolocation){ document.body.innerText='GEO_UNSUPPORTED'; return; }
  navigator.geolocation.getCurrentPosition(function(pos){
    const lat = pos.coords.latitude.toFixed(6);
    const lon = pos.coords.longitude.toFixed(6);
    const url = new URL(window.location.href);
    url.searchParams.set('lat', lat);
    url.searchParams.set('lon', lon);
    const a = document.createElement('a');
    a.href = url.toString();
    a.target = '_top';
    a.rel = 'opener';
    document.body.appendChild(a);
    a.click();
  }, function(err){
    document.body.innerText = 'GEO_ERROR:' + (err && err.message ? err.message : 'DENIED');
  }, { enableHighAccuracy:true, timeout:10000, maximumAge:0 });
})();
</script>
            """,
            height=0,
        )
    return None, None

# -----------------------------------------------------------
# メイン
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_material_css()

    # ヒーローヘッダー
    st.markdown(
        f"""
<div class="app-hero jp-sans">
  <div class="app-hero-title">🏗️ {APP_TITLE}</div>
  <div class="app-hero-sub">スマートフォン最適化 / 画像（可視・赤外）＋RAG（BM25/ブースト）＋Gemini 2.5 Flash</div>
</div>
        """,
        unsafe_allow_html=True
    )

    # --- 1) 質問 ---
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">1) 質問</div>', unsafe_allow_html=True)
    user_q = st.text_input("例：外壁タイルのひび割れ基準と推定寿命", "", placeholder="分析したいテーマ・質問を入力")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 2) 画像入力（可視/赤外） ---
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">2) 画像入力（可視/赤外）</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**可視画像（どちらか1つ）**")
        vis_file = st.file_uploader("可視画像を選択（JPG/PNG）", type=["jpg", "jpeg", "png"], key="vis_up")
        vis_cam = st.camera_input("カメラで撮影（可視）", key="vis_cam")
    with colB:
        st.markdown("**赤外線（IR）画像（任意：外付けカメラ写真）**")
        ir_file = st.file_uploader("IR画像を選択（JPG/PNG）", type=["jpg", "jpeg", "png"], key="ir_up")

        with st.expander("IRメタデータ（任意・精度向上）"):
            ir_emiss = st.text_input("放射率 ε（例: 0.95）", "")
            ir_tref  = st.text_input("反射温度 T_ref [℃]（例: 20）", "")
            ir_tamb  = st.text_input("外気温 T_amb [℃]（例: 22）", "")
            ir_rh    = st.text_input("相対湿度 RH [%]（例: 65）", "")
            ir_dist  = st.text_input("撮影距離 [m]（例: 5）", "")
            ir_ang   = st.text_input("撮影角度 [°]（例: 10）", "")
    st.markdown('</div>', unsafe_allow_html=True)

    # 画像ハンドリング
    vis_img = None
    if vis_cam is not None:
        vis_img = Image.open(vis_cam)
    elif vis_file is not None:
        vis_img = Image.open(vis_file)

    ir_img = None
    if ir_file is not None:
        ir_img = Image.open(ir_file)

    # プレビュー（カード）
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    pv_col1, pv_col2 = st.columns(2)
    with pv_col1:
        if vis_img is not None:
            st.image(vis_img, caption="可視画像", use_container_width=True)
    with pv_col2:
        if ir_img is not None:
            st.image(ir_img, caption="IR画像", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 3) 位置情報（端末の現在地） ---
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">3) 位置情報（現在地 or 手入力）</div>', unsafe_allow_html=True)

    lat_val: Optional[float] = None
    lon_val: Optional[float] = None

    if HAVE_GEO:
        loc = st_geolocation(key="geoloc", label="📍 現在地を取得（ブラウザの位置情報を許可してください）")
        if isinstance(loc, dict):
            lat_val = loc.get("latitude") or loc.get("lat")
            lon_val = loc.get("longitude") or loc.get("lon")
            try:
                if lat_val is not None: lat_val = float(lat_val)
                if lon_val is not None: lon_val = float(lon_val)
            except Exception:
                lat_val, lon_val = None, None
    else:
        st.info("現在地取得：標準コンポーネントが未インストールのため簡易方式を使用します。下のボタン押下後、ブラウザ許可→自動で位置が入力されます。")
        if st.button("📍 現在地を取得（簡易方式）"):
            geolocate_fallback_via_query_params(show_widget=True)
        lat_val, lon_val = geolocate_fallback_via_query_params(show_widget=False)

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        lat_str = "" if lat_val is None else f"{lat_val:.6f}"
        lat_str = st.text_input("緯度（例：35.6804）", value=lat_str, key="lat_manual")
    with c2:
        lon_str = "" if lon_val is None else f"{lon_val:.6f}"
        lon_str = st.text_input("経度（例：139.7690）", value=lon_str, key="lon_manual")
    with c3:
        st.caption("※ 端末の位置情報が取得できない場合は、緯度経度を手入力してください。")

    # 地図
    lat_f, lon_f = None, None
    try:
        lat_f = float(lat_str) if lat_str else None
        lon_f = float(lon_str) if lon_str else None
    except Exception:
        lat_f, lon_f = None, None

    if lat_f is not None and lon_f is not None:
        m = folium.Map(location=[lat_f, lon_f], zoom_start=18, tiles="OpenStreetMap")
        folium.Marker([lat_f, lon_f], tooltip="対象地点").add_to(m)
        st_folium(m, height=300, use_container_width=True)
    else:
        st.info("地図表示：緯度経度が未指定です。位置情報の許可を与えるか、手入力してください。")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 4) 実行 ---
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">4) 解析の実行</div>', unsafe_allow_html=True)
    run = st.button("🔎 Geminiで詳細分析（結果のみ表示）", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------
    # 解析フロー
    # -------------------------------------------------------
    if run:
        if not user_q:
            st.error("質問を入力してください。")
            return

        # RAG検索（BM25 + ブースト）
        have_ir = ir_img is not None
        snippets = rag_search(user_q, have_ir=have_ir, k=MAX_SNIPPETS)

        # 画像 → Gemini parts
        image_parts = []
        if vis_img is not None:
            image_parts.append(image_to_inline_part(vis_img))
        if ir_img is not None:
            image_parts.append(image_to_inline_part(ir_img))

        # IRメタ（任意）
        ir_meta = {
            "has_ir": have_ir,
            "emissivity": (locals().get('ir_emiss') or "不明").strip(),
            "t_ref": (locals().get('ir_tref') or "不明").strip(),
            "t_amb": (locals().get('ir_tamb') or "不明").strip(),
            "rh": (locals().get('ir_rh') or "不明").strip(),
            "dist": (locals().get('ir_dist') or "不明").strip(),
            "angle": (locals().get('ir_ang') or "不明").strip(),
        }

        # プロンプト作成
        prompt = build_master_prompt(user_q, snippets, ir_meta)

        # Gemini API Key
        try:
            api_key = st.secrets["gemini"]["API_KEY"]
        except (KeyError, FileNotFoundError):
            st.error("Gemini API Key が未設定です。.streamlit/secrets.toml に [gemini].API_KEY を設定してください。")
            return

        with st.spinner("Geminiが分析中…"):
            try:
                result = call_gemini(api_key, prompt, image_parts)
                report_md = extract_text_from_gemini(result)
            except requests.HTTPError as e:
                st.error(f"APIエラー: {e.response.status_code} {e.response.reason}\n{e.response.text[:500]}")
                return
            except Exception as e:
                st.error(f"呼び出しエラー: {e}")
                return

        if not report_md:
            st.warning("レポート出力が空でした。入力内容（質問・画像・PDF）をご確認ください。")
            return

        # ---- 5) 解析結果（総合評価を先頭に） ----
        st.markdown('<div class="md-card good-shadow jp-report">', unsafe_allow_html=True)
        st.markdown('<div class="md-title">5) 解析結果</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 総合評価ブロック抽出
        summary_block = None
        try:
            pattern = r"(?:^|\n)##?\s*総合評価[\s\S]*?(?=\n##?\s|\Z)"
            m = re.search(pattern, report_md, re.IGNORECASE)
            if m:
                summary_block = m.group(0)
        except Exception:
            summary_block = None

        if summary_block:
            st.markdown('<div class="md-card good-shadow jp-report">', unsafe_allow_html=True)
            st.markdown('<div class="md-title">🧭 総合評価（要約）</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='jp-report'>{summary_block}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 全文（詳細）はExpander内
        with st.expander("詳細レポートを表示", expanded=(summary_block is None)):
            st.markdown(f"<div class='jp-report'>{report_md}</div>", unsafe_allow_html=True)

        # ダウンロード（共有）
        st.download_button(
            "📄 レポートをダウンロード",
            report_md,
            file_name="building_health_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    # フッタ
    st.markdown("")
    st.caption("© 建物診断くん — 可視/赤外 × RAG（BM25/ブースト）× Gemini 2.5 Flash。モバイル最適化UI。")


if __name__ == "__main__":
    main()
