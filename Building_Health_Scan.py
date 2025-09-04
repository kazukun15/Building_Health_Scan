# -*- coding: utf-8 -*-
# ===========================================================
# 建物診断くん（スマホ対応・先進UI）
# 機能はオリジナルを維持しつつ、UI/UXを大幅に改善。
# RAG（BM25+ブースト）＋ ドメイン知識（一般原則）＋ 画像スクリーニング所見 → Gemini 2.5 Flash
# - 可視/赤外：アップロード or カメラ（1枚ずつ）
# - 端末の現在地（Geolocation）＋ Folium 地図
# - “結果のみ”：総合評価を最初に、詳細は Expander
# - CSSは components.html で非表示挿入
# Python 3.12 互換・軽量依存
# ===========================================================

# Python 3.12: pkgutil.ImpImporter削除回避（古い依存対策）
import pkgutil
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter

import os
import io
import re
import math
import base64
import statistics
import unicodedata
from typing import List, Tuple, Dict, Optional, Any

import streamlit as st
import requests
import PyPDF2
from PIL import Image, ImageFilter

# 地図
import folium
from streamlit_folium import st_folium
from streamlit import components

# geolocation（任意）
HAVE_GEO = False
try:
    from streamlit_geolocation import st_geolocation
    HAVE_GEO = True
except Exception:
    HAVE_GEO = False

APP_TITLE = "建物診断くん"

PDF_SOURCES = [
    ("Structure_Base.pdf", "Structure_Base.pdf"),
    ("上島町 公共施設等総合管理計画", "kamijimachou_Public_facility_management_plan.pdf"),
    ("港区 公共施設マネジメント計画", "minatoku_Public_facility_management_plan.pdf"),
]

MAX_SNIPPETS = 6
MAX_SNIPPET_CHARS = 900

QUERY_SYNONYMS = {
    "ひび割れ": ["ひび割れ", "クラック", "亀裂", "ひび"],
    "浮き": ["浮き", "うき"],
    "剥離": ["剥離", "はく離", "はくり"],
    "含水": ["含水", "浸水", "漏水", "雨水侵入"],
    "赤外線": ["赤外線", "IR", "サーモ", "熱画像", "サーモグラフィ"],
    "基準": ["基準", "判定", "評価", "閾値"],
    "タイル": ["タイル", "磁器質タイル", "モザイクタイル"],
    "ALC": ["ALC", "軽量気泡コンクリート"],
    "コンクリート": ["コンクリート", "RC", "中性化"],
    "防水": ["防水", "塗膜", "シーリング", "伸縮目地"],
    "劣化度": ["劣化度", "グレード", "区分", "A", "B", "C", "D"],
}

# -------------------------- テキスト処理 --------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(s: str) -> List[str]:
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
                tokens.update(tokenize(s))
    return list(tokens)

# -------------------------- PDF → チャンク --------------------------
def extract_chunks_from_pdf(pdf_path: str, title: str,
                            max_chars: int = 900, overlap: int = 120) -> List[Dict[str, Any]]:
    if not os.path.exists(pdf_path):
        return []
    chunks: List[Dict[str, Any]] = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                t = page.extract_text() or ""
                page_text = normalize_text(t)
                if not page_text:
                    continue
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
                            "page_end": i,
                        })
                    pos = end - overlap if (end - overlap) > pos else end
    except Exception as e:
        chunks.append({
            "doc": title, "path": pdf_path,
            "text": f"[PDF読込エラー:{pdf_path}:{e}]",
            "page_start": None, "page_end": None
        })
    return chunks

# -------------------------- BM25 インデックス --------------------------
class BM25Index:
    def __init__(self, k1: float = 1.6, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = 0
        self.avgdl = 0.0
        self.df: Dict[str, int] = {}
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.docs: List[Dict[str, Any]] = []

    @staticmethod
    def _tf(tokens: List[str]) -> Dict[str, int]:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return tf

    def build(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.N = len(docs)
        lengths = []
        for doc_id, d in enumerate(docs):
            tokens = tokenize(d["text"])
            d["_tokens"] = tokens
            tf = self._tf(tokens)
            lengths.append(len(tokens))
            for term, c in tf.items():
                self.df[term] = self.df.get(term, 0) + 1
                self.postings.setdefault(term, []).append((doc_id, c))
        self.doc_len = lengths
        self.avgdl = (sum(lengths) / self.N) if self.N else 0.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score_doc(self, q_tokens: List[str], doc_id: int) -> float:
        score = 0.0
        dl = self.doc_len[doc_id] if doc_id < len(self.doc_len) else 0
        if dl == 0:
            return 0.0
        for term in set(q_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            tf = 0
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

@st.cache_resource(show_spinner=False)
def build_rag() -> Dict[str, Any]:
    all_chunks: List[Dict[str, Any]] = []
    for title, path in PDF_SOURCES:
        all_chunks.extend(extract_chunks_from_pdf(path, title))
    # 追加メタ
    for d in all_chunks:
        txt = d["text"]
        d["_has_numbers"] = bool(re.search(r"\b\d+(\.\d+)?\s*(mm|㎜|％|%|℃)\b", txt))
        d["_has_ir"] = any(k in txt for k in ["赤外線", "サーモ", "IR", "熱画像", "放射率", "反射温度"])
    bm25 = BM25Index()
    if all_chunks:
        bm25.build(all_chunks)
    return {"index": bm25, "docs": all_chunks}

def rag_search(query: str, have_ir: bool, k: int = MAX_SNIPPETS) -> List[Dict[str, Any]]:
    data = build_rag()
    bm25: BM25Index = data["index"]
    docs = data["docs"]
    if not docs:
        return []
    q_tokens = query_expand_tokens(query)
    want_threshold = any(w in query for w in ["基準", "閾値", "幅", "mm", "％", "%", "℃", "温度"])
    boost_muni = "港区" if "港区" in query else ("上島町" if ("上島町" in query or "上嶋町" in query) else None)

    scored: List[Tuple[float, int]] = []
    for doc_id, d in enumerate(docs):
        base = bm25.score_doc(q_tokens, doc_id)
        if want_threshold and d.get("_has_numbers"):
            base *= 1.12
        if have_ir and d.get("_has_ir"):
            base *= 1.10
        if boost_muni and boost_muni in d["doc"]:
            base *= 1.15
        if base > 0:
            scored.append((base, doc_id))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [docs[i] for (_, i) in scored[:k]]
    for t in top:
        if len(t["text"]) > MAX_SNIPPET_CHARS:
            t["text"] = t["text"][:MAX_SNIPPET_CHARS] + "…"
    return top

# ---------------------- 画像の軽量スクリーニング ----------------------
def pil_stats(image: Image.Image) -> Dict[str, float]:
    """PILだけで軽量に画素統計を取る（リサイズして負荷軽減）"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w = 256 if image.width > 256 else image.width
    ratio = w / image.width
    image = image.resize((w, int(image.height * ratio)))
    gray = image.convert("L")
    pixels = list(gray.getdata())
    mean_l = statistics.fmean(pixels) if pixels else 0.0
    stdev_l = statistics.pstdev(pixels) if pixels else 0.0
    edges = gray.filter(ImageFilter.FIND_EDGES)
    epx = list(edges.getdata())
    strong = sum(1 for v in epx if v > 60)
    edge_ratio = strong / len(epx) if epx else 0.0
    hsv = image.convert("HSV")
    sat_vals = [p[1] for p in list(hsv.getdata())]
    sat_mean = statistics.fmean(sat_vals) if sat_vals else 0.0
    return {
        "mean_l": round(mean_l, 2),
        "stdev_l": round(stdev_l, 2),
        "edge_ratio": round(edge_ratio, 4),
        "sat_mean": round(sat_mean, 2),
    }

def analyze_visual(image: Image.Image) -> Dict[str, Any]:
    """可視画像の簡易所見（ひび傾向/汚れ傾向などのスクリーニング）"""
    s = pil_stats(image)
    crack_hint = s["edge_ratio"] > 0.11
    stain_hint = s["sat_mean"] < 70 and s["mean_l"] < 110
    level = "low"
    if crack_hint and stain_hint: level = "mid"
    if s["edge_ratio"] > 0.16: level = "mid"
    if s["edge_ratio"] > 0.22: level = "high"
    return {
        "metrics": s,
        "crack_hint": crack_hint,
        "stain_hint": stain_hint,
        "screening_level": level,
        "note": "画像ベースの簡易傾向。寸法・幅などは画像だけでは確定できません。"
    }

def analyze_ir(image: Image.Image, meta: Dict[str, str]) -> Dict[str, Any]:
    """IR画像の相対温度差（輝度差）を簡易評価"""
    gray = image.convert("L")
    w = 256 if gray.width > 256 else gray.width
    gray = gray.resize((w, int(gray.height * (w / gray.width))))
    vals = list(gray.getdata())
    if not vals:
        return {"has_ir": True, "delta_rel": 0.0, "pattern": "unknown", "note": "画像データ無し"}
    vmin, vmax = min(vals), max(vals)
    delta = vmax - vmin
    stdev = statistics.pstdev(vals)
    pattern = "uniform"
    if stdev > 20 and delta > 40: pattern = "hot/cold spots"
    if stdev > 30 and delta > 60: pattern = "strong hotspots"
    return {
        "has_ir": True,
        "delta_rel": round(delta / 255.0, 3),
        "stdev": round(stdev, 2),
        "pattern": pattern,
        "meta": meta,
        "note": "JPEGの輝度差から見た相対的なムラ評価。放射率/反射温度等の補正無し。"
    }

# ---------------------- ルールベース合成（安全側） ----------------------
def rule_based_grade(vis: Optional[Dict[str, Any]], ir: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """画像スクリーニング所見から A–D の暫定グレードを推定（安全側）"""
    score = 0.0
    reasons = []
    if vis:
        er = vis["metrics"]["edge_ratio"]
        if er > 0.22: score += 3; reasons.append(f"強いエッジ密度(edge_ratio={er})")
        elif er > 0.16: score += 2; reasons.append(f"高めのエッジ密度(edge_ratio={er})")
        elif er > 0.11: score += 1; reasons.append(f"やや高いエッジ密度(edge_ratio={er})")
        if vis.get("stain_hint"): score += 1; reasons.append("低彩度・暗部多めで汚染傾向")
    if ir and ir.get("has_ir"):
        dr = ir["delta_rel"]
        if dr > 0.5: score += 3; reasons.append(f"IR輝度差が大(Δ相対={dr})")
        elif dr > 0.3: score += 2; reasons.append(f"IR輝度差が中(Δ相対={dr})")
        elif dr > 0.15: score += 1; reasons.append(f"IR輝度差がやや大(Δ相対={dr})")
        if ir.get("pattern") in ("strong hotspots",): score += 1; reasons.append("強いホットスポットパターン")
    
    if score <= 1: g = "A"
    elif score <= 3: g = "B"
    elif score <= 5: g = "C"
    else: g = "D"
    return g, "・".join(reasons) if reasons else "顕著な異常は検出されず（画像ベースの簡易判定）"

def rule_based_life(grade: str) -> str:
    """参考の寿命幅"""
    mapping = {"A": "10–20年(参考)", "B": "7–15年(参考)", "C": "3–10年(参考)", "D": "1–5年(参考)"}
    return mapping.get(grade, "未評価")

# ---------------------- ドメイン知識（一般原則） ----------------------
def domain_priors_text() -> str:
    return normalize_text("""
- 写真のみから寸法（ひび幅等）や材料特性を正確に決定することはできない。数値が必要な場合は根拠文献の閾値を引用する。
- 可視：クラック/目地劣化は雨水浸入・仕上げ剥離・躯体劣化につながる可能性がある。
- 赤外線：放射率・反射温度・外気温・湿度・撮影距離/角度・風/日射等の条件に強く影響される。JPEGの疑似温度は絶対値として扱わない。
- 安全確保：剥離・落下リスクがある場合は近接調査（打診/はつり/付着力等）と早期仮設防護を検討する。
- 維持管理：軽微段階での目地/防水補修・再塗装等がライフサイクルコストを低減する。
- 判定は「画像所見」「RAG根拠」「現地条件」の整合で行い、不足があれば“未確定”と表現する。
    """)

# ---------------------- Gemini API ----------------------
def image_to_inline_part(image: Image.Image, max_width: int = 1400) -> Dict:
    if image.mode != "RGB": image = image.convert("RGB")
    if image.width > max_width:
        r = max_width / float(image.width)
        image = image.resize((max_width, int(image.height * r)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/jpeg", "data": b64}}

def call_gemini(api_key: str, prompt_text: str, image_parts: List[Dict]) -> Dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    parts = [{"text": prompt_text}] + image_parts
    payload = {"contents": [{"parts": parts}]}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def extract_text_from_gemini(result: Dict) -> str:
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""

# ---------------------- プロンプト組立 ----------------------
def build_master_prompt(user_q: str, rag_snippets: List[Dict[str, Any]], priors: str,
                        vis_find: Optional[Dict[str, Any]], ir_find: Optional[Dict[str, Any]],
                        rule_grade: str, rule_life: str, ir_meta: Dict) -> str:
    rag_lines = []
    for d in rag_snippets:
        pg = f" p.{d['page_start']}" if d.get("page_start") is not None else ""
        rag_lines.append(f"[{d['doc']}{pg}] {d['text']}")
    rag_text = "\n".join(rag_lines) if rag_lines else "（該当抜粋なし）"

    vis_block = f"可視画像所見(簡易):\n- metrics: {vis_find['metrics']}\n- crack_hint: {vis_find['crack_hint']}\n- stain_hint: {vis_find['stain_hint']}\n- screening_level: {vis_find['screening_level']}\n" if vis_find else ""
    ir_block = ""
    if ir_find:
        ir_block = f"""IR画像所見(相対):
- delta_rel(0-1): {ir_find['delta_rel']}, stdev: {ir_find['stdev']}, pattern: {ir_find['pattern']}
- 注意: {ir_find['note']}
- メタ: ε={ir_meta.get('emissivity','不明')}, T_ref={ir_meta.get('t_ref','不明')}℃, T_amb={ir_meta.get('t_amb','不明')}℃, RH={ir_meta.get('rh','不明')}%, dist={ir_meta.get('dist','不明')}m, angle={ir_meta.get('angle','不明')}°
"""
    prompt = f"""
あなたは非破壊検査・建築・材料学の上級診断士。以下の材料だけを根拠に、日本語で“結果のみ”の短いレポートを作成せよ。
**禁止**：推測での数値化・未出典の閾値記載・一般知識の無根拠引用。根拠が無い項目は「未掲載/未確定」とする。
# 入力
- ユーザー質問: {user_q}
- RAG抜粋:
{rag_text}
- ドメイン一般原則:
{priors}
- 画像スクリーニング所見:
{vis_block}{ir_block}
- ルールベース暫定:
  * 暫定グレード: {rule_grade}
  * 参考寿命: {rule_life}
# 出力仕様（Markdown、“結果のみ”）
1. **総合評価**（A/B/C/D、主因1–2行、暫定を明記可）
2. **推定残存寿命**（幅。RAGに根拠が無ければ「参考（画像・一般原則ベース）」と明示）
3. **主要根拠**（可視/IR/RAGの順で簡潔。各項目に [出典] を付与。数値はRAGにある場合のみ）
4. **推奨対応（優先度順）**（安全側。出典があれば併記）
5. **限界と追加データ要望**（不足根拠、必要な現地確認・NDT）
注意：画像のみで確定できない項目は断定しない。IR JPEGは相対判断に留める。
""".strip()
    return normalize_text(prompt)

# ---------------------- CSS（先進UIデザイン） ----------------------
def inject_advanced_ui_css():
    components.v1.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Sans+JP:wght@400;500;700;900&display=swap" rel="stylesheet">
<style id="advanced-ui-style">
:root {
    --font-sans: 'Inter', 'Noto Sans JP', sans-serif;
    --font-jp: 'Noto Sans JP', sans-serif;
    --primary-color: #0052D4;
    --secondary-color: #65C7F7;
    --accent-color: #9CECFB;
    --text-color: #E2E8F0;
    --bg-color: #0A101A;
    --surface-color: rgba(23, 29, 41, 0.5);
    --surface-border: rgba(255, 255, 255, 0.1);
    --outline-color: rgba(101, 199, 247, 0.6);
    --radius-lg: 24px; --radius-md: 16px; --radius-sm: 12px;
    --shadow-lg: 0 10px 30px rgba(0, 0, 0, 0.2);
    --shadow-md: 0 4px 15px rgba(0, 0, 0, 0.15);
    --tap-min: 48px;
}
body {
    background: var(--bg-color);
    background-image: linear-gradient(160deg, var(--bg-color) 0%, #111827 100%);
    font-family: var(--font-sans);
    color: var(--text-color);
}
.stApp { background: transparent; }
.block-container { padding: 1.5rem 1rem 3rem 1rem !important; max-width: 800px; margin: 0 auto; }
.app-header {
    text-align: center; margin-bottom: 2rem; padding: 1rem;
    background: rgba(0,0,0,0.2); border-radius: var(--radius-lg);
}
.app-header-title {
    font-family: 'Inter', sans-serif; font-weight: 800; font-size: 2.2rem; line-height: 1.2;
    background: linear-gradient(45deg, var(--accent-color), var(--secondary-color));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.app-header-sub { font-size: 0.9rem; color: var(--text-color); opacity: 0.8; margin-top: 0.5rem; }
.step-card {
    background: var(--surface-color); border: 1px solid var(--surface-border);
    border-radius: var(--radius-lg); padding: 1.5rem; margin-bottom: 1.5rem;
    box-shadow: var(--shadow-lg); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    transition: all 0.3s ease;
}
.step-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25); }
.step-title {
    display: flex; align-items: center; font-size: 1.2rem; font-weight: 700;
    color: var(--text-color); margin: 0 0 1rem 0;
    border-bottom: 1px solid var(--surface-border); padding-bottom: 0.75rem;
}
.step-title .icon { font-size: 1.5rem; margin-right: 0.75rem; line-height: 1; }
.stButton>button, .stTextInput input, .stFileUploader label, .stCameraInput label {
    border-radius: var(--radius-sm) !important; font-weight: 600; font-family: var(--font-sans);
    transition: all 0.2s ease; border: 1px solid var(--surface-border) !important;
    background: rgba(255, 255, 255, 0.05) !important; color: var(--text-color) !important;
    min-height: var(--tap-min);
}
.stFileUploader label, .stCameraInput label { display: flex; justify-content: center; align-items: center; text-align: center; }
.stButton>button {
    background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
    border: none !important; box-shadow: var(--shadow-md); font-weight: 700;
}
.stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 20px rgba(101, 199, 247, 0.5); }
:where(button,input,select,textarea):focus-visible { outline: 3px solid var(--outline-color) !important; outline-offset: 2px; }
.stTextInput input { color: var(--text-color) !important; }
.image-preview-container { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.image-preview-container .stImage { border-radius: var(--radius-md); overflow: hidden; }
.report-card {
    background: var(--surface-color); border: 1px solid var(--surface-border);
    border-radius: var(--radius-lg); padding: 1.5rem 1.75rem; box-shadow: var(--shadow-lg);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); font-family: var(--font-jp);
}
.report-card h1, .report-card h2, .report-card h3 { color: var(--accent-color); border-bottom: 1px solid var(--surface-border); padding-bottom: 0.5rem; margin-top: 1.5rem; }
.report-card strong { color: var(--secondary-color); }
.stExpander { border: 1px solid var(--surface-border) !important; border-radius: var(--radius-md) !important; background: rgba(255, 255, 255, 0.05) !important; }
.stFolium { border-radius: var(--radius-md); overflow: hidden; border: 1px solid var(--surface-border); }
</style>""", height=0)

# ---------------------- 位置（フォールバックJS） ----------------------
def geolocate_fallback_via_query_params(show_widget: bool = True) -> Tuple[Optional[float], Optional[float]]:
    params = st.query_params
    lat = params.get("lat")
    lon = params.get("lon")
    if lat and lon:
        try: return float(lat), float(lon)
        except Exception: return None, None
    if show_widget:
        components.v1.html("""
<script>
(function(){
  if (!navigator.geolocation){return;}
  navigator.geolocation.getCurrentPosition(function(pos){
    const lat = pos.coords.latitude.toFixed(6), lon = pos.coords.longitude.toFixed(6);
    const url = new URL(window.location.href);
    url.searchParams.set('lat', lat); url.searchParams.set('lon', lon);
    const a=document.createElement('a'); a.href=url.toString(); a.target='_top';
    document.body.appendChild(a); a.click();
  }, function(){}, {enableHighAccuracy:true, timeout:10000, maximumAge:0});
})();
</script>""", height=0)
    return None, None

# ---------------------- メイン ----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    inject_advanced_ui_css()

    st.markdown(f"""
<div class="app-header">
  <div class="app-header-title">🏗️ {APP_TITLE}</div>
  <div class="app-header-sub">AI Building Diagnostics</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="step-card"><div class="step-title"><span class="icon">📝</span>1. Diagnostic Query</div>', unsafe_allow_html=True)
    user_q = st.text_input("例：外壁タイルのひび割れ基準と推定寿命", "", placeholder="分析したいテーマ・質問を入力...", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-card"><div class="step-title"><span class="icon">📸</span>2. Image Input (Visible / IR)</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Visible Image**")
        vis_file = st.file_uploader("Upload JPG/PNG", type=["jpg","jpeg","png"], key="vis_up")
        vis_cam  = st.camera_input("Take a Photo", key="vis_cam")
    with colB:
        st.markdown("**Infrared (IR) Image (Optional)**")
        ir_file = st.file_uploader("Upload IR JPG/PNG", type=["jpg","jpeg","png"], key="ir_up")
        with st.expander("IR Metadata (Optional)"):
            ir_emiss, ir_tref, ir_tamb = st.text_input("Emissivity ε", "0.95"), st.text_input("Ref. Temp T_ref[°C]", "20"), st.text_input("Amb. Temp T_amb[°C]", "22")
            ir_rh, ir_dist, ir_ang = st.text_input("RH[%]", "65"), st.text_input("Distance[m]", "5"), st.text_input("Angle[°]", "10")
    
    vis_img = Image.open(vis_cam) if vis_cam else (Image.open(vis_file) if vis_file else None)
    ir_img  = Image.open(ir_file) if ir_file else None

    if vis_img or ir_img:
        st.markdown('<div class="image-preview-container">', unsafe_allow_html=True)
        if vis_img: st.image(vis_img, caption="Visible Image", use_container_width=True)
        if ir_img: st.image(ir_img, caption="IR Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-card"><div class="step-title"><span class="icon">📍</span>3. Location (Auto or Manual)</div>', unsafe_allow_html=True)
    lat_val, lon_val = None, None
    if HAVE_GEO:
        loc = st_geolocation(key="geoloc", label="Get Current Location")
        if isinstance(loc, dict):
            try:
                lat_val, lon_val = float(loc.get("latitude")), float(loc.get("longitude"))
            except (TypeError, ValueError): pass
    else:
        st.info("コンポーネント未導入のため簡易方式です。")
        if st.button("現在地を取得（簡易）"): geolocate_fallback_via_query_params(show_widget=True)
        lat_val, lon_val = geolocate_fallback_via_query_params(show_widget=False)
    
    c1, c2 = st.columns(2)
    lat_str = c1.text_input("Latitude", value=f"{lat_val:.6f}" if lat_val else "", key="lat_manual")
    lon_str = c2.text_input("Longitude", value=f"{lon_val:.6f}" if lon_val else "", key="lon_manual")
    
    try: lat_f, lon_f = (float(lat_str) if lat_str else None), (float(lon_str) if lon_str else None)
    except (TypeError, ValueError): lat_f, lon_f = None, None

    if lat_f is not None and lon_f is not None:
        m = folium.Map(location=[lat_f, lon_f], zoom_start=18, tiles="CartoDB positron")
        folium.Marker([lat_f, lon_f], tooltip="Target Location").add_to(m)
        st_folium(m, height=300, use_container_width=True)
    else:
        st.info("緯度経度を指定すると地図が表示されます。")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Analyze Now (RAG + Vision + AI)", use_container_width=True):
        if not user_q:
            st.error("ステップ1で質問を入力してください。")
            return

        with st.spinner("AIが分析中です..."):
            try:
                vis_find = analyze_visual(vis_img) if vis_img else None
                ir_meta = {"has_ir": ir_img is not None, "emissivity": ir_emiss, "t_ref": ir_tref, "t_amb": ir_tamb, "rh": ir_rh, "dist": ir_dist, "angle": ir_ang}
                ir_find = analyze_ir(ir_img, ir_meta) if ir_img else None
                rule_grade, rule_reason = rule_based_grade(vis_find, ir_find)
                rule_life = rule_based_life(rule_grade)
                snippets = rag_search(user_q, have_ir=ir_img is not None, k=MAX_SNIPPETS)
                priors = domain_priors_text()
                image_parts = [part for img in [vis_img, ir_img] if img and (part := image_to_inline_part(img))]
                prompt = build_master_prompt(user_q, snippets, priors, vis_find, ir_find, rule_grade, rule_life, ir_meta)
                
                api_key = st.secrets.get("gemini", {}).get("API_KEY")
                if not api_key:
                    st.error("Gemini APIキーが設定されていません。")
                    return
                
                result = call_gemini(api_key, prompt, image_parts)
                report_md = extract_text_from_gemini(result)
            
            except requests.HTTPError as e:
                st.error(f"APIエラー: {e.response.status_code} {e.response.reason}\n{e.response.text[:500]}")
                return
            except Exception as e:
                st.error(f"予期せぬエラーが発生しました: {e}")
                return

        if not report_md:
            st.warning("レポートが空です。入力を見直してください。")
            return

        st.markdown('<div class="step-card" style="margin-top: 2rem;"><div class="step-title"><span class="icon">📊</span>5. Analysis Result</div></div>', unsafe_allow_html=True)
        try:
            summary_pattern = r"(?:^|\n)##?\s*総合評価[\s\S]*?(?=\n##?\s|\Z)"
            summary_match = re.search(summary_pattern, report_md, re.IGNORECASE)
            summary_block = summary_match.group(0) if summary_match else None
        except Exception:
            summary_block = None

        if summary_block:
            st.markdown(f'<div class="report-card">{summary_block}</div>', unsafe_allow_html=True)

        with st.expander("詳細レポートと診断根拠を表示", expanded=(not summary_block)):
            st.markdown('<div class="report-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown(f"**（参考）画像＋一般原則による暫定評価**\n- **グレード:** `{rule_grade}` ({rule_reason})\n- **参考寿命:** `{rule_life}`\n---")
            st.markdown(report_md, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.download_button("📄 レポートをダウンロード", report_md, "building_report.md", "text/markdown", use_container_width=True)

    st.caption("© 建物診断くん — AI Building Diagnostics with Gemini 2.5 Flash.")

if __name__ == "__main__":
    main()
