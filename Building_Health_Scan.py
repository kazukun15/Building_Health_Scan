# -*- coding: utf-8 -*-
# ===========================================================
# 建物診断くん（スマホ対応・マテリアルUI）
# RAG（BM25+ブースト）＋ ドメイン知識（一般原則）＋ 画像スクリーニング所見 → Gemini 2.5 Flash
# - 可視/赤外：アップロード or カメラ（1枚ずつ）
# - 端末の現在地（Geolocation）＋ Folium 地図
# - “結果のみ”：総合評価を最初に、詳細は Expander
# - CSSは components.html で非表示挿入（生CSSが画面に出ない）
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
    # 小さくしてから統計
    w = 256 if image.width > 256 else image.width
    ratio = w / image.width
    image = image.resize((w, int(image.height * ratio)))
    # 明度
    gray = image.convert("L")
    pixels = list(gray.getdata())
    mean_l = statistics.fmean(pixels) if pixels else 0.0
    stdev_l = statistics.pstdev(pixels) if pixels else 0.0
    # エッジ量（簡易）
    edges = gray.filter(ImageFilter.FIND_EDGES)
    epx = list(edges.getdata())
    # 強いエッジの割合（0–255 のうち > 60 を強エッジとみなす簡易指標）
    strong = sum(1 for v in epx if v > 60)
    edge_ratio = strong / len(epx) if epx else 0.0
    # 彩度
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
    # とても素朴なルール群（安全側）：エッジ密度が高いと“クラック傾向”
    crack_hint = s["edge_ratio"] > 0.11
    stain_hint = s["sat_mean"] < 70 and s["mean_l"] < 110  # 低彩度かつ暗め→汚染/雨だれ傾向の可能性
    level = "low"
    if crack_hint and stain_hint:
        level = "mid"
    if s["edge_ratio"] > 0.16:
        level = "mid"
    if s["edge_ratio"] > 0.22:
        level = "high"
    return {
        "metrics": s,
        "crack_hint": crack_hint,
        "stain_hint": stain_hint,
        "screening_level": level,
        "note": "画像ベースの簡易傾向。寸法・幅などは画像だけでは確定できません。"
    }

def analyze_ir(image: Image.Image, meta: Dict[str, str]) -> Dict[str, Any]:
    """IR画像の相対温度差（輝度差）を簡易評価（JPEGの相対指標。絶対温度は扱わない）"""
    gray = image.convert("L")
    w = 256 if gray.width > 256 else gray.width
    gray = gray.resize((w, int(gray.height * (w / gray.width))))
    vals = list(gray.getdata())
    if not vals:
        return {"has_ir": True, "delta_rel": 0.0, "pattern": "unknown", "note": "画像データ無し"}
    vmin, vmax = min(vals), max(vals)
    delta = vmax - vmin  # 0–255
    stdev = statistics.pstdev(vals)
    # 相対評価
    pattern = "uniform"
    if stdev > 20 and delta > 40:
        pattern = "hot/cold spots"
    if stdev > 30 and delta > 60:
        pattern = "strong hotspots"
    return {
        "has_ir": True,
        "delta_rel": round(delta / 255.0, 3),  # 0–1 の相対差指標
        "stdev": round(stdev, 2),
        "pattern": pattern,
        "meta": meta,
        "note": "JPEGの輝度差から見た相対的なムラ評価。放射率/反射温度等の補正無し。"
    }

# ---------------------- ルールベース合成（安全側） ----------------------
def rule_based_grade(vis: Optional[Dict[str, Any]], ir: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """
    画像スクリーニング所見から A–D の暫定グレードを推定（安全側）。
    - 可視のエッジ比/汚れ傾向
    - IRのムラ（delta_rel, pattern）
    """
    score = 0.0
    reasons = []
    if vis:
        er = vis["metrics"]["edge_ratio"]
        if er > 0.22:
            score += 3; reasons.append(f"強いエッジ密度（edge_ratio={er}）")
        elif er > 0.16:
            score += 2; reasons.append(f"高めのエッジ密度（edge_ratio={er}）")
        elif er > 0.11:
            score += 1; reasons.append(f"やや高いエッジ密度（edge_ratio={er}）")
        if vis.get("stain_hint"):
            score += 1; reasons.append("低彩度・暗部多めで汚染傾向")
    if ir and ir.get("has_ir"):
        dr = ir["delta_rel"]
        if dr > 0.5:
            score += 3; reasons.append(f"IR輝度差が大（Δ相対={dr}）")
        elif dr > 0.3:
            score += 2; reasons.append(f"IR輝度差が中（Δ相対={dr}）")
        elif dr > 0.15:
            score += 1; reasons.append(f"IR輝度差がやや大（Δ相対={dr}）")
        if ir.get("pattern") in ("strong hotspots",):
            score += 1; reasons.append("強いホットスポットパターン")
    # スコア → グレード
    if score <= 1:
        g = "A"
    elif score <= 3:
        g = "B"
    elif score <= 5:
        g = "C"
    else:
        g = "D"
    return g, "・".join(reasons) if reasons else "顕著な異常は検出されず（画像ベースの簡易判定）"

def rule_based_life(grade: str) -> str:
    """参考の寿命幅（画像と一般原則の暫定。RAGの根拠があればLLM側で上書き）"""
    mapping = {
        "A": "10–20年（参考）",
        "B": "7–15年（参考）",
        "C": "3–10年（参考）",
        "D": "1–5年（参考）",
    }
    return mapping.get(grade, "未評価")

# ---------------------- ドメイン知識（一般原則） ----------------------
def domain_priors_text() -> str:
    """
    安全側の一般原則だけを列挙（具体的閾値はRAGに無い限り使わない）
    """
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
    if image.mode != "RGB":
        image = image.convert("RGB")
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

# ---------------------- プロンプト組立 ----------------------
def build_master_prompt(user_q: str,
                        rag_snippets: List[Dict[str, Any]],
                        priors: str,
                        vis_find: Optional[Dict[str, Any]],
                        ir_find: Optional[Dict[str, Any]],
                        rule_grade: str,
                        rule_life: str,
                        ir_meta: Dict) -> str:
    # RAG抜粋
    rag_lines = []
    for d in rag_snippets:
        pg = ""
        if d.get("page_start") is not None:
            pg = f" p.{d['page_start']}"
        rag_lines.append(f"[{d['doc']}{pg}] {d['text']}")
    rag_text = "\n".join(rag_lines) if rag_lines else "（該当抜粋なし）"

    vis_block = "" if not vis_find else f"""可視画像所見(簡易):
- metrics: {vis_find['metrics']}
- crack_hint: {vis_find['crack_hint']}
- stain_hint: {vis_find['stain_hint']}
- screening_level: {vis_find['screening_level']}
"""
    ir_block = ""
    if ir_find:
        ir_block = f"""IR画像所見(相対):
- delta_rel(0-1): {ir_find['delta_rel']}, stdev: {ir_find['stdev']}, pattern: {ir_find['pattern']}
- 注意: {ir_find['note']}
- メタ: ε={ir_meta.get('emissivity','不明')}, T_ref={ir_meta.get('t_ref','不明')}℃, T_amb={ir_meta.get('t_amb','不明')}℃, RH={ir_meta.get('rh','不明')}%, dist={ir_meta.get('dist','不明')}m, angle={ir_meta.get('angle','不明')}°
"""

    prompt = f"""
あなたは非破壊検査・建築・材料学の上級診断士。
以下の材料だけを根拠に、日本語で“結果のみ”の短いレポートを作成せよ。
**禁止**：推測での数値化・未出典の閾値記載・一般知識の無根拠引用。根拠が無い項目は「未掲載/未確定」とする。

# 入力
- ユーザー質問: {user_q}

- RAG抜粋（出典ページ付き。これ以外を根拠にしない）:
{rag_text}

- ドメイン一般原則（方針・注意点。閾値は含まない）:
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

# ---------------------- CSS（表示されない注入） ----------------------
def inject_material_css():
    components.v1.html(
        """
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;900&family=Noto+Serif+JP:wght@500;700&display=swap" rel="stylesheet">
<style id="app-style">
:root{
  --mdc-primary:#2962ff; --mdc-secondary:#00b8d4;
  --mdc-bg:#f7f9fc; --mdc-surface:#ffffff; --mdc-outline:rgba(0,0,0,.08);
  --radius:16px; --shadow:0 6px 18px rgba(0,0,0,.08); --tap-min:44px;
}
@media (prefers-color-scheme: dark){
  :root{ --mdc-bg:#0f1115; --mdc-surface:#171a21; --mdc-outline:rgba(255,255,255,.08); }
}
.block-container{padding-top:2.6rem !important;padding-bottom:2rem;}
body{background:var(--mdc-bg);}
.jp-sans{font-family:'Noto Sans JP',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Kaku Gothic ProN','Hiragino Sans','Meiryo',sans-serif!important;line-height:1.7;}
.jp-report *{font-family:'Noto Sans JP',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Kaku Gothic ProN','Hiragino Sans','Meiryo',sans-serif!important;line-height:1.6;}
.app-hero{background:linear-gradient(135deg,var(--mdc-primary),var(--mdc-secondary));color:#fff;border-radius:20px;box-shadow:var(--shadow);padding:14px 16px;margin:0 0 14px 0;}
.app-hero-title{font-weight:900;font-size:1.45rem;line-height:1.25;margin:0 0 4px 0;text-shadow:0 1px 2px rgba(0,0,0,.18);}
.app-hero-sub{font-weight:500;font-size:.95rem;opacity:.95;margin:0;}
.md-card{background:var(--mdc-surface);border-radius:var(--radius);box-shadow:var(--shadow);padding:1rem 1.1rem;margin:0 0 1rem 0;border:1px solid var(--mdc-outline);}
.md-title{font-size:1.1rem;font-weight:700;margin:0 0 .6rem 0;}
.stButton>button,.stTextInput input,.stFileUploader label,.stCameraInput label{min-height:var(--tap-min);border-radius:12px!important;font-weight:600;}
.stButton>button{background:linear-gradient(135deg,var(--mdc-primary),var(--mdc-secondary))!important;color:#fff!important;border:none!important;box-shadow:var(--shadow);}
:where(button,input,select,textarea):focus-visible{outline:3px solid color-mix(in srgb,var(--mdc-primary) 60%, white);outline-offset:2px;border-radius:12px;}
</style>
        """,
        height=0,
    )

# ---------------------- 位置（フォールバックJS） ----------------------
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
  if (!navigator.geolocation){return;}
  navigator.geolocation.getCurrentPosition(function(pos){
    const lat = pos.coords.latitude.toFixed(6);
    const lon = pos.coords.longitude.toFixed(6);
    const url = new URL(window.location.href);
    url.searchParams.set('lat', lat);
    url.searchParams.set('lon', lon);
    const a=document.createElement('a'); a.href=url.toString(); a.target='_top'; document.body.appendChild(a); a.click();
  }, function(){}, {enableHighAccuracy:true, timeout:10000, maximumAge:0});
})();
</script>
            """,
            height=0,
        )
    return None, None

# ---------------------- メイン ----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_material_css()

    st.markdown(
        f"""
<div class="app-hero jp-sans">
  <div class="app-hero-title">🏗️ {APP_TITLE}</div>
  <div class="app-hero-sub">スマホ最適 / 画像（可視・赤外）× RAG × ドメイン知識 × Gemini 2.5 Flash</div>
</div>
        """,
        unsafe_allow_html=True
    )

    # 1) 質問
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">1) 質問</div>', unsafe_allow_html=True)
    user_q = st.text_input("例：外壁タイルのひび割れ基準と推定寿命", "", placeholder="分析したいテーマ・質問を入力")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2) 画像
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">2) 画像入力（可視/赤外）</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**可視（どちらか1つ）**")
        vis_file = st.file_uploader("可視画像（JPG/PNG）", type=["jpg","jpeg","png"], key="vis_up")
        vis_cam  = st.camera_input("カメラで撮影（可視）", key="vis_cam")
    with colB:
        st.markdown("**赤外線（IR）画像（任意）**")
        ir_file = st.file_uploader("IR画像（JPG/PNG）", type=["jpg","jpeg","png"], key="ir_up")
        with st.expander("IRメタデータ（任意・精度向上）"):
            ir_emiss = st.text_input("放射率 ε（例:0.95）", "")
            ir_tref  = st.text_input("反射温度 T_ref[℃]（例:20）", "")
            ir_tamb  = st.text_input("外気温 T_amb[℃]（例:22）", "")
            ir_rh    = st.text_input("相対湿度 RH[%]（例:65）", "")
            ir_dist  = st.text_input("撮影距離[m]（例:5）", "")
            ir_ang   = st.text_input("撮影角度[°]（例:10）", "")
    st.markdown('</div>', unsafe_allow_html=True)

    vis_img = Image.open(vis_cam) if vis_cam is not None else (Image.open(vis_file) if vis_file is not None else None)
    ir_img  = Image.open(ir_file) if ir_file is not None else None

    # プレビュー
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        if vis_img: st.image(vis_img, caption="可視画像", use_container_width=True)
    with p2:
        if ir_img:  st.image(ir_img, caption="IR画像", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3) 位置
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">3) 位置情報（現在地 or 手入力）</div>', unsafe_allow_html=True)
    lat_val: Optional[float] = None
    lon_val: Optional[float] = None
    if HAVE_GEO:
        loc = st_geolocation(key="geoloc", label="📍 現在地を取得（ブラウザで許可）")
        if isinstance(loc, dict):
            lat_val = loc.get("latitude") or loc.get("lat")
            lon_val = loc.get("longitude") or loc.get("lon")
            try:
                if lat_val is not None: lat_val = float(lat_val)
                if lon_val is not None: lon_val = float(lon_val)
            except Exception:
                lat_val, lon_val = None, None
    else:
        st.info("コンポーネント未インストールのため簡易方式です。下のボタン押下→許可で自動入力。")
        if st.button("📍 現在地を取得（簡易方式）"):
            geolocate_fallback_via_query_params(show_widget=True)
        lat_val, lon_val = geolocate_fallback_via_query_params(show_widget=False)

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        lat_str = "" if lat_val is None else f"{lat_val:.6f}"
        lat_str = st.text_input("緯度", value=lat_str, key="lat_manual")
    with c2:
        lon_str = "" if lon_val is None else f"{lon_val:.6f}"
        lon_str = st.text_input("経度", value=lon_str, key="lon_manual")
    with c3:
        st.caption("※ 取得できない場合は手入力してください。")

    # 地図
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
        st.info("地図表示：緯度経度が未指定です。")

    st.markdown('</div>', unsafe_allow_html=True)

    # 4) 実行
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">4) 解析の実行</div>', unsafe_allow_html=True)
    run = st.button("🔎 詳細分析（RAG + 画像 + ドメイン知識）", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        if not user_q:
            st.error("質問を入力してください。")
            return

        # 画像所見（簡易）
        vis_find = analyze_visual(vis_img) if vis_img else None
        ir_meta = {
            "has_ir": ir_img is not None,
            "emissivity": (locals().get('ir_emiss') or "不明").strip(),
            "t_ref": (locals().get('ir_tref') or "不明").strip(),
            "t_amb": (locals().get('ir_tamb') or "不明").strip(),
            "rh": (locals().get('ir_rh') or "不明").strip(),
            "dist": (locals().get('ir_dist') or "不明").strip(),
            "angle": (locals().get('ir_ang') or "不明").strip(),
        }
        ir_find = analyze_ir(ir_img, ir_meta) if ir_img else None

        # ルールベース暫定
        rule_grade, rule_reason = rule_based_grade(vis_find, ir_find)
        rule_life = rule_based_life(rule_grade)

        # RAG検索
        snippets = rag_search(user_q, have_ir=ir_img is not None, k=MAX_SNIPPETS)

        # ドメイン一般原則
        priors = domain_priors_text()

        # 画像 → Gemini parts
        image_parts = []
        if vis_img: image_parts.append(image_to_inline_part(vis_img))
        if ir_img:  image_parts.append(image_to_inline_part(ir_img))

        # プロンプト
        prompt = build_master_prompt(
            user_q=user_q,
            rag_snippets=snippets,
            priors=priors,
            vis_find=vis_find,
            ir_find=ir_find,
            rule_grade=rule_grade,
            rule_life=rule_life,
            ir_meta=ir_meta
        )

        # API
        try:
            api_key = st.secrets["gemini"]["API_KEY"]
        except (KeyError, FileNotFoundError):
            st.error("Gemini API Key が未設定です。.streamlit/secrets.toml に [gemini].API_KEY を設定してください。")
            return

        with st.spinner("Gemini 2.5 Flash が分析中…"):
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
            st.warning("レポートが空でした。入力（質問/画像/PDF）を見直してください。")
            return

        # 5) 結果表示（総合評価 → 詳細）
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
            st.markdown('<div class="md-title">🧭 総合評価（先頭要約）</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='jp-report'>{summary_block}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("詳細レポートを表示", expanded=(summary_block is None)):
            # 先頭にルールベース暫定（参考）も添えて可視化
            st.markdown("**（参考）画像＋一般原則による暫定評価**")
            st.markdown(f"- 暫定グレード: `{rule_grade}`（{rule_reason}）")
            st.markdown(f"- 参考寿命: `{rule_life}`")
            st.markdown("---")
            st.markdown(f"<div class='jp-report'>{report_md}</div>", unsafe_allow_html=True)

        st.download_button(
            "📄 レポートをダウンロード",
            report_md,
            file_name="building_health_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.caption("© 建物診断くん — 可視/赤外 × RAG × ドメイン知識 × Gemini 2.5 Flash。")


if __name__ == "__main__":
    main()
