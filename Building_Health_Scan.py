# -*- coding: utf-8 -*-
# ===========================================================
# 建物診断くん（不具合修正版 / スマホ対応 / マテリアルUI / 複数画像 / RAG / Gemini 2.5 Flash）
# - 重大修正1: 「解析する」を押す→検証エラーで止まる問題
#     → バリデーションを先に実行し、問題点を一覧表示。合格時のみ進捗バーを開始。
# - 重大修正2: 現在地が取得できない問題
#     → 明示ボタン付きのJSコンポーネント（user-gesture起点）で geolocation を呼び出し、
#        取得座標を URL クエリへ反映→同タブでソフトリロードする方式に変更。
#        iOS/一部ブラウザの自動実行制限を回避。
# - 追加: 詳細ログ（デバッグモード）とタイムアウト強化、早期リターン時の進捗 UI を即時消去。
# - 既存機能は維持（複数画像/IRメタ/地図/高速モード/RAG/Web併用/ダウンロード等）。
# ===========================================================

# Python 3.12: pkgutil.ImpImporter 削除対策（古い依存向け）
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
from datetime import date
from typing import List, Tuple, Dict, Optional, Any

import streamlit as st
import requests
import PyPDF2
from PIL import Image, ImageFilter

import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components

# 任意：背面カメラ
HAVE_BACK_CAM = False
try:
    from streamlit_back_camera_input import back_camera_input  # pip install streamlit-back-camera-input
    HAVE_BACK_CAM = True
except Exception:
    HAVE_BACK_CAM = False

APP_TITLE = "建物診断くん"

PDF_SOURCES = [
    ("Structure_Base.pdf", "Structure_Base.pdf"),
    ("上島町 公共施設等総合管理計画", "kamijimachou_Public_facility_management_plan.pdf"),
    ("港区 公共施設マネジメント計画", "minatoku_Public_facility_management_plan.pdf"),
]

MAX_SNIPPETS = 8
MAX_SNIPPET_CHARS = 1000
MAX_IMAGES_TOTAL = 8  # 可視+IR 合計

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

# ---------------------- Web検索（任意・公的サイト優先） ----------------------
@st.cache_data(ttl=600, show_spinner=False)
def web_search_snippets_cached(query: str, max_items: int = 3) -> List[Dict[str, Any]]:
    return _web_search_snippets_impl(query, max_items)

def _web_search_snippets_impl(query: str, max_items: int = 3) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    try:
        if "bing" in st.secrets and st.secrets["bing"].get("API_KEY"):
            api = st.secrets["bing"]["API_KEY"]
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": api}
            params = {"q": query, "count": max_items * 3, "mkt": "ja-JP", "textDecorations": False}
            r = requests.get(url, headers=headers, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            web_pages = (data.get("webPages") or {}).get("value") or []
            for item in web_pages:
                title = item.get("name", "")
                snippet = item.get("snippet", "")
                link = item.get("url", "")
                if not title or not link:
                    continue
                score = 0
                if ".go.jp" in link:
                    score += 3
                if link.lower().endswith(".pdf"):
                    score += 2
                results.append({"title": title, "snippet": snippet, "url": link, "score": score})
        elif "serpapi" in st.secrets and st.secrets["serpapi"].get("API_KEY"):
            api = st.secrets["serpapi"]["API_KEY"]
            url = "https://serpapi.com/search.json"
            params = {"engine": "google", "q": query, "hl": "ja", "num": max_items * 3, "api_key": api}
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            for item in data.get("organic_results", []):
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                if not title or not link:
                    continue
                score = 0
                if ".go.jp" in link:
                    score += 3
                if link.lower().endswith(".pdf"):
                    score += 2
                results.append({"title": title, "snippet": snippet, "url": link, "score": score})
        else:
            return []

        results.sort(key=lambda x: x["score"], reverse=True)
        trimmed: List[Dict[str, Any]] = []
        for r in results:
            trimmed.append({
                "doc": f"WEB: {r['title']} ({r['url']})",
                "path": r["url"],
                "text": normalize_text(r["snippet"])[:MAX_SNIPPET_CHARS],
                "page_start": None, "page_end": None,
                "_web": True
            })
            if len(trimmed) >= max_items:
                break
        return trimmed
    except Exception:
        return []

# ---------------------- 画像の軽量スクリーニング ----------------------
def pil_stats(image: Image.Image, target_w: int = 256) -> Dict[str, float]:
    if image.mode != "RGB":
        image = image.convert("RGB")
    w = target_w if image.width > target_w else image.width
    ratio = w / image.width if image.width else 1.0
    image = image.resize((w, max(1, int(image.height * ratio))))
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

def analyze_visual(image: Image.Image, target_w: int = 256) -> Dict[str, Any]:
    s = pil_stats(image, target_w=target_w)
    crack_hint = s["edge_ratio"] > 0.11
    stain_hint = s["sat_mean"] < 70 and s["mean_l"] < 110
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

def analyze_ir(image: Image.Image, meta: Dict[str, str], target_w: int = 256) -> Dict[str, Any]:
    gray = image.convert("L")
    w = target_w if gray.width > target_w else gray.width
    gray = gray.resize((w, max(1, int(gray.height * (w / (gray.width or 1))))))
    vals = list(gray.getdata())
    if not vals:
        return {"has_ir": True, "delta_rel": 0.0, "pattern": "unknown", "note": "画像データ無し"}
    vmin, vmax = min(vals), max(vals)
    delta = vmax - vmin
    stdev = statistics.pstdev(vals)
    pattern = "uniform"
    if stdev > 20 and delta > 40:
        pattern = "hot/cold spots"
    if stdev > 30 and delta > 60:
        pattern = "strong hotspots"
    return {
        "has_ir": True,
        "delta_rel": round(delta / 255.0, 3),
        "stdev": round(stdev, 2),
        "pattern": pattern,
        "meta": meta,
        "note": "JPEGの輝度差から見た相対的なムラ評価。放射率/反射温度等の補正無し。"
    }

# ---------------------- ルールベース合成（安全側） ----------------------
def rule_based_grade(vis_list: List[Dict[str, Any]], ir_list: List[Dict[str, Any]]) -> Tuple[str, str]:
    score = 0.0
    reasons = []
    if vis_list:
        er_max = max(v["metrics"]["edge_ratio"] for v in vis_list)
        if er_max > 0.22:
            score += 3; reasons.append(f"強いエッジ密度（edge_ratio_max={er_max:.3f}）")
        elif er_max > 0.16:
            score += 2; reasons.append(f"高めのエッジ密度（edge_ratio_max={er_max:.3f}）")
        elif er_max > 0.11:
            score += 1; reasons.append(f"やや高いエッジ密度（edge_ratio_max={er_max:.3f}）")
        if any(v.get("stain_hint") for v in vis_list):
            score += 1; reasons.append("低彩度・暗部多めで汚染傾向（可視複数枚の所見）")
    if ir_list:
        dr_max = max(i["delta_rel"] for i in ir_list)
        if dr_max > 0.5:
            score += 3; reasons.append(f"IR輝度差が大（Δ相対_max={dr_max:.2f}）")
        elif dr_max > 0.3:
            score += 2; reasons.append(f"IR輝度差が中（Δ相対_max={dr_max:.2f}）")
        elif dr_max > 0.15:
            score += 1; reasons.append(f"IR輝度差がやや大（Δ相対_max={dr_max:.2f}）")
        if any(i.get("pattern") == "strong hotspots" for i in ir_list):
            score += 1; reasons.append("強いホットスポットパターンが一部に存在")
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
    mapping = {
        "A": "10–20年（参考）",
        "B": "7–15年（参考）",
        "C": "3–10年（参考）",
        "D": "1–5年（参考）",
    }
    return mapping.get(grade, "未評価")

# ---------------------- ドメイン知識（一般原則） ----------------------
def domain_priors_text() -> str:
    return normalize_text(
        """
- 写真のみから寸法（ひび幅等）や材料特性を正確に決定することはできない。数値が必要な場合は根拠文献の閾値を引用する。
- 可視：クラック/目地劣化は雨水浸入・仕上げ剥離・躯体劣化につながる可能性がある。
- 赤外線：放射率・反射温度・外気温・湿度・撮影距離/角度・風/日射等の条件に強く影響される。JPEGの疑似温度は絶対値として扱わない。
- 安全確保：剥離・落下リスクがある場合は近接調査（打診/はつり/付着力等）と早期仮設防護を検討する。
- 維持管理：軽微段階での目地/防水補修・再塗装等がライフサイクルコストを低減する。
- 判定は「画像所見」「RAG根拠」「現地条件」の整合で行い、不足があれば“未確定”と表現する。
- 追加調査メニュー：散水試験、打診、付着力、鉄筋腐食指標、中性化試験、含水率等。必要に応じ採否を検討する。
        """
    )

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
    # モデル：Gemini 2.5 Flash
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

# ---------------------- プロンプト（出典必須・強制） ----------------------
def build_master_prompt(user_q: str,
                        rag_snippets: List[Dict[str, Any]],
                        priors: str,
                        vis_list: List[Dict[str, Any]],
                        ir_list: List[Dict[str, Any]],
                        rule_grade: str,
                        rule_life: str,
                        ir_meta_note: str,
                        web_snippets: Optional[List[Dict[str, Any]]] = None) -> str:
    rag_lines = []
    for d in rag_snippets:
        pg = f" p.{d['page_start']}" if d.get("page_start") else ""
        rag_lines.append(f"[{d['doc']}{pg}] {d['text']}")
    if web_snippets:
        for d in web_snippets:
            rag_lines.append(f"[{d['doc']}] {d['text']}")
    rag_text = "\n".join(rag_lines) if rag_lines else "（該当抜粋なし）"

    def pack_vis(v):
        m = v['metrics']
        return f"- edge_ratio={m['edge_ratio']} meanL={m['mean_l']} sat={m['sat_mean']} crack_hint={v['crack_hint']} stain_hint={v['stain_hint']} level={v['screening_level']}"
    vis_block = "\n".join([pack_vis(v) for v in vis_list]) or "（可視画像なし）"

    def pack_ir(i):
        return f"- delta_rel={i['delta_rel']} stdev={i['stdev']} pattern={i['pattern']}"
    ir_block = "\n".join([pack_ir(i) for i in ir_list]) or "（IR画像なし）"

    today = date.today().strftime("%Y年%m月%d日")

    prompt = f"""
あなたは非破壊検査・建築・材料学の上級診断士。国土交通省（MLIT）関連文書の適合性を重視し、与えたRAG抜粋の範囲内で簡潔かつ正確に**診断レポート**を作成する。
**禁止**：推測での数値化（閾値・ひび幅等）／未出典の断定。根拠が無い場合は「未掲載／未確定」と明示する。

# 入力
- 作成日: {today}
- ユーザー質問: {user_q}

- RAG抜粋（出典ページ付き、これ以外は根拠にしない）:
{rag_text}

- ドメイン一般原則（閾値は含まない）:
{priors}

- 可視画像所見（複数）:
{vis_block}

- IR画像所見（複数・相対評価）:
{ir_block}
{ir_meta_note}

- ルールベース暫定:
  * 暫定グレード: {rule_grade}
  * 参考寿命: {rule_life}

# 出力仕様（Markdown、“結果のみ”でセクション順序・見出しを厳守。Word貼付け前提で箇条書き多用）
- 先頭に **総合評価（A/B/C/D、主因1–2行）** を明示
- 「推定残存寿命」は**幅**で記載。RAG根拠が無ければ「参考（画像・一般原則ベース）」と注記
- IRは相対指標であることを明示し、日射/雨直後など条件の留意点を記す
- **重要：各主要所見（原因候補・基準適合・推奨対策）には、可能な限り行末に必ず `［出典: 文書名/ページ or URL］` を最低1件併記すること。根拠がRAGに無い場合は「未掲載」と明記する。**
""".strip()
    return normalize_text(prompt)

# ---------------------- CSS（非表示注入） ----------------------
def inject_material_css():
    components.html(
        """
<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
<link href=\"https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;900&display=swap\" rel=\"stylesheet\">
<style id=\"app-style\">
:root{--mdc-primary:#2962ff;--mdc-secondary:#00b8d4;--mdc-bg:#f7f9fc;--mdc-surface:#ffffff;--mdc-outline:rgba(0,0,0,.08);--radius:16px;--shadow:0 6px 18px rgba(0,0,0,.08);--tap-min:44px}
@media (prefers-color-scheme: dark){:root{--mdc-bg:#0f1115;--mdc-surface:#171a21;--mdc-outline:rgba(255,255,255,.08)}}
.block-container{padding-top:2.2rem!important;padding-bottom:2rem}
body{background:var(--mdc-bg)}
.jp-sans{font-family:'Noto Sans JP',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Kaku Gothic ProN','Hiragino Sans','Meiryo',sans-serif!important;line-height:1.7}
.jp-report *{font-family:'Noto Sans JP',system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI','Hiragino Kaku Gothic ProN','Hiragino Sans','Meiryo',sans-serif!important;line-height:1.6}
.app-hero{background:linear-gradient(135deg,var(--mdc-primary),var(--mdc-secondary));color:#fff;border-radius:20px;box-shadow:var(--shadow);padding:14px 16px;margin:0 0 14px 0}
.app-hero-title{font-weight:900;font-size:1.45rem;line-height:1.25;margin:0 0 4px 0;text-shadow:0 1px 2px rgba(0,0,0,.18)}
.app-hero-sub{font-weight:500;font-size:.95rem;opacity:.95;margin:0}
.md-card{background:var(--mdc-surface);border-radius:var(--radius);box-shadow:var(--shadow);padding:1rem 1.1rem;margin:0 0 1rem 0;border:1px solid var(--mdc-outline)}
.md-title{font-size:1.1rem;font-weight:700;margin:0 0 .6rem 0}
.stButton>button,.stTextInput input,.stFileUploader label,.stCameraInput label{min-height:var(--tap-min);border-radius:12px!important;font-weight:600}
.stButton>button{background:linear-gradient(135deg,var(--mdc-primary),var(--mdc-secondary))!important;color:#fff!important;border:none!important;box-shadow:var(--shadow)}
:where(button,input,select,textarea):focus-visible{outline:3px solid color-mix(in srgb,var(--mdc-primary) 60%, white);outline-offset:2px;border-radius:12px}
</style>
        """,
        height=0,
    )

# ---------------------- 位置（JSボタン式フォールバック） ----------------------
def geolocate_with_button(label: str = "📍 現在地を取得") -> Tuple[Optional[float], Optional[float]]:
    params = st.experimental_get_query_params()
    lat = params.get("lat", [None])[0]
    lon = params.get("lon", [None])[0]
    lat_res = float(lat) if lat else None
    lon_res = float(lon) if lon else None

    # ボタン押下時、JSで取得→URL書き換え→ソフトリロード
    clicked = st.button(label, use_container_width=True)
    if clicked:
        components.html(
            """
<script>
(function(){
  function setQuery(lat, lon){
    const url = new URL(window.location.href);
    url.searchParams.set('lat', lat.toFixed(6));
    url.searchParams.set('lon', lon.toFixed(6));
    if (window.top === window){
      window.location.replace(url.toString());
    }else{
      // iframe内でも同一オリジンなら遷移
      window.location.href = url.toString();
    }
  }
  function onErr(e){ alert('位置情報の取得に失敗しました: ' + (e && e.message ? e.message : '許可が必要です')); }
  if (!navigator.geolocation){ onErr({message:'本ブラウザは位置情報に未対応です'}); return; }
  navigator.geolocation.getCurrentPosition(function(pos){ setQuery(pos.coords.latitude, pos.coords.longitude); }, onErr, {enableHighAccuracy:true, timeout:10000, maximumAge:0});
})();
</script>
            """,
            height=0,
        )
    return lat_res, lon_res

# ---------------------- セッション（ギャラリー） ----------------------
def ensure_galleries():
    if "vis_gallery" not in st.session_state:
        st.session_state["vis_gallery"] = []
    if "ir_gallery" not in st.session_state:
        st.session_state["ir_gallery"] = []

def add_to_gallery(pil_img: Image.Image, key: str):
    if pil_img is None:
        return
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.width > 1400:
        r = 1400 / float(pil_img.width)
        pil_img = pil_img.resize((1400, int(pil_img.height * r)))
    st.session_state[key].append(pil_img)
    total = len(st.session_state["vis_gallery"]) + len(st.session_state["ir_gallery"])
    while total > MAX_IMAGES_TOTAL:
        if st.session_state["ir_gallery"]:
            st.session_state["ir_gallery"].pop(0)
        elif st.session_state["vis_gallery"]:
            st.session_state["vis_gallery"].pop(0)
        total = len(st.session_state["vis_gallery"]) + len(st.session_state["ir_gallery"])

def remove_from_gallery(key: str, idx: int):
    try:
        if key in st.session_state and 0 <= idx < len(st.session_state[key]):
            st.session_state[key].pop(idx)
    except Exception:
        pass

# ---------------------- メイン ----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_material_css()
    ensure_galleries()

    st.markdown(
        f"""
<div class=\"app-hero jp-sans\">
  <div class=\"app-hero-title\">🏗️ {APP_TITLE}</div>
  <div class=\"app-hero-sub\">スマホ最適 / 複数画像（可視・赤外）× RAG × Web併用（任意）× ドメイン知識 × Gemini 2.5 Flash</div>
</div>
        """,
        unsafe_allow_html=True
    )

    # 高速モード・デバッグ
    col_mode1, col_mode2 = st.columns(2)
    with col_mode1:
        fast_mode = st.toggle("⚡ 高速モード（代表画像・低解像度・RAG縮小・Web検索OFF）", value=True)
    with col_mode2:
        debug_mode = st.toggle("🐞 デバッグログを表示", value=False)

    # 1) 質問
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">1) 質問</div>', unsafe_allow_html=True)
    user_q = st.text_input("例：外壁タイルのひび割れ基準と推定寿命、雨漏り原因特定など", "", placeholder="分析したいテーマ・質問を入力")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2) Web検索併用（任意）
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">2) RAG不足時の Web検索 併用（任意・公的サイト優先）</div>', unsafe_allow_html=True)
    use_web = st.toggle("RAGで不足する場合は Web検索も併用（.go.jp / PDF を優先・出典URL併記）", value=False, disabled=fast_mode)
    st.caption("Bing または SerpAPI の APIキーが secrets に設定されている場合のみ有効。高速モードでは既定でOFFです。")
    st.markdown('</div>', unsafe_allow_html=True)

    # 3) 画像（複数・背面カメラ優先）
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">3) 画像入力（可視/赤外：複数可）</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        st.markdown("**可視（複数）**")
        if HAVE_BACK_CAM:
            st.caption("📷 リアカメラで撮影 → 『可視に追加』でギャラリーへ")
            bc_img = back_camera_input(key="vis_back")
            if st.button("➕ 撮影画像を可視に追加", use_container_width=True):
                try:
                    if bc_img is not None:
                        if isinstance(bc_img, Image.Image):
                            add_to_gallery(bc_img, "vis_gallery")
                        else:
                            add_to_gallery(Image.open(io.BytesIO(bc_img)), "vis_gallery")
                except Exception as e:
                    if debug_mode:
                        st.exception(e)
                    st.warning("撮影画像の追加に失敗しました。")
        else:
            st.info("背面カメラコンポーネントが未インストールのため、標準カメラにフォールバックします。")

        vis_cam = st.camera_input("カメラで撮影（フォールバック）", key="vis_cam")
        if st.button("➕ 撮影画像を可視に追加（フォールバック）", use_container_width=True) and vis_cam is not None:
            try:
                add_to_gallery(Image.open(vis_cam), "vis_gallery")
            except Exception as e:
                if debug_mode:
                    st.exception(e)
                st.warning("撮影画像の追加に失敗しました。")

        vis_files = st.file_uploader("可視画像を選択（複数可：JPG/PNG）", type=["jpg","jpeg","png"], accept_multiple_files=True, key="vis_up")
        if vis_files and st.button("📥 選択した可視画像をすべて追加", use_container_width=True):
            for f in vis_files:
                try:
                    add_to_gallery(Image.open(f), "vis_gallery")
                except Exception:
                    pass

    with colB:
        st.markdown("**赤外線（IR）（複数）**")
        ir_files = st.file_uploader("IR画像を選択（複数可：JPG/PNG）", type=["jpg","jpeg","png"], accept_multiple_files=True, key="ir_up")
        if ir_files and st.button("📥 選択したIR画像をすべて追加", use_container_width=True):
            for f in ir_files:
                try:
                    add_to_gallery(Image.open(f), "ir_gallery")
                except Exception:
                    pass
        with st.expander("IRメタデータ（任意・精度向上）"):
            st.session_state["ir_emiss"] = st.text_input("放射率 ε（例:0.95）", st.session_state.get("ir_emiss",""))
            st.session_state["ir_tref"]  = st.text_input("反射温度 T_ref[℃]（例:20）", st.session_state.get("ir_tref",""))
            st.session_state["ir_tamb"]  = st.text_input("外気温 T_amb[℃]（例:22）", st.session_state.get("ir_tamb",""))
            st.session_state["ir_rh"]    = st.text_input("相対湿度 RH[%]（例:65）", st.session_state.get("ir_rh",""))
            st.session_state["ir_dist"]  = st.text_input("撮影距離[m]（例:5）", st.session_state.get("ir_dist",""))
            st.session_state["ir_ang"]   = st.text_input("撮影角度[°]（例:10）", st.session_state.get("ir_ang",""))

    st.markdown('</div>', unsafe_allow_html=True)

    # ギャラリー（削除ボタン付）
    if st.session_state["vis_gallery"] or st.session_state["ir_gallery"]:
        st.markdown('<div class="md-card">', unsafe_allow_html=True)
        st.markdown('<div class="md-title">画像ギャラリー（上限8枚／古い順に入替）</div>', unsafe_allow_html=True)

        if st.session_state["vis_gallery"]:
            st.markdown("**可視**")
            cols = st.columns(4)
            for i, img in enumerate(st.session_state["vis_gallery"]):
                with cols[i % 4]:
                    st.image(img, use_container_width=True)
                    if st.button(f"🗑️ 削除 可視 {i+1}", key=f"del_vis_{i}", use_container_width=True):
                        remove_from_gallery("vis_gallery", i)
                        st.experimental_rerun()

        if st.session_state["ir_gallery"]:
            st.markdown("**IR**")
            cols2 = st.columns(4)
            for j, img in enumerate(st.session_state["ir_gallery"]):
                with cols2[j % 4]:
                    st.image(img, use_container_width=True)
                    if st.button(f"🗑️ 削除 IR {j+1}", key=f"del_ir_{j}", use_container_width=True):
                        remove_from_gallery("ir_gallery", j)
                        st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # 4) 位置（ボタン式 geolocation）
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">4) 位置情報（現在地 or 手入力）</div>', unsafe_allow_html=True)

    lat_val, lon_val = geolocate_with_button()

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        lat_str = "" if lat_val is None else f"{lat_val:.6f}"
        lat_str = st.text_input("緯度", value=lat_str, key="lat_manual")
    with c2:
        lon_str = "" if lon_val is None else f"{lon_val:.6f}"
        lon_str = st.text_input("経度", value=lon_str, key="lon_manual")
    with c3:
        st.caption("※ 『現在地を取得』で許可が必要です。取得できない場合は手入力してください。")

    try:
        lat_f = float(lat_str) if lat_str else None
        lon_f = float(lon_str) if lon_str else None
    except Exception:
        lat_f, lon_f = None, None

    if lat_f is not None and lon_f is not None:
        m = folium.Map(location=[lat_f, lon_f], zoom_start=18, tiles="OpenStreetMap")
        folium.Marker([lat_f, lon_f], tooltip="対象地点").add_to(m)
        st_folium(m, height=300, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 5) 実行
    st.markdown('<div class="md-card">', unsafe_allow_html=True)
    st.markdown('<div class="md-title">5) 解析の実行</div>', unsafe_allow_html=True)
    run = st.button("🔎 解析する", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        # 先にバリデーション（ここで止まるときは明示表示）
        problems = []
        if not user_q:
            problems.append("質問を入力してください。")
        if not (st.session_state["vis_gallery"] or st.session_state["ir_gallery"]):
            problems.append("少なくとも1枚の画像（可視またはIR）を追加してください。")
        if problems:
            st.error("\n".join(f"• {p}" for p in problems))
            if debug_mode:
                st.write({"debug": "validation_failed", "problems": problems})
            st.stop()

        # 進捗バー開始
        progress = st.progress(0, text="RAG準備中…")
        step = 10
        progress.progress(step)

        # RAG（高速モードでは上位Kを縮小）
        try:
            k_snip = 4 if fast_mode else MAX_SNIPPETS
            snippets = rag_search(user_q, have_ir=bool(st.session_state["ir_gallery"]), k=k_snip)
        except Exception as e:
            progress.empty()
            st.error("RAG 準備に失敗しました。PDFの配置や権限を確認してください。")
            if debug_mode:
                st.exception(e)
            st.stop()

        step = 35; progress.progress(step, text="画像所見の集約中…")

        # 画像所見（高速モードでは低解像度統計）
        vis_target_w = 128 if fast_mode else 256
        ir_target_w = 128 if fast_mode else 256
        try:
            vis_pairs = [(img, analyze_visual(img, target_w=vis_target_w)) for img in st.session_state["vis_gallery"]]
            ir_pairs  = [(img, analyze_ir(img, {
                "emissivity": (st.session_state.get('ir_emiss') or "不明").strip(),
                "t_ref": (st.session_state.get('ir_tref') or "不明").strip(),
                "t_amb": (st.session_state.get('ir_tamb') or "不明").strip(),
                "rh": (st.session_state.get('ir_rh') or "不明").strip(),
                "dist": (st.session_state.get('ir_dist') or "不明").strip(),
                "angle": (st.session_state.get('ir_ang') or "不明").strip(),
            }, target_w=ir_target_w)) for img in st.session_state["ir_gallery"]]
        except Exception as e:
            progress.empty()
            st.error("画像処理でエラーが発生しました。画像形式・サイズをご確認ください。")
            if debug_mode:
                st.exception(e)
            st.stop()

        vis_list = [v for (_, v) in vis_pairs]
        ir_list  = [i for (_, i) in ir_pairs]
        step = 55; progress.progress(step, text="暫定評価を計算中…")

        # 暫定評価
        rule_grade, rule_reason = rule_based_grade(vis_list, ir_list)
        rule_life = rule_based_life(rule_grade)

        # Web検索（任意）
        step = 65; progress.progress(step, text="Web検索（任意）を実行中…")
        web_snips: List[Dict[str, Any]] = []
        if use_web and not fast_mode:
            try:
                web_snips = web_search_snippets_cached(user_q, max_items=3)
            except Exception as e:
                if debug_mode:
                    st.exception(e)

        step = 75; progress.progress(step, text="プロンプトを作成中…")

        # 画像送信の選抜（高速モード：可視上位2 + IR上位1）
        image_parts: List[Dict[str, Any]] = []
        if fast_mode:
            if vis_pairs:
                vis_sorted = sorted(vis_pairs, key=lambda p: p[1]['metrics']['edge_ratio'], reverse=True)
                for img, _ in vis_sorted[:2]:
                    image_parts.append(image_to_inline_part(img, max_width=1000))
            if ir_pairs:
                ir_sorted = sorted(ir_pairs, key=lambda p: p[1]['delta_rel'], reverse=True)
                for img, _ in ir_sorted[:1]:
                    image_parts.append(image_to_inline_part(img, max_width=1000))
        else:
            for img, _ in vis_pairs:
                image_parts.append(image_to_inline_part(img, max_width=1400))
            for img, _ in ir_pairs:
                image_parts.append(image_to_inline_part(img, max_width=1400))

        ir_meta_note = (
            "注: 赤外線画像はJPEG相対評価。放射率/反射温度/外気温/湿度/距離/角度/風/日射等に影響され、"
            "絶対温度は扱わない。雨直後・散水直後・非日射時間帯での再撮影が有効。"
        )

        priors = domain_priors_text()
        prompt = build_master_prompt(
            user_q=user_q,
            rag_snippets=snippets,
            priors=priors,
            vis_list=vis_list,
            ir_list=ir_list,
            rule_grade=rule_grade,
            rule_life=rule_life,
            ir_meta_note=ir_meta_note,
            web_snippets=web_snips if (use_web and not fast_mode) else None
        )

        step = 85; progress.progress(step, text="Gemini API に送信中…")

        # API キー
        try:
            api_key = st.secrets["gemini"]["API_KEY"]
        except (KeyError, FileNotFoundError):
            progress.empty()
            st.error("Gemini API Key が未設定です。.streamlit/secrets.toml に [gemini].API_KEY を設定してください。")
            st.stop()

        # 呼び出し
        try:
            result = call_gemini(api_key, prompt, image_parts)
            report_md = extract_text_from_gemini(result)
        except requests.HTTPError as e:
            progress.empty()
            st.error(f"APIエラー: {e.response.status_code} {e.response.reason}\n{e.response.text[:500]}")
            if debug_mode:
                st.write({"payload_chars": len(prompt), "images": len(image_parts)})
            st.stop()
        except Exception as e:
            progress.empty()
            st.error(f"呼び出しエラー: {e}")
            if debug_mode:
                st.write({"payload_chars": len(prompt), "images": len(image_parts)})
                st.exception(e)
            st.stop()

        step = 95; progress.progress(step, text="レポート整形中…")

        if not report_md:
            progress.empty()
            st.warning("レポートが空でした。入力（質問/画像/PDF）を見直してください。")
            st.stop()

        # 結果表示：総合評価 → 詳細
        summary_block = None
        try:
            pattern = r"(?:^|\n)##?\s*総合評価[\s\S]*?(?=\n##?\s|\Z)"
            m = re.search(pattern, report_md, re.IGNORECASE)
            if m:
                summary_block = m.group(0)
        except Exception:
            summary_block = None

        st.markdown('<div class="md-card good-shadow jp-report">', unsafe_allow_html=True)
        st.markdown('<div class="md-title">6) 解析結果</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if summary_block:
            st.markdown('<div class="md-card good-shadow jp-report">', unsafe_allow_html=True)
            st.markdown('<div class="md-title">🧭 総合評価（先頭要約）</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='jp-report'>{summary_block}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("詳細レポートを表示（Word貼付け可）", expanded=(summary_block is None)):
            st.markdown("**（参考）画像＋一般原則による暫定評価**")
            st.markdown(f"- 暫定グレード: `{rule_grade}`（{rule_reason}）")
            st.markdown(f"- 参考寿命: `{rule_life}`")
            st.markdown("---")
            st.markdown(f"<div class='jp-report'>{report_md}</div>", unsafe_allow_html=True)
            st.markdown("###### 📋 Word貼付け用テキスト（全選択→コピー）")
            st.text_area("", value=report_md, height=260, label_visibility="collapsed")

        # ダウンロード（MD / TXT with BOM）
        md_bytes = report_md.encode("utf-8")
        txt_bytes = report_md.encode("utf-8-sig")  # BOM付き
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📄 レポートをダウンロード（Markdown）",
                md_bytes,
                file_name="building_health_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                "📄 レポートをダウンロード（TXT／Word向け・BOM付）",
                txt_bytes,
                file_name="building_health_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        progress.progress(100, text="完了")
        progress.empty()

    st.caption("© 建物診断くん — 複数画像 × RAG × Web併用（任意）× ドメイン知識 × Gemini 2.5 Flash。")

if __name__ == "__main__":
    main()
