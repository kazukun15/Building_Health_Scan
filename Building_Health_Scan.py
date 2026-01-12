# -*- coding: utf-8 -*-
# ===========================================================
# 建物診断くん Pro（壁・タイル・防水シート特化 / RAG制御 / 顕微鏡統合版）
# Refactored by World Class Programmer
# ===========================================================

import pkgutil
# インポートエラー回避のためのパッチ
if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = pkgutil.zipimporter

import os
import io
import re
import math
import base64
import statistics
import unicodedata
import time
from datetime import date
from typing import List, Tuple, Dict, Optional, Any, Set

import streamlit as st
import requests
import PyPDF2
from PIL import Image, ImageFilter

import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components

# --- Optional Dependencies ---
HAVE_BACK_CAM = False
try:
    from streamlit_back_camera_input import back_camera_input
    HAVE_BACK_CAM = True
except ImportError:
    pass

# ===========================================================
# 1. Config & Constants
# ===========================================================

APP_TITLE = "建物診断くん Pro"
APP_VERSION = "2.1.0 (Microscope Edition)"

# PDFソース定義（お手元のPDFファイル名に合わせてください）
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

# システム制限値
MAX_SNIPPETS = 8
MAX_SNIPPET_CHARS = 1000
MAX_IMAGES_TOTAL = 12 # 顕微鏡用に少し増加

# 類語辞書
QUERY_SYNONYMS = {
    "ひび割れ": ["ひび割れ", "クラック", "亀裂", "ひび"],
    "浮き": ["浮き", "うき"],
    "剥離": ["剥離", "はく離", "はくり"],
    "含水": ["含水", "浸水", "漏水", "雨水侵入", "雨漏り"],
    "赤外線": ["赤外線", "IR", "サーモ", "熱画像", "サーモグラフィ"],
    "基準": ["基準", "判定", "評価", "閾値", "許容"],
    "タイル": ["タイル", "磁器質タイル", "モザイクタイル"],
    "ALC": ["ALC", "軽量気泡コンクリート"],
    "コンクリート": ["コンクリート", "RC", "中性化"],
    "防水": ["防水", "塗膜", "シーリング", "伸縮目地", "防水シート", "シート防水"],
    "劣化度": ["劣化度", "グレード", "区分", "A", "B", "C", "D"],
}

# キーワード定義
MATERIAL_KEYWORDS: Dict[str, List[str]] = {
    "タイル": ["タイル", "磁器質タイル", "モザイクタイル", "外壁タイル", "目地"],
    "壁": ["壁", "外壁", "内壁", "パラペット", "躯体", "モルタル", "仕上げ"],
    "防水シート": ["防水シート", "シート防水", "塩ビシート", "ゴムシート", "屋上防水", "立上り", "端末"],
    "コンクリート": ["コンクリート", "RC", "鉄筋コンクリート"],
    "シーリング": ["シーリング", "コーキング", "目地材"],
}

DEFECT_KEYWORDS: Dict[str, List[str]] = {
    "ひび割れ": ["ひび割れ", "クラック", "亀裂"],
    "浮き": ["浮き", "うき"],
    "剥離": ["剥離", "はく離", "はくり", "欠損", "落下"],
    "漏水": ["漏水", "雨漏り", "浸水", "水漏れ", "雨水侵入"],
    "ふくれ": ["ふくれ", "膨れ", "ブリスター"],
    "破断": ["破断", "裂け", "切れ", "破れ"],
    "端部": ["端部", "端末", "めくれ", "剥がれ"],
    "ドレン": ["ドレン", "排水口", "ルーフドレン", "詰まり"],
}

# 診断テンプレート
TEMPLATES = {
    "外壁（タイル）": {
        "material_tags": ["タイル", "壁", "シーリング"],
        "checklist": [
            "全景（建物全体・方位が分かる）",
            "中景（症状が出ている面）",
            "近景（スケール入り：定規/コイン等）",
            "顕微鏡（目地の欠損・浮き口）",
            "開口部周り（サッシ・水切り・取り合い）",
            "可能なら打診位置（浮き疑い）",
        ],
        "rag_preset": "base_focus",
    },
    "壁（仕上げ・躯体）": {
        "material_tags": ["壁", "コンクリート", "シーリング"],
        "checklist": [
            "全景（建物全体・方位）",
            "中景（症状が出ている面）",
            "近景（スケール入り）",
            "顕微鏡（ひび割れ内部・エフロ）",
            "雨掛かりライン（汚れ筋・白華）",
            "開口部・取り合い（クラック起点になりやすい）",
        ],
        "rag_preset": "base_focus",
    },
    "屋上（防水シート）": {
        "material_tags": ["防水シート", "防水", "ドレン", "端部"],
        "checklist": [
            "全景（屋上全体：勾配・排水系が分かる）",
            "シート継ぎ目（ジョイント・溶着部）",
            "立上り（パラペット取り合い）",
            "端末押さえ（金物・シール）",
            "顕微鏡（シート表面の紫外線劣化/亀裂）",
            "ふくれ/浮き/破れの近景（スケール入り）",
        ],
        "rag_preset": "base_and_plan",
    },
}

SYMPTOMS = [
    "ひび割れ",
    "浮き",
    "剥離・欠損・落下懸念",
    "漏水・雨漏り",
    "ふくれ（ブリスター）",
    "破断・裂け・破れ",
    "端部めくれ・端末不良",
    "シーリング劣化",
    "ドレン詰まり疑い",
    "原因不明（総合）",
]


# ===========================================================
# 2. Helper Functions (Text, Image, Utils)
# ===========================================================

def normalize_text(text: str) -> str:
    """テキストの正規化（NFKC, 改行削除, 空白短縮）"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(s: str) -> List[str]:
    """簡易トークナイザー"""
    s = s.lower()
    s = re.sub(r"[^\w一-龥ぁ-んァ-ン％℃．\.]", " ", s)
    s = s.replace("．", ".")
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


def query_expand_tokens(q: str) -> List[str]:
    """同義語展開によるクエリ拡張"""
    tokens = set(tokenize(q))
    for key, syns in QUERY_SYNONYMS.items():
        if key in q or any(s in q for s in syns):
            for s2 in syns:
                tokens.update(tokenize(s2))
    return list(tokens)


def pil_stats(image: Image.Image, target_w: int = 256) -> Dict[str, float]:
    """画像から簡易的な統計量を抽出"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # リサイズして計算コスト削減
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
    """可視画像の簡易解析"""
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
    """IR（赤外線）画像の簡易解析"""
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


def image_to_inline_part(image: Image.Image, max_width: int = 1400) -> Dict:
    """Gemini API用の画像ペイロード作成"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.width > max_width:
        r = max_width / float(image.width)
        image = image.resize((max_width, int(image.height * r)))
        
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/jpeg", "data": b64}}


# ===========================================================
# 3. RAG Core Logic (PDF Processing & Search)
# ===========================================================

class BM25Index:
    def __init__(self, k1: float = 1.6, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = 0
        self.avgdl = 0.0
        self.df: Dict[str, int] = {}
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.docs: List[Dict[str, Any]] = []

    def _tf(self, tokens: List[str]) -> Dict[str, int]:
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


def extract_chunks_from_pdf(pdf_path: str, title: str, max_chars: int = 900, overlap: int = 120) -> List[Dict[str, Any]]:
    if not os.path.exists(pdf_path):
        return []
    chunks = []
    base = os.path.basename(pdf_path)
    doc_type = DOC_TYPES.get(base, "その他")
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
                            "doc_type": doc_type,
                        })
                    pos = end - overlap if (end - overlap) > pos else end
    except Exception as e:
        chunks.append({
            "doc": title, "path": pdf_path,
            "text": f"[PDF読込エラー:{pdf_path}:{e}]",
            "page_start": None, "page_end": None,
            "doc_type": doc_type,
        })
    return chunks


@st.cache_resource(show_spinner=False)
def build_rag() -> Dict[str, Any]:
    all_chunks = []
    for title, path in PDF_SOURCES:
        all_chunks.extend(extract_chunks_from_pdf(path, title))

    for d in all_chunks:
        txt = d["text"]
        d["_has_numbers"] = bool(re.search(r"\b\d+(\.\d+)?\s*(mm|㎜|％|%|℃)\b", txt))
        d["_has_ir"] = any(k in txt for k in ["赤外線", "サーモ", "IR", "熱画像", "放射率", "反射率", "反射温度"])

        mats = set()
        defs = set()
        for mk, words in MATERIAL_KEYWORDS.items():
            if any(w in txt for w in words):
                mats.add(mk)
        for dk, words in DEFECT_KEYWORDS.items():
            if any(w in txt for w in words):
                defs.add(dk)
        d["_materials"] = mats
        d["_defects"] = defs

    bm25 = BM25Index()
    if all_chunks:
        bm25.build(all_chunks)
    return {"index": bm25, "docs": all_chunks}


def rag_search(query: str, have_ir: bool, k: int, weights: Dict[str, float]) -> List[Dict[str, Any]]:
    data = build_rag()
    bm25: BM25Index = data["index"]
    docs = data["docs"]
    if not docs:
        return []

    q_tokens = query_expand_tokens(query)
    
    # クエリ解析
    q_mats = {mk for mk, words in MATERIAL_KEYWORDS.items() if any(w in query for w in words)}
    q_defs = {dk for dk, words in DEFECT_KEYWORDS.items() if any(w in query for w in words)}
    
    want_threshold = any(w in query for w in ["基準", "閾値", "幅", "mm", "㎜", "％", "%", "℃", "温度", "判定", "許容"])
    want_ir_q = any(w in query for w in ["赤外線", "IR", "サーモ", "熱画像", "サーモグラフィ"])
    
    boost_muni = "港区" if "港区" in query else ("上島町" if ("上島町" in query or "上嶋町" in query) else None)

    scored = []
    for doc_id, d in enumerate(docs):
        base = bm25.score_doc(q_tokens, doc_id)
        if base <= 0:
            continue

        if want_threshold and d.get("_has_numbers"):
            base *= weights["number_boost"]
        if (have_ir or want_ir_q) and d.get("_has_ir"):
            base *= weights["ir_boost"]

        doc_type = d.get("doc_type", "その他")
        if doc_type == "基準":
            base *= weights["base_weight"]
        elif doc_type == "計画":
            base *= weights["plan_weight"]
        else:
            base *= weights["other_weight"]

        mats = d.get("_materials", set())
        defs = d.get("_defects", set())
        mat_match = bool(q_mats & mats)
        def_match = bool(q_defs & defs)

        if mat_match:
            base *= weights["material_boost"]
        if def_match:
            base *= weights["defect_boost"]
        if mat_match and def_match:
            base *= weights["mat_def_synergy"]

        if boost_muni and boost_muni in d.get("doc", ""):
            base *= weights["muni_boost"]

        scored.append((base, doc_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [docs[i] for (score, i) in scored[:k]]

    for t in top_docs:
        if len(t["text"]) > MAX_SNIPPET_CHARS:
            t["text"] = t["text"][:MAX_SNIPPET_CHARS] + "…"
    return top_docs


# ===========================================================
# 4. Prompt & Report Logic
# ===========================================================

def build_auto_question(template_name: str, symptoms: List[str], free_text: str, want_threshold: bool, want_plan: bool) -> str:
    """ユーザー入力からRAG検索に最適な質問文を生成"""
    base = f"対象: {template_name}。症状: {', '.join(symptoms) if symptoms else '未選択'}。"
    add = free_text.strip()
    asks = [
        "観察所見から想定される原因候補（取り合い、目地、端末、排水、下地等）を整理してほしい",
        "落下/剥離/漏水拡大などのリスクと、一次対応（立入制限・仮設防護・応急止水等）の優先度を示してほしい",
        "恒久対策の選択肢（補修・部分改修・全面改修）を、優先順位つきで提案してほしい",
        "追加調査（打診、散水、含水率、付着力、端末開口確認など）の目的と方法を提案してほしい"
    ]

    if want_threshold:
        asks.insert(0, "基準（許容・判定・注意喚起）に該当する記述があれば根拠付きで示してほしい（数値は出典必須）")
    if want_plan:
        asks.insert(1, "維持管理計画・更新優先度・LCC観点で、緊急度/重要度の整理もしてほしい")

    tags = []
    # タグ付けロジック
    for t in TEMPLATES.get(template_name, {}).get("material_tags", []):
        tags.append(t)
    for s in symptoms:
        if "剥離" in s or "落下" in s: tags.append("剥離")
        if "漏水" in s or "雨漏り" in s: tags.append("漏水")
        if "ひび" in s: tags.append("ひび割れ")
        if "ふくれ" in s: tags.append("ふくれ")
        if "破断" in s or "破れ" in s: tags.append("破断")
        if "端部" in s: tags.append("端部")
        if "ドレン" in s: tags.append("ドレン")
        if "シーリング" in s: tags.append("シーリング")

    tag_str = " / ".join(sorted(set(tags))) if tags else ""

    q = f"""{base}
補足: {add if add else '（なし）'}

依頼:
- {chr(10)+'- '.join(asks)}

検索キーワード（RAG補助）: {tag_str}
""".strip()
    return normalize_text(q)


def call_gemini(api_key: str, prompt_text: str, image_parts: List[Dict], max_retries: int = 3) -> Dict:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + api_key
    headers = {"Content-Type": "application/json"}
    parts = [{"text": prompt_text}] + image_parts
    payload = {"contents": [{"parts": parts}]}

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (503, 429) and attempt < max_retries:
                wait = min(12, 2 ** attempt + 1)
                st.toast(f"Gemini混雑中（{status}）。{wait}秒後に再試行します...", icon="⚠️")
                time.sleep(wait)
                last_err = e
                continue
            last_err = e
            break
        except Exception as e:
            last_err = e
            break
            
    if last_err:
        raise last_err
    raise RuntimeError("Gemini API call failed.")


def build_master_prompt(user_q, rag_snippets, priors, vis_list, ir_list, micro_list, rule_grade, rule_life, ir_meta_note):
    """プロンプト構築（Gemini送信内容）: 顕微鏡対応"""
    # RAG情報の整形
    labeled_snips = []
    counter = 1
    blocks = []
    
    def format_snips(snips, label):
        nonlocal counter
        lines = []
        for d in snips:
            rid = f"R{counter}"
            dd = dict(d)
            dd["_rag_id"] = rid
            labeled_snips.append(dd)
            pg = f" p.{dd['page_start']}" if dd.get("page_start") else ""
            lines.append(f"[{rid} {label} {dd.get('doc','')}{pg}] {dd.get('text','')}")
            counter += 1
        return "\n".join(lines)

    base_snips = [d for d in rag_snippets if d.get("doc_type") == "基準"]
    plan_snips = [d for d in rag_snippets if d.get("doc_type") == "計画"]
    other_snips = [d for d in rag_snippets if d.get("doc_type") not in ("基準", "計画")]

    if base_snips: blocks.append("# RAG基準候補:\n" + format_snips(base_snips, "基準"))
    if plan_snips: blocks.append("# RAG計画・マネジメント:\n" + format_snips(plan_snips, "計画"))
    if other_snips: blocks.append("# RAGその他参考:\n" + format_snips(other_snips, "参考"))
    
    rag_text = "\n\n".join(blocks) if blocks else "（該当抜粋なし）"

    # 画像情報の整形
    vis_block = "\n".join([f"- edge={v['metrics']['edge_ratio']} crack={v['crack_hint']} level={v['screening_level']}" for v in vis_list]) or "（可視画像なし）"
    ir_block = "\n".join([f"- delta={i['delta_rel']} pattern={i['pattern']}" for i in ir_list]) or "（IR画像なし）"
    
    # 顕微鏡情報の整形 (New)
    micro_block = "（顕微鏡画像なし）"
    if micro_list:
        micro_block = "※1000倍顕微鏡画像が含まれています。微細な亀裂、結晶化、腐食パターン、表面テクスチャを材料科学的に分析してください。"

    today = date.today().strftime("%Y年%m月%d日")

    prompt = f"""
あなたは非破壊検査・建築・材料学の上級診断士。国土交通省（MLIT）関連文書の適合性を重視し、与えたRAG抜粋の範囲内で**診断レポート**を作成する。
禁止：推測での数値化（閾値・ひび幅等）／未出典の断定。根拠が無い場合は「未掲載／未確定」と明示。

# 入力
- 作成日: {today}
- ユーザー質問: {user_q}

- RAG抜粋（ID付き。原則これ以外は根拠にしない）:
{rag_text}

- 一般原則:
{priors}

- 可視画像所見:
{vis_block}

- IR画像所見（相対評価）:
{ir_block}
{ir_meta_note}

- 顕微鏡画像所見（詳細解析用）:
{micro_block}

- ルールベース暫定:
  * 暫定グレード: {rule_grade}
  * 参考寿命: {rule_life}

# 出力仕様（Markdown）
- 先頭：総合評価（A/B/C/D、主因1–2行、確度も一言）
- 基準・計画との整合：RAG ID（例：［根拠: R1,R3］）を必ず付与。なければ［根拠: 未掲載（一般原則ベース）］
- 顕微鏡分析（ある場合）：ミクロ視点での劣化メカニズム（エフロ、初期腐食、紫外線劣化等）への言及。
- リスク（落下・漏水拡大等）→ 応急対応 → 恒久対策（選択肢）→ 追加調査（目的/方法）を必ず含める
- IRは相対指標であり、日射/雨直後等の条件影響を必ず書く
""".strip()
    return normalize_text(prompt), labeled_snips


def rule_based_grade(vis_list, ir_list):
    """画像特徴量のみからの簡易判定ロジック"""
    score = 0.0
    reasons = []
    
    if vis_list:
        er_max = max(v["metrics"]["edge_ratio"] for v in vis_list)
        if er_max > 0.22: score += 3; reasons.append(f"強いエッジ密度（{er_max:.3f}）")
        elif er_max > 0.16: score += 2; reasons.append(f"高めのエッジ密度（{er_max:.3f}）")
        elif er_max > 0.11: score += 1; reasons.append(f"やや高いエッジ密度（{er_max:.3f}）")
        
        if any(v.get("stain_hint") for v in vis_list):
            score += 1; reasons.append("低彩度・暗部多めで汚染傾向")
            
    if ir_list:
        dr_max = max(i["delta_rel"] for i in ir_list)
        if dr_max > 0.5: score += 3; reasons.append(f"IR輝度差が大（{dr_max:.2f}）")
        elif dr_max > 0.3: score += 2; reasons.append(f"IR輝度差が中（{dr_max:.2f}）")
        elif dr_max > 0.15: score += 1; reasons.append(f"IR輝度差がやや大（{dr_max:.2f}）")
        
        if any(i.get("pattern") == "strong hotspots" for i in ir_list):
            score += 1; reasons.append("強いホットスポット")

    if score <= 1: g = "A"
    elif score <= 3: g = "B"
    elif score <= 5: g = "C"
    else: g = "D"
    
    return g, "・".join(reasons) if reasons else "顕著な異常なし（画像ベース）"


# ===========================================================
# 5. UI Components & CSS
# ===========================================================

def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700;900&display=swap');
        html, body, [class*="css"] {
            font-family: 'Noto Sans JP', sans-serif;
            color: #333;
        }
        .hero-container {
            background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            box-shadow: 0 4px 15px rgba(0, 184, 212, 0.3);
            margin-bottom: 2rem;
        }
        .hero-title {
            font-size: 1.8rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
        }
        .hero-subtitle {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .app-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #E5E7EB;
            box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        }
        .card-header {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 1rem;
            border-bottom: 2px solid #F3F4F6;
            padding-bottom: 0.5rem;
            color: #1F2937;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 8px;
            background-color: #F9FAFB;
            border: 1px solid #E5E7EB;
            padding: 0 20px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2563EB;
            color: white !important;
            border: none;
        }
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
        }
        div[data-testid="stToast"] {
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_header():
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title">🏗️ {APP_TITLE}</div>
            <div class="hero-subtitle">
                AI Building Diagnosis / Gemini 2.5 Flash / RAG Fail-Safe System<br>
                可視・IR・<strong>1000倍顕微鏡</strong>のマルチモーダル解析に対応
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def ensure_session_state():
    if "vis_gallery" not in st.session_state: st.session_state["vis_gallery"] = []
    if "ir_gallery" not in st.session_state: st.session_state["ir_gallery"] = []
    if "micro_gallery" not in st.session_state: st.session_state["micro_gallery"] = [] # New

def add_image_to_gallery(img: Image.Image, type_key: str):
    """ギャラリーへ画像追加（リサイズ・上限管理含む）"""
    if img is None: return
    if img.mode != "RGB": img = img.convert("RGB")
    
    # メモリ節約のためリサイズして保持
    if img.width > 1200:
        r = 1200 / float(img.width)
        img = img.resize((1200, int(img.height * r)))
        
    st.session_state[type_key].append(img)
    st.toast("画像を追加しました", icon="📸")
    
    # 上限管理（FIFO）
    all_len = sum(len(st.session_state[k]) for k in ["vis_gallery", "ir_gallery", "micro_gallery"])
    if all_len > MAX_IMAGES_TOTAL:
        # 古いものから消す優先度: IR > Vis > Micro
        if st.session_state["ir_gallery"]: st.session_state["ir_gallery"].pop(0)
        elif st.session_state["vis_gallery"]: st.session_state["vis_gallery"].pop(0)
        elif st.session_state["micro_gallery"]: st.session_state["micro_gallery"].pop(0)

# ===========================================================
# 6. Main App
# ===========================================================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    inject_custom_css()
    ensure_session_state()
    
    # サイドバー（設定・ステータス）
    with st.sidebar:
        st.markdown("### ⚙️ システム設定")
        fast_mode = st.toggle("⚡ 高速モード", value=True, help="RAG検索数と画像サイズを縮小してレスポンスを優先します。")
        debug_mode = st.toggle("🐞 デバッグモード", value=False)
        
        st.divider()
        st.markdown("### 📊 ステータス")
        
        # API Key check
        has_api_key = False
        try:
            if st.secrets["gemini"]["API_KEY"]:
                has_api_key = True
        except:
            pass
            
        if has_api_key:
            st.success("API Key: OK")
        else:
            st.error("API Key: 未設定")
            st.caption(".streamlit/secrets.toml を確認してください")

        # PDF check
        missing = [p for _, p in PDF_SOURCES if not os.path.exists(p)]
        if missing:
            st.warning(f"PDF不足: {len(missing)}件")
        else:
            st.success("RAG Data: OK")
            
        st.divider()
        st.caption(f"Ver {APP_VERSION}")

    render_header()

    # メインタブ
    tab1, tab2, tab3, tab4 = st.tabs([
        "① 診断テンプレ", 
        "② 画像アップロード", 
        "③ 位置情報", 
        "④ 診断実行"
    ])

    # --- Tab 1: テンプレートと質問 ---
    with tab1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📋 診断条件の設定</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            template_name = st.selectbox("対象部位（テンプレート）", list(TEMPLATES.keys()))
            want_threshold = st.checkbox("基準値（判定基準）を優先検索", value=True)
        with col2:
            symptoms = st.multiselect("確認された症状", SYMPTOMS, default=["原因不明（総合）"])
            want_plan = st.checkbox("保全計画・LCC観点を含める", value=(template_name == "屋上（防水シート）"))
            
        free_text = st.text_area("補足情報（任意）", placeholder="例：顕微鏡画像はひび割れ深部を撮影。築30年。", height=80)
        
        # 自動生成ロジック
        auto_q = build_auto_question(template_name, symptoms, free_text, want_threshold, want_plan)
        
        with st.expander("📝 自動生成されたRAG検索クエリを確認", expanded=True):
            st.text_area("検索プロンプト（編集可）", value=auto_q, height=140, label_visibility="collapsed")
            
        st.info("💡 **撮影チェックリスト**: " + " / ".join(TEMPLATES[template_name]["checklist"]) )
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 2: 画像入力 (顕微鏡対応) ---
    with tab2:
        col_input, col_gallery = st.columns([1, 2])
        
        with col_input:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📷 画像追加</div>', unsafe_allow_html=True)
            
            # Sub-tabs for image types
            tab_vis, tab_ir, tab_micro = st.tabs(["可視画像", "IR画像", "🔬 顕微鏡"])
            
            with tab_vis:
                if HAVE_BACK_CAM:
                    bc_img = back_camera_input(key="vis_back")
                    if bc_img:
                        if st.button("現在の映像を追加", key="btn_add_back"):
                            add_image_to_gallery(Image.open(io.BytesIO(bc_img)), "vis_gallery")
                vis_cam = st.camera_input("インカメラ撮影", key="vis_cam", label_visibility="collapsed")
                if vis_cam: add_image_to_gallery(Image.open(vis_cam), "vis_gallery")
                vis_files = st.file_uploader("フォルダから選択", type=["jpg","png"], accept_multiple_files=True, key="vis_up")
                if vis_files:
                    for f in vis_files: add_image_to_gallery(Image.open(f), "vis_gallery")

            with tab_ir:
                st.caption("赤外線サーモグラフィ画像")
                ir_files = st.file_uploader("IR画像を選択", type=["jpg","png"], accept_multiple_files=True, key="ir_up")
                if ir_files:
                    for f in ir_files: add_image_to_gallery(Image.open(f), "ir_gallery")
                
                with st.expander("IR撮影条件（メタデータ）"):
                    st.session_state["ir_emiss"] = st.text_input("放射率 ε", "0.95")
                    st.session_state["ir_tref"] = st.text_input("反射温度 [℃]", "20.0")

            with tab_micro:
                st.caption("1000倍顕微鏡などの拡大画像")
                st.info("微細なひび割れ、汚れの結晶、塗膜の荒れなどをアップロードしてください。")
                micro_files = st.file_uploader("顕微鏡画像を選択", type=["jpg","png"], accept_multiple_files=True, key="micro_up")
                
                # PC直結顕微鏡がWebカメラ認識される場合の対応
                use_micro_cam = st.checkbox("顕微鏡カメラ入力を使う")
                if use_micro_cam:
                    micro_cam = st.camera_input("顕微鏡カメラ", key="micro_cam")
                    if micro_cam: add_image_to_gallery(Image.open(micro_cam), "micro_gallery")

                if micro_files:
                    for f in micro_files: add_image_to_gallery(Image.open(f), "micro_gallery")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_gallery:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            total_imgs = sum(len(st.session_state[k]) for k in ["vis_gallery", "ir_gallery", "micro_gallery"])
            st.markdown(f'<div class="card-header">🖼️ ギャラリー ({total_imgs}/{MAX_IMAGES_TOTAL})</div>', unsafe_allow_html=True)
            
            if total_imgs == 0:
                st.info("画像がありません。左側から追加してください。")
            
            # Helper to display gallery grid
            def show_grid(key, label):
                if st.session_state[key]:
                    st.caption(f"**{label}**")
                    cols = st.columns(4)
                    for i, img in enumerate(st.session_state[key]):
                        with cols[i % 4]:
                            st.image(img, use_container_width=True)
                            if st.button("削除", key=f"del_{key}_{i}", use_container_width=True):
                                st.session_state[key].pop(i)
                                st.rerun()

            show_grid("vis_gallery", "可視画像")
            show_grid("ir_gallery", "IR画像")
            show_grid("micro_gallery", "🔬 顕微鏡画像")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 3: 位置情報 ---
    with tab3:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📍 診断場所</div>', unsafe_allow_html=True)
        
        col_map, col_coords = st.columns([2, 1])
        with col_coords:
            st.write("位置情報を指定（任意）")
            lat_in = st.number_input("緯度", value=35.658581, format="%.6f")
            lon_in = st.number_input("経度", value=139.745433, format="%.6f")
        with col_map:
            m = folium.Map(location=[lat_in, lon_in], zoom_start=16)
            folium.Marker([lat_in, lon_in], tooltip="対象").add_to(m)
            st_folium(m, height=300, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 4: 実行 ---
    with tab4:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🚀 診断実行</div>', unsafe_allow_html=True)
        
        valid = True
        if not (st.session_state["vis_gallery"] or st.session_state["ir_gallery"] or st.session_state["micro_gallery"]):
            st.error("⚠️ 画像が1枚もありません。診断できません。")
            valid = False
        if not has_api_key:
            st.error("⚠️ Gemini API Keyが設定されていません。")
            valid = False
        
        if valid:
            if st.button("AI診断を開始する (Gemini 2.5)", type="primary", use_container_width=True):
                with st.spinner("診断中... RAG検索と画像解析を実行しています"):
                    # 1. RAG Search (Real Logic)
                    weights = {
                        "base_weight": 1.18, "plan_weight": 1.18, "other_weight": 1.0,
                        "material_boost": 1.15, "defect_boost": 1.15, "mat_def_synergy": 1.1,
                        "number_boost": 1.15, "ir_boost": 1.12, "muni_boost": 1.15
                    }
                    if TEMPLATES[template_name]["rag_preset"] == "base_focus":
                        weights["base_weight"] = 1.25
                    
                    k = 4 if fast_mode else MAX_SNIPPETS
                    # IRがあるか、顕微鏡があるか？（顕微鏡もIR同様に詳細解析フラグとして扱うか検討）
                    have_ir_or_micro = bool(st.session_state["ir_gallery"] or st.session_state["micro_gallery"])
                    snippets = rag_search(auto_q, have_ir_or_micro, k, weights)
                    
                    # 2. Image Analysis (Simple stats)
                    vis_list = [analyze_visual(img, 128 if fast_mode else 256) for img in st.session_state["vis_gallery"]]
                    ir_list = [analyze_ir(img, {}, 128 if fast_mode else 256) for img in st.session_state["ir_gallery"]]
                    micro_list = [analyze_visual(img, 256) for img in st.session_state["micro_gallery"]] # 顕微鏡も一応統計を取る

                    grade, reason = rule_based_grade(vis_list, ir_list)
                    life_map = {"A": "10-20年", "B": "7-15年", "C": "3-10年", "D": "1-5年"}
                    
                    # 3. Prompt Construction
                    prompt, labeled_rag = build_master_prompt(
                        user_q=auto_q,
                        rag_snippets=snippets,
                        priors="写真のみから寸法断定不可。安全側判断を優先。",
                        vis_list=vis_list,
                        ir_list=ir_list,
                        micro_list=micro_list, # 顕微鏡リスト
                        rule_grade=grade,
                        rule_life=life_map.get(grade, "不明"),
                        ir_meta_note="注: IRは相対温度。"
                    )
                    
                    # 4. Gemini Payload Construction
                    img_parts = []
                    
                    # 高速モード時の画像選抜
                    # 可視: 上位2枚, IR: 上位1枚, 顕微鏡: 全て(重要なので)
                    target_vis = st.session_state["vis_gallery"][:2] if fast_mode else st.session_state["vis_gallery"]
                    target_ir = st.session_state["ir_gallery"][:1] if fast_mode else st.session_state["ir_gallery"]
                    target_micro = st.session_state["micro_gallery"] # 顕微鏡は減らさない
                    
                    for img in target_vis:
                        img_parts.append(image_to_inline_part(img))
                    for img in target_ir:
                        img_parts.append(image_to_inline_part(img))
                    for img in target_micro:
                        # 顕微鏡は高解像度で送る
                        img_parts.append(image_to_inline_part(img, max_width=1600))
                    
                    try:
                        res = call_gemini(st.secrets["gemini"]["API_KEY"], prompt, img_parts)
                        report_text = res["candidates"][0]["content"]["parts"][0]["text"]
                        
                        st.success("診断完了！")
                        st.markdown('<div class="app-card">', unsafe_allow_html=True)
                        st.markdown("### 📝 診断レポート")
                        st.markdown(report_text)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download Buttons
                        st.download_button(
                            "レポートをダウンロード (Markdown)", 
                            report_text.encode("utf-8"), 
                            "report.md", 
                            "text/markdown"
                        )
                        
                        with st.expander("📚 参照した文献根拠 (RAG)"):
                            for d in labeled_rag:
                                st.markdown(f"**[{d.get('_rag_id')}] {d.get('doc')}**")
                                st.caption(d.get('text'))
                                st.divider()
                                
                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")
                        if debug_mode: st.exception(e)
                        
                        # Fallback
                        st.warning("⚠️ フェイルセーフモードで簡易結果を表示します")
                        st.markdown(f"### 暫定評価: {grade}")
                        st.write(f"理由: {reason}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
