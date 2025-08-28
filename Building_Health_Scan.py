# -*- coding: utf-8 -*-
# ===========================================================
# 建物診断くん（スマホ対応・マテリアルデザイン風UI・日本語フォント適用）
# - 可視/赤外 画像：アップロード + カメラ
# - EXIFからGPS抽出 → Folium地図に表示（手入力も可）
# - 3PDFをRAG（軽量スコア）→ Gemini 2.0 Flash で詳細分析
# - 結果のみ表示：総合評価カードを先頭に、詳細は展開
# - レポートDL（共有）
# - レポート表示に Noto Sans JP/Noto Serif JP を適用
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
from exif import Image as ExifImage   # pip install exif
import folium                         # pip install folium
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
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
2. **主要根拠（可視/IR/NDT）**  # 画像から読み取れる事実に限定
3. **MLIT基準との関係**  # RAG内の文書名＋節/見出しで簡潔に
4. **推奨対応（優先度順）**
5. **限界と追加データ要望**  # 精度向上に必要な追補情報

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
# UI: マテリアルデザイン風 + 日本語フォント適用CSS
# -----------------------------------------------------------
def inject_material_css():
    st.markdown("""
    <!-- Google Fonts: Noto Sans JP / Noto Serif JP -->
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=Noto+Serif+JP:wght@500;700&display=swap" rel="stylesheet">
    <style>
    :root{
      --mdc-primary:#2962ff;
      --mdc-primary-variant:#0039cb;
      --mdc-secondary:#00b8d4;
      --mdc-bg:#f7f9fc;
      --mdc-surface:#ffffff;
      --mdc-on-primary:#ffffff;
      --radius:16px;
      --shadow:0 6px 18px rgba(0,0,0,.08);
    }
    .block-container{padding-top:1rem;padding-bottom:2rem;}
    body{background:var(--mdc-bg);}
    /* 日本語フォントクラス */
    .jp-sans{
      font-family:'Noto Sans JP', -apple-system, BlinkMacSystemFont, 'Segoe UI',
        'Hiragino Kaku Gothic ProN','Hiragino Sans','Meiryo', sans-serif !important;
      line-height:1.7;
      letter-spacing:0.01em;
    }
    .jp-serif{
      font-family:'Noto Serif JP','Hiragino Mincho ProN','Yu Mincho',serif !important;
      line-height:1.8;
      letter-spacing:0.01em;
    }
    .jp-report h1,.jp-report h2,.jp-report h3,
    .jp-report p,.jp-report li,.jp-report table{
      font-family:'Noto Sans JP', -apple-system, BlinkMacSystemFont, 'Segoe UI',
        'Hiragino Kaku Gothic ProN','Hiragino Sans','Meiryo', sans-serif !important;
    }
    .md-card{
      background:var(--mdc-surface);
      border-radius:var(--radius);
      box-shadow:var(--shadow);
      padding:1rem 1.1rem;
      margin-bottom:1rem;
      border:1px solid rgba(0,0,0,.04);
    }
    .md-title{font-size:1.1rem;font-weight:700;margin:0.2rem 0 0.6rem 0;}
    .md-sub{opacity:.85;font-size:.95rem;}
    .eval-chip{
      display:inline-block;padding:.35rem .7rem;border-radius:999px;
      color:#fff;font-weight:700;letter-spacing:.02em;
      background:linear-gradient(135deg, var(--mdc-primary), var(--mdc-secondary));
    }
    .primary-btn button{
      background:linear-gradient(135deg, var(--mdc-primary), var(--mdc-secondary)) !important;
      color:#fff !important;border:none !important;
      box-shadow:var(--shadow);
    }
    .good-shadow{box-shadow:var(--shadow);}
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------
# メイン
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_material_css()

    st.markdown(f"### 🏗️ {APP_TITLE}")
    st.caption("スマートフォン最適化 / 画像（可視・赤外）＋RAG＋Gemini 2.0 Flash で詳細分析")

    # --- 入力行（スマホ向けに縦構成、カラムは自動で縦積みになる） ---
    st.markdown("#### 1) 質問")
    user_q = st.text_input("例：外壁タイルのひび割れ基準と推定寿命", "", placeholder="分析したいテーマ・質問を入力")

    st.markdown("#### 2) 画像入力（可視/赤外）")
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

    # 画像ハンドリング
    vis_img = None
    vis_src_bytes = None
    if vis_cam is not None:
        vis_img = Image.open(vis_cam)
        vis_src_bytes = vis_cam.getvalue()
    elif vis_file is not None:
        vis_img = Image.open(vis_file)
        vis_src_bytes = vis_file.getvalue()

    ir_img = None
    ir_src_bytes = None
    if ir_file is not None:
        ir_img = Image.open(ir_file)
        ir_src_bytes = ir_file.getvalue()

    # プレビュー
    pv_col1, pv_col2 = st.columns(2)
    with pv_col1:
        if vis_img is not None:
            st.image(vis_img, caption="可視画像", use_container_width=True)
    with pv_col2:
        if ir_img is not None:
            st.image(ir_img, caption="IR画像", use_container_width=True)

    # --- 位置情報（EXIF→GPS or 手入力） ---
    st.markdown("#### 3) 位置情報（自動 or 手入力）")
    lat, lon = None, None
    if vis_src_bytes:
        gps = extract_gps_from_image(vis_src_bytes)
        if gps:
            lat, lon = gps
    if lat is None and ir_src_bytes:
        gps = extract_gps_from_image(ir_src_bytes)
        if gps:
            lat, lon = gps

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        lat = st.text_input("緯度（例：35.6804）", value="" if lat is None else f"{lat:.6f}")
    with c2:
        lon = st.text_input("経度（例：139.7690）", value="" if lon is None else f"{lon:.6f}")
    with c3:
        st.caption("※ 画像EXIFに位置情報が含まれていれば自動表示。無い場合は手入力してください。")

    # 地図表示
    map_row = st.container()
    with map_row:
        try:
            lat_f = float(lat) if lat else None
            lon_f = float(lon) if lon else None
        except Exception:
            lat_f, lon_f = None, None

        if lat_f is not None and lon_f is not None:
            m = folium.Map(location=[lat_f, lon_f], zoom_start=18, tiles="OpenStreetMap")
            folium.Marker([lat_f, lon_f], tooltip="対象地点").add_to(m)
            st_folium(m, height=300, use_container_width=True)
        else:
            st.info("地図表示：緯度経度が未指定です。EXIFに位置が無い場合は手入力してください。")

    # --- 実行ボタン ---
    st.markdown("#### 4) 解析の実行")
    run = st.button("🔎 Geminiで詳細分析（結果のみ表示）", use_container_width=True)

    # -------------------------------------------------------
    # 解析フロー
    # -------------------------------------------------------
    if run:
        if not user_q:
            st.error("質問を入力してください。")
            return

        # RAGロード＆Top-K抽出
        corpus = load_rag_corpus()
        snippets = topk_chunks(user_q, corpus, k=4)

        # 画像 → Gemini parts
        image_parts = []
        if vis_img is not None:
            image_parts.append(image_to_inline_part(vis_img))
        if ir_img is not None:
            image_parts.append(image_to_inline_part(ir_img))

        # IRメタ
        ir_meta = {
            "has_ir": ir_img is not None,
            "emissivity": (ir_emiss or "不明").strip(),
            "t_ref": (ir_tref or "不明").strip(),
            "t_amb": (ir_tamb or "不明").strip(),
            "rh": (ir_rh or "不明").strip(),
            "dist": (ir_dist or "不明").strip(),
            "angle": (ir_ang or "不明").strip(),
        }

        # プロンプト作成
        prompt = build_master_prompt(user_q, snippets, ir_meta)

        # Gemini API Key
        try:
            api_key = st.secrets["gemini"]["API_KEY"]
        except KeyError:
            st.error("Gemini API Key が未設定です。.streamlit/secrets.toml に [gemini].API_KEY を設定してください。")
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
            st.warning("レポート出力が空でした。入力内容（質問・画像・PDF）をご確認ください。")
            return

        # ---- レポート（日本語フォント適用） ----
        # 「## 総合評価」〜 次見出し直前までを抽出し、Noto Sans JPで表示
        summary_block = None
        try:
            pattern = r"(?:^|\n)##\s*総合評価[\s\S]*?(?=\n##\s|\Z)"
            m = re.search(pattern, report_md)
            if m:
                summary_block = m.group(0)
        except Exception:
            summary_block = None

        st.markdown("#### 5) 解析結果")
        if summary_block:
            st.markdown('<div class="md-card good-shadow jp-report">', unsafe_allow_html=True)
            st.markdown('<div class="md-title">🧭 総合評価（要約）</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='jp-report'>{summary_block}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 全文（詳細）はExpander内を日本語フォントで
        with st.expander("詳細レポートを表示", expanded=(summary_block is None)):
            st.markdown(f"<div class='jp-report'>{report_md}</div>", unsafe_allow_html=True)

        # ダウンロード（共有）— Markdownはフォントを埋め込まない点に注意
        st.download_button(
            "📄 レポートをダウンロード",
            report_md,
            file_name="building_health_report.md",
            mime="text/markdown",
            use_container_width=True
        )

        # 参考：使用したRAG抜粋（こちらも日本語フォントで）
        with st.expander("（参考）使用したRAG抜粋"):
            for src, ch in snippets:
                snippet = ch[:600] + ('...' if len(ch) > 600 else '')
                st.markdown(f"<div class='jp-report'><b>{src}</b>：{snippet}</div>", unsafe_allow_html=True)

    # フッタ
    st.markdown("")
    st.caption("© 建物診断くん — 可視/赤外 × RAG × Gemini。モバイル最適化UI。")


if __name__ == "__main__":
    main()
