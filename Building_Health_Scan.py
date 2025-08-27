# --- Python 3.12.x 互換の軽量実装（重依存なし） ---
# * 画像: アップロード（可視/赤外）＆カメラ撮影（可視）
# * PDF RAG: 純Pythonの簡易スコアリングでTop-K抽出
# * Gemini 2.0 Flash: マルチモーダル（画像base64）で詳細分析
# * スマホ向けUI: 大ボタン/見出し/縦配置、結果のみ表示＋DL

import base64
import io
import os
import re
import unicodedata
from typing import List, Tuple, Dict

import streamlit as st
import requests
import PyPDF2
from PIL import Image

# ----------------------------
# ユーティリティ
# ----------------------------
def normalize_text(text: str) -> str:
    # 全角→半角、改行→空白、連続空白の圧縮
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
    # 文字数ベースで素直に分割（オーバーラップ付き）
    chunks = []
    i = 0
    N = len(text)
    while i < N:
        end = min(i + max_chars, N)
        chunk = text[i:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        i = end - overlap if end - overlap > i else end
    return chunks

def tokenize(s: str) -> List[str]:
    s = s.lower()
    # 記号除去＆日本語も含めて簡易トークン化
    s = re.sub(r"[^a-z0-9一-龥ぁ-んァ-ン_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

def simple_retrieval_score(query: str, chunk: str) -> float:
    # クエリ語の一致個数＋頻度を素朴に加点（軽量・非依存）
    q_tokens = set(tokenize(query))
    c_tokens = tokenize(chunk)
    if not q_tokens or not c_tokens:
        return 0.0
    score = 0.0
    for qt in q_tokens:
        score += c_tokens.count(qt) * 1.0
        if qt in c_tokens:
            score += 0.5
    # わずかに長さペナルティ（冗長防止）
    score = score / (1.0 + len(chunk) / 2000.0)
    return score

def topk_chunks(query: str, corpus: List[Tuple[str, str]], k: int = 4) -> List[Tuple[str, str]]:
    # corpus: (source_name, chunk_text)
    scored = []
    for src, ch in corpus:
        scored.append((simple_retrieval_score(query, ch), src, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, src, ch in scored[:k]:
        if s <= 0:
            continue
        out.append((src, ch))
    # ひとつも当たらないときは最初の短いものを保険で1つ
    if not out and corpus:
        out = [corpus[0]]
    return out

def image_to_inline_part(image: Image.Image, mime: str = "image/jpeg", max_width: int = 1400) -> Dict:
    # JPEGに統一（PNGもJPEG化）。幅を抑えてサイズ削減。
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.width > max_width:
        r = max_width / float(image.width)
        image = image.resize((max_width, int(image.height * r)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": mime, "data": b64}}

# ----------------------------
# Gemini 呼び出し
# ----------------------------
def call_gemini(api_key: str, prompt_text: str, image_parts: List[Dict]) -> Dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    # contentsは順序が重要：まず指示テキスト、続いて画像パーツを渡す
    parts = [{"text": prompt_text}]
    parts.extend(image_parts)
    payload = {"contents": [{"parts": parts}]}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()

def extract_text_from_gemini(result: Dict) -> str:
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""

# ----------------------------
# マスタープロンプト（簡潔・結果主義）
# ----------------------------
def build_master_prompt(user_q: str,
                        rag_snippets: List[Tuple[str, str]],
                        ir_meta: Dict) -> str:
    # RAGコンテキスト整形（出所明記）
    rag_block = []
    for src, txt in rag_snippets:
        rag_block.append(f"[{src}] {txt}")
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
"""
    return normalize_text(prompt)

# ----------------------------
# RAG準備（3PDFを統合）
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_rag_corpus() -> List[Tuple[str, str]]:
    sources = [
        ("Structure_Base.pdf", "Structure_Base.pdf"),
        ("上島町 公共施設等総合管理計画", "kamijimachou_Public_facility_management_plan.pdf"),
        ("港区 公共施設マネジメント計画", "minatoku_Public_facility_management_plan.pdf"),
    ]
    corpus: List[Tuple[str, str]] = []
    for name, path in sources:
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text, max_chars=900, overlap=120)
        for ch in chunks:
            if len(ch) >= 40:
                corpus.append((name, ch))
    # 空の場合も返す（後段で保険処理）
    return corpus

# ----------------------------
# Streamlit アプリ
# ----------------------------
def main():
    st.set_page_config(page_title="Building Health Scan (Gemini)", layout="wide")
    st.markdown("## Building Health Scan — スマホ対応（可視/赤外 + Gemini 詳細分析）")

    # 入力：ユーザー質問
    user_q = st.text_input("質問（例：外壁タイルのひび割れ基準と推定寿命）", "")

    # 画像入力UI（スマホ前提：縦並び・大見出し）
    st.markdown("### 画像入力")
    st.markdown("**可視画像（どちらか1つ）**")
    col1, col2 = st.columns(2)
    with col1:
        vis_file = st.file_uploader("可視画像を選択（JPG/PNG）", type=["jpg", "jpeg", "png"], key="vis_up")
    with col2:
        vis_cam = st.camera_input("カメラで撮影（可視）", key="vis_cam")

    st.markdown("**赤外線（IR）画像（任意・外付けカメラの写真をアップロード）**")
    ir_file = st.file_uploader("IR画像を選択（JPG/PNG）", type=["jpg", "jpeg", "png"], key="ir_up")

    with st.expander("IRメタデータ（任意・精度向上）"):
        ir_emiss = st.text_input("放射率 ε（例: 0.95）", "")
        ir_tref  = st.text_input("反射温度 T_ref [℃]（例: 20）", "")
        ir_tamb  = st.text_input("外気温 T_amb [℃]（例: 22）", "")
        ir_rh    = st.text_input("相対湿度 RH [%]（例: 65）", "")
        ir_dist  = st.text_input("撮影距離 [m]（例: 5）", "")
        ir_ang   = st.text_input("撮影角度 [°]（例: 10）", "")

    # 表示用プレビュー
    vis_img = None
    if vis_cam is not None:
        vis_img = Image.open(vis_cam)
    elif vis_file is not None:
        vis_img = Image.open(vis_file)
    if vis_img is not None:
        st.image(vis_img, caption="可視画像", use_container_width=True)

    ir_img = None
    if ir_file is not None:
        ir_img = Image.open(ir_file)
        st.image(ir_img, caption="IR画像", use_container_width=True)

    # 実行ボタン
    run = st.button("🔎 Geminiで詳細分析（結果のみ表示）", use_container_width=True)

    if run:
        if not user_q:
            st.error("質問を入力してください。")
            return

        # RAG準備
        corpus = load_rag_corpus()
        snippets = topk_chunks(user_q, corpus, k=4)

        # 画像→Geminiパーツ
        image_parts = []
        if vis_img is not None:
            image_parts.append(image_to_inline_part(vis_img, mime="image/jpeg"))
        if ir_img is not None:
            image_parts.append(image_to_inline_part(ir_img, mime="image/jpeg"))

        # IRメタ整形
        ir_meta = {
            "has_ir": ir_img is not None,
            "emissivity": ir_emiss.strip() or "不明",
            "t_ref": ir_tref.strip() or "不明",
            "t_amb": ir_tamb.strip() or "不明",
            "rh": ir_rh.strip() or "不明",
            "dist": ir_dist.strip() or "不明",
            "angle": ir_ang.strip() or "不明",
        }

        # プロンプト作成
        prompt = build_master_prompt(user_q, snippets, ir_meta)

        # Gemini呼び出し
        try:
            api_key = st.secrets["gemini"]["API_KEY"]
        except KeyError:
            st.error("Gemini API Key が未設定です。.streamlit/secrets.toml に設定してください。")
            return

        with st.spinner("Geminiが分析中…"):
            try:
                result = call_gemini(api_key, prompt, image_parts)
                report = extract_text_from_gemini(result)
            except requests.HTTPError as e:
                st.error(f"APIエラー: {e.response.text[:500]}")
                return
            except Exception as e:
                st.error(f"呼び出しエラー: {e}")
                return

        if not report:
            st.warning("出力が空でした。プロンプトや画像/質問内容を見直してください。")
            return

        # 結果のみ表示
        st.markdown("## 解析結果")
        st.markdown(report)
        st.download_button("レポートをダウンロード", report, file_name="report.md", mime="text/markdown", use_container_width=True)

        # 簡易トレース（任意で表示）
        with st.expander("（参考）使用したRAG抜粋", expanded=False):
            for src, ch in snippets:
                st.markdown(f"**{src}**：{ch[:500]}{'...' if len(ch)>500 else ''}")

if __name__ == "__main__":
    main()
