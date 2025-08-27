# app.py
# ─────────────────────────────────────────────────────────────
# 建築劣化診断エキスパート（RAG限定・結果のみMarkdown出力）
# 要件：
#  - 入力（ユーザー質問／画像キャプション／点検メタ／建物・材料情報／NDT値／RAG抜粋）の範囲だけを根拠に判定
#  - MLITの数値・閾値はRAG抜粋内に出てくる場合のみ使用（出所を簡潔併記）
#  - 「結果のみ」を1本のMarkdownで表示（総合評価→根拠→基準→対応→ASCII図→限界）
# 注意：外部通信なし。推測が必要な場合は「未確認」「追加情報が必要」を明示。
# ─────────────────────────────────────────────────────────────

import streamlit as st
import json
import re
from datetime import datetime

st.set_page_config(page_title="建築劣化診断（RAG限定）", layout="wide")

# ========== ユーティリティ ==========
def parse_json(text: str, default):
    if not text or str(text).strip() == "":
        return default, None
    try:
        return json.loads(text), None
    except Exception as e:
        return default, f"JSONパース失敗: {e}"

def find_doc_refs(rag_text: str):
    """
    RAG抜粋から出所（PDF名らしき文字列）と見出し（章/節/項/見出しらしき行）を抽出。
    取得できない場合はNoneを返す。数値閾値も簡便抽出。
    """
    docs = []
    thresholds = []
    if not rag_text:
        return docs, thresholds

    # PDF名（簡便抽出）
    pdf_names = set(re.findall(r'([\w\-/\.]+\.pdf)', rag_text, flags=re.IGNORECASE))

    # 見出しらしき行（章・節・項・見出し）
    headings = set()
    for line in rag_text.splitlines():
        line = line.strip()
        if re.search(r'(第[一二三四五六七八九十百千\d]+[章節項])', line):
            headings.add(line)
        elif re.search(r'(見出し|定義|判定|基準|評価|測定方法|適用範囲)', line):
            headings.add(line)

    # 数値＋単位（例：0.2mm, 10年, 50℃, 0.3 MPa, 200 mV）
    for m in re.finditer(r'([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|m|℃|%|年|MPa|mV|ppm)', rag_text):
        thresholds.append(m.group(0))

    # ペアリング（単純にすべてのPDF×見出しを列挙）
    for pdf in pdf_names:
        if headings:
            for h in headings:
                docs.append((pdf, h))
        else:
            docs.append((pdf, None))

    return docs, thresholds

def has_ir_meta(text: str):
    """IRメタ(ε, Tr, ambient, RH, span, distance, angle, ΔT等)の記載確認（単純キーワード）"""
    if not text:
        return False, []
    keys = ['ε', 'emissivity', 'Tr', 'reflected', 'ambient', 'Ta', 'RH', 'span', 'distance', 'angle', 'ΔT', 'deltaT']
    found = [k for k in keys if re.search(rf'\b{k}\b', text, flags=re.IGNORECASE)]
    return len(found) >= 2, found

def keywords_grade(visible_text: str, ir_text: str):
    """
    可視/IRキャプションに含まれる語から、定性的な劣化度を推定。
    ※数値閾値は使わない。RAGがない限り“参考レベル”の判定。
    """
    text = f"{visible_text or ''}\n{ir_text or ''}"

    # D相当のキーワード（落下・安全リスク）
    kw_D = ['剥落', '落下', '爆裂', '鉄筋露出', '広範囲な剥離', '危険', '緊急', '大面積剥離', '構造安全', '脱落']
    # C相当（部分補修が必要）
    kw_C = ['剥離', '浮き', '錆汁', '腐食', '漏水', 'エフロ', 'ひび密集', '中程度', '膨れ', 'はらみ', '空洞']
    # B相当（軽微）
    kw_B = ['微細ひび', '汚染', '退色', 'チョーキング', '白華', '軽微', '風化', 'ヘアクラック']

    def contains_any(kws):
        return any(kw in text for kw in kws)

    if contains_any(kw_D):
        return 'D', ['落下・安全リスクが示唆される語句を含む（定性的判定）']
    if contains_any(kw_C):
        return 'C', ['部分補修を要する可能性を示す語句を含む（定性的判定）']
    if contains_any(kw_B):
        return 'B', ['軽微な劣化の可能性を示す語句を含む（定性的判定）']
    return 'A', ['顕著な劣化を示す語句なし（定性的判定）']

def pick_life_from_rag(rag_text: str, building_meta: dict):
    """
    RAG内に「～年」「寿命」「更新周期」等があれば抽出。
    さらに「築年」等がある場合、設計耐用年数の“残り”を試算（※RAG内の数値に限定）。
    見つからなければ None を返す。
    """
    if not rag_text:
        return None, "RAG内に寿命・更新周期の記載が見当たりません。"

    # 範囲表現の抽出（10〜15年 / 10-15年）
    m_range = re.search(r'(\d{1,3})\s*[〜\-–—]\s*(\d{1,3})\s*年', rag_text)
    if m_range:
        lo, hi = int(m_range.group(1)), int(m_range.group(2))
        return (lo, hi), "RAG内に寿命/更新の年幅が記載。"

    # 単一年数（例：更新周期15年、耐用年数60年）
    m_single = re.search(r'(耐用年数|更新周期|寿命)\D*?(\d{1,3})\s*年', rag_text)
    if m_single:
        years = int(m_single.group(2))
        # 築年→残存寿命の概算（RAG由来の設計年数がある場合のみ）
        b_year = None
        for key in ['築年', 'built_year', 'year_built', '建築年']:
            if key in building_meta:
                try:
                    b_year = int(str(building_meta[key])[:4])
                    break
                except:
                    pass
        if b_year:
            age = datetime.now().year - b_year
            remain = max(years - age, 0)
            return (max(remain - 2, 0), remain + 2), f"RAG内の{years}年に対し、築年{b_year}から残存概算（±2年の幅）。"
        else:
            # 幅がないと使いにくいので±2年の幅を付与（記載元はRAG）
            return (max(years - 2, 0), years + 2), "RAG内の年数に±2年幅を付与。"

    return None, "RAG内に寿命・更新周期の定量記載が見当たりません。"

def build_ascii_map(building_meta: dict, visible_text: str, ir_text: str):
    """
    簡易ASCII図。位置情報がなければ枠＋凡例のみ（“記号配置は省略”）。
    """
    face = None
    for k in ['面', '方位', '外壁方位', '立面']:
        if k in building_meta and str(building_meta[k]).strip():
            face = str(building_meta[k]).strip()
            break
    face_label = face if face else "面:不明"

    legend = "| ■=剥離/浮き ▲=高温域(IR) ＝＝＝=ひび想定"
    # 位置が不明なので記号配置は行わない（根拠のない可視化を避ける）
    lines = [
        f"[{face_label}]============================",
        legend,
        "| （分布情報が不足のため、記号配置は省略）",
        "| ",
        "| "
    ]
    return "\n".join(lines)

def summarize_ndt(ndt: dict):
    """
    NDT/試験値を列挙（解釈はRAGの裏付けがない限り“値の提示のみ”）。
    """
    if not ndt:
        return []
    items = []
    for k, v in ndt.items():
        try:
            if isinstance(v, (int, float)):
                items.append(f"- {k}: {v}")
            else:
                items.append(f"- {k}: {json.dumps(v, ensure_ascii=False)}")
        except:
            items.append(f"- {k}: {v}")
    return items

def derive_overall_grade(visible_text: str, ir_text: str, rag_text: str):
    """
    総合グレード：可視/IRの定性＋RAGの数値閾値“記載の有無”を加味。
    （※閾値の適用はRAGが提示する場合のみ。ここでは“存在確認のみ”。）
    """
    grade, reasons = keywords_grade(visible_text, ir_text)

    # RAGに「基準/閾値/判定」語があれば、根拠の信頼度を一段上げる注記のみ
    rag_has_numeric = bool(re.search(r'(mm|mV|MPa|ppm|℃|%|年)', rag_text or ""))
    note = None
    if rag_has_numeric:
        note = "RAGに定量的情報（単位付き数値）が含まれるため、判定根拠の信頼度は相対的に高い（適用は当該条項に限定）。"
    else:
        note = "RAGに閾値の明示がないため、定性的判定の位置づけ（定量判定は保留）。"

    return grade, reasons, note

def build_actions(grade: str, ir_meta_ok: bool):
    """
    推奨対応（優先度）。IRメタ不足の場合の注意喚起も追加。
    """
    urgent = []
    soon = []
    plan = []

    if grade == 'D':
        urgent.append("⚠ 落下・剥落リスク箇所の仮防護・立入制限を即時実施")
        urgent.append("⚠ 打診・近接目視での危険部同定（必要に応じて夜間/雨後も）")
        soon.append("赤外再撮影＋打診（IRは日較差・風条件を考慮）")
        soon.append("剥離/浮きの範囲確定後、広範補修計画の立案")
        plan.append("原因分析（漏水起因/鉄筋腐食/施工不良等）に基づく改修設計")
    elif grade == 'C':
        urgent.append("⚠ 剥離・浮きが疑われる面の打診で危険部の早期抽出")
        soon.append("部分補修（樹脂注入/断面修復等）の具体化")
        plan.append("仕上げ・防水更新周期の前倒し検討")
    elif grade == 'B':
        soon.append("観察強化（半年〜1年間隔）と微細ひびの幅計測ログ化")
        plan.append("局所補修の要否判定（汚染/チョーキング対策含む）")
    else:  # A
        plan.append("定期点検のみ（現状維持）")

    if not ir_meta_ok:
        soon.append("IRメタ（ε/Tr/ΔT/距離/角度/湿度）を整えた再撮影で信頼度向上")

    return urgent, soon, plan

def ensure_markdown_sections(md: str):
    """
    出力が指定の6セクションを満たすか最低限チェック。
    """
    required = ["**総合評価**", "**主要根拠（可視/IR/NDT）**", "**MLIT基準との関係**",
                "**推奨対応（優先度付き）**", "**簡易イメージ図（ASCII）**", "**限界と追加データ要望**"]
    missing = [r for r in required if r not in md]
    return missing

# ========== UI ==========
st.title("建築劣化診断エキスパート（RAG限定・結果のみ出力）")

with st.sidebar:
    st.markdown("### 入力")
    user_question = st.text_area("ユーザー質問（任意）", height=80, placeholder="例：この南面外壁の劣化度と推奨対策は？")

    st.markdown("**画像キャプション（Markdown）**\n- 可視/IRそれぞれ1枚以内を想定\n- IRは ε/Tr/ambient/RH/span/distance/angle/ΔT 等を併記推奨")
    captions_visible = st.text_area("可視キャプション", height=120, placeholder='例：南面タイル、1F～3Fでひび密集、錆汁あり、部分的に浮き疑い…')
    captions_ir = st.text_area("IRキャプション", height=120, placeholder='例：type: "ir", emissivity=0.94, Tr=20℃, ambient=18℃, RH=60%, span=5℃, distance=5m, angle=10°, ΔT=3℃。2F梁端に高温域。')

    st.markdown("**点検メタ（JSON）**")
    inspection_meta_text = st.text_area("INSPECTION_META_JSON", height=120, placeholder='{"date":"2025-08-27","weather":"晴","ambient_temp":"30℃","humidity":"70%","recent_rain":"24h無"}')

    st.markdown("**建物・材料情報（JSON）**")
    building_meta_text = st.text_area("BUILDING_META_JSON", height=140, placeholder='{"用途":"庁舎","構造":"RC","築年":1998,"仕上げ":"タイル張り","補修履歴":"2012年打診・部分注入","面":"南面"}')

    st.markdown("**NDT/試験値（JSON）**")
    ndt_text = st.text_area("NDT_JSON（任意）", height=140, placeholder='{"中性化深さ_mm":12.5,"半セル電位_mV":-320,"かぶり_mm":25,"付着強度_MPa":0.8}')

    st.markdown("**RAG抜粋（Top-K）**\n- MLIT等の該当抜粋のみを貼付\n- ここに**ある内容だけ**を根拠に使用")
    rag_text = st.text_area("<<<RAG_CONTEXT>>>", height=240, placeholder="Structure_Base.pdf p.12『判定基準』…\nminatoku_...pdf 3.2『タイル仕上げの浮き判定』…\n（0.2mm以上のひび… 等）")

    run = st.button("診断レポートを生成")

# ========== 実行 ==========
if run:
    # JSONパース
    inspection_meta, err1 = parse_json(inspection_meta_text, {})
    building_meta, err2 = parse_json(building_meta_text, {})
    ndt, err3 = parse_json(ndt_text, {})

    errors = [e for e in [err1, err2, err3] if e]
    if errors:
        st.error("入力エラー：\n- " + "\n- ".join(errors))
        st.stop()

    # IRメタ確認
    ir_ok, ir_found = has_ir_meta(captions_ir)

    # RAG出所と閾値抽出
    doc_refs, thresholds = find_doc_refs(rag_text)

    # 総合グレード（定性）＋RAG数値の有無メモ
    grade, g_reasons, rag_numeric_note = derive_overall_grade(captions_visible, captions_ir, rag_text)

    # 寿命推定（RAG由来のみ）
    life_range, life_note = pick_life_from_rag(rag_text, building_meta)

    # NDT羅列（解釈はRAG裏付けがない限り提示のみ）
    ndt_items = summarize_ndt(ndt)

    # 推奨対応
    urgent, soon, plan = build_actions(grade, ir_ok)

    # ASCII図
    ascii_map = build_ascii_map(building_meta, captions_visible, captions_ir)

    # 主要根拠（可視/IR/NDT）
    visible_bullets = []
    if captions_visible and captions_visible.strip():
        for line in [l.strip("- •\t ") for l in captions_visible.splitlines() if l.strip()]:
            visible_bullets.append(f"- {line}")
    else:
        visible_bullets.append("- 可視所見：未入力")

    ir_bullets = []
    if captions_ir and captions_ir.strip():
        for line in [l.strip("- •\t ") for l in captions_ir.splitlines() if l.strip()]:
            ir_bullets.append(f"- {line}")
        if not ir_ok:
            ir_bullets.append("- IRメタ（ε/Tr/ΔT/距離/角度/湿度 等）が不足し、含水/剥離推定の信頼度が低下")
    else:
        ir_bullets.append("- IR所見：未入力")

    # MLIT基準との関係
    mlit_lines = []
    if doc_refs:
        # 出所（文書名＋見出し）を列挙
        listed = set()
        for pdf, head in doc_refs:
            label = f"{pdf}" + (f" の {head}" if head else "")
            if label not in listed:
                mlit_lines.append(f"- {label}（出所）")
                listed.add(label)
        if thresholds:
            # RAG内で見つかった定量表現の例示
            mlit_lines.append(f"- RAG内の数値例：{', '.join(sorted(set(thresholds))[:6])}（適用は当該条項範囲に限定）")
    else:
        mlit_lines.append("- RAG内に該当条項の明示的な出所が見当たりません。")

    # 数値閾値が未掲載の場合の明示
    if not re.search(r'(mm|mV|MPa|ppm|℃|%|年)', rag_text or ""):
        mlit_lines.append("- 当該数値閾値は文脈に未掲載（定量判定は保留）。")

    # 推定残存寿命の文章化
    if life_range:
        life_text = f"{life_range[0]}–{life_range[1]}年"
        life_note_text = life_note
    else:
        life_text = "未確定（追加情報が必要）"
        life_note_text = life_note

    # 総合評価（主原因の要約：可視/IRの語句ベース）
    main_cause = "；".join(g_reasons)

    # 結果Markdown（“結果のみ”）
    md = []
    md.append("1. **総合評価**")
    md.append(f" - グレード：**{grade}**（{main_cause}）")
    md.append(f" - 推定残存寿命：**{life_text}**")
    if life_range:
        md.append(f"   - 備考：{life_note_text}")
    md.append("")

    md.append("2. **主要根拠（可視/IR/NDT）**")
    md.append(" - 可視所見")
    md.extend([f"   {b}" for b in visible_bullets])
    md.append(" - IR所見")
    md.extend([f"   {b}" for b in ir_bullets])
    if ndt_items:
        md.append(" - NDT/試験（値の提示。解釈はRAG根拠がある場合のみ）")
        md.extend([f"   {b}" for b in ndt_items])
    else:
        md.append(" - NDT/試験：未入力")
    md.append(f" - 付記：{rag_numeric_note}")
    md.append("")

    md.append("3. **MLIT基準との関係**")
    md.extend(mlit_lines)
    md.append("")

    md.append("4. **推奨対応（優先度付き）**")
    if urgent:
        md.append(" - 1) 緊急（安全/落下・漏水の恐れ）")
        md.extend([f"   - {x}" for x in urgent])
    else:
        md.append(" - 1) 緊急：該当なし")
    if soon:
        md.append(" - 2) 早期（1年以内）")
        md.extend([f"   - {x}" for x in soon])
    else:
        md.append(" - 2) 早期：該当なし")
    if plan:
        md.append(" - 3) 計画（中期計画に組入れ）")
        md.extend([f"   - {x}" for x in plan])
    else:
        md.append(" - 3) 計画：該当なし")
    md.append("")

    md.append("5. **簡易イメージ図（ASCII）**")
    md.append("```")
    md.append(build_ascii_map(building_meta, captions_visible, captions_ir))
    md.append("```")
    md.append("")

    md.append("6. **限界と追加データ要望**")
    need_more = [
        "ひび割れ幅の実測値（mm）と分布、面積率（%）",
        "打診結果（浮き/剥離の範囲 m²）と代表写真",
        "IRメタ（ε/Tr/ambient/RH/span/distance/angle/撮影時ΔT条件）の明記",
        "NDT（半セル・中性化・付着強度・Cl- 等）の測定位置と値の対応表"
    ]
    md.extend([f"- {x}" for x in need_more])

    output_md = "\n".join(md)

    # セクション検査（念のため）
    missing = ensure_markdown_sections(output_md)
    if missing:
        st.warning("以下のセクションが不足しています：" + ", ".join(missing))

    st.markdown("### 診断レポート（結果のみ）")
    st.markdown(output_md)

    st.download_button(
        label="Markdownをダウンロード",
        data=output_md.encode("utf-8"),
        file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )

# 使い方メモ（UI上は表示しないほうが良ければコメント化してください）
with st.expander("使い方メモ（実装者向け）"):
    st.markdown("""
- RAG抜粋は FAISS などでTop-K抽出した3–6チャンクを貼付。**ここにある内容のみ**根拠として使用します。
- IRは `ε/Tr/ambient/RH/span/distance/angle/ΔT` 等をキャプションに含めると信頼度判定に反映します。
- MLIT数値閾値はRAG中に見つかった場合のみ例示・適用（出所を「文書名＋節/見出し」で簡潔併記）。
- 寿命推定はRAG内の「寿命/耐用年数/更新周期」記載に限定。築年があれば残存概算（±2年幅）を算出します。
- 出力は1本のMarkdownのみ（“総合評価を最初に”）。
""")
