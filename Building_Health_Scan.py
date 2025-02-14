import streamlit as st
import requests
import json
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

@st.cache_resource(show_spinner=False)
def load_caption_model():
    """
    BLIP画像キャプションモデルとプロセッサをロードする
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_image_caption(image, processor, model):
    """
    画像からキャプションを生成する関数
    """
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_analysis_report(prompt_text):
    """
    Gemini API (gemini-2.0-flash) を呼び出し、テキストプロンプトに基づく解析レポートを生成する関数
    """
    try:
        api_key = st.secrets["gemini"]["API_KEY"]
    except KeyError:
        return {"error": "Gemini API Key is missing. Please add it to your .streamlit/secrets.toml file."}
    
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt_text}]}
        ]
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("建物診断君")
    st.write(
        "このアプリは、アップロードされた画像から自動でキャプションを生成し、"
        "その内容を元に国土交通省基準に沿った外壁・物体の状態分析レポートを生成します。"
    )
    
    # サイドバー: 画像入力方法の選択
    st.sidebar.header("画像入力オプション")
    input_method = st.sidebar.selectbox("画像入力方法を選択してください", ["アップロード", "カメラ撮影"])
    
    image = None
    if input_method == "アップロード":
        uploaded_file = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    elif input_method == "カメラ撮影":
        captured_image = st.camera_input("カメラで画像を撮影してください")
        if captured_image is not None:
            image = Image.open(captured_image)
    
    # 画像表示
    if image is not None:
        st.subheader("入力画像")
        st.image(image, caption="選択された画像", use_container_width=True)
    
    # 分析実行ボタン
    if st.button("分析実行"):
        if image is None:
            st.error("画像が選択されていません。画像をアップロードまたは撮影してください。")
        else:
            st.write("キャプション生成中です。しばらくお待ちください...")
            
            # BLIPモデルのロード（キャッシュ利用）
            processor, model = load_caption_model()
            
            # 画像キャプション生成
            caption = generate_image_caption(image, processor, model)
            st.write("生成されたキャプション:")
            st.write(caption)
            
            # 画像をJPEG形式で保存する前にRGBモードに変換（今後の拡張用）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # プロンプト作成：生成されたキャプションを含め、詳細レポートの生成を指示
            prompt = (
                f"アップロードされた画像のキャプション: '{caption}'.\n\n"
                "上記キャプションに基づき、以下のレポートテンプレートに沿った、国土交通省の基準に準拠する外壁・物体の状態分析レポートを生成してください。\n\n"
                "【レポートテンプレート】\n"
                "-----------------------------\n"
                "【外壁状態分析レポート】\n\n"
                "1. 調査概要\n"
                "　- 調査対象、調査目的、調査方法、調査日、調査者など\n\n"
                "2. 調査結果\n"
                "　- ひび割れ、浮き・剥離、変色・汚染などの評価（各項目でA～D評価）\n\n"
                "3. 劣化度評価（総合評価）\n"
                "　- A: 健全、B: 軽微、C: 中程度、D: 重大\n\n"
                "4. 推定寿命\n"
                "　- 現在の状態からの推定寿命、補修後の推定寿命\n\n"
                "5. 補修・改修計画（提案）\n"
                "　- 優先順位、具体的な補修方法、使用材料、概算費用など\n"
                "-----------------------------\n\n"
                "※画像の詳細な解析は行えないため、上記テンプレートに基づいた一般的なレポートを作成してください。"
            )
            st.write("Gemini API にプロンプトを送信中です...")
            
            # Gemini API を呼び出して解析結果を取得
            result = generate_analysis_report(prompt)
            if "error" in result:
                st.error(f"API呼び出しエラー: {result['error']}")
            else:
                # JSONからレポートテキストを抽出（例: candidates[0] -> content -> parts[0] -> text）
                try:
                    report_text = result["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    st.error("レポート抽出中にエラーが発生しました: " + str(e))
                    return
                
                st.subheader("分析結果レポート")
                st.markdown(report_text)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "このアプリは、画像キャプション生成モデルとGemini APIを組み合わせ、"
        "画像の情報をテキスト化してレポート生成に活用します。\n"
        "APIキーはst.secretsにて管理してください。"
    )

if __name__ == "__main__":
    main()
