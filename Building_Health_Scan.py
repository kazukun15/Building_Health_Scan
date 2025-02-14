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
    st.title("Facade Insight with Image Captioning")
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
            
            # 画像がアップロードされたことを前提としたプロンプトを作成
            prompt = (
                f"アップロードされた画像のキャプション: '{caption}'.\n\n"
                "上記キャプションに基づき、国土交通省の基準に沿った外壁や物体の状態分析レポートを生成してください。\n"
                "非破壊検査、建築業、材料学の専門知識を活用し、劣化度をA、B、C、Dで評価し、推定寿命および補修・改修計画も含めた詳細なレポートを作成してください。"
            )
            st.write("Gemini API にプロンプトを送信中です...")
            
            # Gemini API を呼び出し解析結果を取得
            result = generate_analysis_report(prompt)
            if "error" in result:
                st.error(f"API呼び出しエラー: {result['error']}")
            else:
                st.subheader("分析結果")
                st.json(result)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "このアプリは、画像キャプション生成モデルとGemini APIを組み合わせ、"
        "画像の情報をテキスト化してレポート生成に活用します。\n"
        "APIキーはst.secretsにて管理してください。"
    )

if __name__ == "__main__":
    main()
