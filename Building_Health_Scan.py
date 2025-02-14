import streamlit as st
import requests
import json
import io
from PIL import Image

def generate_analysis_report(prompt_text):
    """
    Gemini API (gemini-2.0-flash) を呼び出し、
    テキストプロンプトに基づく解析レポートを生成する関数
    """
    try:
        api_key = st.secrets["gemini"]["API_KEY"]
    except KeyError:
        return {"error": "Gemini API Key is missing. Please add it to your .streamlit/secrets.toml file."}
    
    # Gemini API エンドポイント（モデル: gemini-2.0-flash）
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()  # HTTPエラーがあれば例外発生
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("Facade Insight")
    st.write(
        "外壁や物体の状態を分析するアプリです。"
        "写真をアップロードまたは撮影し、国土交通省基準に基づく詳細なレポートを生成します。"
    )
    
    # サイドバーで画像入力方法を選択
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
    
    # 画像が選択された場合、表示する
    if image is not None:
        st.subheader("入力画像")
        # use_container_width=True を使用してコンテナ幅に合わせて表示
        st.image(image, caption="選択された画像", use_container_width=True)
    
    # 分析実行ボタン
    if st.button("分析実行"):
        if image is None:
            st.error("画像が選択されていません。画像をアップロードまたは撮影してください。")
        else:
            st.write("分析中です。しばらくお待ちください...")
            
            # 画像をJPEG形式で保存する前にRGBモードに変換
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # ※ 現状、Gemini API には画像データ自体は送信できないため、
            # 画像に基づいたシナリオを想定するテキストプロンプトを生成します。
            prompt = (
                "以下の画像に基づいて、国土交通省の基準に沿い外壁や物体の状態を分析してください。"
                "非破壊検査、建築業、材料学の専門知識を用い、劣化度をA、B、C、Dなどで定量的に評価し、"
                "推定される寿命を含む詳細なレポートを生成してください。"
                "なお、画像そのものは参照できないため、アップロードされた画像をもとに想定される状況を推定してください。"
            )
            
            # Gemini API を呼び出して解析結果を取得
            result = generate_analysis_report(prompt)
            
            if "error" in result:
                st.error(f"API呼び出しエラー: {result['error']}")
            else:
                st.subheader("分析結果")
                st.json(result)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "このアプリは Streamlit と Gemini API (gemini-2.0-flash) を利用して外壁や物体の状態を分析します。\n"
        "API キーは st.secrets を利用して安全に管理されています。"
    )

if __name__ == "__main__":
    main()
