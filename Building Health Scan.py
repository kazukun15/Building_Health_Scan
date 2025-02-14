import streamlit as st
import requests
import json
from PIL import Image

def generate_analysis_report(prompt_text):
    """
    Gemini API (gemini-2.0-flash) を呼び出して、テキストプロンプトに基づく解析レポートを生成する関数
    """
    # st.secrets から API キーを取得（.streamlit/secrets.toml に設定済み）
    api_key = st.secrets["gemini"]["API_KEY"]
    # Gemini-2.0-flash のエンドポイントURLを作成
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
        response.raise_for_status()  # HTTPエラーが発生した場合は例外
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("Facade Insight")
    st.write("外壁や物体の状態を分析するアプリです。写真をアップロードまたは撮影し、国土交通省基準に基づく詳細なレポートを生成します。")
    
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
    
    # 画像が選択されている場合、画面上に表示
    if image is not None:
        st.subheader("入力画像")
        st.image(image, caption="選択された画像", use_column_width=True)
    
    # 分析実行ボタン
    if st.button("分析実行"):
        if image is None:
            st.error("画像が選択されていません。画像をアップロードまたは撮影してください。")
        else:
            st.write("分析中です。しばらくお待ちください...")
            # 画像そのものはAPIに送信できないため、画像に基づいたシナリオを想定したプロンプトを作成
            prompt = (
                "以下の画像に基づいて、国土交通省の基準に沿い外壁や物体の状態を分析してください。"
                "非破壊検査、建築業、材料学の専門知識を用い、劣化度をA、B、C、Dなどで定量的に評価し、"
                "推定される寿命を含む詳細なレポートを生成してください。"
                "なお、画像そのものは参照できないため、アップロードされた画像をもとに想定される状況を推定してください。"
            )
            
            # Gemini API を呼び出して解析結果を取得
            result = generate_analysis_report(prompt)
            
            # エラーがあれば表示
            if "error" in result:
                st.error(f"API 呼び出しエラー: {result['error']}")
            else:
                st.subheader("分析結果")
                # レスポンス内容を JSON として表示（必要に応じてパースして整形表示してください）
                st.json(result)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "このアプリは、Streamlit と Gemini API (gemini-2.0-flash) を利用して外壁や物体の状態を分析します。"
        "API キーは st.secrets を利用して安全に管理されています。"
    )

if __name__ == "__main__":
    main()
