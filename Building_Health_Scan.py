import streamlit as st
import requests
from PIL import Image
import io

def analyze_image(image_bytes):
    """
    Gemini API を呼び出し、画像解析を実行する関数
    シークレット機能を利用して API キーとエンドポイントを管理しています。

    ※シークレット情報は、.streamlit/secrets.toml などに以下のように設定してください:
        [gemini]
        API_KEY = "YOUR_GEMINI_API_KEY"
        API_ENDPOINT = "https://api.gemini.example.com/analyze"
    """
    # シークレットから API キーとエンドポイントを取得
    api_key = st.secrets["gemini"]["API_KEY"]
    endpoint = st.secrets["gemini"]["API_ENDPOINT"]
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    # 専門知識（非破壊検査、建築業、材料学）を付与したプロンプトを作成
    prompt = (
        "以下の画像について、非破壊検査の専門的知識、建築業の専門知識、材料学の専門知識を用い、"
        "国土交通省の基準に基づいた詳細な状態分析レポートを作成してください。"
        "レポートには、劣化度をA、B、C、Dなどの定量評価および推定される寿命を含めてください。"
    )
    
    files = {
        "image": ("uploaded_image.jpg", image_bytes, "image/jpeg")
    }
    
    data = {"prompt": prompt}
    
    try:
        response = requests.post(endpoint, headers=headers, data=data, files=files)
        response.raise_for_status()  # HTTPエラーがあれば例外を発生
        return response.json()  # APIは JSON 形式のレスポンスを返すと想定
    except Exception as e:
        return {"error": f"API呼び出しに失敗しました: {e}"}

def main():
    st.title("外壁・物体状態分析アプリ")
    st.write("Gemini API を利用して、画像から状態を分析するアプリです。")
    
    # サイドバーで画像入力方法の選択
    st.sidebar.header("画像入力オプション")
    input_method = st.sidebar.selectbox("画像入力方法を選択してください", ["アップロード", "カメラ撮影"])
    
    image = None
    
    if input_method == "アップロード":
        uploaded_file = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # 画像ファイルの読み込み
            image = Image.open(uploaded_file)
    elif input_method == "カメラ撮影":
        captured_image = st.camera_input("カメラで画像を撮影してください")
        if captured_image is not None:
            image = Image.open(captured_image)
    
    if image is not None:
        st.subheader("入力画像")
        st.image(image, caption="選択された画像", use_column_width=True)
        
        if st.button("分析実行"):
            st.write("画像を分析中です。しばらくお待ちください...")
            # 画像をバイナリデータに変換
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # Gemini API を呼び出して解析結果を取得
            result = analyze_image(img_bytes)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("分析結果")
                # ※以下はサンプルの結果表示です。実際の API レスポンス形式に合わせて調整してください。
                st.write("**劣化度評価:** ", result.get("deterioration_rating", "評価情報なし"))
                st.write("**推定寿命:** ", result.get("estimated_lifetime", "寿命情報なし"))
                st.write("**詳細レポート:**")
                st.text(result.get("detailed_report", "レポート情報なし"))
    
    st.sidebar.markdown("---")
    st.sidebar.info("※ このアプリは、国土交通省の基準に基づく分析を行い、"
                    "非破壊検査、建築業、材料学の専門知識を活用して画像解析を実施します。")

if __name__ == "__main__":
    main()
