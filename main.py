import streamlit as st
import requests
from PIL import Image

FASTAPI_URL = "https://devanagari-ocr-app.onrender.com/predict"  
st.set_page_config(page_title="Devanagari OCR", page_icon="🔤", layout="centered")


st.title("🕉️ Devanagari OCR – Handwritten Character Recognition")
st.write("Upload a Devanagari character image to get predictions from multiple trained models (CNN, MobileNetV2, Fine-tuned).")

uploaded_file = st.file_uploader("📤 Upload an image (JPG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    invert_option = st.checkbox("Invert image before sending", value=True, help="Enable this if models were trained on inverted images")

    if st.button("🔍 Predict"):
        try:
            with st.spinner("Processing and predicting..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{FASTAPI_URL}?invert={invert_option}", files=files)

                if response.status_code == 200:
                    results = response.json()

                    st.subheader("📊 Predictions")
                    for model_name, result in results.items():
                        if "error" in result:
                            st.error(f"{model_name}: {result['error']}")
                        else:
                            st.success(f"**{model_name}** → {result['predicted_class']} ({result['confidence']*100:.2f}% confidence)")
                else:
                    st.error(f"Server error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("🚫 Could not connect to FastAPI server. Make sure it's running on port 8000.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

else:
    st.info("👆 Upload an image above to start.")

st.markdown("---")
st.caption("Built using FastAPI + Streamlit")
