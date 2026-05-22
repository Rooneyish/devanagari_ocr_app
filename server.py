from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
import os
import gc  

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = FastAPI(
    title="Devanagari OCR API",
    description="Recognize handwritten Devanagari characters using multiple trained models."
)

MODEL_PATHS = {
    "CNN": "models/cnn_model.keras",
    "MobileNetV2": "models/mobilenetv2_best.keras",
    "MobileNetV2FineTuned": "models/mobilenetv2_finetuned_best.keras"
}

models = {}

classes = [
    'character_01_ka', 'character_02_kha', 'character_03_ga', 'character_04_gha', 'character_05_kna',
    'character_06_cha', 'character_07_chha', 'character_08_ja', 'character_09_jha', 'character_10_yna',
    'character_11_taamatar', 'character_12_thaa', 'character_13_daa', 'character_14_dhaa', 'character_15_adna',
    'character_16_tabala', 'character_17_tha', 'character_18_da', 'character_19_dha', 'character_20_na',
    'character_21_pa', 'character_22_pha', 'character_23_ba', 'character_24_bha', 'character_25_ma',
    'character_26_yaw', 'character_27_ra', 'character_28_la', 'character_29_waw', 'character_30_motosaw',
    'character_31_petchiryakha', 'character_32_patalosaw', 'character_33_ha', 'character_34_chhya',
    'character_35_tra', 'character_36_gya', 'digit_0', 'digit_1', 'digit_2', 'digit_3', 'digit_4',
    'digit_5', 'digit_6', 'digit_7', 'digit_8', 'digit_9'
]

def preprocessing(image_bytes, invert=True):
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise ValueError("Uploaded file is not a valid image")
    
    img_rgb = img.convert("RGB").resize((32, 32))
    img_gray = img.convert("L").resize((32, 32))

    if invert:
        img_rgb = ImageOps.invert(img_rgb)
        img_gray = ImageOps.invert(img_gray)

    img_array_rgb = np.array(img_rgb, dtype=np.float32) / 255.0
    img_array_gray = np.array(img_gray, dtype=np.float32) / 255.0
    img_array_rgb = np.expand_dims(img_array_rgb, axis=0)
    img_array_gray = np.expand_dims(img_array_gray, axis=(0, -1))

    return img_array_rgb, img_array_gray

@app.get("/")
def home():
    return {"status": "healthy", "message": "Devanagari OCR API is live!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict the Devanagari character by loading models sequentially to respect RAM boundaries."""
    try:
        contents = await file.read()
        img_array_rgb, img_array_gray = preprocessing(contents, invert=True)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {e}"})
    
    results = {}
    
    for model_name, path in MODEL_PATHS.items():
        try:
            current_model = load_model(path)
            
            if model_name == "CNN":
                prediction = current_model.predict(img_array_gray, verbose=0)
            else:
                prediction = current_model.predict(img_array_rgb, verbose=0)

            pred_idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            results[model_name] = {
                "predicted_class": classes[pred_idx],
                "confidence": round(confidence, 4)
            }
            
            del current_model
            from tensorflow.keras import backend as K
            K.clear_session()
            gc.collect()

        except Exception as e:
            results[model_name] = {"error": f"Prediction failed: {e}"}

    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)