from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import shutil
from pathlib import Path

from utils.preprocess import prepare_image

# Ensure uploads directory exists
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Mount static files for uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model on startup
MODEL_PATH = os.path.join("model", "road_segmentation_model.h5")
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict_mask(file: UploadFile = File(...)):
    """
    Inference endpoint.
    Retrieves the uploaded image, preprocesses it, runs it through the model,
    and returns a base64 encoded PNG mask.
    """
    if model is None:
        return {"error": "Model not loaded"}

    # 1. Read file
    contents = await file.read()
    print(f"Received file: {file.filename}, size: {len(contents)} bytes")
    # Save the uploaded image to the uploads directory
    upload_path = UPLOAD_DIR / file.filename
    with open(upload_path, "wb") as f:
        f.write(contents)

    # 2. Preprocess the image
    try:
        input_tensor = prepare_image(contents)
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}

    # 3. Model prediction
    try:
        # returns shape like (1, 256, 256, 1) or (1, 256, 256)
        pred = model.predict(input_tensor)

        # 4. Thresholding at 0.5 (binary mask)
        # Squeeze out batch dimension, and channel dimension if it's 1
        mask_array = np.squeeze(pred[0])
        binary_mask = (mask_array > 0.5).astype(np.uint8) * 255

        # 5. Convert to PNG image
        mask_img = Image.fromarray(binary_mask, mode="L")
        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_bytes = buffered.getvalue()

        # 6. Encode to base64
        mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')

        return {
            "mask": mask_b64,
            "original_url": f"/uploads/{file.filename}"
        }
    except Exception as e:
        return {"error": f"Failed during inference: {e}"}

@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    """
    Converts any uploaded image (especially TIFF) to a browser-safe JPEG base64 string
    so the frontend can display it in the preview panels.
    """
    contents = await file.read()
    try:
        orig_img = Image.open(io.BytesIO(contents))
        if orig_img.mode != "RGB":
            orig_img = orig_img.convert("RGB")
        orig_buffered = io.BytesIO()
        orig_img.save(orig_buffered, format="JPEG")
        orig_b64 = base64.b64encode(orig_buffered.getvalue()).decode('utf-8')
        return {"image_b64": orig_b64}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
