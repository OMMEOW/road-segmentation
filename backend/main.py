from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import time
import base64
import os

app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_mask(file: UploadFile = File(...)):
    """
    Mock inference endpoint.
    Retrieves the uploaded image, simulates some processing time,
    and returns a mock base64 mask response.
    """
    # 1. Read file to verify upload succeeded
    contents = await file.read()
    print(f"Received file: {file.filename}, size: {len(contents)} bytes")
    
    # 2. Simulate model inference taking 1.5 seconds
    time.sleep(1.5)
    
    # 3. Create a dummy base64 mask result.
    # In a real scenario, this would be your model output:
    # predicted_mask_base64 = base64.b64encode(your_mask_bytes).decode('utf-8')
    
    # Since we need a real image representation to render in UI, we'll return a 1x1 fully white transparent pixel or dummy base64
    # Let's generate a basic base64 png string representation (a 100x100 white mask, to symbolize the output)
    # Using a known small base64 pixel (red data to check its working)
    dummy_mask_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAgMBAHwD3OAAAAAASUVORK5CYII=" 
        # Actually, let's provide a very simple repeating pattern for the base64, or just a small transparent red/white block.
        # This is a 1x1 transparent red pixel:
        # iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==
    )
    
    # For now, let's return this simple image as base64. It will fill the container in the frontend.
    mask_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    
    return {
        "mask": mask_b64
        # "mask_url": "path_to_mask_image" # alternative
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
