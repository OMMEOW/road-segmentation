import io
from PIL import Image
import numpy as np

def prepare_image(image_bytes: bytes) -> np.ndarray:
    """
    Reads image bytes, resizes to 256x256, converts to an array,
    normalizes to [0, 1], and adds a batch dimension.
    Target shape: (1, 256, 256, 3)
    """
    # Open image using Pillow
    img = Image.open(io.BytesIO(image_bytes))
    
    # Ensure it's RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Resize to exactly 256x256
    img = img.resize((256, 256))
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension -> (1, 256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
