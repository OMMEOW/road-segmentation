import cv2
import numpy as np

def prepare_image(image_bytes: bytes) -> np.ndarray:
    """
    Reads image bytes, resizes to 256x256, converts to an array,
    applies CLAHE on the Lightness (L) channel, normalizes to [0, 1],
    and adds a batch dimension. This perfectly matches the training processing.
    Target shape: (1, 256, 256, 3)
    """
    # Decode bytes to OpenCV BGR image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image from bytes.")

    # Resize to 256x256
    img = cv2.resize(img, (256, 256))

    # Convert BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge channels back and convert to BGR
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert to float and normalize to [0, 1]
    img = img / 255.0

    # Add batch dimension -> (1, 256, 256, 3)
    img_array = np.expand_dims(img, axis=0)
    
    return img_array
