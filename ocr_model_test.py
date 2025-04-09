import os
import json
import torch
import cv2
import numpy as np
from ocr_model import CRNN  # Import only the model class

# Paths
MODEL_PATH = "ocr_crnn.pth"
CHAR_MAP_FILE = "char_to_idx.json"
TEST_IMAGE = "cropped_screens/screen_1738254291_597.png"  # Change this to the image you want to test

# Load character mapping
if not os.path.exists(CHAR_MAP_FILE):
    raise FileNotFoundError(f"⚠️ Character mapping file {CHAR_MAP_FILE} not found!")

with open(CHAR_MAP_FILE, "r") as f:
    char_to_idx = json.load(f)

# Reverse mapping for decoding predictions
idx_to_char = {v: k for k, v in char_to_idx.items()}

# Load trained OCR model
num_classes = len(char_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CRNN(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set model to evaluation mode


def preprocess_image(image_path, target_size=(120, 100)):
    """Preprocess an image for OCR model (resizing while maintaining aspect ratio)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"⚠️ Image {image_path} not found or cannot be read!")

    h, w = img.shape
    target_w, target_h = target_size

    # Compute scaling factor while keeping aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    # Resize the image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a white background (255)
    padded_img = np.ones((target_h, target_w), dtype=np.uint8) * 255  

    # Compute padding to center the image
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2

    # Place resized image at the center
    padded_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized_img

    # Normalize & Convert to Tensor
    padded_img = padded_img.astype(np.float32) / 255.0  # Normalize
    padded_img = torch.tensor(padded_img).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]

    return padded_img.to(device)


def decode_text(output):
    """Convert model output to readable text while removing consecutive blanks."""
    _, preds = torch.max(output, 2)  # Get max probability index
    preds = preds.squeeze().tolist()  # Flatten the tensor properly

    decoded_text = ""
    last_char = None
    for idx in preds:
        if isinstance(idx, list):  # Fix issue where it's still a nested list
            idx = idx[0]
        if idx in idx_to_char:
            char = idx_to_char[idx]
            if char != "<BLANK>":  # Ignore blank tokens
                if char != last_char:  # Prevent consecutive duplicates
                    decoded_text += char
            last_char = char  # Track previous char

    return decoded_text.strip()  # Remove extra spaces


def predict_text(image_path):
    """Perform OCR prediction on a given image."""
    img = preprocess_image(image_path)

    with torch.no_grad():
        output = model(img).permute(1, 0, 2)  # Shape: [B, W, C]

    return decode_text(output)


# Run test
if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE):
        raise FileNotFoundError(f"⚠️ Test image {TEST_IMAGE} not found!")

    result = predict_text(TEST_IMAGE)
    print(f"🔹 Predicted Text: {result}")
