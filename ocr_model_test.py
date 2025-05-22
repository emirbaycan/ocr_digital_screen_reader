import os
import json
import torch
import cv2
import numpy as np
import argparse
from ocr_model import CRNN  # Import only the model class

def preprocess_image(image_path, target_size=(100, 120), device="cpu"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"⚠️ Image {image_path} not found or cannot be read!")
    h, w = img.shape
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_img = np.ones((target_h, target_w), dtype=np.uint8) * 255
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    padded_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized_img
    padded_img = padded_img.astype(np.float32) / 255.0
    padded_img = torch.tensor(padded_img).unsqueeze(0).unsqueeze(0)
    return padded_img.to(device)

def decode_text(output, idx_to_char):
    _, preds = torch.max(output, 2)
    preds = preds.squeeze().tolist()
    decoded_text = ""
    last_char = None
    for idx in preds:
        if isinstance(idx, list):
            idx = idx[0]
        char = idx_to_char.get(idx, "")
        if char != "<BLANK>":
            if char != last_char:
                decoded_text += char
        last_char = char
    return decoded_text.strip()

def predict_text(image_path, model, idx_to_char, device="cpu"):
    img = preprocess_image(image_path, device=device)
    with torch.no_grad():
        output = model(img).permute(1, 0, 2)
    return decode_text(output, idx_to_char)

def main():
    parser = argparse.ArgumentParser(description="OCR Digital Screen Reader CLI")
    parser.add_argument("--model", type=str, default="ocr_crnn.pth", help="Path to trained OCR model .pth file")
    parser.add_argument("--charmap", type=str, default="char_to_idx.json", help="Path to character mapping json file")
    parser.add_argument("--input", type=str, default="test_image.png", help="Path to input image file")
    parser.add_argument("--output", type=str, default=None, help="(Optional) Path to output .txt file for prediction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu")
    args = parser.parse_args()

    # Check files
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"⚠️ Model file {args.model} not found!")
    if not os.path.exists(args.charmap):
        raise FileNotFoundError(f"⚠️ Character mapping file {args.charmap} not found!")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"⚠️ Input image {args.input} not found!")

    # Load char map
    with open(args.charmap, "r") as f:
        char_to_idx = json.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}

    # Load model
    num_classes = len(char_to_idx)
    model = CRNN(num_classes).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()

    # Predict
    try:
        result = predict_text(args.input, model, idx_to_char, device=args.device)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return

    print(f"🔹 Predicted Text: {result}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fout:
            fout.write(result + "\n")
        print(f"✅ Saved result to {args.output}")

if __name__ == "__main__":
    main()
