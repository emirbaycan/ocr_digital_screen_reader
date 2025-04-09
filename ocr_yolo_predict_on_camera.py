import os
import cv2
import av
import time
import torch
import numpy as np
import re  # For digit filtering
from ultralytics import YOLO
from ocr_model import CRNN  # Import only model class
import json

# RTSP Stream URL
username = "admin"
password = "password"
ip = "192.168.1.98"
port = "554"
rtsp_url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/onvif1"

# Load YOLOv8 model
model = YOLO("runs/detect/train10/weights/best.pt")
model.overrides["verbose"] = False

# Load OCR model
MODEL_PATH = "ocr_crnn.pth"
CHAR_MAP_FILE = "char_to_idx.json"

# Load character mapping
with open(CHAR_MAP_FILE, "r") as f:
    char_to_idx = json.load(f)
idx_to_char = {v: k for k, v in char_to_idx.items()}

# Initialize OCR model
num_classes = len(char_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model = CRNN(num_classes).to(device)
ocr_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
ocr_model.eval()

# Debugging: Verify CNN feature size before inference
dummy_input = torch.zeros(1, 1, 120, 100).to(device)  # Use the same size as in training
with torch.no_grad():
    cnn_out = ocr_model.cnn(dummy_input)  # Run the CNN on a dummy image
    _, c, h, w = cnn_out.shape
    feature_size = c * h
    print(f"🔹 CNN Feature Size (Inference): {feature_size}")

# Ensure LSTM input size matches
if feature_size != ocr_model.feature_size:
    print(f"⚠️ Feature size mismatch! Expected: {ocr_model.feature_size}, Got: {feature_size}")
    ocr_model.feature_size = feature_size  # Force the correct feature size

# Output directory for cropped screens
output_folder = "camera_images"
os.makedirs(output_folder, exist_ok=True)

container = av.open(rtsp_url, options={
    
    "rtsp_transport": "udp",  # Force UDP
    "flags": "low_delay",  # Enable low-latency mode
    "max_delay": "100000",  # Reduce latency
    "buffer_size": "8192000",  # Increase buffer
    "reorder_queue_size": "32",  # Store more packets
    "threads": "auto",
    "hwaccel": "cuda"
})

stream = container.streams.video[0]
stream_time_base = stream.time_base
last_saved_time = 0
save_interval = 1  # Save one frame per second

# Open a window for real-time display
cv2.namedWindow("YOLO + OCR Live Detection", cv2.WINDOW_NORMAL)

# Ensure window is created before setting properties
time.sleep(2)

# Set window to fullscreen (corrected window name)
cv2.setWindowProperty("YOLO + OCR Live Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def preprocess_image(img, target_size=(120, 100)):
    """Ensures input images match the expected size for the OCR model."""

    # Swap dimensions to match the expected format (Height, Width)
    resized_img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    # Convert to float32, normalize, and convert to tensor
    resized_img = resized_img.astype(np.float32) / 255.0
    tensor_img = torch.tensor(resized_img).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, H, W]

    return tensor_img

def decode_text(output):
    """Convert model output to readable text while handling tensor format correctly."""
    _, preds = torch.max(output, 2)  # Get max probability index
    preds = preds.squeeze().tolist()  # Convert to a flat list

    decoded_text = ""
    last_char = None
    for idx in preds:
        if isinstance(idx, list):  # Ensure single integer values
            idx = idx[0]
        if idx in idx_to_char:
            char = idx_to_char[idx]
            if char != "<BLANK>" and char != last_char:  # Ignore consecutive duplicates
                decoded_text += char
            last_char = char  # Track previous character

    return decoded_text.strip()  # Remove extra spaces

def is_valid_meter_reading(text):
    """Check if OCR output is exactly 3 digits (ignores non-numeric characters)."""
    digits_only = re.sub(r"\D", "", text)  # Remove non-digit characters
    return len(digits_only) == 3

save_data = False

# Process frames from RTSP stream
for frame in container.decode(video=0):
    if frame.pts is None:
        continue

    frame_time = frame.pts * stream_time_base
    img = frame.to_ndarray(format='bgr24')  # Convert to numpy (BGR format)

    # Run YOLO object detection
    results = model(img, conf=0.5)

    detected = False  # Track if any valid detection occurs
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            confidence = box.conf[0]  # Confidence score

            # Crop the detected screen
            cropped_screen = img[y1:y2, x1:x2]
            gray_screen = cv2.cvtColor(cropped_screen, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            processed_screen = preprocess_image(gray_screen)

            # Run OCR model
            with torch.no_grad():
                ocr_output = ocr_model(processed_screen).permute(1, 0, 2)
            predicted_text = decode_text(ocr_output)

            # Ensure prediction is exactly 3 digits
            if is_valid_meter_reading(predicted_text):
                detected = True

                # Save cropped screen image with OCR result as filename
                if frame_time - last_saved_time >= save_interval:
                    last_saved_time = frame_time
                    timestamp = int(time.time())
                    filename = os.path.join(output_folder, f"screen_{timestamp}_{predicted_text}.png")
                    if save_data:
                        cv2.imwrite(filename, cropped_screen)  # Save only the cropped screen
                        print(f"✅ Saved: {filename} | Prediction: {predicted_text}")
                    else: None
                        

    # Display annotated frame if detection occurs
    if detected:
        cv2.imshow("YOLO + OCR Live Detection", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
container.close()
