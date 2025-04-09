import os
import json

# Paths
IMAGE_FOLDER = "cropped_screens"
LABEL_FILE = "ocr_labels.json"
OUTPUT_FILE = "ocr_dataset.txt"

# Load labels
with open(LABEL_FILE, "r") as f:
    labels = json.load(f)

# Write dataset file
with open(OUTPUT_FILE, "w") as f:
    for img_name, text in labels.items():
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        f.write(f"{img_path} {text}\n")

print(f"✅ Dataset saved to {OUTPUT_FILE}")