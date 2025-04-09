import os
import json

# Paths
LABEL_FILE = "ocr_labels.json"
DATASET_FILE = "ocr_dataset.txt"
IMAGE_FOLDER = "cropped_screens"

# Load existing labels
if os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, "r") as f:
        labels = json.load(f)
else:
    labels = {}

# Check which images actually exist
valid_labels = {}
missing_images = []

for img_name, label in labels.items():
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    if os.path.exists(img_path):
        valid_labels[img_name] = label
    else:
        missing_images.append(img_name)

# Save cleaned labels
with open(LABEL_FILE, "w") as f:
    json.dump(valid_labels, f, indent=4)

# Recreate `ocr_dataset.txt`
with open(DATASET_FILE, "w") as f:
    for img_name, label in valid_labels.items():
        f.write(f"{os.path.join(IMAGE_FOLDER, img_name)} {label}\n")

# Output cleanup results
print(f"✅ Dataset cleaned! Removed {len(missing_images)} missing images.")
if missing_images:
    print(f"🗑 Removed: {missing_images[:5]}{'...' if len(missing_images) > 5 else ''}")
print(f"📂 Remaining images: {len(valid_labels)}")