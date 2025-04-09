import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ocr_model import CRNN  # Import only the model class

# Paths
LABEL_FILE = "ocr_labels.json"
DATASET_FILE = "ocr_dataset.txt"
CHAR_MAP_FILE = "char_to_idx.json"
MODEL_SAVE_PATH = "ocr_crnn.pth"
CHECKPOINT_DIR = "checkpoints"
TRAIN_SAMPLE_DIR = "training_samples"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TRAIN_SAMPLE_DIR, exist_ok=True)

# Load labels
if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(f"⚠️ Label file {LABEL_FILE} not found!")

with open(LABEL_FILE, "r") as f:
    labels = json.load(f)

# Load or create character mapping
if os.path.exists(CHAR_MAP_FILE):
    with open(CHAR_MAP_FILE, "r") as f:
        char_to_idx = json.load(f)
else:
    unique_chars = set("".join(labels.values()))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(unique_chars)}
    char_to_idx["<BLANK>"] = 0
    with open(CHAR_MAP_FILE, "w") as f:
        json.dump(char_to_idx, f, indent=4)

num_classes = len(char_to_idx)
idx_to_char = {v: k for k, v in char_to_idx.items()}
print(f"🔢 Num Classes: {num_classes}")

# Image Augmentation & Normalization
transform = A.Compose([
    A.Resize(120, 100),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.ImageCompression(quality_range=(90, 90), p=0.2),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2()
])

# Dataset Definition
class OCRDataset(Dataset):
    def __init__(self, dataset_file):
        self.samples = []
        with open(dataset_file, "r") as f:
            for line in f.readlines():
                img_path, label = line.strip().split(" ", 1)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"⚠ Image not found: {img_path}")

        transformed = transform(image=img)
        img = transformed["image"]

        label_indices = [char_to_idx.get(c, char_to_idx["<BLANK>"]) for c in label]

        # Save every 10th image for debugging
        if idx % 10 == 0:
            debug_img_path = os.path.join(TRAIN_SAMPLE_DIR, f"sample_{idx}.png")
            cv2.imwrite(debug_img_path, img.cpu().numpy().squeeze() * 255)

        return img, torch.tensor(label_indices, dtype=torch.long)


if __name__ == "__main__":
    # Training Configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 500
    SAVE_INTERVAL = 10
    LEARNING_RATE = 0.0005
    VALIDATION_SPLIT = 0.1
    ACCURACY_THRESHOLD = 0.95  # Stop training if accuracy reaches this value
    EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for X epochs

    # Load Dataset
    dataset = OCRDataset(DATASET_FILE)
    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    num_workers = min(4, os.cpu_count())  # Use max 4 workers or available CPU cores
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Using device: {device}")

    model = CRNN(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CTCLoss()
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # Resume training if previous model exists
    if os.path.exists(MODEL_SAVE_PATH):
        print("📂 Resuming training from previous checkpoint...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    best_val_loss = float("inf")
    best_accuracy = 0.0
    early_stop_counter = 0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss, correct_chars, total_chars = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)
                target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)
                loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            # Character-level Accuracy
            predictions = torch.argmax(outputs, dim=2)
            correct_chars += (predictions == labels[:, :outputs.size(1)]).sum().item()
            total_chars += labels.numel()

        accuracy = correct_chars / total_chars
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / epoch) * (NUM_EPOCHS - epoch)

        print(f"📌 Epoch [{epoch}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"⏳ Elapsed: {elapsed_time:.2f}s | Estimated Remaining: {remaining_time:.2f}s")

        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)
                target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)
                loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, target_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"📊 Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss or accuracy > best_accuracy:
            best_val_loss = avg_val_loss
            best_accuracy = accuracy
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            early_stop_counter += 1

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("🛑 Early stopping triggered. No improvement in validation loss.")
            break

    print("✅ Model training complete and saved.")
