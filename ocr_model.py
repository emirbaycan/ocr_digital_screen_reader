import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Paths
LABEL_FILE = "ocr_labels.json"
DATASET_FILE = "ocr_dataset.txt"
CHAR_MAP_FILE = "char_to_idx.json"
MODEL_SAVE_PATH = "ocr_crnn.pth"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load Labels
if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(f"⚠️ Label file {LABEL_FILE} not found!")

with open(LABEL_FILE, "r") as f:
    labels = json.load(f)

# Load or Create Character Mapping
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



# Dataset Class
class OCRDataset(Dataset):
    def __init__(self, dataset_file, target_size=(120, 100)):
        self.samples = []
        with open(dataset_file, "r") as f:
            for line in f.readlines():
                img_path, label = line.strip().split(" ", 1)
                self.samples.append((img_path, label))
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"⚠ Image not found: {img_path}")

        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0  
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        label_indices = [char_to_idx.get(c, char_to_idx["<BLANK>"]) for c in label]

        return img, torch.tensor(label_indices, dtype=torch.long)

class CRNN(nn.Module):
    def __init__(self, num_classes, input_size=(120, 100)):  # Default input size
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Compute feature size dynamically from input shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)  # Create a fake input
            cnn_out = self.cnn(dummy_input)  # Forward pass through CNN
            _, c, h, w = cnn_out.shape  # Extract width for LSTM
            self.feature_size = c * h  # Final feature size

        # Initialize LSTM & FC Layer
        self.rnn = nn.LSTM(self.feature_size, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # Pass through CNN
        batch_size, channels, height, width = x.shape  

        x = x.permute(0, 3, 1, 2)  # (batch, width, height, channels)
        x = x.contiguous().view(batch_size, width, -1)  # Reshape for LSTM
        
        x, _ = self.rnn(x)  # LSTM expects `self.feature_size`
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=2)  # Apply log_softmax




if __name__ == "__main__":
    dataset = OCRDataset(DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CTCLoss()

    num_epochs = 500
    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss, correct_chars, total_chars = 0, 0, 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long).to(device)

            loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=2)
            for i in range(len(labels)):
                pred_seq = predictions[i][:target_lengths[i]].tolist()
                true_seq = labels[i][:target_lengths[i]].tolist()
                correct_chars += sum(1 for a, b in zip(pred_seq, true_seq) if a == b)
                total_chars += len(true_seq)

        accuracy = correct_chars / total_chars
        elapsed_time = time.time() - start_time
        print(f"📌 Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Elapsed: {elapsed_time:.2f}s")

        if accuracy >= 0.99:
            print("🎉 Stopping early - model reached target accuracy!")
            break

    torch.save(model.state_dict(), "ocr_crnn.pth")
    print("✅ OCR Model trained and saved as ocr_crnn.pth")

if __name__ == "__main__":
    print(f"🔢 Num Classes: {num_classes}")  # ✅ Only print when directly executed