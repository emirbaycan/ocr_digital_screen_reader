# ocr_digital_screen_reader

**ocr_digital_screen_reader** is an open-source Python tool for reading numbers or text from digital screens in images—especially designed for use on cropped screen segments from devices such as meters, oscilloscopes, digital panels, or lab equipment.

---

## Features

- **Deep learning-based OCR**: Uses a CRNN model for accurate character recognition.
- **Ready for digit and character screens**: Trained on images of real device displays.
- **Command-line interface**: Predict text from a single image or batch easily.
- **Extensible**: Ready for integration into larger automated pipelines or real-time workflows.

---

## Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/emirbaycan/ocr_digital_screen_reader.git
cd ocr_digital_screen_reader
pip install -r requirements.txt
```

---

## Usage

**Predict text from a digital screen image:**

```bash
python ocr_model_test.py \
  --model ocr_crnn.pth \
  --charmap char_to_idx.json \
  --input test_image.png
```

- By default, the script uses `ocr_crnn.pth` as model, `char_to_idx.json` as character map, and `test_image.png` as input.
- You can change these paths with `--model`, `--charmap`, and `--input` arguments.
- To save prediction to a file:  
  ```bash
  python ocr_model_test.py --input your_image.png --output result.txt
  ```

**Example Output:**
```
🔹 Predicted Text: 358
```

---

## Model Training

You can train your own OCR model using your own device screens or other digit/text datasets.

1. **Dataset Preparation**
   - List each image path and its label in `ocr_dataset.txt` (one per line: `<image_path> <label>`).
   - Character mappings are stored in `char_to_idx.json` (auto-generated from your labels).

2. **Training**
   - Use the training script to start training:
     ```bash
     python ocr_model_train.py
     ```
   - Training settings (epochs, batch size, etc.) can be edited at the top of the script.

3. **Augmentation**
   - The training pipeline uses Albumentations for augmenting images (resize, blur, contrast, compression).

---

## File Structure

```
ocr_digital_screen_reader/
│
├── checkpoints/         # Model checkpoints during training
├── cropped_screens/     # Example cropped input images
├── ocr_crnn.pth         # Trained model weights
├── char_to_idx.json     # Character to index mapping
├── ocr_labels.json      # Ground truth labels for dataset images
├── ocr_dataset.txt      # Paths and labels for each image
├── ocr_model_test.py    # Script for predicting text from image
├── ocr_model_train.py   # Script for training the model
├── ocr_model.py         # Model definition (CRNN)
├── requirements.txt     # Python dependencies
└── test_image.png       # Example test image
```

---

## Requirements

All dependencies are listed in `requirements.txt`, including:
- torch
- torchvision
- ultralytics
- albumentations
- numpy
- opencv-python
- Pillow
- matplotlib
- tqdm
- av

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Dataset Annotation

- You can use any annotation method that outputs a simple text file:
  ```
  path/to/image1.png 12345
  path/to/image2.png 67890
  ```
- For larger or custom datasets, adapt the loader in `ocr_create_dataset.py`.

---

## Contributing

Feel free to open issues or pull requests for improvements, bug fixes, or new features.

---

## License

MIT License

---

## Contact

Developed by Emir Baycan  
[GitHub](https://github.com/emirbaycan)

For questions or feedback, open an issue or contact via GitHub.

---
