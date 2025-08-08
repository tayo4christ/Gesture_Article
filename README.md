# Creating a Real-Time Gesture-to-Text Translator Using Python and Mediapipe

A beginner-friendly tutorial project that detects hand gestures in real time and converts them to text using Python, MediaPipe, OpenCV, and scikit-learn.

> This repository accompanies an article prepared for freeCodeCamp Developer News.

## ✨ Features
- Real-time hand tracking (21 landmarks per hand) using MediaPipe
- Data collection script for your own gesture dataset
- Scikit-learn classifier training pipeline
- Real-time inference script that overlays predicted gesture text
- Ethical & accessibility checklist
- Diagrams, screenshots, and a sample dataset

## 📦 Project Structure
```
gesture-to-text-translator/
├── data/
│   └── gesture_data.csv              # Sample dataset (landmarks + label)
├── diagrams/
│   ├── pipeline_flowchart.png
│   ├── dataset_example.png
│   └── landmarks_concept.png
├── screenshots/
│   ├── detection_example.png
│   └── text_output.png
├── src/
│   ├── collect_data.py               # Collect labeled gestures via webcam
│   ├── train_model.py                # Train and save a classifier
│   └── gesture_to_text.py            # Real-time inference
├── requirements.txt
├── README.md
└── LICENSE
```

## 🧰 Prerequisites
- Python 3.8+
- A working webcam

## 🔧 Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 🧪 1) Collect Your Own Gesture Data
This opens your webcam, tracks your hand, and saves landmark coordinates (x, y, z for each of the 21 points). Press keys to label captured frames.

```bash
python src/collect_data.py --label thumbs_up --samples 200
python src/collect_data.py --label open_palm --samples 200
python src/collect_data.py --label ok --samples 200
```

All samples are appended to `data/gesture_data.csv`.

## 🧠 2) Train a Classifier
```bash
python src/train_model.py --data data/gesture_data.csv --model data/gesture_model.pkl
```

The script prints accuracy and saves the model.

## ⚡ 3) Run Real-Time Translator
```bash
python src/gesture_to_text.py --model data/gesture_model.pkl
```

The window will display the predicted gesture at the top-left.

## 🧭 How It Works
1. Webcam frames → MediaPipe Hand Tracking
2. 21×(x,y,z) landmarks are flattened into a feature vector
3. A scikit-learn classifier predicts the gesture label
4. The predicted label is drawn on the video frame

See `diagrams/pipeline_flowchart.png` for a visual overview.

## 🔒 Ethics, Privacy, and Accessibility
- Collect consent if recording data from others.
- Avoid storing raw video unless necessary; landmark coordinates are often sufficient.
- Train with diverse data (skin tones, backgrounds, lighting).
- Consider cultural differences in gesture meanings.
- Test in realistic conditions before deployment.

## 🧩 Extending This Project
- Add Text-to-Speech with `pyttsx3` to speak recognized words
- Deploy in-browser using TensorFlow.js (re-implement the model)
- Expand the vocabulary (more classes, phrase composition)
- Support both hands / multi-hand interactions

## 🔗 Canonical Link
https://dev.to/tayo4christ/how-i-built-a-real-time-gesture-to-text-translator-using-python-and-mediapipe-1c75

## 🙌 Acknowledgements
- MediaPipe team for the real-time hand tracking solution
- freeCodeCamp for their editorial guidance

---

**Author:** Omotayo  
**License:** MIT
