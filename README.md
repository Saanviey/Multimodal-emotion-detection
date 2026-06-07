```markdown
# Multimodal Emotion & Sentiment Detection

Audio-visual-text fusion model for simultaneous emotion and sentiment classification using multi-task learning.

## Overview

Trimodal deep learning system that processes **video** (facial expressions), **audio** (speech signals), and **text** (transcripts) through independent encoders, fuses them via an intermediate fusion layer, and jointly predicts:
- **Emotion** — 7 classes: joy, sadness, anger, fear, surprise, disgust, neutral
- **Sentiment** — 3 classes: positive, negative, neutral

## Architecture

```
Video  → ResNet3D-18  → 128-dim ─┐
Text   → BERT         → 128-dim ─┼→ Concat(384) → FusionLayer(256) → Emotion Head
Audio  → CNN+Spectrogram→128-dim ─┘                                → Sentiment Head
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Framework | PyTorch |
| Video encoding | ResNet3D-18 (pretrained) |
| Text encoding | BERT (Hugging Face Transformers) |
| Audio encoding | CNN on mel-spectrograms (Librosa) |
| Video preprocessing | OpenCV |

## Datasets

- **MELD** — primary benchmark (Friends dialogues, 13k+ utterances)
- **RAVDESS** — acted emotional speech/song
- **CREMA-D** — crowd-sourced audio-visual clips
- **AffectNet** — facial expression images

## Project Structure

```
├── encoders/
│   ├── video_encoder.py       # ResNet3D-18 + projection
│   ├── text_encoder.py        # BERT + projection
│   └── audio_encoder.py       # CNN spectrogram encoder
├── fusion/
│   └── fusion_layer.py        # Intermediate fusion (384 → 256)
├── heads/
│   ├── emotion_classifier.py  # 7-class head
│   └── sentiment_classifier.py# 3-class head
├── data/
│   └── preprocessing/         # Frame extraction, spectrogram gen, tokenization
├── train.py
└── evaluate.py
```

## Setup

```bash
pip install torch torchvision transformers librosa opencv-python
```

## Training

```bash
python train.py --dataset meld --epochs 30 --batch_size 16
```

## Evaluation Metrics

Accuracy · Precision · Recall · F1-score (per class and macro-averaged)

## Authors

- Saanvie Yadav (16801172024, CSE-AI 3)
- Aditee Swaroop (00901172024, CSE-AI 3)
```
