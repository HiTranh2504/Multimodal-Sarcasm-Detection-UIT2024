# 🤖 Multimodal Sarcasm Detection with Text-Image Fusion

This project implements a **Multimodal Sarcasm Detection** system that leverages both textual and visual information (caption + image) to classify sarcastic content. The model is designed using **XLM-RoBERTa**, **Vision Transformer (ViT)**, and an innovative **cross-modal adapter-based fusion mechanism**.

---

## 🧠 Project Highlights

- **Multimodal Learning**: Combines features from text and image modalities using custom Transformer-based encoders.
- **Adapter Fine-tuning**: Efficient fine-tuning using lightweight MoBA (Mixture of Bottleneck Adapters) to avoid overfitting.
- **Cross-modal Attention**: Enables mutual interaction between text and image representations.
- **Data Augmentation**: 
  - **Text**: Back-translation and synonym replacement using WordNet & Google Translate.
  - **Image**: Albumentations with geometric and color-space transforms.
- **Early Stopping & Class Balancing**: Uses weighted loss and validation patience for robust training.

---

## 🗂️ Dataset

The training and testing data is from the [UIT-ViMMSD 2024](https://uit.ai) dataset:

- Each sample includes:
  - `caption`: user-generated comment (often noisy and informal)
  - `image`: visual context
  - `label`: one of `['not-sarcasm', 'multi-sarcasm', 'image-sarcasm', 'text-sarcasm']`

---

## 📦 Architecture Overview

```
Text (XLM-R) ─┐                  ┌─> Adapter → Fusion
              ├─> Cross-modal → ┤
Image (ViT) ──┘                  └─> Adapter → Classifier
```

| Component        | Description |
|------------------|-------------|
| `XLM-RoBERTa`    | Encodes captions (supports Vietnamese) |
| `ViT`            | Extracts image features |
| `EncoderLayer`   | Applies multi-head attention and adapter routing |
| `MoBA`           | Mixture of experts dynamically gated |
| `Classifier`     | Combines weighted fusion of text-image to output label |

---

## 🚀 Training Pipeline

1. **Preprocessing**
   - Remove HTML, normalize Unicode, map emojis & slang to standard form
   - Split long captions into smaller chunks if needed
2. **Data Augmentation**
   - Randomly augment text/image depending on the label type
3. **Model Training**
   - Optimizer: `AdamW`
   - Scheduler: `Linear Warmup`
   - Weighted Loss (`CrossEntropyLoss` with class `alpha`)
   - Early Stopping on macro-F1
4. **Evaluation**
   - Compute `Accuracy`, `Macro-F1` after every epoch

---

## 🔍 Inference Strategy

- Long captions are split into meaningful chunks (sentence-level)
- For each chunk, prediction is made
- The prediction with **highest confidence (probability)** is selected as the final label

**Output format:**

```json
{
  "results": {
    "0": "not-sarcasm",
    "1": "text-sarcasm",
    ...
  },
  "phase": "test"
}
```

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Macro F1-score** (used as main evaluation metric)
- **Confusion Matrix** (optional)

---


## 📌 Notes

- Model uses **lightweight adapter tuning**, making it memory-efficient and generalizable.
- Easily extendable to other multimodal tasks like meme detection, sentiment analysis with images, etc.

---
