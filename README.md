# 🦷 Dental Cavity Detection using Deep Learning

A deep learning-based medical image classification project that detects dental cavities from dental images using Transfer Learning with PyTorch.

---

## 🚀 Project Overview

Dental cavities (caries) are one of the most common oral health problems worldwide. Early detection is crucial for proper treatment.

This project leverages Convolutional Neural Networks (CNNs) and transfer learning to automatically classify dental images into:

- ✅ Cavity
- ❌ No Cavity

The model is trained and evaluated on a labeled dental image dataset using a two-phase training strategy (Frozen + Fine-tuning).

---

## 🧠 Key Highlights

- 🔁 Reproducible training (Seed fixed)
- ⚡ GPU support (CUDA enabled)
- 🧊 Transfer Learning with Frozen Layers
- 🔓 Fine-Tuning for improved performance
- 📊 Comprehensive evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Confusion Matrix

---

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📂 Dataset Structure

Dental_Cavity_Dataset/
│
├── train/
│ ├── Cavity/
│ └── No_Cavity/
│
└── test/
├── Cavity/
└── No_Cavity/



Images are resized to **224x224** for compatibility with pretrained CNN models.

---

## ⚙️ Training Strategy

The model was trained in two phases:

### Phase 1: Frozen Training
- Pretrained backbone layers frozen
- Learning Rate: 1e-3
- Epochs: 5

### Phase 2: Fine-Tuning
- Backbone layers unfrozen
- Learning Rate: 1e-4
- Epochs: 25
- Early stopping with patience = 6

This approach allows the model to first learn task-specific classification before fine-tuning deeper feature representations.

---

## 📊 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Shah Jahan  
Computer Science Student | AI & ML Enthusiast  

- Email: shahjahan.bscsf22@iba-suk.edu.pk

---

## ⭐ If you found this project useful, consider giving it a star!

### 1️⃣ Clone Repository
git clone https://github.com/ShahJahanBrohii/convnext-dental-cavity-detection.git


