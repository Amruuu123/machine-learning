# 🧠 Bayesian Pneumonia Detection using Chest X-rays

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project implements a **Bayesian Convolutional Neural Network (BCNN)** to detect **pneumonia** from chest X-ray images, using **uncertainty estimation** for reliable diagnosis and **Grad-CAM** for explainable AI. Built with PyTorch and deployed via a Flask web application.

---

## 🚀 Features

- ✅ **Bayesian CNN** for predictive uncertainty (using [`torchbnn`](https://github.com/Hzzone/torch-bnn))
- ✅ **Chest X-ray classification**: Pneumonia vs Normal
- ✅ **Grad-CAM visualization** to interpret model decisions
- ✅ **Flask Web App** for user-friendly image upload and prediction
- ✅ **Early Stopping**, **Model Checkpointing**, and **GPU Acceleration**
- ✅ **Metrics**: Accuracy, Confusion Matrix, Model Size, Inference Time

---

## 📁 Dataset

The model is trained on the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle.

- `train/`, `val/`, `test/` directories
- Two classes: `NORMAL`, `PNEUMONIA`

---

## 🧠 Model Overview

- Backbone: ResNet18 / ResNet34
- Bayesian Layers: Applied using `torchbnn` for modeling uncertainty
- Loss: `nn.CrossEntropyLoss()` + `KL Divergence Loss`
- Optimizer: Adam
- Explainability: Grad-CAM for visual heatmaps

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/bayesian-pneumonia-detection.git
cd bayesian-pneumonia-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
