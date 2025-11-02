# Anomaly Detection in Time Series Using Wavelet Domain Information

This project is part of the IE 643 Deep Learning: Theory & Practice course.  
The goal is to detect anomalies in multivariate time series data using a Vision Transformer (ViT) model enhanced with wavelet domain features for improved temporal and frequency pattern extraction.

---

## Project Overview

- Utilizes the Jena Climate Dataset with 14 weather-related features recorded every 10 minutes over several years.
- Applies discrete wavelet transform (DWT) to extract both time and frequency domain information from time series windows.
- Uses a pretrained ViT (google/vit-base-patch16-224-in21k) as the backbone for feature extraction.
- Introduces a custom adapter network on top of ViT to predict anomaly scores.
- Supports both training (adapter weights only) and inference with an easy-to-use Streamlit interface accepting CSV uploads.
- Visualizes anomaly scores and highlights detected anomalies on the uploaded time series.

---

## Folder Contents

- `app.py`: The main Streamlit app implementing the full pipeline from data upload to anomaly visualization.
- `training.ipynb`: Jupyter notebook containing end-to-end training code and experiments.
- `vit_adapter_wavelet_jena.pth`: Saved adapter model weights for inference.
- `requirements.txt`: Python dependencies to run the app and training.

---

## Features

- Sliding window extraction and preprocessing of time series data.
- Wavelet transform integrated inside the model for enhanced feature representation.
- Adapter training with frozen ViT backbone for efficient fine-tuning.
- Synthetic anomaly injection for evaluation with Excess Mass, Mass Volume, and ROC-AUC metrics.
- Streamlit app interface for uploading data, predicting anomalies, and visualizing results interactively.

---

## Usage

1. Install required packages from `requirements.txt`.
2. Run the Streamlit app: `streamlit run app.py`
3. Upload your time series CSV file via the app interface.
4. View anomaly scores and highlighted points plotted over your time series features.
5. (Optional) Inject synthetic anomalies and evaluate model performance quantitatively.

---

## Training Notebook (`training.ipynb`)

- Contains detailed training scripts for adapter fine-tuning on Jena Climate data.
- Implements synthetic anomaly injection for controlled evaluation experiments.
- Performs calculation of advanced metrics (Excess Mass, Mass Volume, ROC-AUC) to assess anomaly detection quality.
- Includes visualization of training loss and anomaly detection results.
- Ideal for reproducing training and evaluation or extending the methodology.


