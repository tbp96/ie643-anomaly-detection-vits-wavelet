import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTModel
from pytorch_wavelets import DWTForward
import matplotlib.pyplot as plt

# ======== Parameters (same as training) ================
window_size = 24
step = 4
resize_target = (96, 96)

#=================== Model definition =======================
class ViTWithWaveletAdapter(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.vit = vit_model
        self.adapter = nn.Sequential(
            nn.Linear(vit_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        Yl, Yh = self.dwt(x)
        detail_bands = Yh[0]
        processed_bands = []
        target_size = Yl.shape[2:]
        target_batch = Yl.shape[0]
        for band in detail_bands:
            while band.dim() > 4:
                band = band.squeeze(dim=-1)
            if band.shape[0] == 1 and target_batch > 1:
                band = band.repeat(target_batch, *[1] * (band.dim() - 1))
            if band.shape[2:] != target_size:
                band = nn.functional.interpolate(band, size=target_size, mode='bilinear', align_corners=False)
            processed_bands.append(band)
        Yh_cat = torch.cat(processed_bands, dim=1)
        x_wave = torch.cat([Yl, Yh_cat], dim=1)
        x_wave_upsampled = nn.functional.interpolate(x_wave, size=(224, 224), mode='bilinear', align_corners=False)
        x_in = x_wave_upsampled.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        outputs = self.vit(pixel_values=x_in)
        pooled = outputs.pooler_output
        return self.adapter(pooled)

# ============= Dataset for inference ==================
class TestDataset(Dataset):
    def __init__(self, images):
        self.images = images.astype(np.float32)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img = pil_img.resize(resize_target[::-1], Image.BILINEAR)
        return transforms.ToTensor()(pil_img)


@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    for param in vit_model.parameters():
        param.requires_grad = False
    model = ViTWithWaveletAdapter(vit_model).to(device)
    model.adapter.load_state_dict(torch.load('vit_adapter_wavelet_jena.pth', map_location=device))
    model.eval()
    return model, device

def create_windows(data, window_size, step):
    X = []
    length = data.shape[0]
    if length < window_size:
        pad_width = window_size - length
        data = np.vstack([data, np.zeros((pad_width, data.shape[1]))])
        length = window_size
    for i in range(0, length - window_size + 1, step):
        X.append(data[i:i+window_size])
    return np.stack(X)

def predict_anomaly_scores(model, device, values):
    patches = create_windows(values, window_size, step)
    images = patches.transpose(0, 2, 1)
    dataset = TestDataset(images)
    dataloader = DataLoader(dataset, batch_size=8)

    scores = []
    with torch.no_grad():
        for xb in dataloader:
            xb = xb.to(device)
            pred = model(xb)
            scores.extend(pred.squeeze().abs().cpu().numpy())
    return np.array(scores), patches

######## ================== Visualization functions for High score anomaly only ######## =======================

def plot_results(values, scores, timestamps, step, window_size):
    time_series_length, num_features = values.shape

    window_indices = np.arange(len(scores))
    time_indices = window_indices * step + window_size // 2
    time_indices = np.clip(time_indices, 0, len(timestamps) - 1).astype(int)

    threshold = np.percentile(scores, 95)
    anomaly_mask = scores > threshold
    anomaly_indices = time_indices[anomaly_mask]

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot all features faded
    for i in range(num_features):
        ax.plot(range(time_series_length), values[:, i], alpha=0.5, label=f'Feature {i+1}')

    # Normalize and scale anomaly scores
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    scaled_scores = scores_norm * (values.max() - values.min()) + values.min()

    ax.plot(time_indices, scaled_scores, color='black', linewidth=2, label='Anomaly Score (scaled)')

    # Highlight anomaly points with circle markers
    anomaly_feature_values = values[anomaly_indices, 0]
    ax.scatter(anomaly_indices, anomaly_feature_values, color='red', s=100, marker='o', label='Anomaly')

    ax.set_title('Multivariate Time Series with Anomaly Scores and Highlighted Anomalies')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Feature Value / Scaled Anomaly Score')
    ax.legend(loc='upper right')
    ax.grid(True)

    st.pyplot(fig)

def display_anomalies_with_timestamps(test_df, scores, step, window_size):
    # Parse datetime column (adjust column name to actual CSV)
    test_df['datetime'] = pd.to_datetime(test_df['Date Time'])  # Change 'Date Time' if needed
    
    # Calculate windows center indices
    window_indices = np.arange(len(scores))
    time_indices = window_indices * step + window_size // 2

    # Get datetime values as numpy array
    datetime_values = test_df['datetime'].values

    # Threshold for anomaly detection (95th percentile)
    threshold = np.percentile(scores, 95)
    anomaly_mask = scores > threshold

    # Filter anomaly time indices safely
    anomaly_time_indices = time_indices[anomaly_mask]
    anomaly_time_indices = anomaly_time_indices[anomaly_time_indices < len(datetime_values)].astype(int)

    # Extract timestamps of anomaly windows
    anomaly_datetimes = datetime_values[anomaly_time_indices]

    st.write("### Detected anomaly points with date/time and anomaly scores:")
    for dt, score in zip(anomaly_datetimes, scores[anomaly_mask]):
        st.write(f"**Time:** {pd.to_datetime(dt)}  -  **Anomaly Score:** {score:.4f}")


# ######## ================== Visualization functions for Two sided anomaly ######## =======================

# def detect_two_sided_anomalies(scores, low_percentile=5, high_percentile=95):
#     low_threshold = np.percentile(scores, low_percentile)
#     high_threshold = np.percentile(scores, high_percentile)
#     anomaly_mask = (scores <= low_threshold) | (scores >= high_threshold)
#     return anomaly_mask, low_threshold, high_threshold

# def display_anomalies_with_timestamps(test_df, scores, step, window_size):
#     test_df['datetime'] = pd.to_datetime(test_df['Date Time'])  # Change 'Date Time' if needed
#     # Calculate window centers
#     window_indices = np.arange(len(scores))
#     time_indices = window_indices * step + window_size // 2

#     datetime_values = test_df['datetime'].values

#     anomaly_mask, low_thr, high_thr = detect_two_sided_anomalies(scores)

#     anomaly_time_indices = time_indices[anomaly_mask]
#     anomaly_time_indices = anomaly_time_indices[anomaly_time_indices < len(datetime_values)].astype(int)
#     anomaly_datetimes = datetime_values[anomaly_time_indices]

#     st.write("### Detected anomaly points with date/time and anomaly scores:")
#     for dt, score in zip(anomaly_datetimes, scores[anomaly_mask]):
#         st.write(f"**Time:** {pd.to_datetime(dt)}  -  **Anomaly Score:** {score:.4f}")



# def plot_results(values, scores, timestamps, step, window_size):
#     time_series_length, num_features = values.shape
#     window_indices = np.arange(len(scores))
#     time_indices = window_indices * step + window_size // 2
#     time_indices = np.clip(time_indices, 0, len(timestamps) - 1).astype(int)

#     # Detecting anomalies at both tails
#     anomaly_mask, low_thresh, high_thresh = detect_two_sided_anomalies(scores)
#     anomaly_indices = time_indices[anomaly_mask]

#     fig, ax = plt.subplots(figsize=(16, 8))
#     # Plotting features
#     for i in range(num_features):
#         ax.plot(range(time_series_length), values[:, i], alpha=0.5, label=f'Feature {i+1}')
#     # Plotting anomaly scores
#     scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
#     scaled_scores = scores_norm * (values.max() - values.min()) + values.min()
#     ax.plot(time_indices, scaled_scores, color='black', linewidth=2, label='Anomaly Score (scaled)')
#     # Highlighting anomalies
#     anomaly_feature_values = values[anomaly_indices, 0]
#     ax.scatter(anomaly_indices, anomaly_feature_values, color='red', s=100, marker='o', label='Anomaly')
#     # Final touches
#     ax.set_title('Multivariate Time Series with Anomaly Scores and Highlights')
#     ax.set_xlabel('Time Index')
#     ax.set_ylabel('Feature Value / Scaled Anomaly Score')
#     ax.legend(loc='upper right')
#     ax.grid(True)
#     # Show plot
#     st.pyplot(fig)

    
    
def main():
    st.title("Anomaly Detection for Time Series Climate Data Using ViT + Wavelet Domain Information")
    st.write("Upload a CSV file containing datetime and feature columns to detect anomalies.")

    uploaded_file = st.file_uploader("Upload your CSV file with datetime and features", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        datetime_col_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_col_candidates:
            datetime_col = datetime_col_candidates[0]
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            timestamps = df[datetime_col].values
        else:
            st.error("No datetime column found. Please include a datetime column in the CSV.")
            return

        # Select numeric columns only (features)
        feature_df = df.select_dtypes(include=[np.number])
        values = feature_df.values
        # Standardize features
        values = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-8)

        model, device = load_model()
        scores, patches = predict_anomaly_scores(model, device, values)
        plot_results(values, scores, timestamps, step=4, window_size=24)

        # Display anomaly times and scores in text
        #display_anomalies_with_timestamps(df, scores, step=4, window_size=24)
        display_anomalies_with_timestamps(df, scores, step=4, window_size=24)

if __name__ == "__main__":
    main()