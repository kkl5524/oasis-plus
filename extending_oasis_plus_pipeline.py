import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import shap
from transformers import AutoModel, AutoTokenizer

# Data Loading & Preprocessing
def load_data(path):
    return pd.read_csv(path)

def preprocess(df, structured_vars, temporal_vars):
    # Drop rows with >40% missing
    df = df[df.isnull().mean(axis=1) <= 0.4]

    # Structured variables: median imputation + standardization
    imputer = SimpleImputer(strategy="median")
    df[structured_vars] = imputer.fit_transform(df[structured_vars])
    scaler = StandardScaler()
    df[structured_vars] = scaler.fit_transform(df[structured_vars])

    # Temporal variables: forward fill
    df[temporal_vars] = df[temporal_vars].fillna(method="ffill")

    return df

# Temporal Encoder (GRU)
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, ts_batch):
        _, h = self.gru(ts_batch)
        return h[-1]

# Fusion Layer (Late Fusion)
class FusionLayer(nn.Module):
    def __init__(self, dims_dict):
        """
        dims_dict example:
            {"structured_dim": 16, "temporal_dim": 32, "text_dim": 768}
        """
        super().__init__()
        total = sum(dims_dict.values())
        self.fuse = nn.Sequential(
            nn.Linear(total, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_dim = 64

    def forward(self, struct_emb, temporal_emb):
        x = torch.cat([struct_emb, temporal_emb], dim=1)
        return self.fuse(x)

# Autoencoder for Latent Features
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def train_autoencoder(X_struct, latent_dim=8, epochs=50):
    model = Autoencoder(input_dim=X_struct.shape[1], latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_struct), batch_size=32, shuffle=True)

    for _ in range(epochs):
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        _, latents = model(X_struct)

    return latents

# Clustering Latent Features
def run_clustering(latent_embeddings, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(latent_embeddings)
    silhouette = silhouette_score(latent_embeddings, labels)
    ch = calinski_harabasz_score(latent_embeddings, labels)
    return labels, silhouette, ch

# Training Data Builder
def build_training_data(latents, df, binary_vars, static_vars, clusters, strategy):
    base = np.hstack([latents, df[binary_vars + static_vars].values])

    if strategy == "cluster_feature":
        X = np.hstack([base, clusters.reshape(-1, 1)])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(df["IN_HOSPITAL_MORTALITY"].values, dtype=torch.float32).unsqueeze(1)
    elif strategy == "per_cluster":
        data = {}
        y = df["IN_HOSPITAL_MORTALITY"].values
        for c in np.unique(clusters):
            idx = clusters == c
            X_c = base[idx]
            y_c = y[idx]
            data[c] = (torch.tensor(X_c, dtype=torch.float32), torch.tensor(y_c, dtype=torch.float32).unsqueeze(1))
        return None, data
    else:
        raise ValueError("Invalid strategy")

# Mortality Classifier
class MortalityClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_model(X, y, epochs=50, lr=1e-4):
    model = MortalityClassifier(X.shape[1])
    crit = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = crit(pred, y)
        loss.backward()
        opt.step()
    return model

# SHAP Explanation
def run_shap(model, X, feature_names=None, multimodal=False):
    if multimodal:
        X_concat = torch.cat([X["structured"], X["temporal"], X["text"]], dim=1)
        explainer = shap.DeepExplainer(model, X_concat)
        shap_vals = explainer.shap_values(X_concat)
        shap.summary_plot(shap_vals, X_concat.numpy(), feature_names)
        return

    explainer = shap.DeepExplainer(model, X)
    shap_vals = explainer.shap_values(X)
    shap.summary_plot(shap_vals, X.numpy(), feature_names)

if __name__ == "__main__":
    structured_vars = ['oasis_age','oasis_heartrate','oasis_meanbp','oasis_resprate','oasis_temp']
    temporal_vars = ['oasis_heartrate','oasis_meanbp','oasis_resprate','oasis_temp']
    binary_vars = ['oasis_electivesurgery','oasis_mechvent']
    static_vars = ['oasis_age','oasis_preiculos']
    strategy = "cluster_feature"

    # Load and preprocess
    df = load_data("output/data/test/oasis_test.csv")
    df = preprocess(df, structured_vars, temporal_vars)

    # Structured latent embeddings
    X_struct = torch.tensor(df[structured_vars].values, dtype=torch.float32)
    latents = train_autoencoder(X_struct)

    # Temporal embeddings
    ts_encoder = TemporalEncoder(input_dim=len(temporal_vars))
    ts_input = torch.tensor(df[temporal_vars].values.reshape(df.shape[0], 1, len(temporal_vars)), dtype=torch.float32)
    temporal_emb = ts_encoder(ts_input)

    # Fusion
    dims_dict = {
        "structured_dim": latents.shape[1],
        "temporal_dim": temporal_emb.shape[1]
    }
    fusion = FusionLayer(dims_dict)
    fused_emb = fusion(latents, temporal_emb)

    # Detach fused embeddings so we don't backprop through autoencoder/GRU
    fused_emb_detached = fused_emb.detach()

    # Train mortality classifier
    y = torch.tensor(df["IN_HOSPITAL_MORTALITY"].values, dtype=torch.float32).unsqueeze(1)
    model = train_model(fused_emb_detached, y)

    # Save model
    torch.save(model.state_dict(), "mortality_model.pt")

    # Load model example
    model_loaded = MortalityClassifier(fused_emb.shape[1])
    model_loaded.load_state_dict(torch.load("mortality_model.pt"))
    model_loaded.eval()

    # Make sure your classifier is in evaluation mode
    model.eval()

    # Predictions
    with torch.no_grad():
        y_pred = model(fused_emb).squeeze()  # fused_emb as input

    # True labels
    y_true = torch.tensor(df["IN_HOSPITAL_MORTALITY"].values, dtype=torch.float32)

    # Compute AUROC
    auroc = roc_auc_score(y_true.numpy(), y_pred.numpy())
    print(f"AUROC (after fusion): {auroc:.4f}")
