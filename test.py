import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix



# -------------------------------
# Load data
# -------------------------------
patients = pd.read_csv("mimic_demo/PATIENTS.csv")
admissions = pd.read_csv("mimic_demo/ADMISSIONS.csv")
icustays = pd.read_csv("mimic_demo/ICUSTAYS.csv")
chartevents = pd.read_csv("mimic_demo/CHARTEVENTS.csv")
d_items = pd.read_csv("mimic_demo/D_ITEMS.csv")

# -------------------------------
# Preprocess dates and ICU-admission merges
# -------------------------------
admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
patients['dob'] = pd.to_datetime(patients['dob'], errors='coerce')
icustays['intime'] = pd.to_datetime(icustays['intime'], errors='coerce')

df = icustays.merge(admissions, on=['subject_id', 'hadm_id'], how='left')
df = df.merge(patients[['subject_id','dob']], on='subject_id', how='left')

# Age at ICU admission
age = df['intime'].dt.year - df['dob'].dt.year
before_bday = ((df['intime'].dt.month < df['dob'].dt.month) |
               ((df['intime'].dt.month == df['dob'].dt.month) &
                (df['intime'].dt.day < df['dob'].dt.day)))
df['age'] = (age - before_bday.astype(int)).clip(0, 120)

# Pre-ICU length of stay (days)
df['pre_icu_los'] = ((df['intime'] - df['admittime']).dt.total_seconds() / (24*3600)).clip(lower=0)

# -------------------------------
# Structured features list
# -------------------------------
structured_vars = ['age','pre_icu_los']

# -------------------------------
# Surgery flag
# -------------------------------
surgery_items = d_items[d_items['label'].str.contains('surgery', case=False, na=False) &
                        ~d_items['label'].str.contains('consult', case=False, na=False)]
surgery_ids = surgery_items['itemid'].tolist()

chartevents['value'] = pd.to_numeric(chartevents['value'], errors='coerce')
surgery_events = chartevents[chartevents['itemid'].isin(surgery_ids)]
surgery_flag = surgery_events.groupby('icustay_id')['value'].max().reset_index()
surgery_flag.rename(columns={'value':'surgery_flag'}, inplace=True)

df = df.merge(surgery_flag, left_on='icustay_id', right_on='icustay_id', how='left')
df['surgery_flag'] = df['surgery_flag'].fillna(0)
structured_vars.append('surgery_flag')

# -------------------------------
# Vital signs & GCS
# -------------------------------
vital_labels = {
    'heart_rate':'heart rate',
    'systolic_bp':'systolic blood pressure',
    'diastolic_bp':'diastolic blood pressure',
    'mean_bp':'mean arterial pressure',
    'resp_rate':'respiratory rate',
    'temp':'temperature',
    'gcs_total':'gcs total',
    'urine_out':'urine out',
    'ventilation':'ventilation'
}

for feat, label in vital_labels.items():
    items = d_items[d_items['label'].str.contains(label, case=False, na=False)]
    itemids = items['itemid'].tolist()
    if len(itemids)==0:
        continue
    events = chartevents[chartevents['itemid'].isin(itemids)]
    feat_df = events.groupby('icustay_id')['value'].mean().reset_index()
    feat_df.rename(columns={'value':feat}, inplace=True)
    df = df.merge(feat_df, left_on='icustay_id', right_on='icustay_id', how='left')
    df[feat] = df[feat].fillna(0)
    structured_vars.append(feat)

# -------------------------------
# Mortality label
# -------------------------------
mortality_var = 'hospital_expire_flag'

# -------------------------------
# Preprocess structured features
# -------------------------------
imputer = SimpleImputer(strategy='median')
df[structured_vars] = imputer.fit_transform(df[structured_vars])

scaler = StandardScaler()
df[structured_vars] = scaler.fit_transform(df[structured_vars])

# -------------------------------
# Temporal features (all CHARTEVENTS)
# -------------------------------
temporal_vars = chartevents['itemid'].unique().tolist()
df_temp = chartevents.pivot_table(index='icustay_id', columns='itemid', values='value', aggfunc='mean')
df_temp.columns = df_temp.columns.astype(str)
temporal_vars = df_temp.columns.tolist()
df = df.merge(df_temp, left_on='icustay_id', right_index=True, how='left')
df[temporal_vars] = df[temporal_vars].fillna(0)

# -------------------------------
# Convert to torch tensors
# -------------------------------
X_struct = torch.tensor(df[structured_vars].values, dtype=torch.float32)
X_temp = torch.tensor(df[temporal_vars].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(df[mortality_var].values, dtype=torch.float32).unsqueeze(1)

# -------------------------------
# Temporal Encoder (GRU)
# -------------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]

ts_encoder = TemporalEncoder(input_dim=len(temporal_vars))
temporal_emb = ts_encoder(X_temp)

# -------------------------------
# Fusion layer
# -------------------------------
class FusionLayer(nn.Module):
    def __init__(self, structured_dim, temporal_dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(structured_dim + temporal_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    def forward(self, struct_emb, temporal_emb):
        x = torch.cat([struct_emb, temporal_emb], dim=1)
        return self.fuse(x)

fusion = FusionLayer(structured_dim=X_struct.shape[1], temporal_dim=temporal_emb.shape[1])
fused_emb = fusion(X_struct, temporal_emb)

# -------------------------------
# Mortality classifier
# -------------------------------
class MortalityClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = MortalityClassifier(input_dim=fused_emb.shape[1])

# -------------------------------
# Train model
# -------------------------------
def train_model(model, ts_encoder, fusion, X_struct, X_temp, y, epochs=100, lr=1e-3):
    model.train()
    opt = torch.optim.Adam(list(model.parameters()) + 
                           list(ts_encoder.parameters()) + 
                           list(fusion.parameters()), lr=lr)
    crit = nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        temporal_emb = ts_encoder(X_temp)
        fused_emb = fusion(X_struct, temporal_emb)
        pred = model(fused_emb)
        loss = crit(pred, y)
        loss.backward()
        opt.step()
    return model

model = train_model(model, ts_encoder, fusion, X_struct, X_temp, y)

# -------------------------------
# AUROC evaluation
# -------------------------------
model.eval()
with torch.no_grad():
    temporal_emb = ts_encoder(X_temp)
    fused_emb = fusion(X_struct, temporal_emb)
    preds = model(fused_emb).squeeze().numpy()
    auroc = roc_auc_score(y.numpy(), preds)

print(f"AUROC: {auroc:.4f}")

# -------------------------------
# SHAP evaluation
# -------------------------------
fused_emb_np = fused_emb.detach().numpy()
background = fused_emb_np[np.random.choice(fused_emb_np.shape[0], 100, replace=False)]
explainer = shap.DeepExplainer(model, torch.tensor(background, dtype=torch.float32))
shap_values = explainer.shap_values(torch.tensor(fused_emb_np, dtype=torch.float32))
shap.summary_plot(shap_values, fused_emb_np, feature_names=[f'feat_{i}' for i in range(fused_emb_np.shape[1])])

# -------------------------------
# ROC Curve and other metrics
# -------------------------------
y_true = y.numpy().squeeze()
y_scores = preds

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Compute AUC
roc_auc = auc(fpr, tpr)

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Mortality Prediction')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Threshold predicted probabilities at 0.5
y_pred = (preds >= 0.5).astype(int)
y_true = y.numpy().squeeze().astype(int)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Metrics
accuracy = accuracy_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred)  # same as TPR
specificity = tn / (tn + fp)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall, TPR): {sensitivity:.4f}")
print(f"Specificity (TNR): {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")
