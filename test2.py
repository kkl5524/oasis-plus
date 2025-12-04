import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import torch
import torch.nn as nn

# -------------------------------
# Load MIMIC demo data and items
# -------------------------------
patients = pd.read_csv("mimic_demo/PATIENTS.csv")
admissions = pd.read_csv("mimic_demo/ADMISSIONS.csv")
icustays = pd.read_csv("mimic_demo/ICUSTAYS.csv")
chartevents = pd.read_csv("mimic_demo/CHARTEVENTS.csv")
d_items = pd.read_csv("mimic_demo/D_ITEMS.csv") 

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# -------------------------------
# Compute age at ICU admission
# -------------------------------
admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
patients['dob'] = pd.to_datetime(patients['dob'], errors='coerce')

df = icustays.merge(admissions, on=['subject_id', 'hadm_id'], how='left')
df = df.merge(patients[['subject_id', 'dob']], on='subject_id', how='left')

age = df['admittime'].dt.year - df['dob'].dt.year
before_bday = ((df['admittime'].dt.month < df['dob'].dt.month) |
               ((df['admittime'].dt.month == df['dob'].dt.month) &
                (df['admittime'].dt.day < df['dob'].dt.day)))
df['age'] = (age - before_bday.astype(int)).clip(0, 120)

# -------------------------------
# Add admission reason cluster
# -------------------------------
admissions['diagnosis'] = admissions['diagnosis'].fillna('unknown').str.lower()
vectorizer = TfidfVectorizer(max_features=500)
X_text = vectorizer.fit_transform(admissions['diagnosis'])

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
admissions['admission_cluster'] = kmeans.fit_predict(X_text)

df = df.merge(admissions[['hadm_id', 'admission_cluster']], on='hadm_id', how='left')

# -------------------------------
# Structured variables
# -------------------------------

# Surgery (exclude consults)
surgery_ids = d_items[
    d_items['label'].str.contains('surgery', case=False, na=False) &
    ~d_items['label'].str.contains('consult', case=False, na=False)
]['itemid'].tolist()

df['surgery'] = chartevents[
    chartevents['itemid'].isin(surgery_ids)
].groupby('icustay_id')['value'].count().reindex(df['icustay_id'], fill_value=0)
df['surgery'] = (df['surgery'] > 0).astype(int)

structured_vars = ['age', 'admission_cluster'] + ['surgery']
mortality_var = 'hospital_expire_flag'

# -------------------------------
# Temporal vars
# -------------------------------
chartevents['value'] = pd.to_numeric(chartevents['value'], errors='coerce')
temporal_itemids = d_items[d_items['label'].str.contains(
    'heart rate|bp|blood pressure|arterial pressure|respiratory rate|ventilation|temperature|gcs total|urine out',
    case=False, na=False
)]['itemid'].tolist()

df_temp = chartevents[chartevents['itemid'].isin(temporal_itemids)]
df_temp = df_temp.pivot_table(index='icustay_id', columns='itemid', values='value', aggfunc='mean')
df_temp = df_temp.fillna(0)
temporal_vars = df_temp.columns.tolist()

df = df.merge(df_temp, left_on='icustay_id', right_index=True, how='left')
df[temporal_vars] = df[temporal_vars].fillna(0)

# -------------------------------
# Torch tensors
# -------------------------------
X_struct = torch.tensor(df[structured_vars].values, dtype=torch.float32)
X_temp = torch.tensor(df[temporal_vars].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(df[mortality_var].values, dtype=torch.float32).unsqueeze(1)

# -------------------------------
# Models
# -------------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]

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

# -------------------------------
# Train with class weighting
# -------------------------------
ts_encoder = TemporalEncoder(input_dim=len(temporal_vars))
fusion = FusionLayer(structured_dim=X_struct.shape[1], temporal_dim=16)
model = MortalityClassifier(input_dim=32)

pos_weight = torch.tensor(2.0)  # Adjust based on class imbalance
crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

opt = torch.optim.Adam(list(model.parameters()) + list(ts_encoder.parameters()) + list(fusion.parameters()), lr=1e-3)
epochs = 50

for _ in range(epochs):
    opt.zero_grad()
    temporal_emb = ts_encoder(X_temp)
    fused_emb = fusion(X_struct, temporal_emb)
    pred = model(fused_emb)
    loss = crit(pred, y)
    loss.backward()
    opt.step()

# -------------------------------
# Predictions & metrics
# -------------------------------
model.eval()
with torch.no_grad():
    temporal_emb = ts_encoder(X_temp)
    fused_emb = fusion(X_struct, temporal_emb)
    pred_probs = model(fused_emb).squeeze().numpy()

# Optimal threshold
fpr, tpr, thresholds = roc_curve(y.numpy(), pred_probs)
gmeans = np.sqrt(tpr * (1-fpr))
optimal_idx = np.argmax(gmeans)
optimal_threshold = thresholds[optimal_idx]
preds = (pred_probs >= optimal_threshold).astype(int)

# Metrics
accuracy = accuracy_score(y.numpy(), preds)
sensitivity = recall_score(y.numpy(), preds)
tn, fp, fn, tp = confusion_matrix(y.numpy(), preds).ravel()
specificity = tn / (tn + fp)
precision_val = precision_score(y.numpy(), preds)
f1 = f1_score(y.numpy(), preds)
auroc = roc_auc_score(y.numpy(), pred_probs)

print(f"AUROC: {auroc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision_val:.4f}")
print(f"F1-score: {f1:.4f}")
