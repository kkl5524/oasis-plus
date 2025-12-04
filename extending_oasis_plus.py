df = load_data("DATA.csv")

df = preprocess(
    df,
    structured_vars=['age','GCS','HR','BP','RR','temperature','urine_output'],
    temporal_vars=['HR','BP','RR','temperature','SpO2','urine_output']
)

structured_vars = ['age','GCS','HR','BP','RR','temperature','urine_output']
X_struct = torch.tensor(df[structured_vars].values, dtype=torch.float32)

latents = train_autoencoder(X_struct)  # output shape: (num_samples, latent_dim)

import torch

structured_vars = ['age','GCS','HR','BP','RR','temperature','urine_output']
X_struct = torch.tensor(df[structured_vars].values, dtype=torch.float32)

latents = train_autoencoder(X_struct)  # output shape: (num_samples, latent_dim)

clusters, sil_score, ch_score = run_clustering(latents.numpy(), k=3)
print("Silhouette:", sil_score, "Calinski-Harabasz:", ch_score)

strategy = "cluster_feature"  # or "per_cluster"
binary_vars = ['elective_surgery','ventilation']
static_vars = ['age','pre_ICU_LoS']

X, per_cluster_data = build_training_data(
    latents.numpy(),
    df,
    binary_vars=binary_vars,
    static_vars=static_vars,
    clusters=clusters,
    strategy=strategy
)

y = torch.tensor(df["mortality"].values, dtype=torch.float32).unsqueeze(1)
model = train_model(X, y)

models = {}
for c, (X_c, y_c) in per_cluster_data.items():
    models[c] = train_model(X_c, y_c)

preds = model(X)  # output: tensor of probabilities
pred_labels = (preds >= 0.5).int()  # threshold = 0.5 by default

preds = torch.zeros(len(df))
for c, (X_c, _) in per_cluster_data.items():
    idx = clusters == c
    preds[idx] = models[c](X_c).squeeze()
pred_labels = (preds >= 0.5).int()

feature_names = structured_vars + binary_vars + static_vars + ['cluster_id']
run_shap(model, X, feature_names=feature_names)
