import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, matthews_corrcoef, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from xgboost import XGBClassifier

def split_data(df, train_size=0.7, valid_size=0.2, test_size=0.1, random_state=42):
    """
    Splits the DataFrame into train, validation, and test sets.
    """
    # First split off the train set
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state)
    # Then split the remaining data into validation and test
    valid_ratio = valid_size / (valid_size + test_size)
    valid_df, test_df = train_test_split(temp_df, train_size=valid_ratio, random_state=random_state)
    return train_df, valid_df, test_df



folders = ['train', 'valid', 'test']
base_path = 'output/data'  # Change this if you want a different base directory

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f'Created folder: {folder_path}')

df = pd.read_csv('output/data/oasis_plus.csv')
train_df, valid_df, test_df = split_data(df)

train_df.to_csv(os.path.join(base_path, 'train', 'oasis_train.csv'), index=False)
valid_df.to_csv(os.path.join(base_path, 'valid', 'oasis_valid.csv'), index=False)
test_df.to_csv(os.path.join(base_path, 'test', 'oasis_test.csv'), index=False)

# Helping functions

def impute_data(in_file):
    """
    Load and impute test data.
    :param in_file csv file including test data. The 10 clinical variables used for computing OASIS score should be named as shown in the oasis_variables_dict.
    :return: imputed data as a DataFrame object
    """
    oasis_variables_dict = {
     'oasis_age': 21.5,  # normal range is [18, 24]
     'oasis_heartrate': 60.5, # normal range is [33,88]
     'oasis_meanbp': 102.34,  # normal range is [61.33,143.44]
     'oasis_resprate': 17.5,  # normal range is [13,22]
     'oasis_temp': 36.64,  # normal range is [36.40, 36.88]
     'oasis_mechvent': 0,   # 'No' correspnds to zero subscore
     'oasis_electivesurgery': 1,  # 'Yes' corresponds to zero subscore
     'oasis_preiculos': 14.5    # normal range is [4.95, 24]
    }
    df = pd.read_csv(in_file)
    for key, value in oasis_variables_dict.items():
        if key in df.columns:
            df[key].fillna(value, inplace=True)
    return df

def get_labels(probs, cutoff=0.5):
    return (probs >= cutoff).astype(int)

def evaluate(Y_true, Y_pred, cutoff=0.5):
    """
    Given true and predicted probabilities, return Accuracy, Sensitivity, Specificity,  Matthew Correlation Coefficients, and AUC scores.
    :param Y_true:
    :param Y_pred:
    :param cutoff:
    :return: Accuracy, Sensitivity, Specificity,  Matthew Correlation Coefficients, and AUC scores.
    """
    Y_score = get_labels(Y_pred, cutoff)
    mcc = matthews_corrcoef(Y_true, Y_score)
    acc = accuracy_score(Y_true, Y_score)
    auc = roc_auc_score(Y_true, Y_pred)
    cm = confusion_matrix(Y_true, Y_score, labels=[1,0])
    #print(cm)
    tp = cm[0,0]
    fp = cm[1,0]
    tn = cm[1,1]
    fn = cm[0,1]
    ap = tp + fn
    an = tn + fp
    total = ap + an
    # compute Sn and Sp
    sn = tp/ap if ap > 0 else 0
    sp = tn/an if an > 0 else 0

    # return TP, FN, FP, TN, total, acc, Sn, Sp, MCC, AUC
    return  np.array([acc, sn, sp, mcc, auc])

# Parameters user-specific parameters

# test data
test_file = 'output/data/test/oasis_test.csv'  # Please insert the file name and path (e.g., ./data/meta_severity_clean_test.csv)
test_lbl = 'IN_HOSPITAL_MORTALITY'   # Please insert the name of the column including the true labels, otherwise use None
threshold = 0.10     # Please insert the threshold for binarizing OASIS+ scores (default is 0.10)

# file for saving OASIS+ predicted scores
out_file = 'output/data/oasis_preds.csv'   # Please insert the file name and path (e.g., ./data/oasis_preds.csv)

# load training data
output_folder = 'results'
os.makedirs(output_folder, exist_ok=True)

train_file = 'output/data/train/oasis_train.csv'
model_file = 'results/oasis_xgb200.joblib'
filter_file = 'results/oasis_filter.joblib'

os.makedirs(os.path.dirname(model_file), exist_ok=True)

train_df = pd.read_csv(train_file)
y_train = train_df['IN_HOSPITAL_MORTALITY'].values

# Initialize features, XGB model, and normalization filter
features = [
 'oasis_age',
 'oasis_heartrate',
 'oasis_meanbp',
 'oasis_resprate',
 'oasis_temp',
 'oasis_mechvent',
 'oasis_electivesurgery', 
 'oasis_preiculos'
]

classifier = XGBClassifier(objective="binary:logistic", n_estimators=200, n_jobs=4, random_state=123)
minMax_filter = MinMaxScaler()

# Train and output the learned model

X_train = train_df[features].values

minMax_filter.fit(X_train)
X_train = minMax_filter.transform(X_train)
classifier.fit(X_train, y_train)

print('\nDeploying the learned model')
dump(classifier, model_file) 
dump(minMax_filter, filter_file)
print('Done!')

oasis_variables = [
 'oasis_age',
 'oasis_heartrate',
 'oasis_meanbp',
 'oasis_resprate',
 'oasis_temp',
 'oasis_mechvent',
 'oasis_electivesurgery',
 'oasis_preiculos'
]
# load OASIS+ model

oasis_model = load(model_file)
oasis_filter = load(filter_file)


# load and impute test data
test_df = impute_data(test_file)
X_test = test_df[oasis_variables].values
if test_lbl is not None:
    y_test = test_df[test_lbl]

# normalize data
X_transformed = oasis_filter.transform(X_test)

# get predictions
preds = oasis_model.predict_proba(X_transformed)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, preds)
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]

print("Optimal threshold (max Sn+Sp-1):", best_threshold)

threshold = best_threshold
pred_lbls = np.zeros_like(preds, dtype=int)
pred_lbls[np.where(preds>=threshold)] = 1

# save predictions
out_df = pd.DataFrame(preds, columns=['OASIS+'])
out_df['in-hospital_mortality'] = pred_lbls
out_df.to_csv(out_file, index=False)
print(out_file + ' saved!')

# Report performance if labels of test data are provided

if test_lbl is not None:
    print ('ACC Sn Sp MCC AUC')
    print(evaluate(y_test, preds, cutoff=threshold))

df = pd.read_csv('output/data/oasis_plus.csv')
for col in ['oasis_age','oasis_heartrate','oasis_meanbp',
            'oasis_resprate','oasis_temp',
            'oasis_mechvent','oasis_electivesurgery','oasis_preiculos']:
    print(col, df[col].nunique(), "unique values")


