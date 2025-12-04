import os
import pandas as pd
import numpy as np

mimic_folder_path = 'output/mimic'

ADMISSIONS = pd.read_csv(os.path.join(mimic_folder_path, 'ADMISSIONS.csv'))
ICUSTAYS = pd.read_csv(os.path.join(mimic_folder_path, 'ICUSTAYS.csv'))
PATIENTS = pd.read_csv(os.path.join(mimic_folder_path, 'PATIENTS.csv'))
PROCEDUREEVENTS_MV = pd.read_csv(os.path.join(mimic_folder_path, 'PROCEDUREEVENTS_MV.csv'))
LABEVENTS = pd.read_csv(os.path.join(mimic_folder_path, 'LABEVENTS.csv'))
CHARTEVENTS = pd.read_csv(os.path.join(mimic_folder_path, 'CHARTEVENTS.csv'))
D_ITEMS = pd.read_csv(os.path.join(mimic_folder_path, 'D_ITEMS.csv'))

ADMISSIONS.columns = ADMISSIONS.columns.str.upper()
ICUSTAYS.columns = ICUSTAYS.columns.str.upper()
PATIENTS.columns = PATIENTS.columns.str.upper()
PROCEDUREEVENTS_MV.columns = PROCEDUREEVENTS_MV.columns.str.upper()
LABEVENTS.columns = LABEVENTS.columns.str.upper()
CHARTEVENTS.columns = CHARTEVENTS.columns.str.upper()
D_ITEMS.columns = D_ITEMS.columns.str.upper()

PROCEDUREEVENTS_MV.rename(columns={'VALUE': 'ORDERCATEGORYNAME'}, inplace=True)
PROCEDUREEVENTS_MV['ORDERCATEGORYNAME'] = PROCEDUREEVENTS_MV['ORDERCATEGORYNAME'].astype(str)
ICUSTAYS.rename(columns={'STAY_ID': 'ICUSTAY_ID'}, inplace=True)

# Compute DOB
PATIENTS['DOB'] = pd.to_datetime({
    'year': PATIENTS['ANCHOR_YEAR'] - PATIENTS['ANCHOR_AGE'],
    'month': PATIENTS['ANCHOR_MONTH'],
    'day': 15  # Use middle of the month as approximation
})

# Ensure datetime columns are in proper format
ADMISSIONS['ADMITTIME'] = pd.to_datetime(ADMISSIONS['ADMITTIME']).dt.tz_localize(None)
ADMISSIONS['DISCHTIME'] = pd.to_datetime(ADMISSIONS['DISCHTIME']).dt.tz_localize(None)
ICUSTAYS['INTIME'] = pd.to_datetime(ICUSTAYS['INTIME']).dt.tz_localize(None)

PATIENTS['DOD'] = pd.to_datetime(PATIENTS.get('DOD', None)).dt.tz_localize(None)

# Create hospital expire flag: 1 if deathdate is within admission period, else 0
ADMISSIONS = ADMISSIONS.merge(PATIENTS[['SUBJECT_ID','DOD']], on='SUBJECT_ID', how='left')
ADMISSIONS['HOSPITAL_EXPIRE_FLAG'] = ((ADMISSIONS['DOD'].notna()) &
                                      (ADMISSIONS['ADMITTIME'] <= ADMISSIONS['DOD']) &
                                      (ADMISSIONS['DISCHTIME'] >= ADMISSIONS['DOD'])).astype(int)

# Merge patient & ICU stay info
df = ICUSTAYS.merge(ADMISSIONS[['HADM_ID','ADMITTIME', 'HOSPITAL_EXPIRE_FLAG']], on='HADM_ID', how='left')
df.rename(columns = {'HOSPITAL_EXPIRE_FLAG': 'IN_HOSPITAL_MORTALITY'}, inplace = True)
df = df.merge(PATIENTS[['SUBJECT_ID','DOB']], on='SUBJECT_ID', how='left')

# Compute age at admission
admit = pd.to_datetime(df['ADMITTIME'], errors='coerce')
dob = pd.to_datetime(df['DOB'], errors='coerce')
age = admit.dt.year - dob.dt.year
before_bday = (
    (admit.dt.month < dob.dt.month) |
    ((admit.dt.month == dob.dt.month) &
     (admit.dt.day < dob.dt.day)))
df['oasis_age'] = (age - before_bday.astype(int)).clip(lower=0, upper=120)

# Define item IDs corresponding to OASIS variables
oasis_itemids = {
    'oasis_heartrate': ['8867-4'],
    'oasis_resprate': ['9279-1'],
    'oasis_temp': ['8310-5'],
    'oasis_sbp': ['8480-6'],
    'oasis_dbp': ['8462-4']
}

# Make sure ITEMID is string
CHARTEVENTS['ITEMID'] = CHARTEVENTS['ITEMID'].astype(str)

def get_first_day(chartevents, itemids):
    """Returns first measurement per hospital admission (HADM_ID)."""
    tmp = chartevents[chartevents['ITEMID'].isin(itemids)].copy()
    tmp['CHARTTIME'] = pd.to_datetime(tmp['CHARTTIME'])

    first = (
        tmp.sort_values(['HADM_ID','CHARTTIME'])
           .groupby('HADM_ID')['VALUENUM']
           .first()
           .reset_index()
    )
    return first

# Extract each variable
for var, ids in oasis_itemids.items():
    print(f"Extracting {var} using ITEMIDs {ids}")
    extracted = get_first_day(CHARTEVENTS, ids)
    extracted.rename(columns={'VALUENUM': var}, inplace=True)
    df = df.merge(extracted, on='HADM_ID', how='left')

# Compute MAP
df['oasis_meanbp'] = (df['oasis_sbp'] + 2*df['oasis_dbp']) / 3

# Pre-ICU LOS
df['oasis_preiculos'] = (df['INTIME'] - df['ADMITTIME']).dt.total_seconds()/(24*3600)

# Ventilation
vent_ids = PROCEDUREEVENTS_MV[PROCEDUREEVENTS_MV['ORDERCATEGORYNAME'].str.contains('ventilation', case=False, na=False)]['HADM_ID'].unique()
df['oasis_mechvent'] = df['HADM_ID'].apply(lambda x: 1 if x in vent_ids else 0)

# elective surgery
PROCEDUREEVENTS_MV['oasis_electivesurgery'] = PROCEDUREEVENTS_MV['ORDERCATEGORYNAME'].str.contains('planned', case=False, na=False)
df['oasis_electivesurgery'] = df['HADM_ID'].apply(lambda x: 1 if x in PROCEDUREEVENTS_MV.loc[PROCEDUREEVENTS_MV['oasis_electivesurgery'], 'HADM_ID'].unique() else 0)

# Save output
os.makedirs('output/data', exist_ok=True)
df.to_csv('output/data/oasis_plus.csv', index=False)

print("\nDONE. Extracted variables")

for col in ['oasis_heartrate', 'oasis_resprate', 'oasis_temp',
            'oasis_sbp', 'oasis_dbp', 'oasis_meanbp']:
    print(col, df[col].nunique())

