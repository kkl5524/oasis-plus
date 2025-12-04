import os
import pandas as pd
import hashlib
from datetime import datetime

# convert to mimic-iii format
input_folder = 'output/synthea_output'
output_folder = 'output/mimic'

os.makedirs(output_folder, exist_ok=True)

patients = pd.read_csv(
    os.path.join(input_folder, "patients.csv"),
    low_memory=False,
    on_bad_lines='skip'
)
patients.columns = patients.columns.str.lower()

encounters = pd.read_csv(
    os.path.join(input_folder, "encounters.csv"),
    low_memory=False,
    on_bad_lines='skip'
)
encounters.columns = encounters.columns.str.lower()

vitals = pd.read_csv(
    os.path.join(input_folder, "observations.csv"),
    low_memory=False,
    on_bad_lines='skip'
)
vitals.columns = vitals.columns.str.lower()

procedures = pd.read_csv(
    os.path.join(input_folder, "procedures.csv"),
    low_memory=False,
    on_bad_lines='skip'
)
procedures.columns = procedures.columns.str.lower()

# PATIENTS.csv table

patients['birthdate'] = pd.to_datetime(patients['birthdate'])
patients['age'] = (datetime.now() - patients['birthdate']).dt.days // 365
mimic_patients = pd.DataFrame({
    'subject_id': patients['id'],
    'gender': patients['gender'],
    'anchor_age': patients['age'],
    'anchor_year': pd.to_datetime(patients['birthdate']).dt.year,
    'anchor_month': pd.to_datetime(patients['birthdate']).dt.month,
    'dod': patients['deathdate'],
})

mimic_patients.to_csv(os.path.join(output_folder, "PATIENTS.csv"), index=False)

# ADMISSIONS.csv table

encounters = encounters[encounters['patient'].isin(patients['id'])]
mimic_adm = pd.DataFrame({
    'hadm_id': encounters['id'],
    'subject_id': encounters['patient'],
    'admittime': encounters['start'],
    'dischtime': encounters['stop'],
    'admission_type': encounters['encounterclass'],
    'diagnosis': encounters.get('reasondescription', None)
})

patients['death_date'] = pd.to_datetime(patients['deathdate'])
encounters['admittime'] = pd.to_datetime(encounters['start'])
encounters['dischtime'] = pd.to_datetime(encounters['stop'])

enc = encounters.merge(
    patients[['id', 'deathdate']],
    left_on='id', right_on='id', how='left'
)

enc['hospital_expire_flag'] = enc.apply(
    lambda row: 1 if pd.notna(row['deathdate']) and row['admittime'] <= row['deathdate'] <= row['dischtime'] else 0,
    axis=1
)

mimic_adm.to_csv(os.path.join(output_folder, "ADMISSIONS.csv"), index=False)

# ICUSTAYS.csv table

icu_enc = encounters[encounters['encounterclass'] == 'inpatient'].copy()

icu_descriptions = [
    'Patient transfer to intensive care unit (procedure)',
    'Admission to intensive care unit (procedure)'
]

icu_enc = icu_enc[icu_enc['description'].isin(icu_descriptions)].copy()

mimic_icu = pd.DataFrame({
    'subject_id': icu_enc['patient'],
    'hadm_id': icu_enc['id'],
    'stay_id': range(1, len(icu_enc)+1),
    'intime': icu_enc['start'],
    'outtime': icu_enc['stop'],
    'los': (pd.to_datetime(icu_enc['stop']) - pd.to_datetime(icu_enc['start'])).dt.total_seconds()/86400
})
mimic_icu.to_csv(os.path.join(output_folder, "ICUSTAYS.csv"), index=False)

# CHARTEVENTS.csv table

vitals = vitals[vitals['patient'].isin(patients['id'])]  # adult filter
mimic_chart = pd.DataFrame({
    'subject_id': vitals['patient'],
    'hadm_id': vitals.get('encounter', None),
    'icustay_id': None,
    'itemid': vitals['code'],
    'charttime': vitals['date'],
    'value': vitals['value'],
    'valuenum': pd.to_numeric(vitals['value'], errors='coerce'),
    'valueuom': vitals.get('units', None),
    'warning': None,
    'error': None
})
mimic_chart.to_csv(os.path.join(output_folder, "CHARTEVENTS.csv"), index=False)

# LABEVENTS.csv table

labs = vitals[vitals['category'] == 'laboratory'].copy()
mimic_lab = pd.DataFrame({
    'subject_id': labs['patient'],
    'hadm_id': labs.get('encounter', None),
    'itemid': labs['code'],
    'charttime': labs['date'],
    'value': labs['value'],
    'valuenum': pd.to_numeric(labs['value'], errors='coerce'),
    'valueuom': labs.get('units', None),
    'flag': None
})
mimic_lab.to_csv(os.path.join(output_folder, "LABEVENTS.csv"), index=False)

# PROCEDUREEVENTS_MV.csv table

procedures = procedures[procedures['patient'].isin(patients['id'])]
mimic_proc = pd.DataFrame({
    'subject_id': procedures['patient'],
    'hadm_id': procedures.get('encounter', None),
    'starttime': procedures['start'],
    'endtime': procedures['stop'],
    'itemid': procedures['code'],
    'value': procedures.get('description', None),
    'valueuom': None,
})
mimic_proc.to_csv(os.path.join(output_folder, "PROCEDUREEVENTS_MV.csv"), index=False)

def map_category(code):
    label = str(code).lower()  # fallback to string search
    if any(x in label for x in ['vital-signs']):
        return 'Vital Signs'
    if any(x in label for x in ['laboratory']):
        return 'Laboratory'
    return "Other"

def map_fluid(code):
  label = str(code).lower()
  if any(x in label for x in ['blood', 'plasma']):
    return 'Blood'
  elif any(x in label for x in ['urine']):
    return 'Urine'
  elif any(x in label for x in ['csf', 'cerebrospinal']):
    return 'CSF'
  else:
    return None

items = vitals[['category', 'code', 'description', 'units']].drop_duplicates()

items['itemid'] = items['code']
items['label'] = items['description']
items['category'] = items['category'].apply(map_category)
items['fluid'] = items['description'].apply(map_fluid)
items['unitname'] = items['units']
items['param_type'] = "Numeric"

mimic_items = items[['itemid','label','fluid','category','unitname','param_type']]
mimic_items.to_csv(os.path.join(output_folder, "D_ITEMS.csv"), index=False)

import zipfile

zip_path = "synthea_mimic_output.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk("synthea/output/mimic"):
        for file in files:
            full_path = os.path.join(root, file)
            zipf.write(full_path)

print(f"All MIMIC-formatted files have been written to {output_folder} and zipped to {zip_path}.")
