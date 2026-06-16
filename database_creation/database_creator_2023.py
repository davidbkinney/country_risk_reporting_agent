"""
database_creator_2023.py

Clean the 2023 world risk poll data and save it as a
.json file for retrieval.
"""

import pyreadstat
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

# -------------------------------------------------
# Classify fields
# -------------------------------------------------

SKIP_FIELDS_2023 = [
    'Country (by alpha)',
    'Country',
    'Country ISO alpha-2 code',
    'Country ISO alpha-3 code',
    'Year of Interview'
    'Study Completion Date',
    'Employed Full Time for an Employer Index (workforce)',
    'Labor Force Participation Index',
    'Wave Year',
    'WGT',
    'Urban/rural -- recoded into 2 groups',
    'Global Region',
    'Region 2 Global',
    'Sampling Stratification',
    'Sampling Stratification 2',
    'Sampling Stage 1 (PSU)',
    'Random Unique Case ID',
    'Weight',
    'Projection weight to be used for multicountry analysis for 2021 data ONLY',
    'Projection weight for time-series or single year analysis',
    'Worried Mean'
]

METADATA_FIELDS_2023 = [
    'Country Name',
    'Gender',
    'Education Level',
    'Age cohort (4 group)',
    'Age',
    'Age - 5 categories',
    'Age - 3 categories',
    'Marital Status',
    'Per Capita Income Quintiles',
    'Primary Occupation/Job Title (Coded Response)',
    'Global region (as used in World Risk Poll analytical report)',
    'World Bank Country Income Classifications (2019-2020 definition)',
    'Urban/Rural',
    'Employment Status'
]

QUANT_FIELDS_2023 = [
    'Worried Index',
    'Experienced Index',
    'Worry and experience gap',
    'Resilience Index',
    'Resilience: Individual Dimension',
    'Resilience: Household Dimension',
    'Resilience: Community Dimension',
    'Resilience: Society Dimension'
]

# -------------------------------------------------
# Read in datasets.
# -------------------------------------------------

path = 'data/_2023/23_wrp.sav' # Your directory may differ!

df, meta = pyreadstat.read_sav(
    path,
    apply_value_formats=True
)

path2 = 'data/trended/trended_wrp.sav' # Your directory may differ!

df2, meta2 = pyreadstat.read_sav(
    path2,
    apply_value_formats=True
)

# -------------------------------------------------
# Clean data and save as JSON.
# -------------------------------------------------

data_2023 = []

for index, row in tqdm(df.iterrows(), total=len(df)):

    person_dict = {
        'metadata': {
            'Survey Year': '2023',
            'ID': row['WPID_RANDOM'],
            'demographics': {},
            'quantitative_data': {}
        },
        'qualitative_data': {}
    }

    seen = set()  # (label, value) pairs

    # =====================================================
    # Pull from 23_wrp.sav
    # =====================================================

    json_data = json.loads(row.to_json())

    for key, value in json_data.items():

        if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
            continue

        label = meta.column_names_to_labels.get(key)

        if label is None or label in SKIP_FIELDS_2023:
            continue

        pair = (label, str(value))

        if pair in seen:
            continue
        seen.add(pair)

        if label in METADATA_FIELDS_2023 or 'Region' in str(label):
            person_dict['metadata']['demographics'][label] = value

        elif label in QUANT_FIELDS_2023:
            person_dict['metadata']['quantitative_data'][label] = value

        else:
            person_dict['qualitative_data'][label] = value


    # =====================================================
    # Pull additional fields from trended_wrp.sav
    # =====================================================

    trended_data = df2[df2['WPID_RANDOM'] == row['WPID_RANDOM']]

    for _, row2 in trended_data.iterrows():

        json_data2 = json.loads(row2.to_json())

        for key, value in json_data2.items():

            if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
                continue

            label = meta2.column_names_to_labels.get(key)

            if label is None or label in SKIP_FIELDS_2023:
                continue

            pair = (label, str(value))

            if pair in seen:
                continue
            seen.add(pair)

            if label in METADATA_FIELDS_2023 or 'Region' in str(label):
                person_dict['metadata']['demographics'][label] = value

            elif label in QUANT_FIELDS_2023:
                person_dict['metadata']['quantitative_data'][label] = value

            else:
                person_dict['qualitative_data'][label] = value

    data_2023.append(person_dict)

# Calculate and add Worry and Experience gap values.
for d in tqdm(data_2023):
    worry = d['metadata']['quantitative_data'].get('Worried Index')
    experienced = d['metadata']['quantitative_data'].get('Experienced Index')

    if pd.notna(worry) and pd.notna(experienced):
        d['metadata']['quantitative_data']['Worry and experience gap'] = (
            worry - experienced
        )
    else:
        d['metadata']['quantitative_data']['Worry and experience gap'] = None

with open('data/_2023/data_2023.json', 'w') as f:
    json.dump(data_2023, f, indent=4)
