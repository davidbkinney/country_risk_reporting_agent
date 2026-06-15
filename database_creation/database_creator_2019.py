"""
database_creator_2019.py

Clean the 2019 world risk poll data and save it as a
.json file for retrieval.
"""

import pyreadstat
import json
from tqdm import tqdm
import numpy as np

# -------------------------------------------------
# Classify fields
# -------------------------------------------------

SKIP_FIELDS_2019 = [
    'Country (by alpha)',
    'Country',
    'Study Completion Date',
    'Employed Full Time for an Employer Index (workforce)',
    'Labor Force Participation Index',
    'Wave Year',
    'Wave year',
    'Country ISO alpha-2 code',
    'Country ISO alpha-3 code',
    'Global region respondent lives in'
    'WGT',
    "Survey wgt",
    "Global region respondent lives in",
    'Urban/rural -- recoded into 2 groups',
    'World Bank Income Categories (2023-24 definiton)',
    'Age - 5 categories',
    "Age - 3 categories",
    'Country/area appears in all 3 waves of WRP',
    'Countries in Wave 3 of WRP also in waves 1 or 2',
    'Global Region',
    'Region 2 Global',
    'Sampling Stratification',
    'Sampling Stratification 2',
    'Sampling Stage 1 (PSU)',
    'Random Unique Case ID',
    "Projection weight for time-series or single year analysis"
]

METADATA_FIELDS_2019 = [
    'Country Name',
    'Gender',
    'Education Level',
    'Age cohort (4 group)',
    'Primary Occupation/Job Title (Coded Response)',
    'Global region (as used in World Risk Poll analytical report)',
    'World Bank Country Income Classifications (2019-2020 definition)',
    'Urban/Rural',
    'Employment Status'
]

QUANT_FIELDS_2019 = [
    'Worried Index',
    'Experienced Index',
    'Worry and experience gap',
]

# -------------------------------------------------
# Read in datasets.
# -------------------------------------------------

path = 'data/_2019/19_wrp.sav' # Your directory may differ!

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

data_2019 = []
for index, row in tqdm(df.iterrows(), total=len(df)):

    person_dict = {
        'metadata': {
            'Survey Year': '2019',
            'ID': row['WPID_RANDOM'],
            'demographics': {},
            'quantitative_data': {}
        },
        'qualitative_data': {}
    }

    seen = set()  # (label, value) pairs

    # =====================================================
    # Pull from 19_wrp.save
    # =====================================================

    json_data = json.loads(row.to_json())

    for key, value in json_data.items():

        if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
            continue

        label = meta.column_names_to_labels.get(key)

        if label is None or label in SKIP_FIELDS_2019:
            continue

        pair = (label, str(value))

        if pair in seen:
            continue
        seen.add(pair)

        if label in METADATA_FIELDS_2019 or 'Region' in str(label):
            person_dict['metadata']['demographics'][label] = value

        elif label in QUANT_FIELDS_2019:
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

            if label is None or label in SKIP_FIELDS_2019:
                continue

            pair = (label, str(value))

            if pair in seen:
                continue
            seen.add(pair)

            if label in METADATA_FIELDS_2019 or 'Region' in str(label):
                person_dict['metadata']['demographics'][label] = value

            elif label in QUANT_FIELDS_2019:
                person_dict['metadata']['quantitative_data'][label] = value

            else:
                person_dict['qualitative_data'][label] = value

    data_2019.append(person_dict)

with open('data/_2019/data_2019.json', 'w') as f:
    json.dump(data_2019, f, indent=4)
