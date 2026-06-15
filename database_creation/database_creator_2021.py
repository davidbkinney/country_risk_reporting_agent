"""
database_creator_2021.py

Clean the 2021 world risk poll data and save it as a
.json file for retrieval.
"""

import pandas as pd
import pyreadstat
import json
from tqdm import tqdm
import numpy as np

# -------------------------------------------------
# Classify fields
# -------------------------------------------------

SKIP_FIELDS_2021 = [
    'Country (by alpha)',
    'Country',
    'Year of interview',
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
    "Projection weight for time-series or single year analysis",
    "Survey wgt",
    "Country ISO alpha-2 code",
    "Country ISO alpha-3 code",
    "Indicator for waves 2 and 3 which included resilience questions"
]

METADATA_FIELDS_2021 = [
    'Country Name',
    'Gender',
    'Education Level',
    'Age cohort (4 group)',
    'Age',
    'Primary Occupation/Job Title (Coded Response)',
    'Global region (as used in World Risk Poll analytical report)',
    'World Bank Country Income Classifications (2019-2020 definition)',
    'Urban/Rural',
    'Employment Status',
    "Global region respondent lives in"
]

QUANT_FIELDS_2021 = ["Resilience Index 0 - 1 scale",
    "Resilience: Individual Dimension 0 - 100 scale",
    "Resilience: Household Dimension 0 - 100 scale",
    "Resilience: Community Dimension 0 - 100 scale",
    "Resilience: Society Dimension 0 - 100 scale",
    "Resilience Index 0 - 100 scale"]


# -------------------------------------------------
# Read in datasets.
# -------------------------------------------------

path = 'data/_2021/21_wrp.sav' # Your directory may differ!

df, meta = pyreadstat.read_sav(
    path,
    apply_value_formats=True
)

path2 = 'data/trended/trended_wrp.sav' # Your directory may differ!

df2, meta2 = pyreadstat.read_sav(
    path2,
    apply_value_formats=True
)

# Pull worry-experience gap data from 21_23_trended_worry_experience.csv

quant_df = pd.read_csv('data/_2021/21_23_trended_worry_experience.csv')
quant_df['Q4_RMw_maxmin'] = pd.to_numeric(quant_df['Q4_RMw_maxmin'], errors='coerce')
quant_df['Q5_RMw_maxmin'] = pd.to_numeric(quant_df['Q5_RMw_maxmin'], errors='coerce')

quant_lookup = quant_df.set_index('WPID_RANDOM')[['Q4_RMw_maxmin', 'Q5_RMw_maxmin']]

# -------------------------------------------------
# Clean data and save as JSON.
# -------------------------------------------------

data_2021 = []

for index, row in tqdm(df.iterrows(), total=len(df)):

    pid = row['WPID_RANDOM']

    if pid in quant_lookup.index:
        worried_index = quant_lookup.loc[pid, 'Q4_RMw_maxmin']
        experienced_index = quant_lookup.loc[pid, 'Q5_RMw_maxmin']
        gap = worried_index - experienced_index
    else:
        worried_index = None
        experienced_index = None
        gap = None

    person_dict = {
        'metadata': {
            'Survey Year': '2021',
            'ID': pid,
            'demographics': {},
            'quantitative_data': {
                "Worried Index": worried_index,
                "Experienced Index": experienced_index,
                "Worry and experience gap": gap
            }
        },
        'qualitative_data': {}
    }

    seen = set()  # (label, value) pairs

    # =====================================================
    # Pull from 19_wrp.sav
    # =====================================================

    json_data = json.loads(row.to_json())

    for key, value in json_data.items():

        if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
            continue

        label = meta.column_names_to_labels.get(key)

        if label is None or label in SKIP_FIELDS_2021:
            continue

        pair = (label, str(value))

        if pair in seen:
            continue
        seen.add(pair)

        if label in METADATA_FIELDS_2021 or 'Region' in str(label):
            person_dict['metadata']['demographics'][label] = value

        elif label in QUANT_FIELDS_2021:
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

            if label is None or label in SKIP_FIELDS_2021:
                continue

            pair = (label, str(value))

            if pair in seen:
                continue
            seen.add(pair)

            if label in METADATA_FIELDS_2021 or 'Region' in str(label):
                person_dict['metadata']['demographics'][label] = value

            elif label in QUANT_FIELDS_2021:
                person_dict['metadata']['quantitative_data'][label] = value

            else:
                person_dict['qualitative_data'][label] = value

    data_2021.append(person_dict)

with open('data/_2021/data_2021.json', 'w') as f:
    json.dump(data_2021, f, indent=4)
