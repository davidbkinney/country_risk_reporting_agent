"""
vector_store_creator_column_2019.py

Creates a vector store containing embeddings of the field
names in the 2019 Gallup World Risk Poll survey.
"""

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import pyreadstat

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/_2019/19_wrp.sav"
CHROMA_DIR = "./vector_stores/_2019/column_db_2019"


# =====================================================
# EMBEDDINGS CLIENT
# =====================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# =====================================================
# LOAD DATA
# =====================================================

df, meta = pyreadstat.read_sav(
    DATA_PATH,
    apply_value_formats=True
)

print("Data loaded.")


# =====================================================
# SEGMENT FIELDS
# =====================================================

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

# =====================================================
# GET UNIQUE COUNTRIES
# =====================================================

columns = [str(c) for c in meta.column_labels if c not in 
           SKIP_FIELDS_2019 + METADATA_FIELDS_2019 + 
           QUANT_FIELDS_2019]

# =====================================================
# MAKE DOCUMENTS
# =====================================================

docs = []
for c in columns:
    docs.append(Document(
        page_content=c
    ))

# =====================================================
# CHROMA INIT
# =====================================================

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# =====================================================
# EMBED DOCUMENTS
# =====================================================

indices = [str(i) for i in range(len(docs))]
texts = [d.page_content for d in docs]
vectors = embeddings.embed_documents(texts)


vectorstore._collection.add(
        embeddings=vectors,
        documents=texts,
        ids=indices
    )

# =====================================================
# FINALIZE
# =====================================================

print("\nDONE")
print("Total indexed:", vectorstore._collection.count())

# =====================================================
# TEST
# =====================================================

results = vectorstore.similarity_search(
    "Tell me about people's sense of safety around drinking water.",
    k=5
)

for r in results:
    print(r.page_content)
