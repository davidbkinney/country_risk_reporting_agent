from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import pyreadstat

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/_2021/21_wrp.sav"
CHROMA_DIR = "./vector_stores/_2021/column_db_2021"


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

# =====================================================
# GET UNIQUE COUNTRIES
# =====================================================

columns = [str(c) for c in meta.column_labels if c not in 
           SKIP_FIELDS_2021 + METADATA_FIELDS_2021 + 
           QUANT_FIELDS_2021]

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