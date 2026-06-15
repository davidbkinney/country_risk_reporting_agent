"""
vector_store_creator_country_wdi.py

Creates a vector store containing embeddings of the country
names in the 2019 Gallup World Risk Poll survey.
"""


import json
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/_2019/data_2019.json"
CHROMA_DIR = "./vector_stores/_2019/country_db_2019"


# =====================================================
# EMBEDDINGS CLIENT
# =====================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# =====================================================
# LOAD DATA
# =====================================================

with open(DATA_PATH, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data):,} rows")

# =====================================================
# GET UNIQUE COUNTRIES
# =====================================================

countries = [d["metadata"]["demographics"]["Country Name"]
             for d in data]

unique_countries = list(set(countries))

# =====================================================
# MAKE DOCUMENTS
# =====================================================

docs = []
for c in unique_countries:
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
    "Tell me about people's anxiety in Spain and Germany.",
    k=5
)

for r in results:
    print(r.page_content)
