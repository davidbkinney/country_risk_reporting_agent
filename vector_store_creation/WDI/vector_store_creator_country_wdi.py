import pandas as pd
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "data/wdi/wdi_mortality_data.csv"
CHROMA_DIR = "./vector_stores/wdi/country_db_wdi"


# =====================================================
# EMBEDDINGS CLIENT
# =====================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# =====================================================
# LOAD DATA
# =====================================================

data = pd.read_csv(DATA_PATH)

data = data.dropna()

print(f"Loaded {len(data):,} rows")

# =====================================================
# GET UNIQUE COUNTRIES
# =====================================================


unique_countries = data['Country Name'].unique()

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