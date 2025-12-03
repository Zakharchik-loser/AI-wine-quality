from sentence_transformers import  SentenceTransformer
from utils.data import df
import chromadb


model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = df["text"].tolist()

embeddings = model.encode(texts)




chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection("wine_data")


collection.upsert(
    documents = df["text"].tolist(),
    ids=[str(i)for i in df["Id"]]
)

results = collection.query(
    query_texts = ["High alcohol wines"],
    n_results = 5

)

print(results)