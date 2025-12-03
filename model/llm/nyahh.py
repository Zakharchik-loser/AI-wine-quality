from sentence_transformers import  SentenceTransformer
from utils.rag import df
import chromadb



model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = df["text"].tolist()

embeddings = model.encode(texts)


chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection("wine_rag")

collection.upsert(
    documents=texts,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(df))]
)

def search(query_text: str, n_results: int = 5):
    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "distances","metadatas"]
    )
        
    return results

