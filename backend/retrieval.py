from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import List

from .config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_MODEL

_embedding_model = None
_reranker_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model

def get_qdrant_client():
    kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**kwargs)

def ensure_collection():
    client = get_qdrant_client()
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

def retrieve_baseline(query: str, top_k: int = 5) -> List:
    """Phase 1: bi-encoder only."""
    embedding_model = get_embedding_model()
    client = get_qdrant_client()
    query_vector = embedding_model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    return results.points

def retrieve_reranker(query: str, top_k: int = 5, initial_k: int = 50) -> List:
    """Phase 2: bi-encoder (top initial_k) -> cross-encoder rerank -> top_k."""
    embedding_model = get_embedding_model()
    client = get_qdrant_client()
    
    query_vector = embedding_model.encode(query).tolist()
    initial_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=initial_k,
    )
    
    if len(initial_results.points) <= top_k:
        return initial_results.points
    
    reranker = get_reranker_model()
    pairs = [
        (query, point.payload["text"]) 
        for point in initial_results.points
    ]
    scores = reranker.predict(pairs)
    
    scored_points = list(zip(initial_results.points, scores))
    scored_points.sort(key=lambda x: x[1], reverse=True)
    
    return [point for point, score in scored_points[:top_k]]

def retrieve(query: str, top_k: int = 5, mode: str = "reranker") -> List:
    if mode == "baseline":
        return retrieve_baseline(query, top_k)
    return retrieve_reranker(query, top_k)