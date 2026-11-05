import sys
sys.path.insert(0, ".")

from backend.retrieval import get_embedding_model, get_reranker_model, get_qdrant_client
from backend.config import COLLECTION_NAME

query = "If my car breaks down in a state office parking lot, do I get paid administrative leave to fix it?"

embedding_model = get_embedding_model()
reranker = get_reranker_model()
client = get_qdrant_client()

# Stage 1: Bi-encoder top-50
query_vector = embedding_model.encode(query).tolist()
initial_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=50,
)

print("=" * 70)
print(f"QUERY: {query}")
print(f"Total chunks in collection: {client.get_collection(COLLECTION_NAME).points_count}")
print(f"Bi-encoder returned: {len(initial_results.points)} candidates")
print("=" * 70)

# Show top 15 bi-encoder results
print("\n--- TOP 15 BI-ENCODER RESULTS ---")
for i, point in enumerate(initial_results.points[:15], 1):
    source = point.payload['source']
    page = point.payload['page']
    score = point.score
    text_preview = point.payload['text'][:120].replace('\n', ' ')
    marker = " <<< HANDBOOK" if "EmployeeHandbook" in source else ""
    print(f"{i:2d}. [{score:.3f}] {source} p{page}{marker}")
    print(f"    {text_preview}...")
    print()

# Stage 2: Cross-encoder scores
pairs = [(query, p.payload["text"]) for p in initial_results.points]
ce_scores = reranker.predict(pairs)

scored = list(zip(initial_results.points, ce_scores))
scored.sort(key=lambda x: x[1], reverse=True)

# Show top 15 after reranking
print("\n--- TOP 15 AFTER CROSS-ENCODER RERANK ---")
for i, (point, ce_score) in enumerate(scored[:15], 1):
    source = point.payload['source']
    page = point.payload['page']
    bi_score = point.score
    text_preview = point.payload['text'][:120].replace('\n', ' ')
    marker = " <<< HANDBOOK" if "EmployeeHandbook" in source else ""
    print(f"{i:2d}. [CE={ce_score:.3f}, BI={bi_score:.3f}] {source} p{page}{marker}")
    print(f"    {text_preview}...")
    print()

# Check: was ANY Handbook chunk in the top-50?
handbook_in_50 = any("EmployeeHandbook" in p.payload['source'] for p in initial_results.points)
print(f"\n>>> EmployeeHandbook chunk in bi-encoder top-50? {handbook_in_50}")

# Check: did ANY Handbook chunk survive to top-5 after rerank?
handbook_in_top5 = any("EmployeeHandbook" in p.payload['source'] for p, _ in scored[:5])
print(f">>> EmployeeHandbook chunk in reranker top-5? {handbook_in_top5}")