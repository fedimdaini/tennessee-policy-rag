import fitz
import tiktoken
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

from .config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
from .retrieval import get_embedding_model, get_qdrant_client

encoding = tiktoken.get_encoding("cl100k_base")

def token_length(text: str) -> int:
    return len(encoding.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=token_length,
)

def extract_pages(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "source": pdf_path, "page": page_num})
    doc.close()
    return pages

def process_pdf(pdf_path: str, min_chunk_tokens: int = 20) -> List[Dict]:
    pages = extract_pages(pdf_path)
    all_chunks = []
    for page in pages:
        chunks = text_splitter.split_text(page["text"])
        for i, chunk_text in enumerate(chunks):
            if token_length(chunk_text) >= min_chunk_tokens:
                all_chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": f"{page['source']}_p{page['page']}_c{i}"
                })
    return all_chunks

def embed_and_upsert(pdf_path: str) -> Dict:
    from qdrant_client.models import PointStruct
    chunks = process_pdf(pdf_path)
    if not chunks:
        return {"chunks_indexed": 0, "source": pdf_path}
    
    embedding_model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
    
    client = get_qdrant_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i].tolist(),
            payload={
                "text": chunks[i]["text"],
                "source": chunks[i]["source"],
                "page": chunks[i]["page"],
                "chunk_id": chunks[i]["chunk_id"],
            }
        )
        for i in range(len(chunks))
    ]
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"chunks_indexed": len(chunks), "source": pdf_path}