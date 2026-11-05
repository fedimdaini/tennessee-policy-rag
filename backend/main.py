from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import shutil

from .generation import generate_answer
from .ingestion import embed_and_upsert
from .retrieval import ensure_collection
from .config import COLLECTION_NAME

app = FastAPI(title="Tennessee Policy RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: str = "reranker"

@app.on_event("startup")
async def startup():
    ensure_collection()

@app.get("/health")
async def health():
    from .retrieval import get_qdrant_client
    try:
        client = get_qdrant_client()
        client.get_collection(COLLECTION_NAME)
        return {"status": "ok", "qdrant": "connected", "collection": COLLECTION_NAME}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unhealthy: {str(e)}")

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        return generate_answer(req.question, top_k=req.top_k, mode=req.mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        result = embed_and_upsert(tmp_path)
        return {
            "filename": file.filename,
            "chunks_indexed": result["chunks_indexed"],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)