from groq import Groq
from typing import Dict
from .config import GROQ_API_KEY, LLM_MODEL
from .retrieval import retrieve

SYSTEM_PROMPT = """You are an enterprise knowledge assistant for Tennessee State Government employees.

Answer the user's question using ONLY the information in the provided context below. Do not use any outside knowledge.

If the context does not contain enough information to answer the question, respond exactly with:
"I don't have enough information to answer that question."

Always be precise with numbers, dates, and specific policy details — do not round or approximate figures from the context.

After your answer, list the sources you used in this format:
Sources: [document name, page X]"""

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

def format_context(retrieved_chunks) -> str:
    context_parts = []
    for chunk in retrieved_chunks:
        label = f"[{chunk.payload['source']}, page {chunk.payload['page']}]"
        context_parts.append(f"{label}\n{chunk.payload['text']}")
    return "\n\n---\n\n".join(context_parts)

def generate_answer(question: str, top_k: int = 5, mode: str = "reranker") -> Dict:
    retrieved_chunks = retrieve(question, top_k=top_k, mode=mode)
    context = format_context(retrieved_chunks)
    
    user_message = f"""Context:
{context}

Question: {question}"""
    
    client = get_groq_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )
    
    answer = response.choices[0].message.content
    sources = [
        {
            "source": c.payload["source"],
            "page": c.payload["page"],
            "score": round(c.score, 3) if hasattr(c, "score") else None,
            "text": c.payload["text"][:300] + "..."
        }
        for c in retrieved_chunks
    ]
    
    return {"answer": answer, "sources": sources, "retrieved_count": len(retrieved_chunks)}