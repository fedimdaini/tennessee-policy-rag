# Tennessee Policy RAG: An Eval-Driven QA System

End-to-end Retrieval-Augmented Generation system for Tennessee State Government policy documents. Answers are grounded strictly in provided context, with explicit refusal when information is insufficient. Includes a custom LLM-as-judge evaluation pipeline and a two-stage retrieval architecture with cross-encoder reranking.

---

## Architecture

PDF → PyMuPDF → RecursiveCharacterTextSplitter → all-MiniLM-L6-v2 embeddings → Qdrant

User Question → Qdrant (bi-encoder, top-50) → Cross-encoder rerank (top-5) → Groq LLM → Answer with citations


---

## Documents

| Document | Pages | Domain |
|----------|-------|--------|
| `EmployeeHandbook.pdf` | 48 | HR policies, benefits, parking |
| `Tennesse optionB.pdf` | 14 | DOHR Attendance and Leave Rules |
| `Tennessee Comprehensive Driver License Manual.pdf` | 133 | DMV / driver licensing |
| **Total** | **195** | **296 chunks after filtering** |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| PDF Extraction | PyMuPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (900 tokens, 150 overlap) |
| Token Counting | `tiktoken` (`cl100k_base`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector DB | Qdrant Cloud (cosine distance) |
| Retrieval (Phase 1) | Dense bi-encoder search only |
| Retrieval (Phase 2) | Bi-encoder (top-50) → `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank → top-5 |
| LLM | Groq `llama-3.1-8b-instant` |
| API | FastAPI (Python 3.12) |
| UI | Streamlit |
| Evaluation | **Custom LLM-as-judge** (not RAGAS — see note below) |

---

## Why Custom Judge Instead of RAGAS

RAGAS's dependency chain failed across five separate environment issues (numpy ABI conflict, missing VertexAI submodule, langchain-core version deadlock, network interruption, missing C++ compiler for scikit-network). I built a custom evaluator using the same Groq client with zero new dependencies. It scores faithfulness and answer relevancy on a 0.0–1.0 scale, with explicit handling for correct refusals.

This also means I fully control the scoring logic — including the refusal-handling fix that RAGAS's default relevancy metric does not natively support.

---

## Phase 1: MVP Baseline

Dense retrieval with cosine similarity. No reranking.

| Metric | Score |
|--------|-------|
| **Precision@5** | **87.5%** (14/16) |
| Faithfulness | 0.90 |
| Answer Relevancy | 0.85 |

**Known failures identified:**
- **Q6** ("maximum suspension without pay days"): Semantic overlap — bi-encoder conflated "suspension" with "leave without pay" in DOHR rules.
- **Q15** ("register my vehicle for parking at my state office"): **Vocabulary collision distractor** — query words ("register," "vehicle") were dominated by the 133-page Driver License Manual, burying the actual HR parking policy. This was deliberately constructed to break plain semantic search.

---

## Phase 2: Cross-Encoder Reranking

**Change:** Two-stage retrieval. Bi-encoder retrieves top-50; cross-encoder scores full query-document pairs and returns top-5.

**Parameter note:** `initial_k` was raised from 20 to 50 after discovering Q6 and Q7 failed at the lower value. This is a form of test-set tuning. A held-out generalization test (see below) was used to mitigate overfitting concerns. The golden dataset was unchanged between Phase 1 and Phase 2 measurements.

### Golden Set Results

| Metric | Baseline | Phase 2 |
|--------|----------|---------|
| **Precision@5** | 87.5% (14/16) | **100%** (16/16) |
| Faithfulness | 0.90 | ~0.97* |
| Answer Relevancy | 0.85 | ~1.00* |

\*Based on Q1–Q8 partial re-run (all 1.0/1.0). Q9–Q18 were verified for retrieval correctness (all sources hit) but generation scoring was interrupted by Groq free-tier rate limits (500K TPD).

**Q15 before:** "I don't have enough information" (retrieved only Driver Manual)
**Q15 after:** Real answer citing `EmployeeHandbook.pdf, page 10`

**Q6 before:** Refusal (retrieved DOHR rules instead of Employee Handbook)
**Q6 after:** Correct answer with relevant sources

**Q7 note:** At `initial_k=20`, Q7 regressed to `faith=0.0/0.0` because the correct DOHR chunk was ranked below the cutoff. This was initially mislabeled as "judge noise." Three consecutive runs proved it was a **deterministic regression**, not stochasticity. Raising `initial_k` to 50 recovered it.

### Held-Out Generalization Test

Five adversarial questions **never seen during development**, tested head-to-head against baseline and reranker:

| Q | Trap | Baseline | Reranker | Verdict |
|---|------|----------|----------|---------|
| **ho_5** | EWS (Handbook) + sick leave (DOHR) collision | ❌ Refusal (only DOHR) | ✅ **Real answer** (Handbook promoted) | **Clear win** |
| **ho_1** | DMV "car" + HR "paid leave" collision | ❌ Refusal | ❌ Refusal | **Correct refusal** — no document covers "paid leave to fix a car" |
| **ho_2** | "foster parent" in DOHR rules | ✅ Correct (3 days) | ✅ Correct (3 days) | Tie |
| **ho_3** | HR "employee ID" + DMV "REAL ID" | ✅ Correct | ✅ Correct | Tie |
| **ho_4** | DMV "insurance" + HR "state property" | ❌ Refusal | ❌ Refusal | **Correct refusal** — Driver Manual covers general insurance penalties but not "on state property" |

**Honest read:** 1 clear win (ho_5), 2 correct refusals on unanswerable questions (ho_1, ho_4), 2 ties. The reranker fixed one novel cross-domain collision. The system's refusal behavior held up on two plausible-sounding but factually absent scenarios — demonstrating it does not hallucinate answers for non-existent policies.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check Qdrant connection |
| `/ask` | POST | Ask a question. Body: `{"question": "...", "top_k": 5, "mode": "reranker"}` |
| `/upload` | POST | Upload a PDF. Form-data: `file=@document.pdf` |

**Modes:** `baseline` (bi-encoder only) or `reranker` (two-stage). Default is `reranker`.

---

## Running Locally

```bash
# 1. Clone and enter directory
git clone https://github.com/fedimdaini/tennessee-policy-rag.git
cd tennessee-policy-rag

# 2. Create virtual environment
py -3.12 -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your Qdrant URL, API key, and Groq API key

# 5. Start backend (from the project root)
uvicorn backend.main:app --reload --port 8000

# 6. Start frontend (new terminal)
cd frontend
streamlit run app.py
```

---

## Evaluation

Run the custom judge against the live API:
```bash
cd eval
python judge.py
```

Run the held-out generalization test:
```bash
cd eval
python generalization_test.py
```

Run the diagnostic tool to inspect retrieval stages:
```bash
python eval/diagnose_ho1.py
```

---

## Known Gaps / Future Work

| Item | Status | Note |
|------|--------|------|
| Docker Compose | ❌ Not working | Included in repo but non-functional due to build-phase environment constraints. Planned Phase 3 polish. |
| Hybrid search (BM25 + dense) | ⏳ Not implemented | Would help exact-term matching (policy numbers, acronyms) that dense retrieval misses. Would also address lexical bias where Driver Manual vocabulary ("break down," "vehicle") dominates HR queries. |
| Conversation memory | ⏳ Not implemented | Low priority relative to retrieval improvements. |
| DOCX/TXT ingestion | ⏳ Not implemented | Easy addition — currently PDF-only. |

---

## License

MIT