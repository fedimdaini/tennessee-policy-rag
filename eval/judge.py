import json
import requests
import re
import statistics
import time

API_URL = "http://localhost:8000"
JUDGE_PROMPT = """You are evaluating a RAG system's answer for quality. Score it on two dimensions.

Question: {question}

Retrieved Context:
{context}

Generated Answer: {answer}

Score the following on a scale of 0.0 to 1.0 (use decimals, e.g. 0.8):

1. FAITHFULNESS: Is every claim in the answer actually supported by the context? A claim not present in the context (even if true in general) counts against this score. An answer that correctly says "I don't have enough information" when the context is irrelevant should score 1.0 for faithfulness. Do NOT penalize the answer for omitting context details it wasn't asked about — only penalize claims that are false or unsupported by the provided context.

2. ANSWER_RELEVANCY: Does the answer directly address what was asked in the question? If the answer correctly refuses because the context does not contain enough information to answer the question, score this as 1.0 — a correct refusal is relevant.

Respond ONLY in this exact format, nothing else:
FAITHFULNESS: <score>
ANSWER_RELEVANCY: <score>
REASONING: <one sentence explaining the faithfulness score>"""

def ask_api(question: str, top_k: int = 5):
    resp = requests.post(
        f"{API_URL}/ask",
        json={"question": question, "top_k": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()

def judge_answer(question: str, context: str, answer: str):
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    resp = requests.post(
        f"{API_URL}/ask",
        json={"question": prompt, "top_k": 5},
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["answer"]

    faith_match = re.search(r"FAITHFULNESS:\s*([\d.]+)", text)
    rel_match = re.search(r"ANSWER_RELEVANCY:\s*([\d.]+)", text)
    reasoning_match = re.search(r"REASONING:\s*(.+)", text)

    return {
        "faithfulness": float(faith_match.group(1)) if faith_match else None,
        "answer_relevancy": float(rel_match.group(1)) if rel_match else None,
        "reasoning": reasoning_match.group(1).strip() if reasoning_match else text,
    }

def main():
    with open("../evaluation/golden_dataset.json", encoding="utf-8") as f:
        golden = json.load(f)

    judged_results = []
    for item in golden:
        if item.get("id") in (16, 17):
            continue  # skip trick questions for judge scoring

        print(f"Judging Q{item['id']}...", end=" ")
        try:
            result = ask_api(item["question"], top_k=5)
            context = "\n\n".join(
                f"[{s['source']}, page {s['page']}]\n{s['text'][:500]}"
                for s in result.get("sources", [])
            )
            scores = judge_answer(item["question"], context, result["answer"])
            judged_results.append({
                "id": item["id"],
                "difficulty": item["difficulty"],
                "question": item["question"],
                **scores,
            })
            print(f"faith={scores['faithfulness']} rel={scores['answer_relevancy']}")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(3)

    faith_scores = [r["faithfulness"] for r in judged_results if r["faithfulness"] is not None]
    rel_scores = [r["answer_relevancy"] for r in judged_results if r["answer_relevancy"] is not None]

    print(f"\nAverage Faithfulness:     {statistics.mean(faith_scores):.2f}")
    print(f"Average Answer Relevancy: {statistics.mean(rel_scores):.2f}")

    with open("judged_results_api.json", "w", encoding="utf-8") as f:
        json.dump(judged_results, f, indent=2)

if __name__ == "__main__":
    main()