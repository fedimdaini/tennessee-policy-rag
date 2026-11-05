import requests
import time

API_URL = "http://localhost:8000"

# HELD-OUT ADVERSARIAL QUESTIONS
# These were NOT in golden_dataset.json and were NOT used to tune initial_k.
# They test whether the reranker generalizes to novel cross-domain distractors.
HELD_OUT = [
    {
        "id": "ho_1",
        "question": "If my car breaks down in a state office parking lot, do I get paid administrative leave to fix it?",
        "expected_domain": "HR / parking policy",
        "trap": "DMV 'car' + HR 'paid leave' vocabulary collision",
    },
    {
        "id": "ho_2",
        "question": "How many days of bereavement leave can I take for a foster parent who passed away?",
        "expected_domain": "DOHR Rules",
        "trap": "Exact term 'foster parent' appears in DOHR rules but is semantically close to 'immediate family'",
    },
    {
        "id": "ho_3",
        "question": "Can I use my state employee ID card to board a domestic flight instead of a REAL ID driver license?",
        "expected_domain": "Driver Manual (REAL ID)",
        "trap": "HR 'employee ID' + DMV 'REAL ID' mixed domains",
    },
    {
        "id": "ho_4",
        "question": "What is the penalty for driving without insurance on state government property?",
        "expected_domain": "Driver Manual",
        "trap": "DMV 'insurance' + HR 'state property' collision",
    },
    {
        "id": "ho_5",
        "question": "Do I need supervisor approval for Emotional Wellbeing Solutions time if I have accrued sick leave?",
        "expected_domain": "Employee Handbook",
        "trap": "EWS (handbook) + sick leave (DOHR) — tests if reranker preserves the right HR source",
    },
]

def test_both_modes(q: dict):
    print(f"\n{'='*60}")
    print(f"[{q['id']}] {q['question']}")
    print(f"Trap: {q['trap']}")
    print(f"Expected domain: {q['expected_domain']}")
    
    for mode in ["baseline", "reranker"]:
        try:
            r = requests.post(
                f"{API_URL}/ask",
                json={"question": q["question"], "top_k": 5, "mode": mode},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            sources = [s["source"] for s in data.get("sources", [])]
            print(f"\n  Mode: {mode}")
            print(f"    Answer: {data['answer'][:120]}...")
            print(f"    Sources: {sources}")
        except Exception as e:
            print(f"\n  Mode: {mode} -> ERROR: {e}")
        time.sleep(2)  # be gentle on Groq rate limits

if __name__ == "__main__":
    print("=" * 60)
    print("HELD-OUT GENERALIZATION TEST")
    print("These questions were NEVER used during reranker development.")
    print("=" * 60)
    
    for q in HELD_OUT:
        test_both_modes(q)
        print()
    
    print("=" * 60)
    print("Done. Compare baseline vs reranker sources for each question.")
    print("=" * 60)