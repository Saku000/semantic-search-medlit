import argparse
import os
from typing import List, Dict, Any

from pathlib import Path
from dotenv import load_dotenv

# === robust .env loading (independent of working directory) ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)
# ==============================================================

from openai import OpenAI

# Reuse your retrieval pipeline (Step 3)
# search.py must expose: semantic_search(query, top_k, index_dir, use_prefilter, min_hits)
from search import semantic_search
print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))

def build_messages(question: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build a strict RAG prompt:
    - Answer ONLY using provided evidence
    - Cite doc_id in square brackets
    - If evidence is insufficient, say so
    """
    evidence_blocks = []
    for r in retrieved:
        doc_id = r.get("doc_id", "")
        title = r.get("title", "")
        text = (r.get("text", "") or "").strip().replace("\n", " ")
        # limit each chunk to control token usage
        text = text[:1200]
        evidence_blocks.append(f"[{doc_id}] {title}\n{text}")

    evidence = "\n\n".join(evidence_blocks)

    system = (
        "You are a careful assistant doing retrieval-augmented QA over medical literature excerpts. "
        "You must NOT hallucinate. Use ONLY the evidence provided. "
        "If the evidence does not support an answer, say 'Insufficient evidence in retrieved documents.' "
        "Cite sources using the doc_id in square brackets, e.g., [071_...]."
    )

    user = f"""Question:
{question}

Evidence (top retrieved excerpts):
{evidence}

Instructions:
- Provide a direct answer in 6-12 bullet points (or fewer if short).
- Every bullet must end with one or more citations like [doc_id].
- If you need to make a cautious inference, label it as "Inference" and still cite.
- Do NOT give medical advice; keep it informational and evidence-based.
"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_openai_chat(messages: List[Dict[str, str]], model: str) -> str:
    """
    Uses Chat Completions API (supported indefinitely per official SDK notes).
    """
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Step 4 (Option B): RAG-style QA based on retrieved results")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--top_k", type=int, default=5, help="How many documents to retrieve")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name (cost-effective default)")
    parser.add_argument("--index_dir", default="index", help="Index directory")
    parser.add_argument("--no_prefilter", action="store_true", help="Disable keyword prefilter in retrieval")
    parser.add_argument("--min_hits", type=int, default=1, help="Prefilter min keyword hits")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in your environment first.")

    # 1) Retrieve evidence using Step 3
    retrieved = semantic_search(
        query=args.question,
        top_k=args.top_k,
        index_dir=args.index_dir,
        use_prefilter=(not args.no_prefilter),
        min_hits=args.min_hits,
    )

    if not retrieved:
        print("No retrieved results. Try a different question or disable prefilter (--no_prefilter).")
        return

    print("=" * 100)
    print("Top retrieved documents:")
    for i, r in enumerate(retrieved, 1):
        print(f"#{i} score={r['score']:.4f} | {r['doc_id']} | {r['title']}")
    print("=" * 100)

    # 2) Ask LLM to answer using only retrieved evidence
    messages = build_messages(args.question, retrieved)
    answer = call_openai_chat(messages, model=args.model)

    print("\nRAG Answer:\n")
    print(answer)
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
