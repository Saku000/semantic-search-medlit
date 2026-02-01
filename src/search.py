import argparse
import json
import os
import re
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# Keyword prefilter utilities
# ----------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "by", "from",
    # very generic terms
    "risk", "factors", "factor", "age",
}

def extract_keywords(query: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", query.lower())
    keywords = [t for t in tokens if t not in STOPWORDS]
    # de-dup while preserving order
    seen = set()
    out = []
    for k in keywords:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out

def keyword_prefilter(metadata: List[Dict[str, Any]], keywords: List[str], min_hits: int = 1) -> List[int]:
    # Return candidate indices
    if not keywords:
        return list(range(len(metadata)))
    cand = []
    for i, item in enumerate(metadata):
        text = (item.get("text", "") or "").lower()
        hits = sum(1 for k in keywords if k in text)
        if hits >= min_hits:
            cand.append(i)
    return cand


# ----------------------------
# Index loading
# ----------------------------

def load_index(index_dir: str = "index") -> Tuple[SentenceTransformer, np.ndarray, List[Dict[str, Any]]]:
    """
    Load embeddings, metadata, and model name from disk.
    """
    emb_path = os.path.join(index_dir, "embeddings.npy")
    meta_path = os.path.join(index_dir, "metadata.json")
    model_path = os.path.join(index_dir, "model.txt")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing {emb_path}. Run build_index.py first.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}. Run build_index.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Run build_index.py first.")

    embeddings = np.load(emb_path)  # (N, D)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(model_path, "r", encoding="utf-8") as f:
        model_name = f.read().strip()

    model = SentenceTransformer(model_name)
    return model, embeddings.astype(np.float32), metadata


# ----------------------------
# Semantic search (doc-level)
# ----------------------------

def semantic_search(
    query: str,
    top_k: int = 5,
    index_dir: str = "index",
    use_prefilter: bool = True,
    min_hits: int = 1,
    fallback_threshold: int = 50,
) -> List[Dict[str, Any]]:
    """
    Perform cosine-similarity-based semantic search.
    Returns top_k distinct documents (doc-level ranking using each doc's best chunk).
    """
    if not query or not query.strip():
        raise ValueError("Query is empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    model, emb_matrix, metadata = load_index(index_dir)

    # Build candidate set
    if use_prefilter:
        keywords = extract_keywords(query)
        cand_idx = keyword_prefilter(metadata, keywords, min_hits=min_hits)

        # If too few candidates, relax filtering; if still empty, fall back to full
        if len(cand_idx) < fallback_threshold:
            cand_idx = keyword_prefilter(metadata, keywords, min_hits=0)
        if len(cand_idx) == 0:
            cand_idx = list(range(len(metadata)))
    else:
        cand_idx = list(range(len(metadata)))

    # Query embedding (normalize for cosine via dot product)
    query_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # Candidate embeddings
    cand_matrix = emb_matrix[cand_idx]  # (Ncand, D)
    # cosine scores
    scores = cand_matrix @ query_vec  # (Ncand,)

    # Doc-level aggregation: keep the best chunk per doc
    best_by_doc: Dict[str, Tuple[float, int]] = {}  # doc_id -> (best_score, cand_pos)
    for cand_pos, s in enumerate(scores):
        item = metadata[cand_idx[cand_pos]]
        doc_id = item.get("doc_id", "")
        if not doc_id:
            continue
        prev = best_by_doc.get(doc_id)
        if (prev is None) or (s > prev[0]):
            best_by_doc[doc_id] = (float(s), cand_pos)

    # Sort docs by best score desc
    ranked = sorted(best_by_doc.items(), key=lambda x: x[1][0], reverse=True)

    results: List[Dict[str, Any]] = []
    for rank, (doc_id, (best_score, cand_pos)) in enumerate(ranked[:top_k], start=1):
        orig_i = cand_idx[cand_pos]
        item = metadata[int(orig_i)]
        results.append(
            {
                "rank": rank,
                "score": float(best_score),
                "doc_id": doc_id,
                "title": item.get("title", ""),
                "chunk_id": item.get("chunk_id", -1),
                "text": item.get("text", ""),
            }
        )

    return results


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Medical Literature Semantic Search (SentenceTransformer + cosine)"
    )
    parser.add_argument("--query", required=True, type=str, help="Search query")
    parser.add_argument("--top_k", default=5, type=int, help="Number of results to return")
    parser.add_argument("--index_dir", default="index", type=str, help="Index directory")
    parser.add_argument("--no_prefilter", action="store_true", help="Disable keyword prefilter")
    parser.add_argument("--min_hits", default=1, type=int, help="Prefilter min keyword hits")
    parser.add_argument("--max_chars", default=1200, type=int, help="Max chars to print per snippet")

    args = parser.parse_args()

    results = semantic_search(
        query=args.query,
        top_k=args.top_k,
        index_dir=args.index_dir,
        use_prefilter=(not args.no_prefilter),
        min_hits=args.min_hits,
    )

    if len(results) == 0:
        print("No results found.")
        return

    for r in results:
        print("=" * 100)
        print(f"#{r['rank']}  score={r['score']:.4f}")
        print(f"{r['title']}  (doc_id={r['doc_id']}, chunk_id={r['chunk_id']})")
        print("-" * 100)
        print((r["text"] or "")[: args.max_chars])
    print("=" * 100)


if __name__ == "__main__":
    main()
