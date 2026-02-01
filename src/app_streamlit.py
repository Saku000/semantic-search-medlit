import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Optional: local folder picker (works best on Windows local run)
def pick_folder_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory()
        root.destroy()
        return folder or None
    except Exception:
        return None


# ---------- Index loading / caching ----------
@st.cache_resource(show_spinner=False)
def load_index(index_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]], str]:
    """
    Load embeddings + metadata + model_name from index_dir.
    Returns:
      embeddings: (N, D) float32
      metadata: list of dicts aligned with embeddings
      model_name: str
    """
    index_path = Path(index_dir)
    emb_path = index_path / "embeddings.npy"
    meta_path = index_path / "metadata.json"
    model_path = index_path / "model.txt"

    if not emb_path.exists() or not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing index files in {index_dir}. Need embeddings.npy, metadata.json, model.txt"
        )

    embeddings = np.load(str(emb_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model_name = model_path.read_text(encoding="utf-8").strip()

    # Safety checks
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"metadata length ({len(metadata)}) != embeddings rows ({embeddings.shape[0]})"
        )

    return embeddings.astype(np.float32), metadata, model_name


@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def keyword_prefilter_candidates(
    metadata: List[Dict[str, Any]],
    query: str,
    min_hits: int = 1,
    extra_keywords: Optional[List[str]] = None,
) -> List[int]:
    """
    Simple prefilter: keep chunks whose text contains >= min_hits keywords.
    Keywords are derived from query tokens plus optional extras.
    """
    q = (query or "").lower()
    tokens = [t.strip(".,;:()[]{}\"'").lower() for t in q.split() if len(t) >= 4]
    # de-dup
    tokens = list(dict.fromkeys(tokens))

    keywords = tokens[:]
    if extra_keywords:
        keywords += [k.lower() for k in extra_keywords if k]

    # Another de-dup
    keywords = list(dict.fromkeys(keywords))

    if not keywords:
        return list(range(len(metadata)))

    cand = []
    for i, m in enumerate(metadata):
        text = (m.get("text") or "").lower()
        hits = sum(1 for k in keywords if k in text)
        if hits >= min_hits:
            cand.append(i)
    return cand


def doc_level_dedup_topk(
    indices: List[int],
    scores: np.ndarray,
    metadata: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Deduplicate by doc_id: keep best scoring chunk per doc, then take top_k docs.
    """
    best_by_doc: Dict[str, Tuple[float, int]] = {}
    for idx in indices:
        doc_id = metadata[idx].get("doc_id", "")
        s = float(scores[idx])
        if doc_id not in best_by_doc or s > best_by_doc[doc_id][0]:
            best_by_doc[doc_id] = (s, idx)

    # Sort by score desc
    ranked = sorted(best_by_doc.values(), key=lambda x: x[0], reverse=True)[:top_k]

    results = []
    for s, idx in ranked:
        m = metadata[idx]
        results.append(
            {
                "score": float(s),
                "doc_id": m.get("doc_id", ""),
                "title": m.get("title", ""),
                "chunk_id": m.get("chunk_id", None),
                "text": m.get("text", ""),
            }
        )
    return results


def semantic_search_local(
    query: str,
    embeddings: np.ndarray,
    metadata: List[Dict[str, Any]],
    embedder,
    top_k: int = 5,
    use_prefilter: bool = True,
    min_hits: int = 1,
    extra_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Streamlit-local semantic search:
    - encode query
    - optional keyword prefilter to narrow candidates
    - cosine similarity via dot product on normalized embeddings
    - doc-level dedup + top_k
    """
    if not query.strip():
        return []

    # query vec
    qvec = embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # candidates
    if use_prefilter:
        cand_idx = keyword_prefilter_candidates(metadata, query, min_hits=min_hits, extra_keywords=extra_keywords)
        if not cand_idx:
            cand_idx = list(range(len(metadata)))
    else:
        cand_idx = list(range(len(metadata)))

    cand_mat = embeddings[cand_idx]  # (M, D), already normalized from build
    # scores for candidates
    cand_scores = cand_mat @ qvec  # (M,)
    # map back to global index scores array
    full_scores = np.full((embeddings.shape[0],), -1e9, dtype=np.float32)
    full_scores[cand_idx] = cand_scores

    # take a larger pool then dedup at doc-level
    pool = sorted(cand_idx, key=lambda i: float(full_scores[i]), reverse=True)[: max(50, top_k * 10)]
    return doc_level_dedup_topk(pool, full_scores, metadata, top_k=top_k)


def build_rag_prompt(question: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    evidence_blocks = []
    for r in retrieved:
        doc_id = r.get("doc_id", "")
        title = r.get("title", "")
        text = (r.get("text") or "").strip().replace("\n", " ")
        text = text[:1200]
        evidence_blocks.append(f"[{doc_id}] {title}\n{text}")
    evidence = "\n\n".join(evidence_blocks)

    system = (
        "You are a careful assistant doing retrieval-augmented QA over medical literature excerpts. "
        "You must NOT hallucinate. Use ONLY the evidence provided. "
        "If the evidence does not support an answer, say: 'Insufficient evidence in retrieved documents.' "
        "Cite sources using doc_id in square brackets, e.g., [071_...]."
    )

    user = f"""Question:
{question}

Evidence (top retrieved excerpts):
{evidence}

Instructions:
- Provide a direct answer in 6-12 bullet points (or fewer if short).
- Every bullet must end with one or more citations like [doc_id].
- If only one document is clearly relevant, explicitly state that most evidence comes from that document.
- Do NOT give medical advice; keep it informational and evidence-based.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_openai_chat(messages: List[Dict[str, str]], model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def run_script(script_path: str, args: List[str]) -> Tuple[int, str]:
    """
    Run a python script via subprocess and capture output.
    Returns (returncode, combined_output).
    """
    cmd = [sys.executable, script_path] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out


# ---------- UI ----------
st.set_page_config(page_title="Semantic Search (MedLit) - WebUI", layout="wide")
st.title("Semantic Search Engine (Medical Literature) — Streamlit WebUI")

with st.sidebar:
    st.header("Paths & Actions")

    # Folder pick / open
    if "raw_dir" not in st.session_state:
        st.session_state.raw_dir = str(Path("data") / "raw")
    if "index_dir" not in st.session_state:
        st.session_state.index_dir = "index"

    colA, colB = st.columns(2)
    with colA:
        if st.button("📂 Pick RAW PDF Folder"):
            picked = pick_folder_dialog()
            if picked:
                st.session_state.raw_dir = picked
            else:
                st.warning("Folder picker unavailable. You can paste the path manually.")
    with colB:
        if st.button("🗂️ Open RAW Folder"):
            try:
                os.startfile(st.session_state.raw_dir)  # Windows
            except Exception:
                st.warning("Open folder failed. This button works best on Windows local runs.")

    st.session_state.raw_dir = st.text_input("RAW PDF folder", st.session_state.raw_dir)

    colC, colD = st.columns(2)
    with colC:
        if st.button("🗂️ Open INDEX Folder"):
            try:
                os.startfile(st.session_state.index_dir)
            except Exception:
                st.warning("Open folder failed. This button works best on Windows local runs.")
    with colD:
        st.session_state.index_dir = st.text_input("Index folder", st.session_state.index_dir)

    st.divider()
    st.subheader("Build Pipeline (optional buttons)")

    st.caption("These buttons run your existing scripts via subprocess for convenience.")
    if st.button("1) Prepare Chunks (PDF → chunks.jsonl)"):
        script = str(Path("src") / "prepare_chunks_from_pdf.py")
        if not Path(script).exists():
            st.error(f"Missing script: {script}")
        else:
            with st.spinner("Running prepare_chunks_from_pdf.py ..."):
                rc, out = run_script(script, [])
            st.code(out)
            if rc == 0:
                st.success("Chunks prepared.")
            else:
                st.error("Chunk preparation failed.")

    if st.button("2) Build Index (chunks → embeddings/index)"):
        script = str(Path("src") / "build_index.py")
        if not Path(script).exists():
            st.error(f"Missing script: {script}")
        else:
            with st.spinner("Running build_index.py ..."):
                rc, out = run_script(script, [])
            st.code(out)
            if rc == 0:
                st.success("Index built.")
            else:
                st.error("Index build failed.")

    st.divider()
    st.header("Mode")

    mode = st.radio("Choose mode", ["Search", "Ask (RAG)"], index=0)

    st.subheader("Retrieval Settings")
    top_k = st.slider("top_k", 1, 20, 5)
    use_prefilter = st.checkbox("Use prefilter (keyword candidate pruning)", value=True)
    min_hits = st.slider("Prefilter min_hits", 1, 5, 1, disabled=not use_prefilter)
    extra_kw = st.text_input("Extra keywords (comma-separated)", value="asthma,copd,obesity,inflammation")
    extra_keywords = [x.strip() for x in extra_kw.split(",") if x.strip()]

    st.subheader("LLM Settings (Ask mode)")
    llm_model = st.text_input("OpenAI model", value="gpt-4o-mini")
    show_evidence = st.checkbox("Show retrieved evidence snippets", value=True)


# Main panel
st.write("")

# Load index lazily
embeddings = metadata = embedder = model_name = None
index_ready = True
try:
    embeddings, metadata, model_name = load_index(st.session_state.index_dir)
    embedder = load_embedder(model_name)
except Exception as e:
    index_ready = False
    st.warning(f"Index not ready: {e}")
    st.info("Use sidebar buttons to prepare chunks + build index, or set correct index folder path.")


query_label = "Query" if mode == "Search" else "Question"
query = st.text_area(query_label, height=120, placeholder="Type here...")

col1, col2 = st.columns([1, 3])
with col1:
    run_btn = st.button("▶ Run", disabled=not index_ready)

with col2:
    st.caption("Tip: Keep prefilter ON for small/heterogeneous corpora; turn it OFF only if your corpus is large and focused.")

if run_btn and index_ready:
    if not query.strip():
        st.error("Please enter a query/question.")
        st.stop()

    with st.spinner("Retrieving top results..."):
        t0 = time.time()
        retrieved = semantic_search_local(
            query=query,
            embeddings=embeddings,
            metadata=metadata,
            embedder=embedder,
            top_k=top_k,
            use_prefilter=use_prefilter,
            min_hits=min_hits,
            extra_keywords=extra_keywords if use_prefilter else None,
        )
        t1 = time.time()

    st.success(f"Retrieved {len(retrieved)} docs in {t1 - t0:.2f}s (doc-level dedup).")

    if not retrieved:
        st.warning("No results found. Try different keywords or disable prefilter.")
        st.stop()

    # Always show results list
    st.subheader("Top Retrieved Documents")
    for i, r in enumerate(retrieved, 1):
        st.markdown(
            f"**#{i}**  score={r['score']:.4f}  \n"
            f"**{r['title']}**  \n"
            f"`doc_id={r['doc_id']}`  `chunk_id={r.get('chunk_id')}`"
        )
        if show_evidence:
            with st.expander("Show snippet"):
                st.write(r["text"])

    if mode == "Search":
        st.info("Search mode returns retrieved documents only (no LLM).")
    else:
        st.subheader("RAG Answer")

        if not os.getenv("OPENAI_API_KEY"):
            st.error("Missing OPENAI_API_KEY in environment. Set it before using Ask (RAG) mode.")
            st.stop()

        messages = build_rag_prompt(query, retrieved)

        with st.spinner("Calling LLM..."):
            try:
                answer = call_openai_chat(messages, model=llm_model)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                st.stop()

        st.write(answer)
        st.caption("Note: The answer is constrained to retrieved excerpts and should cite doc_id sources.")
