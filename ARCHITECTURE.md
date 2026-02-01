# Architecture: Semantic Search Engine (Medical Literature)

This document describes the technical architecture of the semantic search + RAG system, including data flow, indexing, retrieval, and the Streamlit interface.

---

## 1. System Goals

**Primary goal**: Build a semantic retrieval pipeline over a medical literature PDF corpus and support:
- **Semantic search** using embeddings + cosine similarity (Step 3)
- **RAG-style QA** that answers questions grounded in retrieved text chunks (Step 4, Option B)
- **CLI** and **Web UI** usage (Step 5)

Non-goals:
- Training/fine-tuning embedding models
- Full OCR pipeline for scanned PDFs (future work)
- Long-form citation extraction (future work)

---

## 2. High-Level Data Flow

```text
PDFs (data/raw/*.pdf)
   |
   v
[prepare_chunks_from_pdf.py]
   -> extract text
   -> clean & normalize
   -> paragraph splitting
   -> chunking (length bounds)
   v
chunks.jsonl (data/processed/chunks.jsonl)
   |
   v
[build_index.py]
   -> SentenceTransformer embed(chunks)
   -> save embeddings + metadata
   v
index/embeddings.npy + index/metadata.json + index/model.txt
   |
   v
[search.py] / [rag_answer.py] / [app_streamlit.py]
   -> embed(query)
   -> cosine similarity
   -> top-K chunks
   -> (optional) prefilter
   -> (optional) LLM answer
```

## 3. Project Structure
```
semantic-search-medlit/
|- data/
|   |- raw/                    # input PDFs
|   |- processed/              # chunks.jsonl output
|- index/                      # embeddings + metadata artifacts
|- src/
|   |- prepare_chunks_from_pdf.py
|   |- build_index.py
|   |- search.py
|   |- rag_answer.py
|   |- app_streamlit.py
|- .env                        # OPENAI_API_KEY (local only)
|- requirements.txt
|- README.md
|- ARCHITECTURE.md
|- TEAM_CONTRIBUTIONS.md
```

## 4. Data Layer: PDF -> Chunks
### 4.1 Input

Location: data/raw/

Format: PDF research papers

### 4.2 Extraction and Cleaning

prepare_chunks_from_pdf.py performs:

Text extraction (PDF parsing)

Basic normalization (whitespace cleanup, removing repeated noise patterns if present)

Paragraph splitting

Chunking by paragraph with length bounds:

min_len: avoid very short/noisy chunks

max_len: prevent overly long chunks

### 4.3 Output Format: chunks.jsonl

Each line is a JSON object:
```
{
  "doc_id": "071_Application of a Novel Machine Learning...",
  "chunk_id": 4,
  "text": "Asthma is a common chronic disease that affects..."
}
```
Design rationale:

JSONL is streaming-friendly and easy to debug

Chunk-level granularity improves retrieval precision and reduces context size for RAG

## 5. Index Layer: Chunk Embeddings
### 5.1 Model

build_index.py uses Sentence Transformers:

Default: sentence-transformers/all-mpnet-base-v2

Rationale:

Strong general-purpose semantic embeddings

Works well with cosine similarity for dense retrieval

### 5.2 Index Artifacts

Output saved to index/:

embeddings.npy: shape (N_chunks, embedding_dim)

metadata.json: list of metadata aligned with embeddings row indices

model.txt: the model name used (reproducibility)

A strict invariant:

len(metadata) == embeddings.shape[0]

## 6. Retrieval Layer: Semantic Search
### 6.1 Query Embedding

Both search.py and rag_answer.py embed the query using the same SentenceTransformer model saved in index/model.txt.

### 6.2 Similarity

Cosine similarity is computed between:

q_emb (1 x d)

X (N x d)

Then top-K indices are selected and mapped back to metadata + text.

### 6.3 Dedup Strategy (Doc-Level)

If the corpus contains many chunks from the same document, top-K chunk results can be dominated by a single paper.

Recommended approach (if implemented):

Keep at most max_per_doc chunks per doc_id within the returned set

Improves diversity of results and is useful for RAG evidence coverage

## 7. Step 4: RAG (Option B) Answer Generation
### 7.1 What is "RAG" here?

RAG = retrieve top-K evidence chunks, then generate an answer conditioned on those chunks.

### 7.2 Prompt Construction

rag_answer.py builds a prompt using:

The user question

A list of retrieved evidence snippets (truncated to fit context window)

A grounding instruction (answer must be based on evidence; if insufficient, say so)

### 7.3 Keyword Prefilter (Optional)

A lightweight keyword prefilter can be enabled to reduce off-topic retrieval:

Search first retrieves a wider candidate set

Candidates are filtered by a minimum number of keyword hits

Then cosine ranking is applied again on the filtered set (or top-K is selected from the filtered pool)

This is an engineering trade-off:

Pros: reduces obviously irrelevant results when corpus is narrow or heterogeneous

Cons: may reduce recall for paraphrased queries

### 7.4 API Key Handling

OpenAI API key is read from .env:

.env location: project root

Key: OPENAI_API_KEY

Loaded using python-dotenv

The RAG scripts explicitly fail fast if the key is missing.

## 8. Interface Layer
### 8.1 CLI

search.py: semantic search

rag_answer.py: RAG QA

All scripts accept flags such as:

--query / --question

--top_k

(optional) --prefilter toggles

### 8.2 Streamlit Web UI

app_streamlit.py provides:

Mode switch: Search vs Ask (RAG)

Parameter controls: top_k, prefilter toggles, thresholds

Result visualization:

retrieved docs/chunks

RAG answer (Ask mode)

optional evidence view

## 9. Error Handling & Reproducibility
### 9.1 Fail-Fast Checks

Typical runtime checks:

Index files exist before search/RAG

Metadata length matches embedding matrix size

API key exists before calling LLM

Empty query/question is rejected

### 9.2 Reproducibility

Rebuildable artifacts:

data/processed/chunks.jsonl from PDFs

index/* from chunks using build_index.py

Model version is recorded in index/model.txt to ensure embeddings are consistent.

## 10. Known Limitations & Future Improvements

Scanned PDFs: some files have little/no extractable text; OCR is not implemented

Domain mismatch: general embedding models may retrieve broadly related biomedical topics

Reranking: could add a cross-encoder reranker to improve precision

Biomedical embeddings: could test PubMedBERT-based sentence embeddings to improve domain alignment

Citations: could add chunk-to-page mapping and provide citation pointers