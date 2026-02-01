# Semantic Search Engine for Medical Literature

## 1. Project Overview

This project implements a **semantic search engine for medical literature**, enhanced with an optional **Retrieval-Augmented Generation (RAG)** question answering capability.

The system allows users to:
- Perform **semantic search** over a corpus of medical research papers using embeddings and cosine similarity.
- Ask **natural language questions** and receive synthesized answers grounded in retrieved evidence from the literature.
- Interact with the system via **command-line interface (CLI)** or a **Streamlit-based web interface**.

The project follows a modular, end-to-end NLP pipeline covering data preprocessing, embedding-based retrieval, and LLM-based answer generation.

---

## 2. Data

- **Domain**: Medical literature  
- **Source format**: PDF documents  
- The `data/raw/` directory contains a placeholder file (`example.pdf`) for demonstration
purposes only. The actual medical literature PDFs used in experiments are not included
in this repository due to size and licensing considerations.


### Preprocessing
1. Extract raw text from PDFs.
2. Clean and normalize text.
3. Split documents into paragraph-level chunks.
4. Filter very short or noisy text.
5. Save processed chunks to a unified JSONL file.

**Output**:
- `data/processed/chunks.jsonl`

---

## 3. Step 3: Embedding + Semantic Retrieval (Core)

### Embedding Model
- **Sentence Transformers**: `all-mpnet-base-v2`

### Index Construction
- Convert each chunk into a dense embedding.
- Store embeddings and metadata locally.

**Artifacts**:
- `index/embeddings.npy`  
- `index/metadata.json`  
- `index/model.txt`

### Retrieval
- Query embedding + cosine similarity
- Top-K semantic matches returned

---

## 4. Step 4: LLM Enhancement (Option B – RAG)

This project implements **Option B: Answer questions based on retrieved results (RAG-style)**.

### RAG Workflow
1. Retrieve top-K relevant chunks via semantic search.
2. Optionally apply a keyword-based prefilter (engineering trade-off).
3. Construct a prompt containing the question and retrieved evidence.
4. Use an LLM to generate an answer grounded in retrieved documents.

---

## 5. Step 5: Interface / Usage

### Command-Line Interface (CLI)

#### Semantic Search
```bash
python src/search.py --query "asthma comorbidities" --top_k 5
```
RAG Question Answering
```
python src/rag_answer.py --question "What are common comorbidities of asthma?" --top_k 5
```
Web Interface (Streamlit)
```
streamlit run src/app_streamlit.py
```
The web interface supports:

Search vs Ask (RAG) modes

Adjustable top_k

Optional keyword prefilter

Evidence display

## 6. Step 6: Code Quality and Project Structure

The project is modular, readable, and reproducible.
```
semantic-search-medlit/
├── data/
│   ├── raw/
│   └── processed/
├── index/
├── src/
│   ├── prepare_chunks_from_pdf.py
│   ├── build_index.py
│   ├── search.py
│   ├── rag_answer.py
│   └── app_streamlit.py
├── .env
├── requirements.txt
└── README.md
```

## 7. Environment Setup (Virtual Environment Required)
⚠️ This project is intended to be run inside a Python virtual environment.
### 7.1 Create Virtual Environment
```
python -m venv .venv
```
### 7.2 Activate Virtual Environment
#### Windows (PowerShell):
```
.venv\Scripts\Activate.ps1
```
#### macOS / Linux:
```
source .venv/bin/activate
```
### 7.3 Install Dependencies
```
pip install -r requirements.txt
```

## 8. API Key Configuration

The OpenAI API key is required only for Ask (RAG) mode.

Create a .env file in the project root directory:
```
OPENAI_API_KEY=your_api_key_here
```
The project loads this key automatically at runtime using python-dotenv.

## 9. Reproducibility Check

After activating the virtual environment, the following commands should run successfully:
```
python src/prepare_chunks_from_pdf.py
python src/build_index.py
python src/search.py --query "type 2 diabetes cardiovascular risk factors" --top_k 5
python src/rag_answer.py --question "What are common risk factors and complications of type 2 diabetes?" --top_k 5
streamlit run src/app_streamlit.py
```

## 10. Limitations and Future Work
Limited corpus size

No OCR for scanned PDFs

Biomedical-specific embedding models could further improve performance

More advanced reranking and citation grounding could be added
## 11. Summary
This project demonstrates an end-to-end semantic search and RAG pipeline for medical literature, with clear modular design, reproducible setup, and both CLI and web-based interfaces.