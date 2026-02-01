import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    doc_id: str
    title: str
    chunk_id: int
    text: str


def load_chunks_jsonl(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    doc_id=str(obj.get("doc_id", "")),
                    title=str(obj.get("title", "")),
                    chunk_id=int(obj.get("chunk_id", 0)),
                    text=text,
                )
            )
    if not chunks:
        raise ValueError(f"No chunks loaded from {path}")
    return chunks


def save_metadata(chunks: List[Chunk], out_path: str) -> None:
    meta: List[Dict[str, Any]] = []
    for c in chunks:
        meta.append(
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "chunk_id": c.chunk_id,
                "text": c.text,  # demo 方便：直接把 chunk 打出来
            }
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    in_path = "data/processed/chunks.jsonl"
    out_dir = "index"
    os.makedirs(out_dir, exist_ok=True)

    # ✅ 作业稳妥通用模型；你后面想“更医学”，我们再换成 PubMedBERT 类模型
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)

    chunks = load_chunks_jsonl(in_path)
    texts = [c.text for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,            
        show_progress_bar=True,
        normalize_embeddings=True  # cosine = dot
    )

    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings.astype(np.float32))
    save_metadata(chunks, os.path.join(out_dir, "metadata.json"))

    with open(os.path.join(out_dir, "model.txt"), "w", encoding="utf-8") as f:
        f.write(model_name + "\n")

    print(f"Built index: {len(chunks)} chunks")
    print(f"Saved to: {out_dir}\\embeddings.npy, {out_dir}\\metadata.json, {out_dir}\\model.txt")


if __name__ == "__main__":
    main()
