import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF


# ----------------------------
# Text cleaning / splitting
# ----------------------------

NOISE_PATTERNS = [
    # preprint boilerplate
    r"granted bioRxiv a license",
    r"display the preprint in perpetuity",
    r"all rights reserved\.? no reuse allowed without permission",
    r"was not certified by peer review",
    r"this preprint",
    r"doi:\s*10\.\d{4,9}/[-._;()/:a-z0-9]+",

    # obvious headers
    r"^\s*page\s*\d+\s*$",
    r"^\s*\d+\s*$",
]

def clean_text(t: str) -> str:
    # Keep it safe (do NOT do aggressive cross-paragraph deletes here)
    t = t.replace("\u00ad", "")  # soft hyphen
    t = re.sub(r"-\s*\n\s*", "", t)           # de-hyphen line breaks
    t = re.sub(r"[ \t]+", " ", t)             # collapse spaces
    t = re.sub(r"\n{3,}", "\n\n", t)          # collapse many newlines
    return t.strip()

def split_to_paragraphs(t: str) -> List[str]:
    # Split on blank lines; fallback to single newlines if needed
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    # If the PDF has no blank lines, split by newline blocks
    if len(paras) <= 2:
        paras = [p.strip() for p in t.split("\n") if p.strip()]
    return paras

def looks_like_reference_block(p: str) -> bool:
    # Conservative heuristic: many years + many commas
    years = len(re.findall(r"\b(19|20)\d{2}\b", p))
    commas = p.count(",")
    return (years >= 3 and commas >= 8)

def looks_like_affiliations(p: str) -> bool:
    # affiliations / author list blocks are noisy for search
    pl = p.lower()
    if any(k in pl for k in ["department of", "university", "hospital", "center", "institute"]):
        # many commas usually indicates long affiliation lists
        if p.count(",") >= 6:
            return True
        # common pattern like "USA. 18Department of ..."
        if re.search(r"\b(usa|uk|australia|canada|china|germany|france)\b", pl) and re.search(r"\bdepartment of\b", pl):
            return True
    return False

def is_noise_paragraph(p: str) -> bool:
    p_stripped = p.strip()
    if len(p_stripped) < 60:
        return True

    # paragraph-level pattern removals
    pl = p_stripped.lower()
    for pat in NOISE_PATTERNS:
        if re.search(pat, p_stripped, flags=re.IGNORECASE | re.MULTILINE):
            return True

    # too symbol-heavy (tables/garbage)
    non_alnum = sum(1 for ch in p_stripped if not ch.isalnum() and not ch.isspace())
    if non_alnum / max(1, len(p_stripped)) > 0.35:
        return True

    # references-like
    if looks_like_reference_block(p_stripped):
        return True

    # affiliation blocks
    if looks_like_affiliations(p_stripped):
        return True

    return False


def chunk_from_paragraphs(paras: List[str], min_len: int = 250, max_len: int = 1100) -> List[str]:
    """
    Merge consecutive paragraphs into chunks within [min_len, max_len].
    """
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append(" ".join(buf).strip())
            buf = []
            buf_len = 0

    for p in paras:
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue

        # If paragraph itself is very long, split it softly by sentences.
        if len(p) > max_len * 1.5:
            sents = re.split(r"(?<=[\.\?\!])\s+", p)
            for s in sents:
                s = s.strip()
                if not s:
                    continue
                if buf_len + len(s) + 1 <= max_len:
                    buf.append(s)
                    buf_len += len(s) + 1
                else:
                    if buf_len >= min_len:
                        flush()
                    buf.append(s)
                    buf_len = len(s)
            continue

        # Normal merge
        if buf_len + len(p) + 1 <= max_len:
            buf.append(p)
            buf_len += len(p) + 1
        else:
            if buf_len >= min_len:
                flush()
            else:
                # buffer too small but would overflow; flush anyway to avoid mega chunks
                flush()
            buf.append(p)
            buf_len = len(p)

    flush()

    # final cleanup: drop too-short chunks
    chunks = [c for c in chunks if len(c) >= min_len]
    return chunks


# ----------------------------
# PDF extraction
# ----------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        # "text" mode is generally best for scientific PDFs
        parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(parts)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare paragraph-based chunks from PDFs")
    parser.add_argument("--raw_dir", default="data/raw", help="Directory containing PDFs")
    parser.add_argument("--out_path", default="data/processed/chunks.jsonl", help="Output JSONL path")
    parser.add_argument("--min_len", type=int, default=250, help="Min chunk length (chars)")
    parser.add_argument("--max_len", type=int, default=1100, help="Max chunk length (chars)")
    parser.add_argument("--debug", action="store_true", help="Print per-file debug info")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(raw_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {raw_dir.resolve()}")

    all_chunks: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0
    errored = 0

    for f in pdf_files:
        try:
            text = extract_pdf_text(f)
            text = clean_text(text)

            if not text or len(text) < 200:
                skipped += 1
                if args.debug:
                    print(f"[DEBUG] {f.name}: skipped (no/short text)")
                continue

            paras_raw = split_to_paragraphs(text)
            paras = [p for p in paras_raw if not is_noise_paragraph(p)]

            chunks = chunk_from_paragraphs(paras, min_len=args.min_len, max_len=args.max_len)

            # doc_id/title: use filename stem
            # e.g. "071_Application of ..." from "071_Application of ....pdf"
            doc_id = f.stem
            title = f.stem.replace("_", " ")

            # ALWAYS print debug per file when --debug is enabled
            if args.debug:
                print(f"[DEBUG] {f.name}: paras={len(paras)} (raw={len(paras_raw)}), chunks={len(chunks)}")

            # build records
            for chunk_id, chunk_text in enumerate(chunks):
                all_chunks.append(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    }
                )

            processed += 1

        except Exception as e:
            errored += 1
            if args.debug:
                print(f"[DEBUG] {f.name}: error={type(e).__name__}: {e}")
            continue

    # write jsonl
    with open(out_path, "w", encoding="utf-8") as w:
        for obj in all_chunks:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Processed PDFs: {processed}/{len(pdf_files)}")
    print(f"Wrote chunks: {len(all_chunks)} -> {out_path.as_posix()}")

    if skipped or errored:
        print("Some PDFs likely had no extractable text (scanned images) or extraction was too short.")
        if args.debug:
            print(f"[DEBUG] skipped={skipped}, errored={errored}")


if __name__ == "__main__":
    main()
