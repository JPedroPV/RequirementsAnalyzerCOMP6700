"""Extract elements and their requirements from a large PDF with Gemma 3 (GGUF).
Chunks are processed in parallel: each worker process loads its own copy of the model
(a llama.cpp model can't serve two requests at once, so threads sharing one won't help).
LangChain is used only for chunking (RecursiveCharacterTextSplitter).

Setup:
    pip install llama-cpp-python huggingface-hub pymupdf langchain-text-splitters

Usage:
    Fill in the CONFIG section below, then run:
    python pdf_ask_chunked.py

Output (printed and saved to OUTPUT_PATH):
    {
      "Element name": {
        "name": "Element name",
        "requirements": ["req 1", "req 2"]
      },
      ...
    }
"""
import json, yaml, sys
import multiprocessing as mp
from pathlib import Path

import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama

# ============================ CONFIG ============================
PDF_PATH = Path("Documents/docker-cis-v2.pdf")
QUESTION = "List every element and the requirements for each one."
OUTPUT_PATH = PDF_PATH.with_suffix(".yaml")   # set to None to skip saving

REPO = "unsloth/gemma-3-1b-it-GGUF"
FILE = "*BF16.gguf"
ANSWER_TOKENS = 4000
CHUNK_TOKENS = 8000
CHUNK_OVERLAP = 800
N_CTX = 16384         # per worker: one chunk + prompt + answer fits easily
WORKERS = 5           # model copies on the GPU (~3-4 GB each); tune by watching nvidia-smi
MAX_RETRIES = 5       # attempts per chunk before giving up on it
# ================================================================

SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "requirements": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "requirements"],
            },
        }
    },
    "required": ["elements"],
}

# --------------------------- worker side ---------------------------
_llm = None  # one model per worker process


def init_worker(model_path):
    global _llm
    _llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_gpu_layers=-1,
        flash_attn=True,
        n_batch=2048,
        n_ubatch=2048,
        verbose=False,
    )


def _ask(chunk):
    prompt = (
        f"Document text:\n\n{chunk}\n\n"
        f"Task: {QUESTION}\n"
        "Use only this text. Copy requirements faithfully and keep each one short. "
        "List each element once and never repeat a requirement. "
        'If the text contains no elements, return {"elements": []}.'
    )
    r = _llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object", "schema": SCHEMA},
        temperature=0.2,
        repeat_penalty=1.15,      # discourages the repetition loops
        max_tokens=ANSWER_TOKENS,
    )
    choice = r["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]


def _extract(chunk, idx, depth=0):
    for attempt in range(1, MAX_RETRIES + 1):
        content, reason = _ask(chunk)
        try:
            return json.loads(content).get("elements", [])
        except json.JSONDecodeError:
            print(f"Warning: chunk {idx + 1} incomplete JSON "
                  f"(finish_reason={reason}, attempt {attempt}).", file=sys.stderr)
            if reason == "length" and depth < 3 and len(chunk) > 2000:
                # Too much to say for one reply: split at a line break near the middle.
                mid = chunk.rfind("\n", 0, len(chunk) // 2) or len(chunk) // 2
                return _extract(chunk[:mid], idx, depth + 1) + _extract(chunk[mid:], idx, depth + 1)
    print(f"Error: chunk {idx + 1} skipped after {MAX_RETRIES} attempts.", file=sys.stderr)
    return []


def extract_elements(job):
    idx, chunk = job
    return idx, _extract(chunk, idx)


# ---------------------------- main side ----------------------------
def extract_text(path):
    doc = fitz.open(path)
    pages = [f"[Page {i + 1}]\n{page.get_text()}" for i, page in enumerate(doc)]
    if not "".join(p.split("\n", 1)[-1] for p in pages).strip():
        sys.exit("No text found. The PDF is probably scanned images and needs OCR first.")
    return "\n\n".join(pages), len(pages)


def split_text(tok, text):
    """Split into chunks measured in Gemma tokens, preferring page, then paragraph boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda t: len(tok.tokenize(t.encode("utf-8"), add_bos=False)),
        separators=["\n\n[Page ", "\n\n", "\n", ". ", " ", ""],
        keep_separator="start",
    )
    return splitter.split_text(text)


def merge(results):
    """Same name -> combined, de-duplicated requirements."""
    merged = {}
    for el in results:
        name = el.get("name", "").strip()
        if not name:
            continue
        entry = merged.setdefault(name.lower(), {"name": name, "requirements": []})
        seen = {r.lower() for r in entry["requirements"]}
        for req in el.get("requirements", []):
            req = req.strip()
            if req and req.lower() not in seen:
                entry["requirements"].append(req)
                seen.add(req.lower())
    return {v["name"]: v for v in merged.values()}


def main():
    # Tokenizer only (no weights on the GPU): downloads the file once and gives its path.
    tok = Llama.from_pretrained(repo_id=REPO, filename=FILE, vocab_only=True, verbose=False)
    model_path = tok.model_path

    text, n_pages = extract_text(PDF_PATH)
    total = len(tok.tokenize(text.encode("utf-8"), add_bos=False))
    print(f"Document: {n_pages} pages, ~{total:,} tokens", file=sys.stderr)

    chunks = [text] if total <= CHUNK_TOKENS else split_text(tok, text)
    workers = min(WORKERS, len(chunks))
    print(f"{len(chunks)} chunk(s) across {workers} worker(s).", file=sys.stderr)

    # "spawn" is required: CUDA can't be shared with forked child processes.
    ctx = mp.get_context("spawn")
    results = {}
    with ctx.Pool(workers, initializer=init_worker, initargs=(model_path,)) as pool:
        for idx, elements in pool.imap_unordered(extract_elements, enumerate(chunks)):
            results[idx] = elements
            print(f"Finished {len(results)}/{len(chunks)} (chunk {idx + 1})", file=sys.stderr)

    # Merge in document order so element/requirement order is stable run to run.
    found = [el for i in sorted(results) for el in results[i]]
    result = merge(found)

    out = json.dumps(result, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if OUTPUT_PATH:
        out_path = Path(OUTPUT_PATH).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            yaml.safe_dump(result, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        print(f"Saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()