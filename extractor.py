"""Extract elements and their requirements from a large PDF with Gemma 3 (GGUF).
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
import yaml, json, sys

import pymupdf as fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from pathlib import Path

# ============================ CONFIG ============================
PDF_PATH = Path("Documents/docker-cis-v1.pdf")        # <-- path to your PDF (~ works)
QUESTION = "List every element and the requirements for each one."
OUTPUT_PATH = PDF_PATH.with_suffix(".json")         # saves next to the PDF; set to None to skip

REPO = "unsloth/gemma-3-1b-it-GGUF"  # 4B+ supports 128k context; 1B only 32k
FILE = "*Q8_0.gguf"
N_CTX = 32768         # raise (e.g. 65536) if you have the RAM
ANSWER_TOKENS = 5000  # room reserved for the model's JSON reply
CHUNK_TOKENS = 9000  # chunk size when the PDF doesn't fit in one pass
CHUNK_OVERLAP = 1000   # tokens shared between neighbouring chunks
# ================================================================

# The model is forced (via llama.cpp grammar) to return JSON matching this schema.
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


def load_model():
    return Llama.from_pretrained(
        repo_id=REPO,
        filename=FILE,
        n_ctx=N_CTX,
        n_gpu_layers=-1,   # Metal on Apple Silicon
        flash_attn=True,
        verbose=False,
    )


def extract_text(path):
    doc = fitz.open(path)
    pages = [f"[Page {i + 1}]\n{page.get_text()}" for i, page in enumerate(doc)]
    text = "\n\n".join(pages)
    if not "".join(p.split("\n", 1)[-1] for p in pages).strip():
        sys.exit("No text found. The PDF is probably scanned images and needs OCR first.")
    return text, len(pages)


def ntokens(llm, text):
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


def split_with_langchain(llm, text):
    """Split text into chunks measured in Gemma tokens, preferring page, then paragraph boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=lambda t: ntokens(llm, t),
        separators=["\n\n[Page ", "\n\n", "\n", ". ", " ", ""],
        keep_separator="start",
    )
    return splitter.split_text(text)


def extract_elements(llm, chunk):
    """Ask Gemma for the elements/requirements in one chunk; returns a list of dicts."""
    r = llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": (
                f"Document text:\n\n{chunk}\n\n"
                f"Task: {QUESTION}\n"
                "Use only this text. Copy requirements faithfully and keep each one short. "
                'If the text contains no elements, return {"elements": []}.'
            ),
        }],
        response_format={"type": "json_object", "schema": SCHEMA},
        temperature=0.2,
        max_tokens=ANSWER_TOKENS,
    )
    content = r["choices"][0]["message"]["content"]
    try:
        return json.loads(content).get("elements", [])
    except json.JSONDecodeError:
        # Usually means the reply hit max_tokens; raise ANSWER_TOKENS or lower CHUNK_TOKENS.
        print("Warning: a chunk returned incomplete JSON. Restarting", file=sys.stderr)
        return extract_elements(llm, chunk)  # retry


def merge(results):
    """Merge elements found across chunks: same name -> combined, de-duplicated requirements."""
    merged = {}
    for el in results:
        name = el.get("name", "").strip()
        if not name:
            continue
        key = name.lower()
        entry = merged.setdefault(key, {"name": name, "requirements": []})
        seen = {r.lower() for r in entry["requirements"]}
        for req in el.get("requirements", []):
            req = req.strip()
            if req and req.lower() not in seen:
                entry["requirements"].append(req)
                seen.add(req.lower())
    return {v["name"]: v for v in merged.values()}


def main():
    llm = load_model()
    text, n_pages = extract_text(PDF_PATH)
    total = ntokens(llm, text)
    print(f"Document: {n_pages} pages, ~{total:,} tokens", file=sys.stderr)

    if total + ANSWER_TOKENS + 500 < N_CTX:
        chunks = [text]
    else:
        chunks = split_with_langchain(llm, text)
        print(f"Too long for one pass; split into {len(chunks)} chunks.", file=sys.stderr)

    found = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Reading chunk {i}/{len(chunks)}...", file=sys.stderr)
        found.extend(extract_elements(llm, chunk))

    result = merge(found)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if OUTPUT_PATH:
        out_path = Path(OUTPUT_PATH).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            yaml.safe_dump(result, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()