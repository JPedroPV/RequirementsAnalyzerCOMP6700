"""Extract elements and their requirements from a large PDF with Gemma 3 (GGUF).
Chunks are processed in parallel: each worker process loads its own copy of the model
(a llama.cpp model can't serve two requests at once, so threads sharing one won't help).
LangChain is used only for chunking (RecursiveCharacterTextSplitter).

Setup:
    pip install llama-cpp-python huggingface-hub pymupdf langchain-text-splitters

Usage:
    Fill in the CONFIG section below, then run:
    python extractor.py                                  # uses PDF_PATH / PROMPT_TYPE
    python extractor.py --prompt one-shot
    python extractor.py --prompt chain-of-thought --pdf Documents/docker-cis-v1.pdf
    python extractor.py --prompt one-shot --example-pdf ex.pdf --example-yaml ex.yaml

One-shot example (--example-pdf + --example-yaml; otherwise a built-in example is used):
    The PDF is the example input: keep it short (a page or two). Its text has to fit in the
    model's context next to a document chunk, so chunks shrink to make room for it.
    The YAML is the expected output for that PDF, in the same format this script writes:
        Insecure Registries:
          name: Insecure Registries
          requirements:
          - The Docker daemon must not be configured to use insecure registries.

Prompt types:
    zero-shot         instructions only
    one-shot          instructions + one worked example (example PDF text -> its YAML, as JSON)
    chain-of-thought  model writes brief step-by-step reasoning in a "reasoning" field
                      before the "elements" list; the reasoning is discarded afterwards

Every LLM call (prompt + raw reply) is also logged to
OUTPUT_DIR/<pdf name>-<prompt type>-llm-outputs.txt by dump_llm_outputs().

Output (printed and saved to OUTPUT_DIR/<pdf name>-<prompt type>-kdes.yaml):
    {
      "Element name": {
        "name": "Element name",
        "requirements": ["req 1", "req 2"]
      },
      ...
    }
"""
import argparse, json, yaml, sys
import multiprocessing as mp
from pathlib import Path

import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama

# ============================ CONFIG ============================
PDF_PATH = Path("Documents/docker-cis-v2.pdf")
QUESTION = "List every element and the requirements for each one."
PROMPT_TYPE = "zero-shot"            # "zero-shot" | "one-shot" | "chain-of-thought"
OUTPUT_DIR = Path("Extractor_Outputs")   # set to None to skip saving

REPO = "unsloth/gemma-3-1b-it-GGUF"
FILE = "*BF16.gguf"
ANSWER_TOKENS = 4000
CHUNK_TOKENS = 8000
CHUNK_OVERLAP = 800
N_CTX = 16384         # per worker: one chunk + prompt + answer fits easily
WORKERS = 5           # model copies on the GPU (~3-4 GB each); tune by watching nvidia-smi
MAX_RETRIES = 5       # attempts per chunk before giving up on it
# ================================================================

ELEMENTS_SCHEMA = {
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

SCHEMA = {
    "type": "object",
    "properties": {"elements": ELEMENTS_SCHEMA},
    "required": ["elements"],
}

# Chain of thought: "reasoning" comes first so the model thinks before it answers.
COT_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "elements": ELEMENTS_SCHEMA},
    "required": ["reasoning", "elements"],
}

# ----------------------------- prompts -----------------------------
RULES = (
    "Use only this text. Copy requirements faithfully and keep each one short. "
    "List each element once and never repeat a requirement. "
    'If the text contains no elements, return {"elements": []}.'
)

EXAMPLE_TEXT = """\
[Page 20]
2.3 Ensure the default ulimit is configured appropriately (Manual)
Profile Applicability:
• Level 1 - Docker - Linux
Description:
Set the default ulimit options as appropriate in your environment.
Remediation:
Run Docker in daemon mode and pass --default-ulimit as an argument with the
respective ulimits as appropriate in your environment.
2.4 Ensure that the Docker daemon is not configured to use insecure registries (Automated)
Description:
Docker considers a private registry either secure or insecure. You should not use
insecure registries.
Audit:
Verify that no insecure registries are in use."""

EXAMPLE_OUTPUT = {
    "elements": [
        {
            "name": "Default ulimit",
            "requirements": [
                "Set the default ulimit options as appropriate in your environment.",
                "Pass --default-ulimit as an argument when running the Docker daemon.",
            ],
        },
        {
            "name": "Insecure Registries",
            "requirements": [
                "The Docker daemon must not be configured to use insecure registries.",
            ],
        },
    ]
}


def build_zero_shot_prompt(chunk):
    """Instructions only, no examples."""
    return f"Document text:\n\n{chunk}\n\nTask: {QUESTION}\n{RULES}"


def load_example(pdf_path, yaml_path):
    """Load a one-shot example: the PDF's text (input) and its expected-output YAML.

    The YAML uses this script's own output format: {key: {name: str, requirements: [str]}}.
    Returns (example_text, {"elements": [...]}), the shape the model is asked to produce.
    """
    pdf_path, yaml_path = Path(pdf_path).expanduser(), Path(yaml_path).expanduser()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"example PDF not found or not a .pdf file: {pdf_path}")
    if not yaml_path.is_file() or yaml_path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"example YAML not found or not a .yaml/.yml file: {yaml_path}")

    text, _ = extract_text(pdf_path)
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ValueError(f"example YAML is not valid: {err}") from err
    if not isinstance(data, dict) or not data:
        raise ValueError("example YAML must map element names to {name, requirements}")

    elements = []
    for key, el in data.items():
        reqs = el.get("requirements") if isinstance(el, dict) else None
        if not (isinstance(reqs, list) and reqs and all(isinstance(r, str) for r in reqs)):
            raise ValueError(f"example element '{key}' needs a non-empty "
                             "'requirements' list of strings")
        name = el.get("name", key)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"example element '{key}' has an invalid 'name'")
        elements.append({"name": name.strip(), "requirements": [r.strip() for r in reqs]})
    return text, {"elements": elements}


def build_one_shot_prompt(chunk, example_text=EXAMPLE_TEXT, example_output=EXAMPLE_OUTPUT):
    """Instructions plus one worked example showing the expected element/requirement split."""
    return (
        f"Task: {QUESTION}\n"
        "An element is a configurable item, component or setting that the document "
        "places requirements on. Here is one example.\n\n"
        f"Example text:\n\n{example_text}\n\n"
        f"Example answer:\n{json.dumps(example_output, indent=2, ensure_ascii=False)}\n\n"
        "Now do the same for the document text below. Do not copy the example's elements "
        "unless they also appear in this text.\n\n"
        f"Document text:\n\n{chunk}\n\n"
        f"Task: {QUESTION}\n{RULES}"
    )


def build_cot_prompt(chunk):
    """Asks for step-by-step reasoning (in the "reasoning" field) before the answer."""
    return (
        f"Document text:\n\n{chunk}\n\n"
        f"Task: {QUESTION}\n"
        "Think step by step before answering, and write those steps in the \"reasoning\" "
        "field, briefly:\n"
        "1. Find each numbered recommendation or section heading in the text.\n"
        "2. For each one, decide which element (component, file, setting or feature) it is about.\n"
        "3. Collect the requirements for that element from its title, description, audit "
        "and remediation text.\n"
        "4. Merge recommendations that are about the same element.\n"
        "Then put the final answer in the \"elements\" field.\n"
        f"{RULES}"
    )


PROMPT_BUILDERS = {
    "zero-shot": build_zero_shot_prompt,
    "one-shot": build_one_shot_prompt,
    "chain-of-thought": build_cot_prompt,
}

# --------------------------- worker side ---------------------------
_llm = None          # one model per worker process
_prompt_type = None  # set per worker: globals aren't inherited under "spawn"
_prompt_kwargs = {}  # extra builder arguments (the one-shot example)


def init_worker(model_path, prompt_type, prompt_kwargs):
    global _llm, _prompt_type, _prompt_kwargs
    _prompt_type = prompt_type
    _prompt_kwargs = prompt_kwargs
    _llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_gpu_layers=-1,
        flash_attn=True,
        n_batch=2048,
        n_ubatch=2048,
        verbose=False,
    )


def _ask(chunk, log):
    """One model call. Appends {"prompt", "output"} to `log` so every call can be dumped later."""
    prompt = PROMPT_BUILDERS[_prompt_type](chunk, **_prompt_kwargs)
    schema = COT_SCHEMA if _prompt_type == "chain-of-thought" else SCHEMA
    r = _llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object", "schema": schema},
        temperature=0.2,
        repeat_penalty=1.15,      # discourages the repetition loops
        max_tokens=ANSWER_TOKENS,
    )
    choice = r["choices"][0]
    log.append({"prompt": prompt, "output": choice["message"]["content"]})
    return choice["message"]["content"], choice["finish_reason"]


def _extract(chunk, idx, log, depth=0):
    for attempt in range(1, MAX_RETRIES + 1):
        content, reason = _ask(chunk, log)
        try:
            return json.loads(content).get("elements", [])
        except json.JSONDecodeError:
            print(f"Warning: chunk {idx + 1} incomplete JSON "
                  f"(finish_reason={reason}, attempt {attempt}).", file=sys.stderr)
            if reason == "length" and depth < 3 and len(chunk) > 2000:
                # Too much to say for one reply: split at a line break near the middle.
                mid = chunk.rfind("\n", 0, len(chunk) // 2) or len(chunk) // 2
                return (_extract(chunk[:mid], idx, log, depth + 1)
                        + _extract(chunk[mid:], idx, log, depth + 1))
    print(f"Error: chunk {idx + 1} skipped after {MAX_RETRIES} attempts.", file=sys.stderr)
    return []


def extract_elements(job):
    idx, chunk = job
    log = []
    return idx, _extract(chunk, idx, log), log


# ---------------------------- main side ----------------------------
def extract_text(path):
    doc = fitz.open(path)
    pages = [f"[Page {i + 1}]\n{page.get_text()}" for i, page in enumerate(doc)]
    if not "".join(p.split("\n", 1)[-1] for p in pages).strip():
        raise ValueError(f"no text found in {path}. "
                         "The PDF is probably scanned images and needs OCR first.")
    return "\n\n".join(pages), len(pages)


def split_text(tok, text, chunk_tokens=CHUNK_TOKENS):
    """Split into chunks measured in Gemma tokens, preferring page, then paragraph boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=min(CHUNK_OVERLAP, chunk_tokens // 10),
        length_function=lambda t: len(tok.tokenize(t.encode("utf-8"), add_bos=False)),
        separators=["\n\n[Page ", "\n\n", "\n", ". ", " ", ""],
        keep_separator="start",
    )
    return splitter.split_text(text)


def dump_llm_outputs(records, out_path):
    """Write every LLM call to a text file, one block per call:

        *LLM Name*      model used
        *Prompt Used*   full prompt sent (including the document chunk)
        *Prompt Type*   zero-shot / one-shot / chain-of-thought
        *LLM Output*    raw model reply

    `records` is a list of dicts with keys "llm", "prompt", "prompt_type" and "output".
    """
    blocks = [
        f"*LLM Name*\n{r['llm']}\n\n"
        f"*Prompt Used*\n{r['prompt']}\n\n"
        f"*Prompt Type*\n{r['prompt_type']}\n\n"
        f"*LLM Output*\n{r['output']}\n"
        for r in records
    ]
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(("\n" + "=" * 80 + "\n\n").join(blocks), encoding="utf-8")
    return out_path


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


def parse_args():
    p = argparse.ArgumentParser(description="Extract key data elements from a PDF with Gemma 3.")
    p.add_argument("--pdf", type=Path, default=PDF_PATH, help=f"input PDF (default: {PDF_PATH})")
    p.add_argument("--prompt", choices=PROMPT_BUILDERS, default=PROMPT_TYPE,
                   help=f"prompting strategy (default: {PROMPT_TYPE})")
    p.add_argument("--example-pdf", type=Path,
                   help="one-shot example input PDF (use with --example-yaml)")
    p.add_argument("--example-yaml", type=Path,
                   help="expected output YAML for --example-pdf, in this script's output format")
    args = p.parse_args()
    if bool(args.example_pdf) != bool(args.example_yaml):
        p.error("--example-pdf and --example-yaml must be given together")
    if args.example_pdf and args.prompt != "one-shot":
        p.error("--example-pdf/--example-yaml can only be used with --prompt one-shot")
    return args


def main():
    args = parse_args()
    pdf_path, prompt_type = args.pdf, args.prompt

    prompt_kwargs = {}
    if args.example_pdf:
        try:
            ex_text, ex_output = load_example(args.example_pdf, args.example_yaml)
        except ValueError as err:
            sys.exit(f"Error: {err}")
        prompt_kwargs = {"example_text": ex_text, "example_output": ex_output}
        print(f"One-shot example: {args.example_pdf} -> {args.example_yaml}", file=sys.stderr)

    # Tokenizer only (no weights on the GPU): downloads the file once and gives its path.
    tok = Llama.from_pretrained(repo_id=REPO, filename=FILE, vocab_only=True, verbose=False)
    model_path = tok.model_path

    # Whatever the prompt adds besides the chunk (instructions, example) eats into the
    # chunk's share of the context, so shrink chunks when the one-shot example is large.
    n_tokens = lambda s: len(tok.tokenize(s.encode("utf-8"), add_bos=False))
    builder = PROMPT_BUILDERS[prompt_type]
    overhead = n_tokens(builder("", **prompt_kwargs)) + 200  # + chat template / safety margin
    chunk_tokens = min(CHUNK_TOKENS, N_CTX - ANSWER_TOKENS - overhead)
    if chunk_tokens < 1000:
        sys.exit(f"Error: the prompt without the document is ~{overhead:,} tokens, leaving no "
                 f"room for document text in N_CTX={N_CTX:,}. Use a shorter example PDF.")
    if chunk_tokens < CHUNK_TOKENS:
        print(f"Example is large: chunks reduced to {chunk_tokens:,} tokens.", file=sys.stderr)

    try:
        text, n_pages = extract_text(pdf_path)
    except ValueError as err:
        sys.exit(f"Error: {err}")
    total = n_tokens(text)
    print(f"Document: {n_pages} pages, ~{total:,} tokens. Prompt: {prompt_type}", file=sys.stderr)

    chunks = [text] if total <= chunk_tokens else split_text(tok, text, chunk_tokens)
    workers = min(WORKERS, len(chunks))
    print(f"{len(chunks)} chunk(s) across {workers} worker(s).", file=sys.stderr)

    # "spawn" is required: CUDA can't be shared with forked child processes.
    ctx = mp.get_context("spawn")
    results, logs = {}, {}
    with ctx.Pool(workers, initializer=init_worker, initargs=(model_path, prompt_type, prompt_kwargs)) as pool:
        for idx, elements, log in pool.imap_unordered(extract_elements, enumerate(chunks)):
            results[idx], logs[idx] = elements, log
            print(f"Finished {len(results)}/{len(chunks)} (chunk {idx + 1})", file=sys.stderr)

    # Merge in document order so element/requirement order is stable run to run.
    found = [el for i in sorted(results) for el in results[i]]
    result = merge(found)

    llm_name = f"{REPO} ({Path(model_path).name})"
    records = [{"llm": llm_name, "prompt_type": prompt_type, **call}
               for i in sorted(logs) for call in logs[i]]

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if OUTPUT_DIR:
        stem = Path(OUTPUT_DIR).expanduser() / f"{pdf_path.stem}-{prompt_type}"
        out_path = stem.with_name(stem.name + "-kdes.yaml")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            yaml.safe_dump(result, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        print(f"Saved to {out_path}", file=sys.stderr)
        log_path = dump_llm_outputs(records, stem.with_name(stem.name + "-llm-outputs.txt"))
        print(f"LLM log ({len(records)} call(s)) saved to {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()