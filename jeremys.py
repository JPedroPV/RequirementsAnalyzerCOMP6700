import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader

def chunk_pdf(pdf_path, chunk_size=12000, overlap=200):
    reader = PdfReader(pdf_path)

    # Extract text from all pages
    pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]
    text = "\n".join(pages)

    # Split text into overlapping chunks
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


chunks = chunk_pdf("./Documents/docker-cis-v0.pdf")

print(f"Created {len(chunks)} chunks")
print(chunks[1])

# Hugging Face model ID
model_id = "google/gemma-3-1b-it"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model and automatically select available device(s)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)

model.eval()

print("Gemma 3 1B loaded successfully!")


messages = [
    {
        "role": "user",
        "content": f"""Explain this pdf for me: {chunks[1]}"""
    }
]

# Convert chat messages into model input
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Generate a response
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1500000,
        do_sample=False,
    )

# Decode only the newly generated tokens
new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True,
)



print("Gemma:", response)