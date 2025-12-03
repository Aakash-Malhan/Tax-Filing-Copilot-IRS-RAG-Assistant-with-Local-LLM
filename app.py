import json
from pathlib import Path
from collections import defaultdict

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# PATHS / CONSTANTS #

KB_PATH = Path("data/processed/tax_kb_chunks.jsonl")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Globals so we only load models once per Space
_records = None
_embed_model = None
_embeddings = None
_tokenizer = None
_llm_model = None
_device = None


# KB + RETRIEVAL HELPERS #

def load_kb():
    global _records
    if _records is None:
        chunks = []
        with open(KB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        print(f"Loaded {len(chunks)} chunks from {KB_PATH}")
        _records = chunks
    return _records


def build_embeddings():
    global _embed_model, _embeddings
    if _embed_model is None or _embeddings is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        texts = [r["text"] for r in load_kb()]
        print(f"Encoding {len(texts)} chunks...")
        _embeddings = _embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return _embed_model, _embeddings


def retrieve_top_k(query, k=3):
    records = load_kb()
    embed_model, embeddings = build_embeddings()

    query_emb = embed_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]

    scores = np.dot(embeddings, query_emb)
    top_idx = np.argsort(-scores)[:k]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        r = records[idx]
        results.append(
            {
                "rank": rank,
                "score": float(scores[idx]),
                "source_name": r["source_name"],
                "page_number": r["page_number"],
                "doc_type": r.get("doc_type", "unknown"),
                "text": r["text"],
            }
        )
    return results


def build_context_and_citations(results):
    context_blocks = []
    by_source = defaultdict(set)

    for r in results:
        header = f"{r['source_name']} (page {r['page_number']})"
        block = f"SOURCE: {header}\nCONTENT:\n{r['text']}"
        context_blocks.append(block)
        by_source[r["source_name"]].add(r["page_number"])

    context_str = "\n\n-----\n\n".join(context_blocks)

    parts = []
    for src, pages in by_source.items():
        pages_sorted = sorted(pages)
        parts.append(f"{src} (pp. {', '.join(str(p) for p in pages_sorted)})")
    citation_str = "; ".join(parts)

    return context_str, citation_str


# LOCAL LLM HELPERS #

def load_local_llm():
    global _tokenizer, _llm_model, _device
    if _tokenizer is None or _llm_model is None:
        print(f"Loading local LLM: {LLM_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        _llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _llm_model.to(_device)

    return _tokenizer, _llm_model, _device


def generate_answer_local(question, context, citation_str):
    tokenizer, model, device = load_local_llm()

    prompt = f"""You are a careful assistant that answers questions about U.S. federal individual income tax
for individuals. Use ONLY the information in the provided IRS text. If something is unclear or not covered,
say that clearly and recommend consulting the full IRS publication or a tax professional.
Context from IRS documents:
{context}
User question:
{question}
Sources summary:
{citation_str}
Answer clearly and concisely (2–4 short paragraphs or bullet points). Start your reply after 'Answer:'.
Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1800,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in full_text:
        answer = full_text.rsplit("Answer:", 1)[1].strip()
    else:
        answer = full_text[-1000:].strip()

    return answer


# RAG PIPELINE FOR GRADIO #

def run_rag(question: str, k: int = 3):
    question = question.strip()
    if not question:
        return "Please enter a question.", "", ""

    results = retrieve_top_k(question, k=k)
    context_str, citation_str = build_context_and_citations(results)
    answer = generate_answer_local(question, context_str, citation_str)

    # Build markdown for retrieved context
    ctx_md_parts = []
    for r in results:
        header = f"**{r['source_name']} – page {r['page_number']}**  \n"
        meta = f"_score={r['score']:.3f}, type={r['doc_type']}_  \n\n"
        ctx_md_parts.append(header + meta + r["text"] + "\n\n---\n\n")
    ctx_md = "".join(ctx_md_parts)

    return answer, f"**Sources:** {citation_str}", ctx_md


# GRADIO UI #

def gradio_answer(question, k):
    answer, sources, ctx_md = run_rag(question, int(k))
    return answer, sources, ctx_md


with gr.Blocks() as demo:
    gr.Markdown(
        """
# Tax Filing Copilot (IRS RAG, Local LLM)
Ask questions about U.S. federal individual income tax.  
This app uses a retrieval-augmented generation pipeline over official IRS PDFs
(Form 1040 instructions, Pub 17, Pub 502, Pub 503, Pub 596, Pub 970, etc.)
and a local open-source language model (TinyLlama) running on Hugging Face Spaces.
        """
    )

    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            lines=4,
            placeholder="e.g. What medical expenses are deductible?",
        )
        k_slider = gr.Slider(
            label="Number of passages to retrieve (K)",
            minimum=1,
            maximum=5,
            value=3,
            step=1,
        )

    ask_btn = gr.Button("Ask", variant="primary")

    answer_box = gr.Markdown(label="Answer")
    sources_box = gr.Markdown(label="Sources")
    context_box = gr.Markdown(label="Retrieved context")

    ask_btn.click(
        fn=gradio_answer,
        inputs=[question, k_slider],
        outputs=[answer_box, sources_box, context_box],
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
