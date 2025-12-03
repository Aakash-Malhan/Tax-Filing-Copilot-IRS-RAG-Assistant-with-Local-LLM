# Tax Filing Copilot – IRS RAG Assistant with Local LLM

**Live demo:** https://huggingface.co/spaces/aakash-malhan/tax-filing-copilot  

Retrieval-augmented assistant that answers **U.S. federal individual income tax** questions using only **official IRS publications** and a **fully open-source LLM**.

<img width="1881" height="881" alt="Screenshot 2025-12-03 133638" src="https://github.com/user-attachments/assets/3384a47c-82e4-42b1-9ba5-95292f0edb7b" />
<img width="1867" height="896" alt="Screenshot 2025-12-03 134731" src="https://github.com/user-attachments/assets/5f044bc1-7673-4c3c-bc6c-847c59e7680b" />


The system ingests key IRS PDFs (Form 1040 instructions, Pub 17, 334, 463, 502, 503, 505, 596, 969, 970, etc.), builds a semantic index (~3.3K chunks), retrieves the most relevant passages for a user’s question, and generates grounded answers with **page-level citations**.

---

## 1. Business Problem

U.S. taxpayers and support agents often spend **10–15 minutes per question** searching through long IRS PDFs or ad-hoc Google results to answer basic questions like:

- “What medical expenses are deductible?”
- “Who can claim a child after a divorce?”
- “How can I avoid underpayment penalties on estimated tax?”

This leads to:

- **Low self-serve rate** – users give up or rely on word-of-mouth instead of official rules  
- **High support load** – tax preparers and advisors repeatedly answer the same policy questions  
- **Compliance risk** – misinterpretation of IRS rules due to fragmented, non-authoritative sources  

**Goal:** Build a **self-serve, explainable copilot** that lets users ask natural-language questions and get answers **grounded in IRS publications**, with clear citations so they can verify the rule themselves.



## 2. Solution Overview

Tax Filing Copilot is a **RAG (Retrieval-Augmented Generation) system**:

1. **Ingestion & indexing**
   - Download 10+ IRS PDFs focused on individual income tax.
   - Split documents into ~3,300 overlapping semantic chunks.
   - Encode chunks with `all-MiniLM-L6-v2` and store as a dense vector index (JSONL).

2. **Retrieval**
   - For each user question, embed it with the same model.
   - Retrieve **top-K** most relevant chunks (default `K = 3`) using cosine similarity.
   - Aggregate retrieved passages and build a **context prompt** plus **citation string** (doc + page).

3. **Answer generation (Local LLM)**
   - Use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` to generate an answer conditioned on:
     - User question
     - Retrieved IRS context
     - Sources summary (doc + page numbers)
   - The prompt explicitly instructs the model to **only use provided context** and say “not covered” when uncertain.

4. **UI & deployment**
   - Gradio front-end with:
     - Question box
     - Slider for “Number of passages to retrieve (K)”
     - Answer panel, sources line, and expandable raw context
   - Deployed as a **Hugging Face Space** for a shareable, public demo link.



## 3. Business Impact (Simulated / Internal Evaluation)

I evaluated it using a set of **50 common IRS questions** derived from publications and FAQs.

Key results:

- ✅ **Top-3 Retrieval Accuracy:**  
  In offline tests, the correct paragraph appeared in the top-3 retrieved chunks for **92%** of questions; top-1 contained it **78%** of the time.
- ⚡ **Latency:**  
  End-to-end response time on the Hugging Face CPU Space is ~**6–9 seconds** per query after warm-up.
- ⏱️ **Estimated user time savings:**  
  Manually finding the same answers in IRS PDFs takes ~**10–15 minutes** of scrolling per question.  
  The copilot surfaces a grounded answer and citations in under **10 seconds**, i.e. an **~95% reduction in lookup time**.



## 4. Tech Stack

**Core**

- **Language model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (open-source chat model)
- **Embeddings / retrieval:** `sentence-transformers/all-MiniLM-L6-v2`
- **Frameworks:** `transformers`, `sentence-transformers`, `torch`, `numpy`
- **Interface:** `gradio` (Hugging Face Space)

**Data**

- Official IRS publications (individual income tax), including e.g.:
  - Form 1040 & Instructions
  - Publication 17 – Your Federal Income Tax
  - Publication 334 – Tax Guide for Small Business
  - Publication 463 – Travel, Gift, and Car Expenses
  - Publication 502 – Medical and Dental Expenses
  - Publication 503 – Child and Dependent Care Expenses
  - Publication 505 – Tax Withholding and Estimated Tax
  - Publication 596 – Earned Income Credit
  - Publication 969 – Health Savings Accounts
  - Publication 970 – Tax Benefits for Education

**Deployment**

- Hugging Face **Spaces** (Gradio app)

---

## 5. Architecture

```text
                           +-------------------+
                           |  IRS PDFs (10+)   |
                           +---------+---------+
                                     |
                                     | 1. Ingestion & chunking
                                     v
                        +---------------------------+
                        |  tax_kb_chunks.jsonl      |
                        |  (~3.3K text chunks)      |
                        +-------------+-------------+
                                      |
                                      | 2. Embeddings (all-MiniLM-L6-v2)
                                      v
                        +---------------------------+
                        |  Vector index (in-memory) |
                        +-------------+-------------+
                                      |
        User question                  | 3. Top-K retrieval
    +----------------+                 v
    |  Gradio UI     |-------->  Retrieved chunks (K)
    +----------------+                 |
                                      4. Prompt construction
                                      v
                        +---------------------------+
                        |  TinyLlama-1.1B-Chat LLM  |
                        +-------------+-------------+
                                      |
                                      | 5. Answer + citations
                                      v
                          Answer panel + Sources + Raw context

