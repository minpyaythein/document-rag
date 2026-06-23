# DocumentRAG — Chat with your PDF

<div align="center">

[![English](https://img.shields.io/badge/README-English-2563eb?style=for-the-badge)](README.md)
[![日本語](https://img.shields.io/badge/README-日本語-lightgrey?style=for-the-badge)](README.ja.md)

</div>

A PDF-based Q&A chatbot. Upload a PDF, ask a question, and get a streamed answer
grounded in the document — retrieval-augmented generation (RAG) with **Z.AI GLM-5.2**
for answers and Google Gemini embeddings over a FAISS vector store, in one `main.py`.

---

## Features

- **PDF Q&A** — upload a PDF in the sidebar and ask questions about it in plain language.
- **Grounded answers** — the model answers *only* from the document and says so when the
  answer isn't there, so it doesn't make things up.
- **Streamed responses** — answers stream in token-by-token as the model generates them.
- **Index once, not per question** — embedding is cached on the file, so re-asking is fast
  and doesn't re-spend on the same document.
- **Bilingual UI (EN/JA)** — switch the whole interface between English and 日本語 from the
  language selector in the top-right corner.

---

## Tech stack

| Area | Choice |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| PDF parsing | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| Orchestration | [LangChain](https://python.langchain.com) (LCEL) |
| Chat LLM | [Z.AI GLM-5.2](https://z.ai) (OpenAI-compatible) via `langchain-openai` |
| Embeddings | Google Gemini (`gemini-embedding-001`) via `langchain-google-genai` |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| Runtime | Python 3.11+ + [uv](https://docs.astral.sh/uv/) |

---

## Getting started

**Prerequisites:** Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip). Runs locally.

```bash
uv sync
streamlit run main.py    # opens http://localhost:8501

# or with pip:
pip install -r requirements.txt
streamlit run main.py
```

Create a `.env` in the project root with your keys:

```bash
ZAI_API_KEY=your-zai-api-key
ZAI_MODEL=glm-5.2
GOOGLE_API_KEY=your-gemini-api-key
```

### Environment variables

| Variable | Required | Notes |
|---|---|---|
| `ZAI_API_KEY` | yes | Z.AI API key — used for the chat model. Loaded from `.env`. |
| `ZAI_MODEL` | yes | Z.AI chat model id, e.g. `glm-5.2`. |
| `GOOGLE_API_KEY` | yes | Google Gemini API key — used for embeddings. Loaded from `.env`. |

> `.env` is git-ignored — never commit real keys.

---

## How it works

The uploaded PDF is the single source of truth for a session: its text is split into
overlapping chunks, embedded with Gemini, and indexed in FAISS. Each question retrieves
the most relevant chunks, which are fed to **Z.AI GLM-5.2** as the *only* context it may
answer from. Configuration (chunk size, model names, temperature, system prompt) lives as
constants at the top of `main.py`.

For the full design, the request lifecycle, and the reasoning behind each choice, see
**`ARCHITECTURE.md`**.

---

## Further docs

- `ARCHITECTURE.md` — how it works and why (the design, explained).
- `DEVELOPMENT_LOG.md` — dated record of what was built and why.
- `CLAUDE.md` — repo conventions for AI assistants.
