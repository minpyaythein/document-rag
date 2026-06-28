# DocumentRAG — Chat with your PDF

<div align="center">

[![English](https://img.shields.io/badge/README-English-2563eb?style=for-the-badge)](README.md)
[![日本語](https://img.shields.io/badge/README-日本語-lightgrey?style=for-the-badge)](README.ja.md)

</div>

A PDF-based Q&A chatbot. Upload a PDF, ask a question, and get a streamed answer
grounded in the document — retrieval-augmented generation (RAG) with **Z.AI GLM-5.2**
for answers and Google Gemini embeddings over a FAISS vector store, in one `main.py`.

---

## Why I built this

I built DocumentRAG primarily as a **learning project** — to understand how
retrieval-augmented generation actually works end to end, not just in theory. Putting it
together taught me:

- **RAG from the ground up** — chunking, embeddings, vector similarity search, and grounding
  an LLM in retrieved context so it answers from a source instead of making things up.
- **LangChain (LCEL)** — composing a `retriever → prompt → LLM → parser` pipeline with the `|`
  pipe, and streaming tokens out of it.
- **Vector search with FAISS** — turning text into embeddings and querying them in-memory.
- **Wiring multiple providers** — running an OpenAI-compatible chat model (Z.AI GLM) and
  Google Gemini embeddings in one app, and seeing how a clean abstraction keeps them swappable.
- **Prompt design for grounding** — writing a system prompt that keeps the model honest about
  what is (and isn't) in the document.
- **Building a real Streamlit app** — caching expensive work, session state, forms, live token
  streaming, and a bilingual (EN/JA) UI.
- **Production-minded details** — config via `.env`, reproducible installs with `uv`, and
  handling messy inputs (scanned PDFs, API failures) gracefully.

---

## Features

- **PDF Q&A** — upload a PDF in the sidebar and ask questions about it in plain language.
- **Grounded answers** — the model answers *only* from the document and says so when the
  answer isn't there, so it doesn't make things up.
- **Streamed responses** — answers stream in token-by-token as the model generates them,
  and the page auto-scrolls to follow along (it pauses if you scroll up).
- **Stop anytime** — interrupt a long answer mid-stream with the Stop button; whatever was
  generated so far stays on screen.
- **Index once, not per question** — embedding is cached on the file, so re-asking is fast
  and doesn't re-spend on the same document.
- **Bilingual UI (EN/JA)** — switch the whole interface between English and 日本語 from the
  language selector in the top-right corner.
- **Public-deploy guardrails** — a 1MB upload cap plus per-IP rate limits (PDFs per window and
  questions per PDF), shown live in a sidebar usage meter. The limits are held server-side, so
  a page reload doesn't reset them; the uploader locks and auto-unlocks when the window frees.

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
./local_setup.sh         # sync deps + launch — opens http://localhost:8501

# …or do it by hand:
uv sync
streamlit run main.py

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

### Deployment config

`.streamlit/config.toml` holds the public-facing guardrails: `maxUploadSize = 1` (MB cap) and
`toolbarMode = "minimal"` (hides Streamlit's Deploy button). The rate-limit numbers are
constants at the top of `main.py` — `MAX_UPLOADS_PER_WINDOW`, `MAX_QUESTIONS_PER_PDF`, and
`UPLOAD_WINDOW_SECONDS`. The intended host is **Render** (free tier): Streamlit needs a
long-lived server, and the app is stateless (in-memory FAISS), so no database is required.

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
