# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocumentRAG is a PDF-based Q&A chatbot built with Streamlit and LangChain. Users
upload a PDF document via the sidebar, and the app uses **Z.AI GLM-5.2** for answer
generation and **Google Gemini** embeddings over a FAISS vector store for
retrieval-augmented generation (RAG), then streams the answer back token-by-token.

## Tech Stack

- **UI**: Streamlit
- **PDF parsing**: pdfplumber
- **LLM orchestration**: LangChain (langchain, langchain-community, langchain-core, langchain-text-splitters)
- **Chat model**: Z.AI **GLM-5.2** via `langchain-openai` (`ChatOpenAI` pointed at Z.AI's
  OpenAI-compatible endpoint `https://api.z.ai/api/coding/paas/v4`)
- **Embeddings**: Google Gemini `models/gemini-embedding-001` via `langchain-google-genai`
- **Vector store**: FAISS (faiss-cpu)

## Commands

```bash
# Run the app
streamlit run main.py

# Install dependencies (uv is the package manager, Python 3.11+)
uv sync

# Or with pip
pip install -r requirements.txt
```

## Architecture

Single-file app (`main.py`), organized into small functions:

1. `extract_text` — text extraction from the uploaded PDF with pdfplumber
2. `build_retriever` — splits text (langchain-text-splitters), embeds with Gemini,
   and indexes into FAISS. Wrapped in `@st.cache_resource` keyed on the file bytes
   so the document isn't re-embedded on every Streamlit rerun.
3. `build_chain` — wires `retriever -> prompt -> Z.AI GLM (ChatOpenAI) -> StrOutputParser`
4. `main` — Streamlit UI flow: pick language (top-right selector) → upload → (cached) index
   → ask via an `st.form` → stream the answer live with `st.write_stream(chain.stream(...))`

Configuration (chunk size/overlap, model names, temperature, system prompt) lives in
module-level constants at the top of `main.py`.

## Configuration

- `ZAI_API_KEY` (chat) and `GOOGLE_API_KEY` (embeddings) are loaded from `.env` (never hardcode them)
- `ZAI_MODEL` (required) sets the chat model id, e.g. `glm-5.2`
- Python: requires 3.11+ (`pyproject.toml`); dev env pinned to 3.14 via `.python-version`
- Dependencies are pinned in both `requirements.txt` and `pyproject.toml` — keep them in sync
- UI strings are bilingual (EN/JA) in a `TRANSLATIONS` dict in `main.py`; a top-right selector
  sets the language. When changing any UI copy, update **both** the `en` and `ja` entries.
