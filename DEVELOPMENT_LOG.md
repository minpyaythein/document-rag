# DocumentRAG Development Log

A specific, dated record of what was built and decided. Read this when picking up after a break.

---

## Project at a glance

- **Workspace**: local repo `document-rag/`
- **What it is**: a PDF Q&A chatbot — upload a PDF, ask questions, get answers grounded in it (RAG)
- **Stack**: Streamlit 1.55 + LangChain 1.2.x (LCEL) + Z.AI GLM-5.2 (`langchain-openai`, chat) + Google Gemini (`langchain-google-genai`, embeddings) + FAISS (`faiss-cpu` 1.13.2), Python 3.11+ (dev on 3.14), `uv`
- **Models**: chat `glm-5.2` via Z.AI (`temperature=0.3`, `max_tokens=1024`, thinking disabled); embeddings `models/gemini-embedding-001` via Gemini
- **Hosting**: local only (`streamlit run main.py`) — not deployed
- **Repo**: public — `github.com/minpyaythein/document-rag` (`origin/main`)

---

## Origin (before 2026-06-22)

- Started as an early single-file PDF Q&A prototype (Streamlit + LangChain + Gemini + FAISS),
  working name `chatbot`.
- **Naming was inconsistent** across files — the package name, the app header, and the
  README title didn't agree before being unified as **DocumentRAG**.
- **`CLAUDE.md` was wrong**: it described an OpenAI + tiktoken stack the code never used
  (the code is Gemini).

---

## 2026-06-22 → 2026-06-23: Rename + full refactor

### Rename to `document-rag` (2026-06-22 → 2026-06-23)
- Renamed the project to `document-rag` everywhere: `pyproject.toml`
  name + description, app header (now `"DocumentRAG — Chat with your PDF"`), README and
  CLAUDE titles.
- Renamed the folder `chatbot/` → `document-rag/`; the hidden files (`.git`, `.env`,
  `.venv`, `.claude`) moved intact.
- Regenerated `uv.lock` with `uv lock` — swapped the project entry cleanly (resolved 99
  packages, swapped the old package entry for `document-rag`).
- **Gotcha**: renaming the folder leaves `.venv` pointing at the old path. Fix:
  `rm -rf .venv && uv sync`.

### `main.py` refactor (2026-06-23)
- Restructured the flat module-scope script into functions: `extract_text`,
  `build_retriever`, `format_docs`, `build_chain`, `stream_answer`, `main`.
- **Perf fix**: wrapped `build_retriever` in `@st.cache_resource` keyed on the file
  bytes. Streamlit reruns the whole script on every interaction, so previously the PDF
  was re-extracted and re-embedded on *every question* — now a document is indexed once.
- Pulled tunables into module constants (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`,
  `CHAT_MODEL`, `TEMPERATURE`, `SYSTEM_PROMPT`).
- Fixed a duplicate `"5."` in the system prompt (now numbered 1–6).
- Removed dead commented code and the placeholder cat emojis (spinner is `"Thinking..."`,
  cursor is `▌`), added a `GOOGLE_API_KEY` missing-key guard, and dropped the hacky
  `components.html` auto-scroll script.

### Docs corrected (2026-06-23)
- Rewrote `CLAUDE.md` — it claimed an OpenAI/tiktoken stack; corrected to the real Gemini
  stack and the new function architecture.
- Fixed stale README steps (referenced a commented-out `max_tokens` and `st.write` instead
  of the streaming render).

---

## 2026-06-23: Standardized the doc set (setup-project-docs)

- Created `ARCHITECTURE.md` — the interview-prep cheat sheet (30-sec pitch, stack-with-why,
  the big idea, request lifecycle, real code snippets, Q&A, honest trade-offs).
- Reshaped `README.md` from a step-by-step code walkthrough into the operator format
  (Features, Tech stack, Getting started, env vars, Further docs) so it no longer
  duplicates `ARCHITECTURE.md`.
- Created this `DEVELOPMENT_LOG.md`.

---

## 2026-06-23: Bilingual README

- Made the README **bilingual**: created `README.ja.md` (a natural-Japanese twin, not a
  literal gloss) and added language-switch badges to both, kept in lockstep.
- Upgraded the `setup-project-docs` skill to produce the `README.md` + `README.ja.md` pair
  by default; `ARCHITECTURE.md` and `DEVELOPMENT_LOG.md` stay English-only.
- Reviewed the full refactor afterward — no code changed; open items are tracked under
  **Known issues** below.

---

## 2026-06-23: High-tier fixes (code review)

Fixed the three high-priority items from the full code review (`main.py`):
- **Error handling** — wrapped `build_retriever`/`build_chain` and `chain.invoke` in
  `try/except`; API failures (rate limit, network, bad key/model) now show a friendly
  `st.error` instead of a raw traceback.
- **Empty / scanned PDF guard** — `build_retriever` raises a clear `ValueError` when no text
  is extracted, surfaced to the user instead of crashing in `FAISS.from_texts([])`.
- **Page separator** — `extract_text` now joins pages with `\n` so words don't merge across
  page boundaries (was silently degrading chunking/retrieval on multi-page PDFs).

---

## 2026-06-23: Pinned floating deps

- Pinned the two unpinned dependencies to their resolved versions in both `pyproject.toml`
  and `requirements.txt`: `langchain-google-genai==4.2.1`, `python-dotenv==1.2.2`.
  Regenerated `uv.lock` (resolved versions unchanged). Installs are now reproducible and the
  `CLAUDE.md` "pinned in both" claim is accurate.

---

## 2026-06-23: Medium fixes (real streaming, form, Python floor)

- **Real token streaming** — replaced the cosmetic `stream_answer` typewriter (and its
  `time.sleep` loop) with `st.write_stream(chain.stream(question))`, so tokens render live
  as Gemini generates them. Kills the O(n²) re-render and the off-screen problem.
- **Submit form** — wrapped the question box in `st.form` + `st.form_submit_button`, so the
  chain only runs on an explicit submit. An unrelated rerun (e.g. uploading a new PDF) no
  longer re-answers a stale question or re-bills the API.
- **Python floor lowered** — `requires-python` `>=3.14` → `>=3.11` for broader install
  compatibility; `.python-version` stays 3.14 (dev env). Re-locked: resolved versions
  unchanged.
- **Model ID verified** — web-checked `gemini-3-flash-preview`: it's a real *preview* model
  (no 404 risk). The GA `gemini-3.5-flash` is the more stable swap if desired (not changed).

---

## 2026-06-23: Switched chat to Z.AI GLM-5.2

- **Chat model migrated** off Google Gemini to **Z.AI GLM-5.2**, mirroring the portfolio's
  Z.AI setup. Z.AI is OpenAI-compatible, so the chat LLM is now `langchain-openai`'s
  `ChatOpenAI` pointed at `https://api.z.ai/api/coding/paas/v4`, model `glm-5.2`, with
  `max_tokens=1024` and `extra_body={"thinking": {"type": "disabled"}}` (faster first token).
- **Embeddings stay on Gemini** (`gemini-embedding-001`) — hybrid: Z.AI for chat, Gemini for
  the FAISS vector index.
- **New dep**: `langchain-openai==1.1.11` (pulls `openai`, `tiktoken`); re-locked (105 pkgs).
- **New env var**: `ZAI_API_KEY` (chat) alongside `GOOGLE_API_KEY` (embeddings); `main()` now
  guards for both.
- **Chat model is env-driven** — `CHAT_MODEL = os.getenv("ZAI_MODEL")` (required, no default);
  set `ZAI_MODEL` in `.env`. `main()` validates it alongside the API keys. (The portfolio
  keeps a default model id; here it's required.)
- **Verify on first run**: that GLM-5.2 accepts the `thinking: {type: disabled}` body param
  (the one Z.AI-specific extra) — drop it from `THINKING` if the API rejects it.

---

## Frozen facts (keep consistent everywhere)

- **Chat model**: set via `ZAI_MODEL` in `.env` (e.g. `glm-5.2`), through Z.AI, endpoint `https://api.z.ai/api/coding/paas/v4` (`temperature=0.3`, `max_tokens=1024`)
- **Embedding model**: `models/gemini-embedding-001` via Google Gemini
- **Chunking**: 1000 chars, 200 overlap
- These model IDs / the Z.AI endpoint appear in `main.py`, `CLAUDE.md`, and `ARCHITECTURE.md` — change them in lockstep.

---

## Review findings (2026-06-23)

1. ~~**Long answers stream off-screen.**~~ ✅ Fixed 2026-06-23 — switched to real token
   streaming (`st.write_stream(chain.stream(...))`); no more typewriter or off-screen render.
2. ~~**Scanned / empty PDFs crash.**~~ ✅ Fixed 2026-06-23 — `build_retriever` raises a clear
   error when no text is extractable.
3. ~~**`CLAUDE.md` overstates pinning.**~~ ✅ Fixed 2026-06-23 — both deps pinned
   (`langchain-google-genai==4.2.1`, `python-dotenv==1.2.2`), so the claim is now true.
4. ~~**`gemini-3-flash-preview` is unverified**~~ ✅ Superseded 2026-06-23 — chat was switched
   off Gemini entirely to **Z.AI GLM-5.2**, so the Gemini chat-model question is moot.
