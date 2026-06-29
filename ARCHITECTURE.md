# How DocumentRAG Works — interview-prep cheat sheet

A plain-English walk-through of the design and the important code behind
**DocumentRAG**, a PDF Q&A chatbot, written so I can confidently answer questions
about it. For setup and usage, see **`README.md`**; for the dated build history, see
**`DEVELOPMENT_LOG.md`** — this file is the "explain it to an interviewer" companion.

---

## 1. The 30-second pitch

> "It's a **retrieval-augmented generation (RAG)** chatbot built on **Streamlit +
> LangChain**. You upload a PDF; it extracts the text, splits it into overlapping
> chunks, embeds them with **Google Gemini**, and indexes them in a **FAISS** vector
> store. When you ask a question, it retrieves the most relevant chunks, stuffs them
> into the prompt as context, and has **Z.AI GLM-5.2** answer *only* from that context —
> then streams the answer back token-by-token. The whole app is one
> `main.py`, and the expensive embedding step is cached so a document is indexed
> once, not on every keystroke."

That hits: the pattern (RAG), the stack, the grounding story, and the one
performance decision that matters. Everything below is the detail behind it.

---

## 2. Stack — and *why* each piece (the "why" is what gets asked)

| Layer | Choice | Why I picked it |
|---|---|---|
| UI | **Streamlit** | Zero front-end code — file uploader, text input, and live output in pure Python. Perfect for a single-purpose data app. |
| PDF parsing | **pdfplumber** | Reliable per-page text extraction; simpler than PyPDF for plain text. |
| Orchestration | **LangChain (LCEL)** | The `|` pipe syntax wires retriever → prompt → LLM → parser into one declarative chain. Swappable pieces. |
| Chat LLM | **Z.AI GLM-5.2** (`langchain-openai`) | OpenAI-compatible endpoint, so `ChatOpenAI` + a `base_url` is all it takes; strong, low-cost generation with a 1M-token context. |
| Embeddings | **Google Gemini** (`langchain-google-genai`) | `gemini-embedding-001` for the vector index; generous free tier. (Z.AI does chat, Gemini does embeddings — see follow-up.) |
| Vector store | **FAISS** (`faiss-cpu`) | In-memory similarity search, no external DB to run. Ideal for "index one document per session." |
| Runtime | **Python 3.11+ + uv** | `uv` for fast, reproducible installs (`uv.lock`); dev is pinned to 3.14 via `.python-version`. |

**Common follow-ups:**
- *"Why FAISS over a hosted vector DB (Pinecone/pgvector)?"* → The corpus is *one
  uploaded PDF* that lives for the session. An in-memory index is simpler, free, and
  fast enough. A persistent DB only earns its keep when documents are shared/reused
  across sessions.
- *"Why Z.AI GLM for chat, but Gemini for embeddings?"* → GLM-5.2 is a strong, low-cost
  generator that speaks the OpenAI API, so it drops in via `ChatOpenAI(base_url=...)`.
  Gemini stays for embeddings because that path already worked — LCEL makes the two
  providers a clean swap, not a rewrite.

---

## 3. The big idea: the model answers *only* from retrieved context

This is the cleanest thing to explain because it shows a design *principle*, not
just code.

```
PDF ──► extract text ──► split into chunks ──► embed ──► FAISS index
                                                              │
                              question ──► embed ──► similarity search
                                                              │
                                          top-k relevant chunks = {context}
                                                              │
                  prompt("answer ONLY from this context") ──► GLM-5.2 ──► answer
```

The system prompt hard-instructs the model: *"Only use information from the provided
context — do not use outside knowledge. If the information is not in the context, say
so politely."* So the bot can't drift into hallucinated facts — its world is the
chunks FAISS hands it. **One source of truth: the uploaded document.**

---

## 4. Request lifecycle (whiteboard this if asked)

```
[1] User uploads a PDF in the sidebar  (st.file_uploader)
        │
        ▼
[2] build_retriever(file.getvalue())     ← @st.cache_resource
        │   extract text → chunk (1000/200) → Gemini embed → FAISS index
        │   (cached on the file's bytes, so this runs ONCE per document)
        ▼
[3] build_chain(retriever)
        │   retriever | format_docs → {context};  question → passthrough
        │   → prompt → Z.AI GLM (ChatOpenAI) → StrOutputParser
        ▼
[4] User submits a question  (st.form + st.form_submit_button)
        │   chain.stream(question)  → token generator
        ▼
[5] st.write_stream(...)   renders tokens live as GLM-5.2 generates them
```

The subtle bit is **[2]**: Streamlit reruns the *entire script top-to-bottom* on
every interaction (including every keystroke that submits the text box). Without
caching, every question would re-extract the PDF and re-embed every chunk — slow and
billable. `@st.cache_resource` keyed on the file bytes makes indexing happen exactly
once per document.

---

## 5. The pipeline in code

### 5a. Index once, keyed on the file (`build_retriever`)

```python
@st.cache_resource(show_spinner="Indexing your document...")
def build_retriever(file_bytes: bytes):
    text = extract_text(io.BytesIO(file_bytes))
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    ).split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, ...)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store.as_retriever()
```

- **Why `bytes` as the cache key, not the upload object?** Streamlit's `UploadedFile`
  isn't a stable hash key; the raw bytes are. Same PDF → same key → cache hit.
- **Why overlap (200)?** Splitting at hard 1000-char boundaries can cut a sentence in
  half and lose meaning at the seam. Overlapping chunks keep context across the cut.

### 5b. The RAG chain (`build_chain`)

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

LangChain Expression Language (LCEL): the dict runs **in parallel** — the retriever
fetches + formats chunks into `{context}` while the raw question passes through into
`{question}`. Both fill the prompt template, GLM-5.2 answers, and `StrOutputParser`
strips the response down to a plain string.

### 5c. Streaming UX — form, busy state, and a Stop button

```python
with st.form("ask"):
    question = st.text_input("…", key="question")
    submitted = st.form_submit_button("Ask", disabled=st.session_state.asking)

# Stop button (shown only while streaming) interrupts the run below.
if st.session_state.asking and stop:
    st.session_state.asking = False
    st.rerun()

if st.session_state.asking:
    st.write_stream(capture_answer(chain.stream(st.session_state.question)))
```

A few things are happening here:

- **Real streaming** — `chain.stream()` yields tokens as GLM-5.2 generates them and
  `st.write_stream` renders each live, not a cosmetic typewriter.
- **Submit-only** — wrapping the box in a **form** means the chain runs only on an explicit
  submit, so an unrelated rerun (uploading a new PDF) never re-answers a stale question or
  re-bills the API.
- **Busy state** — clicking Ask flips an `asking` flag and reruns, so the button re-renders
  **disabled** with a spinner. A generator holds the spinner open only until the first token,
  then clears it as the answer rolls in.
- **Stop = a deliberate interrupt.** Streamlit runs one script at a time, but *clicking any
  widget while a run is in flight makes it abort that run and rerun.* The Stop button
  weaponises exactly that: the click interrupts the blocking `write_stream`, and the rerun
  lands with the click registered, where the `asking` flag is dropped. `capture_answer` saves
  each token to session state as it streams, so the **partial answer survives** the interrupt
  instead of vanishing. (One catch: Streamlit signals the abort with an internal exception, so
  the `except` around streaming re-raises control-flow exceptions rather than mistaking them
  for an LLM error.)
- **Auto-scroll** — a 0-height `components.html` iframe runs a script that scrolls the main
  container (`[data-testid="stMain"]`, *not* the window) to the bottom as tokens arrive,
  pausing if you scroll up. `st.markdown` can't do this — it strips `<script>`.

---

## 6. Rate limiting & deploy guardrails

Made public-deploy-ready (target: a free-tier host) without a database. Three concerns:
cap upload size, cap API spend, and hide Streamlit's dev chrome.

### 6a. Server-side, IP-keyed rate limiter — *the reload-proof part*

The naive approach — counting in `st.session_state` — resets the moment the user reloads the
page, because a reload starts a fresh session. So the limiter state lives in the **server
process** instead: a single dict returned by `@st.cache_resource` (one object shared across
*all* sessions and reruns), keyed by client IP. It's a direct port of the portfolio site's
in-memory `rate-limit.ts` — the same fixed-window `{count, reset_at}` buckets and IP keying.

```python
@st.cache_resource
def _rate_state() -> dict:          # one map, process-wide (survives reloads)
    return {"buckets": {}, "last_prune": 0.0}

def consume_limit(state, key, limit, window, now) -> bool:
    bucket = state["buckets"].get(key)
    if bucket is None or bucket["reset_at"] <= now:
        state["buckets"][key] = {"count": 1, "reset_at": now + window}
        return True                  # fresh/expired window → new bucket
    if bucket["count"] >= limit:
        return False                 # full → reject
    bucket["count"] += 1
    return True
```

Two caps, two bucket namespaces: `upload:{ip}` (PDFs per window) and `q:{ip}:{file_id}`
(questions per document per window). The client key prefers `X-Forwarded-For` (set by the
host's proxy, stable across reloads); with no proxy (local dev) it falls back to a random id parked
in the **URL query string** — which survives a reload, where `st.session_state` would not (a
reload starts a fresh session, so a session-based key would silently reset the limit). Upload
slots are consumed **before** embedding (the
costly step); question slots at submit (each attempt is a billed call, so a stopped answer
still counts). A read-only `peek_limit` drives the live usage meter without spending a slot.

**Why it survives a reload:** the buckets are in server memory, not the session. To get past
the cap a user must re-upload — which itself spends an upload slot from the same IP bucket.

### 6b. Dropzone lock + auto-unlock

At the cap, only the uploader's **dropzone** (Browse + drag-drop) is locked, via injected CSS
— *not* `file_uploader(disabled=True)`, which would also disable the file's remove (×) button
and trap the current PDF on screen. Because Streamlit only reruns on interaction, a locked
uploader would stay locked after the window frees up; an invisible `@st.fragment(run_every=
"2s")` polls and fires `st.rerun(scope="app")` the instant the cooldown ends, so the dropzone
re-enables on its own with no click. The count drives both the lock and the meter in one render.

### 6c. Config (`.streamlit/config.toml`)

`maxUploadSize = 1` (MB) is the hard stop against a large PDF OOM-killing a 512MB free-tier
container; `toolbarMode = "minimal"` hides the Deploy button. The rate-limit numbers are constants
in `main.py` (`MAX_UPLOADS_PER_WINDOW`, `MAX_QUESTIONS_PER_PDF`, `UPLOAD_WINDOW_SECONDS`).

### 6d. Monitoring & alerting — two layers

Liveness and correctness are different questions, so they get different probes:

- **External liveness (UptimeRobot → `/healthz`).** A keyword monitor expects `ok`. It uses
  `/healthz` rather than `/_stcore/health` because the latter 303-redirects anonymous clients
  into Streamlit's auth/wake flow (never reaching `ok`). It catches the platform being down —
  but `/healthz` is an *edge*-layer response, so it can't tell you the RAG itself broke.
- **In-app error mirror (`discord_alert.py`).** That gap is why `report_error()` exists: the two
  real upstream-failure `except` blocks in `main()` — embedding (Gemini) and chat streaming
  (Z.AI) — mirror the exception to Discord, throttled to one alert per signature per 5 min, on a
  daemon thread so the user's error UI isn't blocked. User-input errors (scanned PDF) and
  Streamlit control-flow exceptions (Stop/rerun) are filtered out. Off unless
  `DISCORD_ERROR_WEBHOOK_URL` is set. (No top-level catch-all: `st.rerun()` throws constantly, so
  a global handler would be noise.)

---

## 7. Configuration

All tunables live as module-level constants at the top of `main.py`, so behavior
changes in one place: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`, `CHAT_MODEL`
(set via `ZAI_MODEL` in `.env`, e.g. `glm-5.2`), `ZAI_BASE_URL`, `TEMPERATURE`, `MAX_TOKENS`,
`THINKING`, and `SYSTEM_PROMPT`.
`temperature=0.3` keeps answers factual; `THINKING = {"type": "disabled"}` skips GLM's
reasoning trace for faster first-token latency. Two keys load from `.env` (never
hardcoded) — `ZAI_API_KEY` (chat) and `GOOGLE_API_KEY` (embeddings) — and `main()` guards
against either being missing. UI copy is centralized in a `TRANSLATIONS` dict (`en`/`ja`)
for the bilingual interface, with the language picked by a top-right selector.

---

## 8. Likely interview Q&A — quick-fire

| Question | One-line answer |
|---|---|
| "What is RAG?" | Retrieve relevant chunks from a vector store, feed them as context, and have the LLM answer only from them — grounding the model in real data. |
| "Walk me through the stack." | Streamlit UI, pdfplumber for text, LangChain LCEL chain, Z.AI GLM-5.2 for generation, Gemini for embeddings, FAISS for vector search. |
| "How do you stop it hallucinating?" | The system prompt forbids outside knowledge and tells it to say "not in the document" when context is missing. |
| "Why chunk overlap?" | So meaning isn't lost at hard split boundaries — adjacent chunks share 200 chars. |
| "What's the performance gotcha?" | Streamlit reruns the whole script per interaction; `@st.cache_resource` stops the PDF being re-embedded every question. |
| "Is it real token streaming?" | Yes — `chain.stream()` yields tokens and `st.write_stream` renders them live as GLM-5.2 generates them. |
| "Can you stop a long answer?" | Yes — the Stop button. Clicking a widget mid-run makes Streamlit abort and rerun, which interrupts `write_stream`; a wrapper saves the partial answer to session state so it isn't lost. |
| "How is the UI bilingual?" | A `TRANSLATIONS` dict keyed `en`/`ja`; a top-right selector sets `st.session_state["lang"]` and every label/message renders through it. Streamlit has no built-in i18n, so the strings are kept in-app. |
| "How would you scale it?" | Persist the index (e.g. pgvector/Chroma), support multiple/large docs, add conversation memory, and cite which chunks an answer came from. |
| "How do you rate-limit without a DB?" | Fixed-window counters in a process-global `@st.cache_resource` dict keyed by client IP — server-side, so a page reload doesn't reset them. Ported from the portfolio's in-memory limiter. |

---

## 9. The honest trade-offs (say these *before* you're asked)

1. **Single-turn, no memory.** Each question is independent — there's no conversation
   history. The latest answer is kept in session state (so it survives reruns and a Stop),
   but only one at a time; a chat history layer would be the next step.
2. **In-memory FAISS, session-scoped.** The index lives in the Streamlit cache and
   dies with the process. Fine for "ask one PDF"; a persistent store is needed to
   reuse documents across sessions or users.
3. **Whole-PDF, single-document.** It indexes one uploaded file at a time and assumes
   the PDF has extractable text (scanned/image PDFs need OCR first).
4. **No source citations.** It answers from retrieved chunks but doesn't show *which*
   chunks — adding page/snippet citations would make answers auditable.
5. **Fixed-size chunking.** A flat 1000/200 split ignores document structure;
   semantic or header-aware splitting would retrieve cleaner context.
6. **Two providers, two keys.** Chat (Z.AI) and embeddings (Gemini) are separate
   services — two API keys, two points of failure. Fine for a small app; using one
   provider for both would simplify ops.
7. **Rate limits are in-memory and IP-keyed.** They survive page reloads (server-side state)
   but reset when the process restarts — e.g. a free-tier container waking from sleep — and IP
   keying is coarse: users behind one NAT share a budget, a new IP/VPN bypasses it. A shared
   store (Upstash Redis) is the upgrade if traffic ever warrants it.
