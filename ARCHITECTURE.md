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

### 5c. Streaming UX (`st.write_stream` + a form)

```python
with st.form("ask"):
    question = st.text_input("Ask a question about your document")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    st.write_stream(chain.stream(question))
```

`chain.stream()` yields the answer token-by-token as GLM-5.2 generates it, and
`st.write_stream` renders each token live — real streaming, not a cosmetic typewriter.
Wrapping the input in a **form** means the chain only runs on an explicit submit, so an
unrelated rerun (like uploading a new PDF) never re-answers a stale question or re-bills
the API.

---

## 6. Configuration

All tunables live as module-level constants at the top of `main.py`, so behavior
changes in one place: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`, `CHAT_MODEL`
(set via `ZAI_MODEL` in `.env`, e.g. `glm-5.2`), `ZAI_BASE_URL`, `TEMPERATURE`, `MAX_TOKENS`,
`THINKING`, and `SYSTEM_PROMPT`.
`temperature=0.3` keeps answers factual; `THINKING = {"type": "disabled"}` skips GLM's
reasoning trace for faster first-token latency. Two keys load from `.env` (never
hardcoded) — `ZAI_API_KEY` (chat) and `GOOGLE_API_KEY` (embeddings) — and `main()` guards
against either being missing.

---

## 7. Likely interview Q&A — quick-fire

| Question | One-line answer |
|---|---|
| "What is RAG?" | Retrieve relevant chunks from a vector store, feed them as context, and have the LLM answer only from them — grounding the model in real data. |
| "Walk me through the stack." | Streamlit UI, pdfplumber for text, LangChain LCEL chain, Z.AI GLM-5.2 for generation, Gemini for embeddings, FAISS for vector search. |
| "How do you stop it hallucinating?" | The system prompt forbids outside knowledge and tells it to say "not in the document" when context is missing. |
| "Why chunk overlap?" | So meaning isn't lost at hard split boundaries — adjacent chunks share 200 chars. |
| "What's the performance gotcha?" | Streamlit reruns the whole script per interaction; `@st.cache_resource` stops the PDF being re-embedded every question. |
| "Is it real token streaming?" | Yes — `chain.stream()` yields tokens and `st.write_stream` renders them live as GLM-5.2 generates them. |
| "How would you scale it?" | Persist the index (e.g. pgvector/Chroma), support multiple/large docs, add conversation memory, and cite which chunks an answer came from. |

---

## 8. The honest trade-offs (say these *before* you're asked)

1. **Single-turn, no memory.** Each question is independent — there's no conversation
   history, and the answer clears on the next interaction (it isn't persisted). A chat
   history layer would be the next step.
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
