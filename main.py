import io
import os
import time
import uuid

import pdfplumber
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
load_dotenv()
# Embeddings run on Google Gemini; chat/generation runs on Z.AI GLM.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"  # OpenAI-compatible endpoint
CHAT_MODEL = os.getenv("ZAI_MODEL")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/gemini-embedding-001"
TEMPERATURE = 0.3
MAX_TOKENS = 1024
# GLM-5.2 is a reasoning model; disable the thinking trace for faster first-token latency.
THINKING = {"type": "disabled"}

# --- Rate limiting (public-deploy guardrails) ---
# Fixed-window limits kept server-side and keyed by client IP (see `_rate_state` /
# `consume_limit`), mirroring the portfolio site's in-memory limiter. Because the state lives
# in the Streamlit *server process* rather than in st.session_state, a browser reload — which
# starts a fresh session — can't reset it. It resets only when the process restarts (e.g. the
# Render dyno waking from sleep). Both caps share one window for simplicity.
UPLOAD_WINDOW_SECONDS = 10 * 60  # 10-minute rolling window
MAX_UPLOADS_PER_WINDOW = 1  # PDFs indexed per IP per window
MAX_QUESTIONS_PER_PDF = 10  # questions per PDF (per IP) per window

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a PDF document.\n\n"
    "Guidelines:\n"
    "1. Provide complete, well-explained answers using the context below.\n"
    "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
    "3. If the context mentions related information, include it to give a fuller picture.\n"
    "4. Only use information from the provided context — do not use outside knowledge.\n"
    "5. Summarize long information, ideally in bullets where needed.\n"
    "6. If the information is not in the context, say so politely.\n\n"
    "Context:\n{context}"
)

# --- Internationalization (UI strings) ---
# Streamlit has no built-in i18n, so UI copy is keyed by language here. The chat
# answers themselves come from the model and follow the user's question language.
LANGUAGES = {"en": "English", "ja": "日本語"}

TRANSLATIONS = {
    "en": {
        "header": "DocumentRAG — Chat with your PDF",
        "sidebar_title": "Your Documents",
        "uploader_label": "Upload your PDF here",
        "uploader_caption": "📄 One PDF at a time — a new upload replaces the current document.",
        "upload_prompt": "Upload a PDF in the sidebar to get started.",
        "indexing": "Indexing your document — you'll be able to chat once it's ready, so hang tight...",
        "missing_env": "Missing required .env value(s): {names}. Add them to your .env file.",
        "scanned_pdf": "No extractable text found — this looks like a scanned or image-only PDF.",
        "index_failed": "Failed to index the document: {error}",
        "retry": "🔄 Retry indexing",
        "question_label": "Ask a question about your document",
        "ask_button": "Ask",
        "thinking": "Generating your answer...",
        "stop_button": "⏹ Stop",
        "answer_failed": "Couldn't get an answer: {error}",
        "limits_header": "📊 Usage limits",
        "usage_pdfs": "Uploads: {used} of {max} used · resets every {minutes} min",
        "usage_questions": "Questions: {used} of {max} used on this PDF",
        "limit_pdfs": "⏳ Upload limit reached — {max} per {minutes} min. Try again in ~{wait} min.",
        "limit_questions": "⏳ Question limit reached — {max} per PDF. Upload a new PDF to keep going.",
    },
    "ja": {
        "header": "DocumentRAG — PDFと対話するチャットボット",
        "sidebar_title": "ドキュメント",
        "uploader_label": "PDFをここにアップロード",
        "uploader_caption": "📄 一度に1つのPDFのみ — 新しいアップロードで現在のドキュメントと置き換わります。",
        "upload_prompt": "サイドバーからPDFをアップロードして始めましょう。",
        "indexing": "ドキュメントをインデックス化しています — 準備が整うとチャットできます。少々お待ちください...",
        "missing_env": "必須の .env 値が不足しています: {names}。.env ファイルに追加してください。",
        "scanned_pdf": "抽出可能なテキストが見つかりません — スキャン画像のみのPDFのようです。",
        "index_failed": "ドキュメントのインデックス作成に失敗しました: {error}",
        "retry": "🔄 インデックスを再試行",
        "question_label": "ドキュメントについて質問してください",
        "ask_button": "質問する",
        "thinking": "回答を生成しています...",
        "stop_button": "⏹ 停止",
        "answer_failed": "回答を取得できませんでした: {error}",
        "limits_header": "📊 利用制限",
        "usage_pdfs": "アップロード: {max}件中{used}件使用 · {minutes}分ごとにリセット",
        "usage_questions": "質問: {max}問中{used}問使用（このPDF）",
        "limit_pdfs": "⏳ アップロード上限に達しました — {minutes}分あたり{max}件まで。約{wait}分後に再試行してください。",
        "limit_questions": "⏳ 質問の上限に達しました — PDFごとに{max}問まで。続けるには新しいPDFをアップロードしてください。",
    },
}

# Streamlit's file-uploader dropzone text ("Drag and drop file here", the size limit, and
# the "Browse files" button) is English-only with no Python API. When the UI is Japanese we
# override it via CSS by hiding the built-in text and injecting Japanese through pseudo-
# elements. These selectors target Streamlit's internal DOM (test-ids), so they may need a
# tweak if a future Streamlit release changes the uploader markup.
UPLOADER_JA_CSS = """
<style>
[data-testid='stFileUploaderDropzoneInstructions'] div span { display: none; }
[data-testid='stFileUploaderDropzoneInstructions'] div::before {
    content: 'ここにPDFをドラッグ＆ドロップ';
    display: block;
}
[data-testid='stFileUploaderDropzoneInstructions'] div small { display: none; }
[data-testid='stFileUploaderDropzoneInstructions'] div::after {
    content: '1ファイル1MBまで • PDF';
    display: block;
    font-size: 0.8rem;
}
[data-testid='stFileUploaderDropzone'] button { font-size: 0; }
[data-testid='stFileUploaderDropzone'] button::after {
    content: 'ファイルを選択';
    font-size: 0.875rem;
}
</style>
"""

# When the upload cap is hit, lock only the dropzone (the Browse button + drag-and-drop) so a
# new file can't be picked. We do this with CSS rather than file_uploader(disabled=True),
# which would also disable the uploaded file's "remove" (×) button — the × sits outside the
# dropzone, so it stays clickable here and the current PDF can still be cleared.
UPLOADER_LOCKED_CSS = """
<style>
[data-testid='stFileUploaderDropzone'] {
    pointer-events: none;
    opacity: 0.5;
}
</style>
"""


# --- Server-side rate limiter (IP-keyed, fixed window) ---
# Direct port of the portfolio site's `rate-limit.ts`: a process-global bucket map that
# survives browser reloads because it lives in the server, not the client session.


@st.cache_resource
def _rate_state() -> dict:
    """One bucket map shared across every session and rerun (the server-side store).

    `@st.cache_resource` returns the same object process-wide — the Streamlit equivalent of
    the portfolio limiter's module-level `const buckets = new Map()`.
    """
    return {"buckets": {}, "last_prune": 0.0}


def _prune(state: dict, now: float) -> None:
    """Drop expired buckets occasionally so the map doesn't grow forever."""
    if now - state["last_prune"] < 60:
        return
    state["last_prune"] = now
    state["buckets"] = {
        key: b for key, b in state["buckets"].items() if b["reset_at"] > now
    }


def peek_limit(state: dict, key: str, now: float) -> tuple[int, float]:
    """Return (used, reset_at) for the active window, or (0, 0.0) if none/expired.

    Read-only — used to render the usage meter without consuming a slot.
    """
    bucket = state["buckets"].get(key)
    if bucket is None or bucket["reset_at"] <= now:
        return 0, 0.0
    return bucket["count"], bucket["reset_at"]


def consume_limit(state: dict, key: str, limit: int, window: float, now: float) -> bool:
    """Consume one slot from a fixed window. Returns True if allowed (and increments).

    Mirrors the portfolio's `checkRateLimit`: a fresh/expired window starts a new bucket;
    a full one is rejected; otherwise the count ticks up.
    """
    buckets = state["buckets"]
    bucket = buckets.get(key)
    if bucket is None or bucket["reset_at"] <= now:
        buckets[key] = {"count": 1, "reset_at": now + window}
        _prune(state, now)
        return True
    if bucket["count"] >= limit:
        return False
    bucket["count"] += 1
    return True


def client_id() -> str:
    """Stable per-client key that survives a page reload.

    Prefers the real IP behind a proxy — Render/Vercel set `X-Forwarded-For` — which is stable
    across reloads. With no proxy (local dev) there's no such header, so we fall back to a
    random id parked in the URL query string. Unlike `st.session_state` (a fresh session every
    reload — which is exactly why the limit kept resetting), a query param rides along on F5,
    so the same browser keeps the same bucket.
    """
    try:
        headers = st.context.headers or {}
        forwarded = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    except Exception:
        pass
    rid = st.query_params.get("rid")
    if not rid:
        rid = uuid.uuid4().hex
        st.query_params["rid"] = rid
    return rid


def extract_text(file) -> str:
    """Read every page of a PDF into a single string."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


@st.cache_resource(show_spinner=False)
def build_retriever(file_bytes: bytes):
    """Chunk, embed, and index the PDF, returning a FAISS retriever.

    Cached on the file's bytes so Streamlit reruns (which happen on every
    question) don't re-embed the same document over and over.
    """
    text = extract_text(io.BytesIO(file_bytes))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError(
            "No extractable text found — this looks like a scanned or image-only PDF."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store.as_retriever()


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(retriever):
    """Wire retriever -> prompt -> LLM -> string output into a RAG chain."""
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=ZAI_API_KEY,
        base_url=ZAI_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body={"thinking": THINKING},
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def stream_with_thinking(stream, message: str):
    """Yield a token stream, showing a spinner only until the first token arrives.

    A plain `st.spinner` around `st.write_stream` would stay up for the whole
    response. Here the spinner covers just the wait for the first token (the model's
    latency) and disappears the moment the answer starts rolling in.
    """
    iterator = iter(stream)
    with st.spinner(message):
        first = next(iterator, None)
    if first is None:  # empty stream — nothing to show
        return
    yield first
    yield from iterator


def capture_answer(stream):
    """Stream tokens through while saving the running text to session state.

    `st.write_stream`'s return value is lost if the run is interrupted (e.g. by the
    Stop button), so the answer is accumulated here instead — whatever was generated
    before a stop stays on screen.
    """
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        st.session_state.answer = "".join(chunks)
        yield chunk


def autoscroll() -> None:
    """Keep the page pinned to the newest streamed text.

    Streamlit doesn't scroll as `st.write_stream` appends tokens, so a long answer
    runs off the bottom of the screen. `st.markdown` can't help (it strips
    <script>), so this drops a 0-height component iframe whose script reaches the
    parent page and, on a short interval, scrolls the main container to the bottom
    as text arrives — pausing if the user scrolls up, and resuming when they return
    to the bottom. The page scrolls an inner container (`[data-testid="stMain"]`),
    not the window, so we target whichever candidate actually overflows. The iframe
    is torn down on the next rerun, so it only follows while an answer is streaming.
    """
    components.html(
        """
        <script>
        const doc = window.parent.document;
        // The scroll container is an inner element, not the window — pick whichever
        // candidate is actually overflowing right now.
        const scroller = () => {
            const els = [
                doc.querySelector('[data-testid="stMain"]'),
                doc.querySelector('section.main'),
                doc.scrollingElement,
                doc.documentElement,
            ].filter(Boolean);
            return els.find((e) => e.scrollHeight > e.clientHeight + 5) || els[0];
        };
        let follow = true;
        // Capture-phase listener catches scrolls on the inner container too (scroll
        // events don't bubble). Pause when the user scrolls up; resume near bottom.
        window.parent.addEventListener('scroll', () => {
            const el = scroller();
            if (el) follow = el.scrollHeight - el.scrollTop - el.clientHeight < 150;
        }, { passive: true, capture: true });
        const id = setInterval(() => {
            const el = scroller();
            if (el && follow) el.scrollTop = el.scrollHeight;
        }, 120);
        // Safety net in case the iframe ever outlives the streaming run.
        setTimeout(() => clearInterval(id), 300000);
        </script>
        """,
        height=0,
    )


@st.fragment(run_every="2s")
def cooldown_refresh(deadline: float) -> None:
    """Poll during an upload cooldown and rerun the whole app once it expires.

    Streamlit only reruns on user interaction, so a locked uploader would stay locked after
    its window frees up until the user clicks something. This invisible fragment ticks on a
    timer and, when the deadline passes, triggers a full-app rerun — which prunes the expired
    upload timestamp and re-enables the dropzone on its own. While the cooldown is still
    running it does nothing, so the rest of the page isn't re-rendered.
    """
    if time.time() >= deadline:
        st.rerun(scope="app")


def main() -> None:
    # Page styling:
    # - widen the sidebar from its default — scoped to aria-expanded='true' so the rule
    #   doesn't pin the collapsed sidebar open and break the fold/unfold arrow;
    # - the language selectbox is a searchable field, so it shows a text cursor and a
    #   blinking caret (as if you could type) — force a pointer cursor and hide the caret
    #   so the switcher reads as a plain clickable dropdown.
    st.markdown(
        "<style>"
        "[data-testid='stSidebar'][aria-expanded='true'] { min-width: 320px; width: 320px; }"
        # Streamlit hides the sidebar's collapse («) arrow with visibility:hidden until you
        # hover the panel; force it always-visible so the collapse control is obvious.
        "[data-testid='stSidebarCollapseButton'] { visibility: visible !important; }"
        "[data-testid='stHeading'] h2 { white-space: nowrap; }"
        "div[data-baseweb='select'] > div,"
        "div[data-baseweb='select'] > div * { cursor: pointer; }"
        "div[data-baseweb='select'] input { caret-color: transparent; }"
        "</style>",
        unsafe_allow_html=True,
    )

    # Language switcher, pinned to the top-right of the page.
    _, lang_col = st.columns([5, 1])
    with lang_col:
        lang = st.selectbox(
            "Language",
            options=list(LANGUAGES),
            format_func=lambda code: LANGUAGES[code],
            label_visibility="collapsed",
            key="lang",
        )
    t = TRANSLATIONS[lang]

    # Streamlit's uploader strings are English-only; override them for the JA UI via CSS.
    if lang == "ja":
        st.markdown(UPLOADER_JA_CSS, unsafe_allow_html=True)

    st.header(t["header"])

    # Pull the server-side, IP-keyed limiter state up front, before drawing the uploader, so
    # the Browse button can be locked the moment the per-window cap is in effect.
    store = _rate_state()
    client = client_id()
    now = time.time()
    window_minutes = UPLOAD_WINDOW_SECONDS // 60
    upload_key = f"upload:{client}"
    uploads_used, uploads_reset = peek_limit(store, upload_key, now)
    uploads_exhausted = uploads_used >= MAX_UPLOADS_PER_WINDOW

    with st.sidebar:
        st.title(t["sidebar_title"])
        file = st.file_uploader(
            t["uploader_label"],
            type=["pdf"],
            # Stable key so switching language (which changes the label) doesn't make
            # Streamlit treat this as a new widget and drop the uploaded PDF.
            key="pdf_uploader",
        )
        st.caption(t["uploader_caption"])
        if uploads_exhausted:
            # Budget already spent coming into this run (a prior upload, or the file was
            # removed mid-cooldown) — lock the dropzone right away. The × stays usable since
            # it sits outside the dropzone.
            st.markdown(UPLOADER_LOCKED_CSS, unsafe_allow_html=True)
        # Filled below once the current file is counted (the uploader is drawn before that
        # happens, so on the first upload the lock state isn't known here yet).
        upload_notice = st.empty()

    # Render the prompt into a placeholder so it can be cleared the instant a file is
    # uploaded. Otherwise Streamlit leaves the previous run's prompt on screen (dimmed)
    # for the whole blocking indexing call, since that slot isn't re-rendered until the
    # run finishes — emptying it here pushes the clear to the browser right away.
    prompt_box = st.empty()
    if file is None:
        # Show the usage meter from the very start, before anything is uploaded.
        if uploads_exhausted:
            # Locked with nothing loaded — e.g. straight after a reload, where the uploader
            # comes back empty but the IP's budget is still spent. Explain the lock and keep
            # the auto-unlock timer ticking, since we return here without reaching the main UI.
            wait = int((uploads_reset - now) // 60) + 1 if uploads_reset else 1
            upload_notice.caption(
                t["limit_pdfs"].format(
                    max=MAX_UPLOADS_PER_WINDOW, minutes=window_minutes, wait=wait
                )
            )
            if uploads_reset:
                cooldown_refresh(uploads_reset)
        st.sidebar.markdown(f"**{t['limits_header']}**")
        st.sidebar.caption(
            t["usage_pdfs"].format(
                used=uploads_used, max=MAX_UPLOADS_PER_WINDOW, minutes=window_minutes
            )
        )
        st.sidebar.caption(
            t["usage_questions"].format(used=0, max=MAX_QUESTIONS_PER_PDF)
        )
        prompt_box.info(t["upload_prompt"])
        return
    prompt_box.empty()

    missing = [
        name
        for name, value in (
            ("ZAI_API_KEY", ZAI_API_KEY),
            ("ZAI_MODEL", CHAT_MODEL),
            ("GOOGLE_API_KEY", GOOGLE_API_KEY),
        )
        if not value
    ]
    if missing:
        st.error(t["missing_env"].format(names=", ".join(missing)))
        return

    # Ask/answer state. `asking` is on while an answer streams: the submit button renders
    # disabled and a spinner shows. The answer (or error) is kept in session state so it
    # survives the rerun that re-enables the button. `consumed_uploads` records which file_ids
    # this session has already charged to the upload bucket, so reruns don't double-count.
    st.session_state.setdefault("asking", False)
    st.session_state.setdefault("answer", None)
    st.session_state.setdefault("answer_error", None)
    st.session_state.setdefault("consumed_uploads", set())

    file_id = getattr(file, "file_id", None)

    # A file_id this session hasn't charged yet is a fresh upload. The locked dropzone normally
    # blocks a new pick once the cap is hit, but a file selected in the brief gap before the
    # lock applies still lands here — so consume a slot *before* embedding (the costly step),
    # which the server-side bucket rejects if the IP's budget is spent. Clear any prior answer.
    if file_id not in st.session_state.consumed_uploads:
        if not consume_limit(
            store, upload_key, MAX_UPLOADS_PER_WINDOW, UPLOAD_WINDOW_SECONDS, now
        ):
            _, reset_at = peek_limit(store, upload_key, now)
            wait = int((reset_at - now) // 60) + 1 if reset_at else 1
            st.warning(
                t["limit_pdfs"].format(
                    max=MAX_UPLOADS_PER_WINDOW, minutes=window_minutes, wait=wait
                )
            )
            return
        st.session_state.consumed_uploads.add(file_id)
        st.session_state.answer = None
        st.session_state.answer_error = None
        st.session_state.asking = False

    # Re-read the buckets now that the current upload is counted, for the meter + lock.
    uploads_used, uploads_reset = peek_limit(store, upload_key, now)
    question_key = f"q:{client}:{file_id}"
    questions_used, _ = peek_limit(store, question_key, now)

    # Lock the dropzone (Browse + drag-drop) if the upload budget is spent — fires in the same
    # render the "1 of 1 used" meter appears, even on the first upload (whose count lands after
    # the uploader was drawn). CSS is global, so injecting here still styles the live uploader.
    if uploads_used >= MAX_UPLOADS_PER_WINDOW:
        st.markdown(UPLOADER_LOCKED_CSS, unsafe_allow_html=True)
        wait = int((uploads_reset - now) // 60) + 1 if uploads_reset else 1
        upload_notice.caption(
            t["limit_pdfs"].format(
                max=MAX_UPLOADS_PER_WINDOW, minutes=window_minutes, wait=wait
            )
        )
        # Tick on a timer and auto-unlock the moment the window frees up, with no click needed.
        if uploads_reset:
            cooldown_refresh(uploads_reset)

    st.sidebar.markdown(f"**{t['limits_header']}**")
    st.sidebar.caption(
        t["usage_pdfs"].format(
            used=uploads_used, max=MAX_UPLOADS_PER_WINDOW, minutes=window_minutes
        )
    )
    st.sidebar.caption(
        t["usage_questions"].format(used=questions_used, max=MAX_QUESTIONS_PER_PDF)
    )

    try:
        with st.spinner(t["indexing"]):
            retriever = build_retriever(file.getvalue())
        chain = build_chain(retriever)
    except ValueError:
        st.error(t["scanned_pdf"])
        return
    except Exception as e:  # embedding/index failure: API error, bad key or model, etc.
        # Often transient (rate limit, network) — let the user re-run the indexing.
        # The error goes in a placeholder so clicking retry clears it before re-indexing.
        error_box = st.empty()
        if st.button(t["retry"]):
            build_retriever.clear()
            st.rerun()
        else:
            error_box.error(t["index_failed"].format(error=e))
        return

    limit_reached = questions_used >= MAX_QUESTIONS_PER_PDF

    with st.form("ask"):
        question = st.text_input(t["question_label"], key="question", autocomplete="off")
        submitted = st.form_submit_button(
            t["ask_button"], disabled=st.session_state.asking or limit_reached
        )
    # Hold the limit notice until the answer has finished (or been stopped) — showing it
    # mid-stream would flash up the moment the question is submitted.
    if limit_reached and not st.session_state.asking:
        st.warning(t["limit_questions"].format(max=MAX_QUESTIONS_PER_PDF))

    # While an answer streams, show a Stop button under the right of the box. Clicking it
    # is a widget interaction, so Streamlit interrupts the in-progress run (the streaming
    # below) and reruns here with `stop` True — at which point we leave the asking state.
    stop = False
    if st.session_state.asking:
        _, stop_col = st.columns([5, 1])
        with stop_col:
            stop = st.button(t["stop_button"], key="stop", use_container_width=True)
    if stop:
        st.session_state.asking = False
        st.rerun()

    # First click: flip into the asking state and rerun so the button comes back disabled
    # before the blocking stream. The guard ignores reruns that aren't an explicit submit
    # (e.g. uploading a new PDF), so a stale question is never re-answered or re-billed.
    if submitted and question and not st.session_state.asking and not limit_reached:
        # Charge the question now (at submit) — each attempt fires a billed API call, so a
        # stopped or failed answer still counts. consume_limit re-checks the cap server-side;
        # if it just filled (a race on the same IP+PDF), skip and let the rerun show the limit.
        if consume_limit(
            store, question_key, MAX_QUESTIONS_PER_PDF, UPLOAD_WINDOW_SECONDS, now
        ):
            st.session_state.asking = True
            st.session_state.answer = None
            st.session_state.answer_error = None
            st.rerun()

    answer_box = st.empty()
    if st.session_state.asking:
        autoscroll()  # follow the answer down the page as it streams
        try:
            with answer_box.container():
                st.write_stream(
                    capture_answer(
                        stream_with_thinking(
                            chain.stream(st.session_state.question), t["thinking"]
                        )
                    )
                )
        except Exception as e:  # genuine LLM failure: rate limit, network, bad model
            # Let Streamlit's control-flow exceptions (the Stop button's rerun/stop)
            # bubble up; only real errors become an answer error.
            if type(e).__name__ in ("RerunException", "StopException"):
                raise
            st.session_state.answer_error = str(e)
        # Reached only when streaming finished or failed on its own (a Stop interrupt
        # raises past here). Leave the asking state, which re-enables the button.
        st.session_state.asking = False
        st.rerun()
    elif st.session_state.answer_error:
        answer_box.error(t["answer_failed"].format(error=st.session_state.answer_error))
    elif st.session_state.answer:
        answer_box.markdown(st.session_state.answer)


if __name__ == "__main__":
    main()
