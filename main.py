import io
import os

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
    content: '1ファイル200MBまで • PDF';
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


def main() -> None:
    # Page styling:
    # - widen the sidebar a little from its default initial width;
    # - the language selectbox is a searchable field, so it shows a text cursor and a
    #   blinking caret (as if you could type) — force a pointer cursor and hide the caret
    #   so the switcher reads as a plain clickable dropdown.
    st.markdown(
        "<style>"
        "[data-testid='stSidebar'] { min-width: 320px; width: 320px; }"
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

    # Render the prompt into a placeholder so it can be cleared the instant a file is
    # uploaded. Otherwise Streamlit leaves the previous run's prompt on screen (dimmed)
    # for the whole blocking indexing call, since that slot isn't re-rendered until the
    # run finishes — emptying it here pushes the clear to the browser right away.
    prompt_box = st.empty()
    if file is None:
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

    # Ask/answer state. `asking` is on while an answer streams: the submit button renders
    # disabled and a spinner shows. The answer (or error) is kept in session state so it
    # survives the rerun that re-enables the button. A different PDF clears the last answer.
    st.session_state.setdefault("asking", False)
    st.session_state.setdefault("answer", None)
    st.session_state.setdefault("answer_error", None)
    file_id = getattr(file, "file_id", None)
    if st.session_state.get("answered_file") != file_id:
        st.session_state.answered_file = file_id
        st.session_state.answer = None
        st.session_state.answer_error = None
        st.session_state.asking = False

    with st.form("ask"):
        question = st.text_input(t["question_label"], key="question", autocomplete="off")
        submitted = st.form_submit_button(
            t["ask_button"], disabled=st.session_state.asking
        )

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
    if submitted and question and not st.session_state.asking:
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
