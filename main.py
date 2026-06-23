import io
import os

import pdfplumber
import streamlit as st
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
        "indexing": "Indexing your document...",
        "missing_env": "Missing required .env value(s): {names}. Add them to your .env file.",
        "scanned_pdf": "No extractable text found — this looks like a scanned or image-only PDF.",
        "index_failed": "Failed to index the document: {error}",
        "retry": "🔄 Retry indexing",
        "question_label": "Ask a question about your document",
        "ask_button": "Ask",
        "answer_failed": "Couldn't get an answer: {error}",
    },
    "ja": {
        "header": "DocumentRAG — PDFと対話するチャットボット",
        "sidebar_title": "ドキュメント",
        "uploader_label": "PDFをここにアップロード",
        "uploader_caption": "📄 一度に1つのPDFのみ — 新しいアップロードで現在のドキュメントと置き換わります。",
        "upload_prompt": "サイドバーからPDFをアップロードして始めましょう。",
        "indexing": "ドキュメントをインデックス化しています...",
        "missing_env": "必須の .env 値が不足しています: {names}。.env ファイルに追加してください。",
        "scanned_pdf": "抽出可能なテキストが見つかりません — スキャン画像のみのPDFのようです。",
        "index_failed": "ドキュメントのインデックス作成に失敗しました: {error}",
        "retry": "🔄 インデックスを再試行",
        "question_label": "ドキュメントについて質問してください",
        "ask_button": "質問する",
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

    with st.form("ask"):
        question = st.text_input(t["question_label"])
        submitted = st.form_submit_button(t["ask_button"])

    # Only invoke on an explicit submit, so an unrelated rerun (e.g. uploading a new
    # PDF) never re-answers a stale question or re-bills the API.
    if submitted and question:
        try:
            st.write_stream(chain.stream(question))
        except Exception as e:  # LLM call failed: rate limit, network, bad model
            st.error(t["answer_failed"].format(error=e))


if __name__ == "__main__":
    main()
