# DocumentRAG — PDFと対話するチャットボット

<div align="center">

[![English](https://img.shields.io/badge/README-English-lightgrey?style=for-the-badge)](README.md)
[![日本語](https://img.shields.io/badge/README-日本語-2563eb?style=for-the-badge)](README.ja.md)

</div>

PDFベースのQ&Aチャットボット。PDFをアップロードして質問すると、文書に基づいた回答が
ストリーミングで返ってきます。**Z.AI GLM-5.2** が回答を生成し、Google Gemini が埋め込みを
担当する RAG（検索拡張生成）を、FAISSベクトルストア上で、1つの `main.py` に実装しています。

---

## 主な機能

- **PDFのQ&A** — サイドバーからPDFをアップロードし、その内容について自然な言葉で質問できます。
- **根拠のある回答** — モデルは文書の内容**のみ**から回答し、文書に答えが無い場合はその旨を
  伝えるため、事実を作り出しません。
- **ストリーミング表示** — モデルが生成するそばから、トークン単位で回答が流れて表示されます。
- **インデックスは一度だけ** — 埋め込みはファイル単位でキャッシュされるため、再質問は高速で、
  同じ文書に対して何度も課金されません。

---

## 技術スタック

| 分野 | 採用技術 |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| PDF解析 | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| オーケストレーション | [LangChain](https://python.langchain.com)（LCEL） |
| チャットLLM | [Z.AI GLM-5.2](https://z.ai)（OpenAI互換、`langchain-openai`） |
| 埋め込み | Google Gemini（`gemini-embedding-001`、`langchain-google-genai`） |
| ベクトルストア | [FAISS](https://github.com/facebookresearch/faiss)（`faiss-cpu`） |
| ランタイム | Python 3.11+ + [uv](https://docs.astral.sh/uv/) |

---

## セットアップ

**前提条件:** Python 3.11+ と [uv](https://docs.astral.sh/uv/)（または pip）。ローカルで動作します。

```bash
uv sync
streamlit run main.py    # http://localhost:8501 が開きます

# または pip の場合:
pip install -r requirements.txt
streamlit run main.py
```

起動前に、プロジェクトルートへ キーを記載した `.env` を作成してください：

```bash
ZAI_API_KEY=your-zai-api-key
ZAI_MODEL=glm-5.2
GOOGLE_API_KEY=your-gemini-api-key
```

### 環境変数

| 変数 | 必須 | 備考 |
|---|---|---|
| `ZAI_API_KEY` | はい | Z.AI の API キー。チャットモデルで使用。`.env` から読み込み。 |
| `ZAI_MODEL` | はい | Z.AI のチャットモデル ID（例: `glm-5.2`）。 |
| `GOOGLE_API_KEY` | はい | Google Gemini の API キー。埋め込みで使用。`.env` から読み込み。 |

> `.env` は git 管理外 — 実際のキーは絶対にコミットしない。

---

## 仕組み

アップロードされたPDFが、そのセッションにおける信頼できる唯一の情報源です。テキストを
オーバーラップ付きのチャンクに分割し、Geminiで埋め込み、FAISSにインデックス化します。
質問ごとに最も関連性の高いチャンクを取得し、それを**唯一の根拠**として **Z.AI GLM-5.2** に
渡して回答させます。設定値（チャンクサイズ、モデル名、temperature、システムプロンプト）は
`main.py` 冒頭の定数にまとまっています。

設計の全体像、リクエストのライフサイクル、各選択の理由については **`ARCHITECTURE.md`**
を参照してください。

---

## 関連ドキュメント

- `ARCHITECTURE.md` — 仕組みと「なぜ」の解説（設計、英語）。
- `DEVELOPMENT_LOG.md` — 構築の経緯を日付付きで記録（英語）。
- `CLAUDE.md` — AI アシスタント向けのリポジトリ規約。
