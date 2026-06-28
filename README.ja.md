# DocumentRAG — PDFと対話するチャットボット

<div align="center">

[![English](https://img.shields.io/badge/README-English-lightgrey?style=for-the-badge)](README.md)
[![日本語](https://img.shields.io/badge/README-日本語-2563eb?style=for-the-badge)](README.ja.md)

</div>

PDFベースのQ&Aチャットボット。PDFをアップロードして質問すると、文書に基づいた回答が
ストリーミングで返ってきます。**Z.AI GLM-5.2** が回答を生成し、Google Gemini が埋め込みを
担当する RAG（検索拡張生成）を、FAISSベクトルストア上で、1つの `main.py` に実装しています。

---

## なぜ作ったか

DocumentRAG は、主に**学習目的**で作りました — 検索拡張生成（RAG）が理論だけでなく、
実際にどう動くのかを端から端まで理解するためです。作る過程で、次のことを学びました：

- **RAG をゼロから** — チャンク分割、埋め込み、ベクトル類似検索、そして取得した文脈に
  LLM を根拠づけ、作り話ではなく出典から答えさせる仕組み。
- **LangChain（LCEL）** — `retriever → prompt → LLM → parser` のパイプラインを `|` で組み、
  そこからトークンをストリーミングする方法。
- **FAISS によるベクトル検索** — テキストを埋め込みに変換し、メモリ上で検索する。
- **複数プロバイダの併用** — OpenAI 互換のチャットモデル（Z.AI GLM）と Google Gemini の
  埋め込みを 1 つのアプリで動かし、きれいな抽象化が差し替えやすさを生むことを体感。
- **根拠づけのためのプロンプト設計** — 文書に何があり何が無いかについて、モデルに正直に
  答えさせるシステムプロンプトの書き方。
- **実用的な Streamlit アプリの構築** — 重い処理のキャッシュ、セッション状態、フォーム、
  リアルタイムのトークンストリーミング、二言語（EN/JA）UI。
- **本番を意識した細部** — `.env` による設定、`uv` での再現可能なインストール、扱いにくい
  入力（スキャンPDF、APIエラー）への丁寧な対処。

---

## 主な機能

- **PDFのQ&A** — サイドバーからPDFをアップロードし、その内容について自然な言葉で質問できます。
- **根拠のある回答** — モデルは文書の内容**のみ**から回答し、文書に答えが無い場合はその旨を
  伝えるため、事実を作り出しません。
- **ストリーミング表示** — モデルが生成するそばから、トークン単位で回答が流れて表示され、
  画面が自動で追従します（上にスクロールすると一時停止します）。
- **いつでも停止** — 長い回答は「停止」ボタンで途中で中断でき、そこまで生成された内容は
  画面に残ります。
- **インデックスは一度だけ** — 埋め込みはファイル単位でキャッシュされるため、再質問は高速で、
  同じ文書に対して何度も課金されません。
- **二言語対応UI（EN/JA）** — 右上の言語セレクターで、画面表示を英語と日本語に切り替えられます。
- **公開デプロイ向けの安全策** — 1MB のアップロード上限に加え、IP単位のレート制限（ウィンドウ
  あたりのPDF数と、PDFごとの質問数）を備え、サイドバーの使用状況メーターにリアルタイム表示します。
  制限はサーバー側で保持されるため、ページを再読み込みしてもリセットされません。上限に達すると
  アップローダーはロックされ、ウィンドウが空くと自動で解除されます。
- **ボット対策（任意）** — [Cloudflare Turnstile](https://www.cloudflare.com/products/turnstile/)
  のゲートでアプリ全体を保護できます。訪問者がチャレンジを通過するまで何も表示されず、通過は
  サーバー側で検証されます。Turnstile のキーが設定されている場合のみ有効になるため、ローカル
  開発では既定でゲートなしで動作します。

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
./local_setup.sh         # 依存関係の同期＋起動 — http://localhost:8501 が開きます

# …手動で行う場合:
uv sync
streamlit run main.py

# または pip の場合:
pip install -r requirements.txt
streamlit run main.py
```

起動前に、プロジェクトルートへ キーを記載した `.env` を作成してください：

```bash
ZAI_API_KEY=your-zai-api-key
ZAI_MODEL=glm-5.2
GOOGLE_API_KEY=your-gemini-api-key

# 任意 — Cloudflare Turnstile ゲートを有効化（有効化には両方が必要）:
TURNSTILE_SITE_KEY=your-turnstile-site-key
TURNSTILE_SECRET_KEY=your-turnstile-secret-key
```

### 環境変数

| 変数 | 必須 | 備考 |
|---|---|---|
| `ZAI_API_KEY` | はい | Z.AI の API キー。チャットモデルで使用。`.env` から読み込み。 |
| `ZAI_MODEL` | はい | Z.AI のチャットモデル ID（例: `glm-5.2`）。 |
| `GOOGLE_API_KEY` | はい | Google Gemini の API キー。埋め込みで使用。`.env` から読み込み。 |
| `TURNSTILE_SITE_KEY` | いいえ | Cloudflare Turnstile の**サイト**キー（公開）。これとシークレットの両方が設定されたときのみゲートが有効。 |
| `TURNSTILE_SECRET_KEY` | いいえ | Cloudflare Turnstile の**シークレット**キー。サーバー側のトークン検証で使用。 |

> `.env` は git 管理外 — 実際のキーは絶対にコミットしない。

### デプロイ設定

`.streamlit/config.toml` が公開時の安全策を設定します：`maxUploadSize = 1`（MBの上限）と
`toolbarMode = "minimal"`（Streamlit の Deploy ボタンを非表示）。レート制限の数値は `main.py`
冒頭の定数 — `MAX_UPLOADS_PER_WINDOW`、`MAX_QUESTIONS_PER_PDF`、`UPLOAD_WINDOW_SECONDS` です。
想定するホストは **Render**（無料枠、`render.yaml` を参照）です。Streamlit は常駐サーバーが
必要で、本アプリはステートレス（FAISS はメモリ上）なため、データベースは不要です。公開デプロイを
ゲートで保護するには、`TURNSTILE_*` の2つのキーを設定します（ホスト名に `*.onrender.com` の
URL と `localhost` を含めた専用の Turnstile ウィジェットを使用してください）。

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
