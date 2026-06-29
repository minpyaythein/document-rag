# DocumentRAG Development Log

A specific, dated record of what was built and decided. Read this when picking up after a break.

---

## Project at a glance

- **Workspace**: local repo `document-rag/`
- **What it is**: a PDF Q&A chatbot — upload a PDF, ask questions, get answers grounded in it (RAG)
- **Stack**: Streamlit 1.55 + LangChain 1.2.x (LCEL) + Z.AI GLM-5.2 (`langchain-openai`, chat) + Google Gemini (`langchain-google-genai`, embeddings) + FAISS (`faiss-cpu` 1.13.2), Python 3.11+ (dev on 3.14), `uv`
- **Models**: chat `glm-5.2` via Z.AI (`temperature=0.3`, `max_tokens=1024`, thinking disabled); embeddings `models/gemini-embedding-001` via Gemini
- **Hosting**: local (`streamlit run main.py`); **Streamlit Community Cloud** is the live deploy (`document-rag-minpyaythein.streamlit.app`) as of 2026-06-29 — deploys from GitHub on push to `main`, Python pinned to 3.12 in the app's Advanced settings.
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

## 2026-06-24: Single-PDF hint + bilingual UI (i18n)

- **Single-PDF hint** — the uploader was already single-file (Streamlit's
  `accept_multiple_files` defaults to off), but nothing told the user. Added an
  always-visible sidebar caption ("One PDF at a time — a new upload replaces the current
  document") plus a matching `help` tooltip.
- **Bilingual UI (EN/JA)** — added in-app i18n. A `TRANSLATIONS` dict (keyed `en`/`ja`) holds
  every UI string; all labels, captions, and error/info messages now render through `t[...]`.
  A language switcher (`st.selectbox`, label hidden) is pinned top-right via
  `st.columns([5, 1])`; the choice lives in `st.session_state["lang"]`, so switching language
  re-renders the whole UI **without dropping the uploaded PDF** (the FAISS index stays cached).
- **Spinner made translatable** — `build_retriever`'s `@st.cache_resource(show_spinner=...)`
  became `show_spinner=False`, with `st.spinner(t["indexing"])` wrapping the call so the
  indexing message follows the selected language.
- **Scope note** — Streamlit's built-in uploader strings ("Drag and drop file here",
  "Browse files", "Limit 200MB per file") stay English; they live inside Streamlit and have no
  public i18n hook. Model answers follow the *question's* language, not the UI toggle.

---

## 2026-06-24: Language-switcher polish + indexing UX fixes

Follow-up pass after i18n landed — styling plus a few state/rendering fixes:
- **Language switcher styling** — the EN/JA selector (top-right) got a pointer cursor and a
  hidden text caret (it's a searchable field underneath), so it reads as a plain dropdown, not
  a text box.
- **Japanese uploader strings** — the JA scope note above is now partly addressed: Streamlit's
  built-in dropzone text is overridden for the JA UI via CSS (`UPLOADER_JA_CSS`, injected only
  when `lang == "ja"`). It targets Streamlit's internal test-ids, so it may need a tweak on a
  Streamlit upgrade.
- **Removed the uploader `?` tooltip** — the single-PDF rule is already shown by the caption
  below it; dropped the now-dead `uploader_help` key.
- **Sidebar widened** to 320px for more breathing room.
- **JA header kept on one line** — the longer Japanese title wrapped; `white-space: nowrap` on
  the heading keeps it single-line.
- **Retry on index failure** — the transient indexing-error branch now shows a "🔄 Retry
  indexing" button that clears the cache and re-runs. It's rendered through an `st.empty()`
  placeholder so clicking it removes the error before retrying. (The scanned-PDF `ValueError`
  has no retry — it isn't recoverable.)
- **Upload prompt no longer lingers dimmed** — it's rendered in a placeholder and emptied the
  instant a file is present, so Streamlit's stale-element dimming doesn't leave it greyed on
  screen during the blocking indexing call.
- **Language switch keeps the PDF** — gave the uploader a stable `key="pdf_uploader"`. Without
  it, the label changing with language made Streamlit treat it as a new widget and drop the
  file (which also forced a re-index); the key keeps the upload and its cached index across a
  switch.

---

## 2026-06-28: Streaming UX — busy state, Stop button, auto-scroll

A pass on the ask/answer experience while a response streams:
- **Busy state** — clicking Ask flips an `st.session_state["asking"]` flag and reruns, so the
  submit button re-renders **disabled** while the answer streams. The "Generating your
  answer..." spinner is held open only until the first token (a `stream_with_thinking`
  generator pulls the first chunk under the spinner, then exits), so it clears the instant
  text starts rolling instead of lingering for the whole response.
- **Stop button** — a right-aligned Stop button appears under the box while streaming and
  interrupts generation. It leans on Streamlit's model: clicking any widget mid-run aborts the
  running script and reruns. A `capture_answer` generator writes each token to
  `st.session_state["answer"]` as it streams, so the **partial answer survives** the interrupt.
  The `except` around `st.write_stream` re-raises Streamlit's control-flow exceptions
  (`RerunException`/`StopException`) so the abort isn't mistaken for an LLM error.
- **Answer persisted** — the latest answer/error lives in session state (cleared on a new
  question or a new PDF, tracked via the uploaded file's id), so it survives the rerun that
  re-enables the button. (Updates ARCHITECTURE trade-off #1, which previously said answers
  weren't persisted.)
- **Auto-scroll** — the page now follows the streamed text down. A 0-height `components.html`
  iframe scrolls the main scroll container (`[data-testid="stMain"]`, *not* the window) to the
  bottom on a short interval, pausing if the user scrolls up and resuming at the bottom.
  Verified against the installed Streamlit that the component iframe is `allow-same-origin`
  (so it can reach the parent page) and that `stMain` is the real scroller — the earlier
  window-scroll attempt did nothing.
- **`autocomplete="off"`** on the question box (native `st.text_input` param) — no browser
  autofill dropdown.
- **`local_setup.sh`** — convenience launcher: checks for `uv`, warns if `.env` is missing,
  runs `uv sync`, then `streamlit run main.py`.

---

## 2026-06-28: Public-deploy guardrails (rate limits, size cap, server-side state)

Prep for a free-tier deploy (Streamlit needs a long-lived server, so Vercel is out; the app is
stateless, so no DB). Added cost/abuse guardrails without a database:

- **`.streamlit/config.toml`** — `maxUploadSize = 1` (MB; was Streamlit's 200MB default — a
  hard stop against OOM on a 512MB free-tier container) and `toolbarMode = "minimal"` (hides the
  Deploy button). Updated the JA uploader CSS text `200MB → 1MB` to match (the EN side reads
  the cap from config automatically).
- **Server-side, IP-keyed rate limiter** — fixed-window `{count, reset_at}` buckets in a
  process-global `@st.cache_resource` dict, keyed by client IP (`X-Forwarded-For`, set by
  the host's proxy). A direct port of the portfolio's
  `server/utils/rate-limit.ts`. Two caps: PDFs per window (`upload:{ip}`) and questions per
  PDF per window (`q:{ip}:{file_id}`). **Key win:** because the state is server-side, a browser
  reload no longer resets the limits — the earlier `st.session_state` version did. Helpers:
  `consume_limit` (≙ `checkRateLimit`), read-only `peek_limit` for the meter, `client_id`
  (≙ `getClientIp`). Window logic verified with a standalone unit test (cap, reload-survival,
  expiry, per-IP isolation).
- **Reload-reset fix (local dev)** — first cut of `client_id` fell back to an id in
  `st.session_state` when no proxy `X-Forwarded-For` was present, so locally every reload (a
  new session) minted a new key and the limit appeared to reset. Fixed by parking the fallback
  id in the **URL query string** (`?rid=...`) instead — it rides along on F5. On a deployed host
  the proxy IP is used directly, so this fallback only matters in local dev.
- **Locked-but-empty UI fix** — after a reload the uploader comes back empty (`file is None`),
  so `main()` returned early — before the lock caption and the auto-unlock fragment. Result:
  dropzone locked with no reason shown and no auto re-enable. Fixed by rendering the cooldown
  caption + uploads meter and scheduling `cooldown_refresh` inside the `file is None` branch
  when the budget is spent.
- **Live usage meter** — bilingual sidebar lines under a "📊 Usage limits" header
  ("Uploads: X of Y · resets every N min", "Questions: X of Y on this PDF"), fed by `peek_limit`.
- **Dropzone lock + auto-unlock** — at the cap, only the dropzone (Browse + drag-drop) is
  locked via CSS (`UPLOADER_LOCKED_CSS`), so the file's × stays usable — unlike
  `file_uploader(disabled=True)`, which disables the whole widget. An invisible
  `@st.fragment(run_every="2s")` reruns the app when the cooldown expires, re-enabling Browse
  with no click. The post-count lock fires in the same render as the meter, so they never desync.
- **State move** — `st.session_state` no longer holds the limits; it keeps only
  `consumed_uploads` (a per-session dedup set so reruns don't double-charge an upload).
- **Limits** (constants in `main.py`): **1 PDF upload per 10-minute window**, **10 questions
  per PDF**. Tunable via `MAX_UPLOADS_PER_WINDOW`, `MAX_QUESTIONS_PER_PDF`,
  `UPLOAD_WINDOW_SECONDS`. The usage meter renders from first load (before any upload), not
  only once a PDF is indexed.
- **Caveats** (same as the portfolio's, already documented there): in-memory resets on a dyno
  restart/sleep; IP keying is coarse (NAT shares a budget, VPN bypasses). Upgrade path noted:
  Upstash Redis if traffic grows.

---

## 2026-06-29: Live on Streamlit Community Cloud

Deployed to **Streamlit Community Cloud** (`document-rag-minpyaythein.streamlit.app`) as the
single host. (An earlier free-tier deploy was trialed and then dropped — Streamlit Cloud felt
faster in normal use, and running one host is simpler.)

- **Deploy**: from `minpyaythein/document-rag` `main`, main file `main.py`, Python pinned to **3.12**
  in Advanced settings (repo `.python-version` is 3.14, which Cloud doesn't offer). Cloud runs
  `main.py` itself and auto-redeploys on push to `main`.
- **Secrets**: pasted as flat TOML keys in the app's Settings → Secrets. Confirmed Streamlit Cloud
  exposes root-level secrets as **environment variables**, so the existing `os.getenv()` config
  loading works unchanged — no `st.secrets` migration needed (keys must stay top-level, not nested).
- **Turnstile**: added `document-rag-minpyaythein.streamlit.app` to the dedicated document-rag
  widget's hostname allowlist (alongside `localhost`).
- **Note**: the rerun-model blanks (Turnstile→main, upload→indexing) are inherent to Streamlit;
  Cloud's only quirk is the explicit click-to-wake screen after idle.
- **Also**: dropped the `"Verifying…"` Turnstile spinner (commit `4d4cf48`) — it barely painted
  before the rerun, so it added noise without filling the gap.

---

## 2026-06-30: Site health monitoring (two layers, ported from the portfolio)

Added uptime + error monitoring, mirroring the portfolio's two-layer design.

- **Layer 1 — external liveness (UptimeRobot).** HTTP(s) **keyword** monitor on
  `https://document-rag-minpyaythein.streamlit.app/healthz`, keyword `ok`, 5-min interval,
  Discord alert contact. Set up in the UptimeRobot dashboard (no code).
  - **Why `/healthz`, not `/_stcore/health`:** verified that `/_stcore/health` **303-redirects
    anonymous clients** (`→ share.streamlit.io/-/auth/app → /-/login?payload=…`) even when the
    app is awake, so a plain monitor never sees `ok` (this is also the "white screen, loads
    forever" symptom in a browser). `/healthz` returns `{"status":"ok"}` directly to anonymous
    clients. App sharing is "anyone with the URL"; the redirect was Streamlit's auth/wake dance.
  - **Caveat:** `/healthz` is answered at Streamlit Cloud's **edge layer** — it proves the
    platform is up, not that the RAG works. Hence Layer 2.
- **Layer 2 — in-app error mirror (Discord).** New `discord_alert.py` (port of the portfolio's
  `server/utils/discord.ts`): `report_error()` fires a throttled (5-min/signature) Discord embed
  on a daemon thread; `post_discord()` is fire-and-forget and fully swallowed. **Stdlib only**
  (`urllib`), so no change to `requirements.txt` / `pyproject.toml`.
  - **Gotcha (fixed):** `post_discord` MUST send an explicit `User-Agent` header — Discord's
    Cloudflare **403-blocks** urllib's default `Python-urllib/x.y` UA, so the first version
    silently failed (403, swallowed) and no alert landed. The portfolio's Node `fetch` never hit
    this because it sends a browser-ish UA. Verified: same payload returns 403 without a UA, 204
    with one.
  - **Wired into the two existing `except` seams in `main.py`:** the embedding/index failure
    (Gemini) and the LLM streaming failure (Z.AI). Filtered out, deliberately: the scanned-PDF
    `ValueError` (user input, the 400 analog) and Streamlit's `RerunException`/`StopException`
    (Stop/rerun control flow). Gated on `DISCORD_ERROR_WEBHOOK_URL` — unset = off, like the
    portfolio's `NUXT_DISCORD_ERROR_WEBHOOK_URL`.
  - **No top-level catch-all** around `main()`: `st.rerun()` raises `RerunException` constantly,
    so a global catch would be noisy/fragile. The two seams cover every genuine upstream failure;
    truly uncaught crashes still surface in Streamlit's own error box.
- **Files**: new `discord_alert.py`; `main.py` (import + `DISCORD_ERROR_WEBHOOK_URL` +
  two `report_error` calls); docs (`CLAUDE.md`, `README.md`/`README.ja.md`, this log, `ARCHITECTURE.md`).
- **Secrets**: add `DISCORD_ERROR_WEBHOOK_URL` to `.env` (local) and Streamlit Cloud → Settings →
  Secrets (live). Can reuse the portfolio's alerts webhook or a dedicated DocumentRAG channel.

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
