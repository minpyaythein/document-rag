"""🚨 In-app error mirror → Discord (Layer 2 monitoring).

Mirrors genuine *upstream* failures (Gemini embedding, Z.AI chat) to a Discord channel so
you find out the RAG is broken WITHOUT watching Streamlit Cloud's logs. This is the
complement to Layer 1: the external UptimeRobot keyword monitor on `/healthz` only proves
the deployment/platform is alive — `/healthz` is answered at Streamlit's edge layer and keeps
saying `ok` even while the actual app is erroring. This module closes that gap.

Ported 1:1 from the portfolio's `server/utils/discord.ts` (postDiscord / reportServerError),
keeping the same behaviour: fire-and-forget, fully swallowed (a webhook failure must NEVER
escalate into another error — especially from inside an error handler), with a short
in-memory throttle so a looping failure can't machine-gun the channel.

Stdlib only (urllib) — no new dependency; the app already uses urllib for Turnstile.
Gated on the caller passing a webhook URL: pass a falsy URL (env var unset) and alerting is
simply off.
"""

import json
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timezone

# In-memory throttle so a looping failure (e.g. Z.AI down → every question errors) can't
# machine-gun the channel. Per-process + reset on restart, which is exactly the blast radius
# we care about: the same error signature within the window is suppressed, and a fresh
# deploy/restart re-arms it (you want to know an error is STILL happening after a restart).
# Streamlit reruns the script body on every interaction but imports this module once, so this
# dict persists across reruns for the life of the process.
ERROR_THROTTLE_SECONDS = 5 * 60
_last_alert_at: dict[str, float] = {}


def post_discord(webhook_url: str, payload: dict) -> None:
    """Low-level POST. Caught + swallowed only — callers never rely on it for correctness.

    Runs synchronously here but is invoked on a daemon thread by report_error, so it never
    blocks rendering the user-facing error.
    """
    try:
        request = urllib.request.Request(
            webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            # An explicit User-Agent is REQUIRED: Discord's Cloudflare 403-blocks urllib's
            # default "Python-urllib/x.y" UA. (The portfolio's Node `fetch` never hit this.)
            headers={
                "Content-Type": "application/json",
                "User-Agent": "DocumentRAG-AlertBot/1.0 (+https://document-rag-minpyaythein.streamlit.app)",
            },
            method="POST",
        )
        urllib.request.urlopen(request, timeout=5)
    except Exception as e:  # noqa: BLE001 — must never escalate out of the error handler
        print(f"[discord] webhook post failed: {e}")


def report_error(webhook_url: str | None, route: str, error: BaseException) -> None:
    """Mirror a genuine upstream failure to Discord, throttled, never raising.

    Args:
        webhook_url: Discord webhook; falsy (env var unset) → alerting off, returns silently.
        route: logical source label, e.g. "chat (Z.AI GLM)" / "indexing (Gemini embeddings)".
        error: the caught exception. Its type name stands in for the portfolio's HTTP status.
    """
    if not webhook_url:
        return  # alerting off

    message = str(error) or "(no message)"
    error_type = type(error).__name__

    # Signature dedup: same route + type + message inside the window stays quiet.
    signature = f"{route}::{error_type}::{message}"[:200]
    now = time.time()
    previous = _last_alert_at.get(signature)
    if previous is not None and now - previous < ERROR_THROTTLE_SECONDS:
        return  # recently alerted — stay quiet
    _last_alert_at[signature] = now

    # Opportunistic prune so the dict can't grow unbounded on a long-lived instance.
    if len(_last_alert_at) > 200:
        for key, at in list(_last_alert_at.items()):
            if now - at > ERROR_THROTTLE_SECONDS:
                del _last_alert_at[key]

    # Discord caps: title 256, description 4096, field value 1024. Stay well under.
    tb = traceback.format_exc()
    stack_block = (
        "```\n" + tb[:900] + "\n```"
        if tb and tb.strip() != "NoneType: None"
        else "_(no traceback)_"
    )

    payload = {
        "username": "🚨 DocumentRAG Alerts",
        "embeds": [
            {
                "title": f"🚨 {route} → {error_type}"[:256],
                "description": message[:1500],
                "color": 0xDC2626,  # red
                "fields": [{"name": "Traceback", "value": stack_block}],
                "footer": {"text": f"source: {route}"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }

    # Fire on a daemon thread so showing the user their error isn't blocked on the webhook
    # round-trip. Daemon so it never holds up process shutdown.
    threading.Thread(
        target=post_discord, args=(webhook_url, payload), daemon=True
    ).start()
