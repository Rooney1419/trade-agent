"""Long-running Telegram bot that routes messages to an Anthropic managed agent.

Send any message to the bot in Telegram and it forwards it to the managed agent,
then sends the agent's reply back to you.

Required env vars:
    ANTHROPIC_API_KEY
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
    MANAGED_AGENT_ID
    MANAGED_ENV_ID

Optional env vars:
    PORT                          (default: 8080, for health check)
    AGENT_TIMEOUT_SECONDS         (default: 180)
    TELEGRAM_SKIP_PENDING_UPDATES (default: true)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock, Thread

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import anthropic
import requests

BETAS = ["managed-agents-2026-04-01"]

TELEGRAM_MAX_CHARS = 4096
TELEGRAM_MAX_RETRIES = 3
TELEGRAM_POLL_TIMEOUT = 30
TELEGRAM_POLL_REQUEST_TIMEOUT = 35
TELEGRAM_RETRY_DELAY = 5
TELEGRAM_POLL_STALE_SECONDS = TELEGRAM_POLL_REQUEST_TIMEOUT + 30
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "180"))


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


SKIP_PENDING_UPDATES = _env_flag("TELEGRAM_SKIP_PENDING_UPDATES", True)


def _get_required_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"Missing required env var: {name}", file=sys.stderr)
        raise SystemExit(1)
    return val


class RuntimeState:
    """Tracks poller and request liveness for the health endpoint."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._poll_last_ok_at: float | None = None
        self._poll_last_error: str | None = None
        self._request_started_at: float | None = None
        self._fatal_error: str | None = None

    def mark_poll_ok(self) -> None:
        with self._lock:
            self._poll_last_ok_at = time.monotonic()
            self._poll_last_error = None

    def mark_poll_error(self, message: str) -> None:
        with self._lock:
            self._poll_last_error = message

    def mark_request_started(self) -> None:
        with self._lock:
            self._request_started_at = time.monotonic()

    def mark_request_finished(self) -> None:
        with self._lock:
            self._request_started_at = None

    def mark_fatal(self, message: str) -> None:
        with self._lock:
            self._fatal_error = message

    def snapshot(self) -> tuple[int, str]:
        with self._lock:
            issues: list[str] = []
            now = time.monotonic()

            if self._fatal_error:
                issues.append(self._fatal_error)

            if self._poll_last_ok_at is None:
                issues.append("telegram poller not ready")
            elif now - self._poll_last_ok_at > TELEGRAM_POLL_STALE_SECONDS:
                issues.append("telegram poller stale")

            if (
                self._request_started_at is not None
                and now - self._request_started_at > AGENT_TIMEOUT_SECONDS + 10
            ):
                issues.append("agent request exceeded timeout window")

            if not issues:
                return 200, "ok"

            if self._poll_last_error:
                issues.append(self._poll_last_error)

            return 503, "; ".join(issues)


def _split_message(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Split text into chunks that fit Telegram's character limit, breaking at newlines."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


def _send_one(payload: dict, token: str) -> bool:
    """Send a single Telegram message with retry."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    for attempt in range(TELEGRAM_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.ok:
                return True

            if resp.status_code == 400 and "can't parse entities" in resp.text.lower():
                payload.pop("parse_mode", None)
                continue

            if resp.status_code == 429:
                retry_after = 5
                try:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 5))
                except (TypeError, ValueError):
                    pass
                print(f"Telegram rate-limited, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            if resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            print(f"Telegram error {resp.status_code}: {resp.text}", file=sys.stderr)
            return False

        except requests.RequestException as exc:
            print(f"Telegram request failed: {exc}", file=sys.stderr)
            time.sleep(2 ** attempt)

    print("Telegram send failed after retries", file=sys.stderr)
    return False


def send_telegram(text: str, token: str, chat_id: str) -> bool:
    """Send a message to Telegram, splitting long messages and using Markdown."""
    parts = _split_message(text)
    ok = True
    for part in parts:
        payload = {
            "chat_id": chat_id,
            "text": part,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        if not _send_one(payload, token):
            ok = False
        if len(parts) > 1:
            time.sleep(0.5)
    return ok


async def query_agent(client: anthropic.AsyncAnthropic, agent_id: str, env_id: str, prompt: str) -> str:
    """Create a session, send the prompt, stream the reply, return full text."""
    chunks: list[str] = []
    stream = None

    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            session = await client.beta.sessions.create(
                agent=agent_id,
                environment_id=env_id,
                betas=BETAS,
            )
            print(f"  session: {session.id}")

            stream = await client.beta.sessions.events.stream(session.id, betas=BETAS)

            await client.beta.sessions.events.send(
                session.id,
                events=[{
                    "type": "user.message",
                    "content": [{"type": "text", "text": prompt}],
                }],
                betas=BETAS,
            )

            async for event in stream:
                if event.type == "agent.message":
                    for block in event.content:
                        if hasattr(block, "text"):
                            chunks.append(block.text)
                elif event.type == "agent.tool_use":
                    print(f"  [tool] {event.name}", flush=True)
                elif event.type == "session.status_idle":
                    break
                elif event.type == "error":
                    print(f"  [error] {event}", file=sys.stderr)
                    return f"Agent error: {event}"
    except TimeoutError:
        print(f"Agent request timed out after {AGENT_TIMEOUT_SECONDS}s", file=sys.stderr)
        return f"Agent timed out after {AGENT_TIMEOUT_SECONDS}s."
    finally:
        if stream is not None:
            try:
                await stream.close()
            except Exception as exc:
                print(f"Failed to close agent stream: {exc}", file=sys.stderr)

    return "".join(chunks)


def _parse_telegram_updates(resp: requests.Response) -> list[dict]:
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError("Telegram response was not a JSON object")

    updates = payload.get("result")
    if not isinstance(updates, list):
        raise ValueError("Telegram response did not include a result list")

    for update in updates:
        if not isinstance(update, dict):
            raise ValueError("Telegram update was not a JSON object")

    return updates


def _bootstrap_telegram_offset(url: str, state: RuntimeState) -> int:
    if not SKIP_PENDING_UPDATES:
        print("Telegram backlog replay enabled; starting from the oldest pending update.")
        return 0

    print("Skipping pending Telegram backlog before entering live mode...")
    offset = 0
    skipped = 0

    while True:
        try:
            resp = requests.get(
                url,
                params={"offset": offset, "timeout": 0, "limit": 100},
                timeout=10,
            )
            if not resp.ok:
                raise RuntimeError(f"Telegram bootstrap error {resp.status_code}: {resp.text}")

            updates = _parse_telegram_updates(resp)
            state.mark_poll_ok()

            if not updates:
                print(f"Skipped {skipped} pending Telegram update(s) on startup.")
                return offset

            for update in updates:
                update_id = update.get("update_id")
                if not isinstance(update_id, int):
                    raise ValueError("Telegram update missing integer update_id")
                offset = update_id + 1
                skipped += 1

        except Exception as exc:
            state.mark_poll_error(f"telegram bootstrap failed: {exc}")
            print(f"Telegram bootstrap failed: {exc}", file=sys.stderr)
            time.sleep(TELEGRAM_RETRY_DELAY)


def poll_telegram_updates(
    token: str,
    chat_id: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    state: RuntimeState,
) -> None:
    """Long-poll Telegram for new messages and push them onto the async queue."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    offset = _bootstrap_telegram_offset(url, state)

    print(f"Polling Telegram for messages (chat_id={chat_id})...")

    while True:
        try:
            resp = requests.get(
                url,
                params={"offset": offset, "timeout": TELEGRAM_POLL_TIMEOUT},
                timeout=TELEGRAM_POLL_REQUEST_TIMEOUT,
            )
            if not resp.ok:
                state.mark_poll_error(f"telegram poll error {resp.status_code}")
                print(f"Telegram poll error: {resp.status_code}", file=sys.stderr)
                time.sleep(TELEGRAM_RETRY_DELAY)
                continue

            updates = _parse_telegram_updates(resp)
            state.mark_poll_ok()

            for update in updates:
                update_id = update.get("update_id")
                if not isinstance(update_id, int):
                    raise ValueError("Telegram update missing integer update_id")

                offset = update_id + 1
                msg = update.get("message", {})
                text = msg.get("text", "").strip()
                sender_chat_id = str(msg.get("chat", {}).get("id", ""))

                if not text or sender_chat_id != chat_id:
                    continue

                print(f"[msg] received {len(text)} chars")
                loop.call_soon_threadsafe(queue.put_nowait, text)

        except Exception as exc:
            state.mark_poll_error(f"telegram poll failed: {exc}")
            print(f"Telegram poll failed: {exc}", file=sys.stderr)
            time.sleep(TELEGRAM_RETRY_DELAY)


class HealthHandler(BaseHTTPRequestHandler):
    state: RuntimeState | None = None

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return

        if self.state is None:
            status, body = 503, "health state unavailable"
        else:
            status, body = self.state.snapshot()

        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8", errors="replace"))

    def log_message(self, *args) -> None:
        pass


def start_health_server(state: RuntimeState) -> None:
    port = int(os.getenv("PORT", "8080"))
    HealthHandler.state = state
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Health server on :{port}")


async def main() -> None:
    anthropic_api_key = _get_required_env("ANTHROPIC_API_KEY")
    token = _get_required_env("TELEGRAM_BOT_TOKEN")
    chat_id = _get_required_env("TELEGRAM_CHAT_ID")
    agent_id = _get_required_env("MANAGED_AGENT_ID")
    env_id = _get_required_env("MANAGED_ENV_ID")
    state = RuntimeState()

    start_health_server(state)

    client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
    queue: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    poll_thread = Thread(
        target=poll_telegram_updates,
        args=(token, chat_id, queue, loop, state),
        daemon=True,
    )
    poll_thread.start()

    print("Ready - send a message in Telegram.\n")

    while True:
        text = await queue.get()

        send_telegram("Thinking...", token, chat_id)

        state.mark_request_started()
        try:
            reply = await query_agent(client, agent_id, env_id, text)
        except Exception as exc:
            print(f"Agent error: {exc}", file=sys.stderr)
            send_telegram(f"Error: {exc}", token, chat_id)
            continue
        finally:
            state.mark_request_finished()

        if not reply.strip():
            send_telegram("Agent returned an empty response.", token, chat_id)
            continue

        send_telegram(reply, token, chat_id)
        print(f"  replied ({len(reply)} chars)")


if __name__ == "__main__":
    asyncio.run(main())
