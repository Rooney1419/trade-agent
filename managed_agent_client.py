"""Long-running Telegram bot that routes messages to an Anthropic managed agent.

Send any message to the bot in Telegram and it forwards it to the managed agent,
then sends the agent's reply back to you.

Required env vars:
    ANTHROPIC_API_KEY
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Optional env vars:
    MANAGED_AGENT_ID   (default: agent_011CZvYHX9XjGTQAv7XnV87c)
    MANAGED_ENV_ID     (default: env_018YG4QFkAZrNuUbat2RC6XU)
    PORT               (default: 8080, for health check)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import anthropic
import requests

BETAS = ["managed-agents-2026-04-01"]

DEFAULT_AGENT_ID = "agent_011CZvYHX9XjGTQAv7XnV87c"
DEFAULT_ENV_ID = "env_018YG4QFkAZrNuUbat2RC6XU"

TELEGRAM_MAX_CHARS = 4096
TELEGRAM_MAX_RETRIES = 3
POLL_INTERVAL = 2


def _get_required_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"Missing required env var: {name}", file=sys.stderr)
        raise SystemExit(1)
    return val


def send_telegram(text: str, token: str, chat_id: str) -> bool:
    """Send a message to Telegram with retry on transient errors."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    if len(text) > TELEGRAM_MAX_CHARS:
        text = text[: TELEGRAM_MAX_CHARS - 20] + "\n\n… [truncated]"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    for attempt in range(TELEGRAM_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.ok:
                return True

            if resp.status_code == 400 and "can't parse entities" in resp.text.lower():
                payload["parse_mode"] = ""
                continue

            if resp.status_code == 429:
                retry_after = int(resp.json().get("parameters", {}).get("retry_after", 5))
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


async def query_agent(client: anthropic.AsyncAnthropic, agent_id: str, env_id: str, prompt: str) -> str:
    """Create a session, send the prompt, stream the reply, return full text."""
    session = await client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        betas=BETAS,
    )
    print(f"  session: {session.id}")

    chunks: list[str] = []

    stream = await client.beta.sessions.events.stream(session.id, betas=BETAS)

    await client.beta.sessions.events.send(
        session.id,
        events=[{
            "type": "user.message",
            "content": [{"type": "text", "text": prompt}],
        }],
        betas=BETAS,
    )

    try:
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
    finally:
        await stream.close()

    return "".join(chunks)


def poll_telegram_updates(token: str, chat_id: str, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    """Long-poll Telegram for new messages and push them onto the async queue."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    offset = 0

    print(f"Polling Telegram for messages (chat_id={chat_id})...")

    while True:
        try:
            resp = requests.get(url, params={"offset": offset, "timeout": 30}, timeout=35)
            if not resp.ok:
                print(f"Telegram poll error: {resp.status_code}", file=sys.stderr)
                time.sleep(5)
                continue

            for update in resp.json().get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "").strip()
                sender_chat_id = str(msg.get("chat", {}).get("id", ""))

                if not text or sender_chat_id != chat_id:
                    continue

                print(f"[msg] {text[:80]}")
                loop.call_soon_threadsafe(queue.put_nowait, text)

        except requests.RequestException as exc:
            print(f"Telegram poll failed: {exc}", file=sys.stderr)
            time.sleep(5)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def start_health_server() -> None:
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Health server on :{port}")


async def main() -> None:
    token = _get_required_env("TELEGRAM_BOT_TOKEN")
    chat_id = _get_required_env("TELEGRAM_CHAT_ID")
    agent_id = os.getenv("MANAGED_AGENT_ID", DEFAULT_AGENT_ID)
    env_id = os.getenv("MANAGED_ENV_ID", DEFAULT_ENV_ID)

    start_health_server()

    client = anthropic.AsyncAnthropic()
    queue: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    poll_thread = Thread(
        target=poll_telegram_updates,
        args=(token, chat_id, queue, loop),
        daemon=True,
    )
    poll_thread.start()

    print("Ready — send a message in Telegram.\n")

    while True:
        text = await queue.get()

        send_telegram("⏳ Thinking...", token, chat_id)

        try:
            reply = await query_agent(client, agent_id, env_id, text)
        except Exception as exc:
            print(f"Agent error: {exc}", file=sys.stderr)
            send_telegram(f"Error: {exc}", token, chat_id)
            continue

        if not reply.strip():
            send_telegram("Agent returned an empty response.", token, chat_id)
            continue

        send_telegram(reply, token, chat_id)
        print(f"  replied ({len(reply)} chars)")


if __name__ == "__main__":
    asyncio.run(main())
