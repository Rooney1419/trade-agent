"""Query an Anthropic managed agent and forward the reply to Telegram.

Usage:
    python managed_agent_client.py                          # uses AGENT_PROMPT env var
    python managed_agent_client.py "summarise the market"   # one-shot prompt

Required env vars:
    ANTHROPIC_API_KEY
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Optional env vars:
    MANAGED_AGENT_ID   (default: agent_011CZvYHX9XjGTQAv7XnV87c)
    MANAGED_ENV_ID     (default: env_018YG4QFkAZrNuUbat2RC6XU)
    AGENT_PROMPT       (default prompt when no CLI arg given)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

import anthropic
import requests

BETAS = ["managed-agents-2026-04-01"]

DEFAULT_AGENT_ID = "agent_011CZvYHX9XjGTQAv7XnV87c"
DEFAULT_ENV_ID = "env_018YG4QFkAZrNuUbat2RC6XU"
DEFAULT_PROMPT = "Give me a morning market briefing. Cover major index moves, sector rotation, notable pre-market movers, and any macro catalysts for today."

TELEGRAM_MAX_CHARS = 4096
TELEGRAM_MAX_RETRIES = 3


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
    print(f"Session: {session.id}")

    chunks: list[str] = []

    async with client.beta.sessions.events.stream(session.id, betas=BETAS) as stream:
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
                        print(block.text, end="", flush=True)
                        chunks.append(block.text)
            elif event.type == "agent.tool_use":
                print(f"\n  [tool] {event.name}", flush=True)
            elif event.type == "session.status_idle":
                break
            elif event.type == "error":
                print(f"\n[error] {event}", file=sys.stderr)
                raise SystemExit(1)

    print()
    return "".join(chunks)


async def main() -> None:
    token = _get_required_env("TELEGRAM_BOT_TOKEN")
    chat_id = _get_required_env("TELEGRAM_CHAT_ID")

    agent_id = os.getenv("MANAGED_AGENT_ID", DEFAULT_AGENT_ID)
    env_id = os.getenv("MANAGED_ENV_ID", DEFAULT_ENV_ID)
    prompt = sys.argv[1] if len(sys.argv) > 1 else os.getenv("AGENT_PROMPT", DEFAULT_PROMPT)

    client = anthropic.AsyncAnthropic()

    print(f"Prompt: {prompt}")
    reply = await query_agent(client, agent_id, env_id, prompt)

    if not reply.strip():
        print("Agent returned empty response, skipping Telegram.")
        raise SystemExit(1)

    header = f"<b>Agent Briefing</b>\n\n"
    ok = send_telegram(header + reply, token, chat_id)
    print("Telegram: sent" if ok else "Telegram: FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
