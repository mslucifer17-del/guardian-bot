is code ko update kar do jo bhi dikkat hai sab thik karo and ise bhot jata powerful bana do
ye meri channel id hai isse aane wali post ko delete na kare -1003330141433, upra code update kar ke de do

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.errors import UniqueViolation

from flask import Flask
from waitress import serve

from telegram import (
    ChatPermissions,
    Message,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Optional AI (Gemini) for final-stage spam classification
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# ----------------------------
# Settings & Initialization
# ----------------------------

@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    database_url: str
    admin_user_ids: Tuple[int, ...]
    port: int
    ai_enabled: bool
    gemini_api_key: Optional[str]
    max_warnings_default: int
    channel_id: Optional[int]   # main allowed channel for posts/forwards
    promotion_text: str
    spam_ai_model: str
    ai_timeout_sec: float
    deletion_delay_sec: int
    min_seconds_between_msgs: float
    min_seconds_between_same_user_msgs: float

    @staticmethod
    def from_env() -> "Settings":
        admin_ids_str = os.environ.get("ADMIN_USER_IDS", "")
        admin_ids = tuple(
            int(x.strip()) for x in admin_ids_str.split(",") if x.strip()
        )
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        db_url = os.environ.get("DATABASE_URL", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", None)
        port = int(os.environ.get("PORT", 8080))
        channel_id_str = os.environ.get("CHANNEL_ID", "").strip()
        channel_id = int(channel_id_str) if channel_id_str else None

        ai_enabled_env = os.environ.get("SPAM_AI_ENABLED", "true").lower() in ("1", "true", "yes")
        ai_enabled = ai_enabled_env and GEMINI_AVAILABLE and bool(gemini_key)

        max_warn = int(os.environ.get("MAX_WARNINGS", 5))
        return Settings(
            telegram_bot_token=telegram_token,
            database_url=db_url,
            admin_user_ids=admin_ids,
            port=port,
            ai_enabled=ai_enabled,
            gemini_api_key=gemini_key,
            max_warnings_default=max_warn,
            channel_id=channel_id,
            promotion_text=os.environ.get("PROMOTION_TEXT", "For promotions, join: https://t.me/+scHqQ2SR0J45NjQ1"),
            spam_ai_model=os.environ.get("SPAM_AI_MODEL", "gemini-2.5-flash"),
            ai_timeout_sec=float(os.environ.get("AI_TIMEOUT_SEC", "7.0")),
            deletion_delay_sec=int(os.environ.get("DELETION_DELAY_SEC", "10")),
            min_seconds_between_msgs=float(os.environ.get("MIN_SECONDS_BETWEEN_MSGS", "2.0")),
            min_seconds_between_same_user_msgs=float(os.environ.get("MIN_SECONDS_BETWEEN_SAME_USER_MSGS", "2.0")),
        )


def validate_settings(settings: Settings) -> None:
    missing: list[str] = []
    if not settings.telegram_bot_token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not settings.database_url:
        missing.append("DATABASE_URL")

    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    if settings.ai_enabled and not (GEMINI_AVAILABLE and settings.gemini_api_key):
        logging.warning("AI enabled but Gemini not properly configured; disabling AI.")
        object.__setattr__(settings, "ai_enabled", False)  # type: ignore


# ----------------------------
# Logging
# ----------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("guardian-bot")


# ----------------------------
# Database Pool & Helpers
# ----------------------------

class Database:
    def __init__(self, dsn: str):
        # ThreadedConnectionPool is safe with asyncio if used briefly per query.
        # Keep DB interactions short and minimal.
        self._pool: pg_pool.ThreadedConnectionPool = pg_pool.ThreadedConnectionPool(
            minconn=1, maxconn=10, dsn=dsn
        )

    @contextmanager
    def conn(self):
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def close(self):
        self._pool.closeall()


db: Optional[Database] = None


def setup_database() -> None:
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS blacklist (
                id SERIAL PRIMARY KEY,
                word TEXT NOT NULL UNIQUE,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS allowed_chats (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT NOT NULL UNIQUE,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS custom_commands (
                id SERIAL PRIMARY KEY,
                command TEXT NOT NULL UNIQUE,
                response TEXT NOT NULL,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reported_spam (
                id SERIAL PRIMARY KEY,
                message TEXT NOT NULL,
                reported_by BIGINT,
                reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forward_whitelist (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL UNIQUE,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS allowed_channels (
                id SERIAL PRIMARY KEY,
                channel_id BIGINT NOT NULL UNIQUE,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_commands (
                id SERIAL PRIMARY KEY,
                command TEXT NOT NULL UNIQUE,
                action_type TEXT NOT NULL,
                parameters JSONB,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bot_settings (
                id SERIAL PRIMARY KEY,
                setting_key TEXT NOT NULL UNIQUE,
                setting_value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()


# ----------------------------
# State & Caches
# ----------------------------

@dataclass
class BotState:
    # in-memory counters
    user_warnings: Dict[int, int]
    user_last_message_time: Dict[int, datetime]

    # allowlists
    blacklist_words: Set[str]
    allowed_chats: Set[int]
    forward_whitelist_users: Set[int]
    allowed_channels: Set[int]

    # dynamic commands cache: command -> {action_type, parameters}
    dynamic_commands: Dict[str, Dict[str, Any]]

    # config settings that can be overridden by DB
    max_warnings: int

    # simple per-user rate limiting across chats
    message_rate_limit_ts: Dict[Tuple[int, int], float]


state = BotState(
    user_warnings={},
    user_last_message_time={},
    blacklist_words=set(),
    allowed_chats=set(),
    forward_whitelist_users=set(),
    allowed_channels=set(),
    dynamic_commands={},
    max_warnings=5,
    message_rate_limit_ts={},
)


def load_blacklist_and_seed_critical(settings: Settings) -> None:
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT word FROM blacklist;")
        rows = cur.fetchall()
        state.blacklist_words = {r[0].lower() for r in rows}

        critical = {
            "cp", "child", "porn", "premium", "collection", "price",
            "payment", "purchase", "desi", "indian", "foreign", "tamil",
            "chinese", "arabian", "bro-sis", "dad-daughter", "pedo"
        }
        inserted = 0
        for w in critical:
            if w not in state.blacklist_words:
                try:
                    cur.execute(
                        "INSERT INTO blacklist (word, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                        (w, settings.admin_user_ids[0] if settings.admin_user_ids else 0),
                    )
                    inserted += 1
                except Exception:
                    pass
        conn.commit()
        if inserted:
            cur.execute("SELECT word FROM blacklist;")
            state.blacklist_words = {r[0].lower() for r in cur.fetchall()}
    logger.info("Loaded %d blacklist words.", len(state.blacklist_words))


def load_allowed_chats() -> None:
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT chat_id FROM allowed_chats;")
        state.allowed_chats = {r[0] for r in cur.fetchall()}
    logger.info("Loaded %d allowed chats.", len(state.allowed_chats))


def load_forward_whitelist() -> None:
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT user_id FROM forward_whitelist;")
        state.forward_whitelist_users = {r[0] for r in cur.fetchall()}
    logger.info("Loaded %d forward-whitelisted users.", len(state.forward_whitelist_users))


def load_allowed_channels(settings: Settings) -> None:
    assert db is not None
    ids: Set[int] = set()
    if settings.channel_id:
        ids.add(settings.channel_id)
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT channel_id FROM allowed_channels;")
        ids |= {r[0] for r in cur.fetchall()}
    state.allowed_channels = ids
    logger.info("Loaded %d allowed channels (incl. configured).", len(state.allowed_channels))


def load_dynamic_commands() -> None:
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT command, action_type, parameters FROM dynamic_commands;")
        state.dynamic_commands = {
            r[0]: {"action_type": r[1], "parameters": (r[2] or {})} for r in cur.fetchall()
        }
    logger.info("Loaded %d dynamic commands.", len(state.dynamic_commands))


def load_bot_settings(settings: Settings) -> None:
    assert db is not None
    state.max_warnings = settings.max_warnings_default
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT setting_key, setting_value FROM bot_settings;")
        for key, value in cur.fetchall():
            if key == "max_warnings":
                try:
                    state.max_warnings = max(1, int(value))
                except ValueError:
                    logger.warning("Invalid max_warnings in DB: %s", value)
    logger.info("MAX_WARNINGS = %d", state.max_warnings)


async def update_setting(key: str, value: str) -> bool:
    assert db is not None
    try:
        with db.conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bot_settings (setting_key, setting_value)
                VALUES (%s, %s)
                ON CONFLICT (setting_key)
                DO UPDATE SET setting_value = EXCLUDED.setting_value, updated_at = CURRENT_TIMESTAMP;
                """,
                (key, value),
            )
            conn.commit()
        return True
    except Exception as e:
        logger.exception("Failed to update setting %s: %s", key, e)
        return False


# ----------------------------
# AI Spam Detection (Optional)
# ----------------------------

SPAM_DETECTION_PROMPT = """You are a vigilant spam detection AI for Telegram. Analyze messages and:
1. Reply with "SPAM" if it contains promotions, scams, ads, or suspicious links.
2. Reply with "OK" for normal messages.
3. Consider cultural context and multiple languages.
4. Pay special attention to child sexual abuse material, adult content, selling, and illegal activities. 
Return only "SPAM" or "OK".
"""

ai_model = None  # will be set if enabled


def ai_init(settings: Settings) -> None:
    global ai_model
    if not settings.ai_enabled:
        logger.info("AI spam detection disabled.")
        return
    try:
        genai.configure(api_key=settings.gemini_api_key)
        ai_model = genai.GenerativeModel(
            model_name=settings.spam_ai_model,
            system_instruction=SPAM_DETECTION_PROMPT
        )
        logger.info("AI model initialized: %s", settings.spam_ai_model)
    except Exception as e:
        logger.warning("Failed to init AI model: %s. Continuing without AI.", e)


async def ai_is_spam(text: str, normalized: str, timeout_sec: float) -> Optional[bool]:
    if ai_model is None:
        return None
    try:
        analysis = f"Original: {text}\nNormalized: {normalized}"
        resp = await asyncio.wait_for(ai_model.generate_content_async(analysis), timeout=timeout_sec)  # type: ignore
        decision = (resp.text or "").strip().upper()
        if "SPAM" in decision:
            return True
        if "OK" in decision:
            return False
    except asyncio.TimeoutError:
        logger.warning("AI timeout.")
    except Exception as e:
        logger.warning("AI error: %s", e)
    return None


# ----------------------------
# Spam Rules (Precompiled)
# ----------------------------

SPAM_PATTERNS = [
    r"lowest price", r"premium collection", r"dm for purchase", r"payment method",
    r"combo novd", r"latest updated", r"desi cp", r"indian cp", r"foreign cp",
    r"tamil cp", r"chinese cp", r"arabians? cp", r"bro-?sis cp", r"dad-?daughter cp",
    r"pedo (?:mom-?son|dad-?daughter) cp", r"premium cp", r"content collection",
    r"price available", r"(?:upi|paypal|crypto|gift card)", r"nude", r"service available",
    r"video call"
]
COMPILED_SPAM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SPAM_PATTERNS]

PAYMENT_TERMS = {"upi", "paypal", "crypto", "gift card", "payment", "purchase", "price", "💰", "📣", "🟢"}

# Link/mention detection:
# We'll rely primarily on Telegram entities to avoid false positives (e.g., emails).
URL_ENTITY_TYPES = {"url", "text_link"}
MENTION_ENTITY_TYPES = {"mention", "text_mention"}

HIDDEN_LINK_REGEXES = [
    re.compile(r"[🟢💰📣⬇️➖➗]+\s*[\w\s]+\s*[🟢💰📣⬇️➖➗]+"),
    re.compile(r"(?:http|https)://[^\s]+", re.IGNORECASE),
    re.compile(r"(?:(?:\w[\w-]*\.)+[A-Za-z]{2,})"),  # domain-like pattern
    re.compile(r"@[\w]+\s*(offer|deal|price|sale)", re.IGNORECASE),
]

NORMALIZE_NONWORD = re.compile(r"[^\w\s@\.\-$]")
NORMALIZE_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = NORMALIZE_NONWORD.sub(" ", text)
    text = NORMALIZE_SPACES.sub(" ", text).strip().lower()
    return text


def contains_hidden_links(text: str) -> bool:
    return any(rx.search(text) for rx in HIDDEN_LINK_REGEXES)


def detect_spam_patterns(text: str) -> Tuple[bool, str]:
    for rx in COMPILED_SPAM_PATTERNS:
        if rx.search(text):
            return True, f"Pattern matched: {rx.pattern}"
    return False, ""


# ----------------------------
# Utilities
# ----------------------------

def is_admin(user_id: int, settings: Settings) -> bool:
    return user_id in settings.admin_user_ids


async def safe_delete_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int) -> None:
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception as e:
        msg = str(e).lower()
        if "message to delete not found" in msg:
            logger.debug("Message %s already deleted in chat %s.", message_id, chat_id)
        elif "timed out" in msg:
            logger.warning("Timeout deleting message %s in chat %s.", message_id, chat_id)
        elif "not enough rights" in msg:
            logger.warning("No rights to delete in chat %s.", chat_id)
        else:
            logger.error("Delete failed for %s in %s: %s", message_id, chat_id, e)


async def delete_message_after_delay(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int, delay: int) -> None:
    await asyncio.sleep(delay)
    await safe_delete_message(context, chat_id, message_id)


async def send_ephemeral(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, settings: Settings, include_promo: bool = True) -> Optional[Message]:
    if not update.effective_message:
        return None
    final = f"{text}\n\n{settings.promotion_text}" if (include_promo and "promotion" not in text.lower()) else text
    msg = await update.effective_message.reply_text(final)
    context.application.create_task(delete_message_after_delay(context, update.effective_chat.id, msg.message_id, settings.deletion_delay_sec))
    return msg


# ----------------------------
# Commands (Admin and User)
# ----------------------------

async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🛡️ Hello! I'm Guardian Bot.\n\nI protect groups from spam and scams with layered detection.\nUse /help (admins) for commands."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can use this command", settings)
        return
    txt = f"""
🛡️ Admin Commands
/addword <w1> <w2> ... — Add blacklist words
/block <w1> <w2> ... — Alias for /addword
/addcommand <name> <response> — Add custom text command
/adddynamic <command> <action_type> <json> — Add dynamic command
/listcommands — List custom & dynamic commands
/allowchat <chat_id> — Allow a chat by ID
/allowthischat — Allow current chat
/listchats — List allowed chats
/allowforward <user_id or reply> — Allow user to forward
/revokeforward <user_id or reply> — Revoke forwarding
/listforwarders — List forward-enabled users
/allowchannel <channel_id> — Allow channel ID to post/forward
/setmaxwarnings <n> — Set ban threshold (Current: {state.max_warnings})
/stats — Show protection stats
/botversion — Show bot version
/reload — Reload settings & caches
"""
    await update.message.reply_text(txt)


async def cmd_botversion(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("✅ Guardian Bot v3.0 — Pooled DB, safer rules, dynamic cmds.")


async def cmd_addword(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can add words", settings)
        return
    words = {w.lower() for w in context.args}
    if not words:
        await send_ephemeral(update, context, "Usage: /addword <word1> <word2> ...", settings)
        return
    assert db is not None
    added = 0
    with db.conn() as conn, conn.cursor() as cur:
        for w in words:
            try:
                cur.execute(
                    "INSERT INTO blacklist (word, added_by) VALUES (%s, %s);",
                    (w, update.effective_user.id),
                )
                added += 1
            except UniqueViolation:
                conn.rollback()
            except Exception:
                conn.rollback()
        conn.commit()
    load_blacklist_and_seed_critical(settings)
    await update.message.reply_text(f"✅ Added {added} word(s) to blacklist")


async def cmd_block(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # alias to addword
    await cmd_addword(update, context)


async def cmd_addcommand(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admins can add commands", settings)
        return
    if len(context.args) < 2:
        await send_ephemeral(update, context, "Usage: /addcommand <name> <response>", settings)
        return
    name = context.args[0].lower().lstrip("/")
    response = " ".join(context.args[1:])
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO custom_commands (command, response, added_by) VALUES (%s, %s, %s);",
                (name, response, update.effective_user.id),
            )
            conn.commit()
            await update.message.reply_text(f"✅ Command /{name} added.")
        except UniqueViolation:
            await update.message.reply_text(f"❌ Command /{name} already exists")
        except Exception as e:
            conn.rollback()
            await update.message.reply_text(f"❌ Error: {e}")


async def cmd_adddynamic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admins can add dynamic commands", settings)
        return

    if len(context.args) < 3:
        await send_ephemeral(update, context, "Usage: /adddynamic <command> <action_type> <json_parameters>", settings)
        return

    name = context.args[0].lower().lstrip("/")
    action = context.args[1].lower()
    try:
        params = json.loads(" ".join(context.args[2:]))
    except json.JSONDecodeError:
        await send_ephemeral(update, context, "❌ Invalid JSON parameters", settings)
        return

    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO dynamic_commands (command, action_type, parameters, added_by) VALUES (%s, %s, %s, %s);",
                (name, action, json.dumps(params), update.effective_user.id),
            )
            conn.commit()
            load_dynamic_commands()
            await update.message.reply_text(f"✅ Dynamic command /{name} added.")
        except UniqueViolation:
            await update.message.reply_text(f"❌ Command /{name} already exists")
        except Exception as e:
            conn.rollback()
            await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listcommands(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT command, response FROM custom_commands;")
        custom = cur.fetchall()
        cur.execute("SELECT command, action_type FROM dynamic_commands;")
        dyn = cur.fetchall()

    lines = ["📋 Available Commands:\n", "🔹 Custom:"]
    for cmd, resp in custom:
        preview = (resp[:50] + "...") if len(resp) > 53 else resp
        lines.append(f"/{cmd} — {preview}")
    lines.append("\n🔹 Dynamic:")
    for cmd, act in dyn:
        lines.append(f"/{cmd} — {act}")

    await update.message.reply_text("\n".join(lines))


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not (update.message and update.message.reply_to_message):
        await send_ephemeral(update, context, "❌ Reply to a spam message to report it.", settings)
        return
    spam_text = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO reported_spam (message, reported_by) VALUES (%s, %s);",
            (spam_text, update.effective_user.id),
        )
        conn.commit()
    await update.message.reply_text("✅ Spam reported. Thanks for helping!")


async def cmd_allowchat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, f"❌ Only admins can allow chats (Your ID: {update.effective_user.id})", settings)
        return
    if not context.args:
        await send_ephemeral(update, context, "Usage: /allowchat <chat_id>", settings)
        return
    try:
        chat_id = int(context.args[0])
    except ValueError:
        await send_ephemeral(update, context, "❌ Invalid chat ID (must be a number).", settings)
        return

    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO allowed_chats (chat_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (chat_id, update.effective_user.id),
        )
        conn.commit()
    state.allowed_chats.add(chat_id)
    await update.message.reply_text(f"✅ Chat {chat_id} allowed.")
    logger.info("Chat %s allowed by admin %s", chat_id, update.effective_user.id)


async def cmd_allowthischat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can use this command.", settings)
        return
    chat_id = update.effective_chat.id
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO allowed_chats (chat_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (chat_id, update.effective_user.id),
        )
        conn.commit()
    state.allowed_chats.add(chat_id)
    await update.message.reply_text(f"✅ Now active in this chat (ID: {chat_id}).")


async def cmd_listchats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can view allowed chats", settings)
        return
    if not state.allowed_chats:
        await update.message.reply_text("No chats are currently allowed.")
        return
    await update.message.reply_text("Allowed chats:\n" + "\n".join(str(cid) for cid in sorted(state.allowed_chats)))


async def cmd_allowforward(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can use this command.", settings)
        return

    target_user_id: Optional[int] = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target_user_id = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target_user_id = int(context.args[0])
        except ValueError:
            pass

    if not target_user_id:
        await send_ephemeral(update, context, "Usage: reply to a user or use /allowforward <user_id>", settings)
        return

    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO forward_whitelist (user_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (target_user_id, update.effective_user.id),
        )
        conn.commit()
    state.forward_whitelist_users.add(target_user_id)
    await update.message.reply_text(f"✅ User {target_user_id} can now forward messages.")


async def cmd_revokeforward(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can use this command.", settings)
        return

    target_user_id: Optional[int] = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target_user_id = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target_user_id = int(context.args[0])
        except ValueError:
            pass

    if not target_user_id:
        await send_ephemeral(update, context, "Usage: reply to a user or use /revokeforward <user_id>", settings)
        return

    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM forward_whitelist WHERE user_id = %s;", (target_user_id,))
        conn.commit()
    state.forward_whitelist_users.discard(target_user_id)
    await update.message.reply_text(f"❌ User {target_user_id} can no longer forward messages.")


async def cmd_listforwarders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can view this list.", settings)
        return
    if not state.forward_whitelist_users:
        await update.message.reply_text("No users are currently allowed to forward messages.")
        return
    await update.message.reply_text("Users allowed to forward:\n" + "\n".join(str(uid) for uid in sorted(state.forward_whitelist_users)))


async def cmd_allowchannel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can allow channels", settings)
        return
    if not context.args:
        await send_ephemeral(update, context, "Usage: /allowchannel <channel_id>", settings)
        return
    try:
        channel_id = int(context.args[0])
    except ValueError:
        await send_ephemeral(update, context, "❌ Invalid channel ID.", settings)
        return

    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO allowed_channels (channel_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (channel_id, update.effective_user.id),
        )
        conn.commit()
    state.allowed_channels.add(channel_id)
    await update.message.reply_text(f"✅ Channel {channel_id} added to allowed channels.")


async def cmd_setmaxwarnings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can change warning threshold", settings)
        return
    if not context.args or len(context.args) != 1:
        await send_ephemeral(update, context, "Usage: /setmaxwarnings <number>", settings)
        return
    try:
        new_max = max(1, int(context.args[0]))
    except ValueError:
        await send_ephemeral(update, context, "❌ Please provide a valid number", settings)
        return

    if await update_setting("max_warnings", str(new_max)):
        state.max_warnings = new_max
        await update.message.reply_text(f"✅ Max warnings set to {new_max}")
        logger.info("Max warnings changed to %d by %s", new_max, update.effective_user.id)
    else:
        await update.message.reply_text("❌ Failed to update warning threshold.")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can view stats", settings)
        return
    txt = f"""
📊 Guardian Bot Statistics

• Blacklisted words: {len(state.blacklist_words)}
• Allowed chats: {len(state.allowed_chats)}
• Allowed forwarders: {len(state.forward_whitelist_users)}
• Allowed channels: {len(state.allowed_channels)}
• Active warnings: {len(state.user_warnings)}
• AI Enabled: {"Yes" if settings.ai_enabled else "No"}
• Dynamic commands: {len(state.dynamic_commands)}
• Max warnings before ban: {state.max_warnings}
"""
    await update.message.reply_text(txt)


async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Only admin can reload settings", settings)
        return
    load_blacklist_and_seed_critical(settings)
    load_allowed_chats()
    load_forward_whitelist()
    load_allowed_channels(settings)
    load_dynamic_commands()
    load_bot_settings(settings)
    await update.message.reply_text("✅ Reloaded configuration & caches.")


# ----------------------------
# Custom & Dynamic Command Handlers
# ----------------------------

async def handle_custom_command(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not (update.message and update.message.text):
        return
    cmd = update.message.text.split()[0][1:].lower()
    assert db is not None
    with db.conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT response FROM custom_commands WHERE command = %s;", (cmd,))
        row = cur.fetchone()
    if row:
        await update.message.reply_text(row[0])


async def handle_dynamic_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not (update.message and update.message.text):
        return
    settings: Settings = context.bot_data["settings"]
    cmd = update.message.text.split()[0][1:].lower()
    if cmd not in state.dynamic_commands:
        return

    entry = state.dynamic_commands[cmd]
    action = entry["action_type"]
    params: Dict[str, Any] = entry.get("parameters") or {}
    try:
        if action == "reply":
            await update.message.reply_text(params.get("text", "No text specified"))
        elif action == "restrict_user":
            if not is_admin(update.effective_user.id, settings):
                await send_ephemeral(update, context, "❌ Only admins can use this command.", settings)
                return
            user_id = params.get("user_id")
            if not user_id:
                await update.message.reply_text("❌ No user_id specified in parameters.")
                return
            perms = ChatPermissions(
                can_send_messages=False,
                can_send_audios=False,
                can_send_documents=False,
                can_send_photos=False,
                can_send_videos=False,
                can_send_video_notes=False,
                can_send_voice_notes=False,
                can_send_polls=False,
                can_add_web_page_previews=False,
                can_change_info=False,
                can_invite_users=False,
                can_pin_messages=False,
            )
            await context.bot.restrict_chat_member(
                chat_id=update.effective_chat.id,
                user_id=int(user_id),
                permissions=perms,
            )
            await update.message.reply_text(f"✅ User {user_id} restricted.")
        elif action == "delete_message":
            if update.message.reply_to_message:
                try:
                    await update.message.reply_to_message.delete()
                    await update.message.delete()
                except Exception as e:
                    await update.message.reply_text(f"❌ Could not delete message: {e}")
            else:
                await update.message.reply_text("❌ Reply to a message to delete it.")
        # Add more actions as needed
    except Exception as e:
        await update.message.reply_text(f"❌ Error executing command: {e}")


# ----------------------------
# Main Message Handler
# ----------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    msg = update.effective_message
    if not msg or not msg.from_user:
        return

    user = msg.from_user
    chat_id = update.effective_chat.id

    # Allow direct posts from explicitly allowed channels
    if update.channel_post and update.channel_post.chat and update.channel_post.chat.id in state.allowed_channels:
        logger.info("Allowed channel post from %s", update.channel_post.chat.id)
        return

    # Ignore all chats not allowlisted
    if chat_id not in state.allowed_chats:
        return

    # Ignore all bot senders, but schedule deletion to keep clean
    if user.is_bot:
        context.application.create_task(delete_message_after_delay(context, chat_id, msg.message_id, settings.deletion_delay_sec))
        return

    # Simple rate-limit per user per chat
    now_ts = time.time()
    key = (user.id, chat_id)
    last_ts = state.message_rate_limit_ts.get(key, 0.0)
    if now_ts - last_ts < settings.min_seconds_between_msgs:
        return
    state.message_rate_limit_ts[key] = now_ts

    # Per-user message spacing (delete flood)
    now_dt = datetime.now()
    last_dt = state.user_last_message_time.get(user.id)
    if last_dt and (now_dt - last_dt).total_seconds() < settings.min_seconds_between_same_user_msgs:
        await safe_delete_message(context, chat_id, msg.message_id)
        return
    state.user_last_message_time[user.id] = now_dt

    # Admins are exempt
    if is_admin(user.id, settings):
        return

    # Forwarding rules:
    # - Allow forwards from allowed channels
    # - Allow forwards by forward-whitelisted users
    if (msg.forward_from or msg.forward_from_chat):
        origin_allowed = (
            (msg.forward_from_chat and msg.forward_from_chat.id in state.allowed_channels)
        )
        user_allowed = user.id in state.forward_whitelist_users
        if not (origin_allowed or user_allowed):
            await enforce_spam(context, chat_id, user.id, msg, "You do not have permission to forward messages.", settings)
            return

    text = msg.text or msg.caption or ""
    text_lower = text.lower()
    normalized = normalize_text(text)
    entities = msg.entities or []
    caption_entities = msg.caption_entities or []
    all_entities = list(entities) + list(caption_entities)

    # Rule pipeline: cheap checks first
    is_spam = False
    reason = ""

    # Links disallowed unless admin; check using entities
    if any(e.type in URL_ENTITY_TYPES for e in all_entities):
        is_spam, reason = True, "Links are not allowed"

    # Mentions disallowed (avoid emails): rely on entity typing
    if not is_spam and any(e.type in MENTION_ENTITY_TYPES for e in all_entities) and not text.startswith("/"):
        is_spam, reason = True, "Mentions are not allowed"

    # Hidden links / obfuscated signals
    if not is_spam and contains_hidden_links(text):
        is_spam, reason = True, "Hidden/obfuscated links detected"

    # Blacklist
    if not is_spam and any(w in text_lower or w in normalized for w in state.blacklist_words):
        is_spam, reason = True, "Blacklisted term detected"

    # Payment/commerce cues
    if not is_spam and any(term in text_lower for term in PAYMENT_TERMS):
        is_spam, reason = True, "Payment terms detected"

    # Pattern library
    if not is_spam:
        matched, why = detect_spam_patterns(text_lower)
        if matched:
            is_spam, reason = True, why

    # AI final check (optional)
    if not is_spam and text and settings.ai_enabled:
        verdict = await ai_is_spam(text, normalized, settings.ai_timeout_sec)
        if verdict is True:
            is_spam, reason = True, "AI detected spam content"

    if is_spam:
        await enforce_spam(context, chat_id, user.id, msg, reason, settings)


async def enforce_spam(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    msg: Message,
    reason: str,
    settings: Settings,
) -> None:
    try:
        await msg.delete()
    except Exception as e:
        if "message to delete not found" not in str(e).lower():
            logger.warning("Delete failed: %s", e)

    # Increment warnings
    state.user_warnings[user_id] = state.user_warnings.get(user_id, 0) + 1
    count = state.user_warnings[user_id]

    if count >= state.max_warnings:
        # Ban
        try:
            await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
        except Exception as e:
            logger.warning("Ban failed: %s", e)
        warn = f"⚠️ <a href='tg://user?id={user_id}'>User</a> has been banned after {state.max_warnings} warnings.\n\n{settings.promotion_text}"
        sent = await context.bot.send_message(chat_id=chat_id, text=warn, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        context.application.create_task(delete_message_after_delay(context, chat_id, sent.message_id, settings.deletion_delay_sec))
        state.user_warnings.pop(user_id, None)
        logger.info("User %s banned. Reason: %s", user_id, reason)
    else:
        warn = f"⚠️ <a href='tg://user?id={user_id}'>User</a>, {reason}. Warning {count}/{state.max_warnings}\n\n{settings.promotion_text}"
        sent = await context.bot.send_message(chat_id=chat_id, text=warn, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        context.application.create_task(delete_message_after_delay(context, chat_id, sent.message_id, settings.deletion_delay_sec))
        logger.info("Spam from %s: %s", user_id, reason)


# ----------------------------
# Error Handler
# ----------------------------

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling update", exc_info=context.error)
    if update and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ Sorry, an error occurred while processing your request. Please try again later."
            )
        except Exception:
            pass


# ----------------------------
# Flask Keep-Alive
# ----------------------------

flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "🛡️ Guardian Bot is running and vigilant!"


def run_flask(port: int):
    serve(flask_app, host="0.0.0.0", port=port)


# ----------------------------
# Application Setup & Main
# ----------------------------

def build_application(settings: Settings) -> Application:
    application = Application.builder().token(settings.telegram_bot_token).build()
    application.add_error_handler(error_handler)

    # Commands
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("botversion", cmd_botversion))
    application.add_handler(CommandHandler("addword", cmd_addword))
    application.add_handler(CommandHandler("block", cmd_block))
    application.add_handler(CommandHandler("addcommand", cmd_addcommand))
    application.add_handler(CommandHandler("adddynamic", cmd_adddynamic))
    application.add_handler(CommandHandler("listcommands", cmd_listcommands))
    application.add_handler(CommandHandler("report", cmd_report))
    application.add_handler(CommandHandler("allowchat", cmd_allowchat))
    application.add_handler(CommandHandler("allowthischat", cmd_allowthischat))
    application.add_handler(CommandHandler("listchats", cmd_listchats))
    application.add_handler(CommandHandler("allowforward", cmd_allowforward))
    application.add_handler(CommandHandler("revokeforward", cmd_revokeforward))
    application.add_handler(CommandHandler("listforwarders", cmd_listforwarders))
    application.add_handler(CommandHandler("allowchannel", cmd_allowchannel))
    application.add_handler(CommandHandler("setmaxwarnings", cmd_setmaxwarnings))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("reload", cmd_reload))

    # Known built-in commands to exclude from custom handler
    known = r'^/(start|help|botversion|addword|block|addcommand|adddynamic|listcommands|report|allowchat|allowthischat|listchats|allowforward|revokeforward|listforwarders|allowchannel|setmaxwarnings|stats|reload)\b'
    application.add_handler(MessageHandler(filters.COMMAND & ~filters.Regex(known), handle_custom_command))

    # Dynamic commands (catch-all after custom)
    application.add_handler(MessageHandler(filters.COMMAND, handle_dynamic_command))

    # All other messages
    application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))

    # Keep settings in bot_data for quick access
    application.bot_data["settings"] = settings
    return application


def main() -> None:
    settings = Settings.from_env()
    validate_settings(settings)

    global db
    db = Database(settings.database_url)
    setup_database()
    load_blacklist_and_seed_critical(settings)
    load_allowed_chats()
    load_forward_whitelist()
    load_allowed_channels(settings)
    load_dynamic_commands()
    load_bot_settings(settings)
    ai_init(settings)

    # Flask keep-alive
    flask_thread = threading.Thread(target=run_flask, args=(settings.port,), daemon=True)
    flask_thread.start()

    app = build_application(settings)

    # Graceful shutdown
    def _shutdown(*_):
        logger.info("Shutting down...")
        try:
            if db:
                db.close()
        finally:
            asyncio.get_event_loop().stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(
        "🛡️ Guardian Bot running | AI: %s | Max warnings: %d | Port: %d",
        "ON" if settings.ai_enabled else "OFF",
        state.max_warnings,
        settings.port,
    )

    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        stop_signals=(signal.SIGINT, signal.SIGTERM),
        close_loop=False,
        timeout=60,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
