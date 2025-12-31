#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🛡️ Guardian Bot v5.0.1 - AI-First Ultra Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• AI-powered spam detection (ALL languages)
• Context-aware (movie requests allowed)
• Bengali, Hindi, Tamil, Urdu, Arabic support
• Database migration fix included
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2
from psycopg2 import pool as pg_pool

from flask import Flask, jsonify
from waitress import serve

from telegram import Message, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.error import BadRequest, Forbidden

# Google Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# ============================================================
# CONFIGURATION
# ============================================================

BOT_VERSION = "5.0.1"
OWNER_CHANNEL_ID = -1003330141433  # Your channel - NEVER deleted

# ============================================================
# AI PROMPT - Heart of the Bot (All Languages)
# ============================================================

AI_SYSTEM_PROMPT = """You are a smart spam detector for Telegram groups. These groups are:
1. Movie Search Group - users ask for movies
2. Friends Chat Group - casual conversations

🎯 YOUR JOB: Detect ONLY harmful content. Be VERY careful not to block normal messages.

✅ ALWAYS ALLOW (Reply "OK"):
• Movie requests: "Pushpa 2 link do", "Avengers download", "KGF Hindi dubbed"
• Content requests: "bhai wo movie bhejo", "send kar do", "link dedo"
• Normal chat: "kaise ho", "hello", "good morning", "thanks bro"
• Questions about movies/shows
• Friendly conversations in ANY language
• Short messages, greetings, emojis
• File sharing requests between friends

❌ BLOCK AS SPAM (Reply "SPAM"):
• Self-promotion: "Join my channel", "Follow my page", "Subscribe now"
• Advertisements: "Buy now", "Discount offer", "Limited time"
• External promotions in ANY language:
  - Bengali: "আমার চ্যানেলে যোগ দিন", "টাকা আয় করুন"
  - Hindi: "मेरा चैनल जॉइन करो", "पैसे कमाओ"
  - Tamil: "என் சேனலில் சேருங்கள்"
  - Any language promoting external links/channels
• Crypto/Trading scams: "Earn money", "Investment opportunity", "Double your crypto"
• Adult service promotion
• Unsolicited business offers
• Mass-forward style messages with channel links

⛔ BLOCK + BAN (Reply "ILLEGAL"):
• Child exploitation (CP, CSAM, minor abuse)
• Drug trafficking
• Weapons sale
• Any illegal content

📋 DETECTION TIPS:
• Obfuscated text: "j.o" + "i.n m.y c.h" = "join my channel" = SPAM
• Mixed language spam: "Bro জয়েন my channel" = SPAM
• Hidden links: "telegram dot me slash..." = SPAM
• Price lists, DM offers = SPAM
• BUT: "bro movie ka link de" = OK (friend asking for movie)

🔤 YOU UNDERSTAND ALL LANGUAGES:
Bengali, Hindi, Tamil, Telugu, Urdu, Arabic, English, Marathi, Gujarati, Kannada, Malayalam, Punjabi, and more.

RESPOND WITH ONLY ONE WORD:
• "OK" - Allow the message
• "SPAM" - Delete message, warn user  
• "ILLEGAL" - Delete message, ban user immediately

Think carefully. Movie requests and friendly chat = OK. Promotions and scams = SPAM."""

# ============================================================
# Settings
# ============================================================

@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    database_url: str
    admin_user_ids: Tuple[int, ...]
    port: int
    gemini_api_key: Optional[str]
    ai_enabled: bool
    max_warnings: int
    channel_id: Optional[int]
    promotion_text: str
    deletion_delay_sec: int
    flood_messages_limit: int
    flood_time_window: int
    ai_timeout_sec: float

    @staticmethod
    def from_env() -> "Settings":
        admin_ids = (6946322342,)
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        db_url = os.environ.get("DATABASE_URL", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", None)
        
        ai_enabled = GEMINI_AVAILABLE and bool(gemini_key)
        
        channel_id_str = os.environ.get("CHANNEL_ID", "").strip()
        channel_id = int(channel_id_str) if channel_id_str else None
        
        return Settings(
            telegram_bot_token=telegram_token,
            database_url=db_url,
            admin_user_ids=admin_ids,
            port=int(os.environ.get("PORT", 8080)),
            gemini_api_key=gemini_key,
            ai_enabled=ai_enabled,
            max_warnings=int(os.environ.get("MAX_WARNINGS", 5)),
            channel_id=channel_id,
            promotion_text=os.environ.get("PROMOTION_TEXT", "For promotions: https://t.me/+scHqQ2SR0J45NjQ1"),
            deletion_delay_sec=int(os.environ.get("DELETION_DELAY_SEC", "10")),
            flood_messages_limit=int(os.environ.get("FLOOD_MESSAGES_LIMIT", "5")),
            flood_time_window=int(os.environ.get("FLOOD_TIME_WINDOW", "10")),
            ai_timeout_sec=float(os.environ.get("AI_TIMEOUT_SEC", "10.0")),
        )


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("guardian")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


# ============================================================
# Database
# ============================================================

class Database:
    _instance: Optional["Database"] = None

    def __init__(self, dsn: str):
        self._pool = pg_pool.ThreadedConnectionPool(minconn=2, maxconn=10, dsn=dsn)

    @classmethod
    def get_instance(cls, dsn: Optional[str] = None) -> "Database":
        if cls._instance is None and dsn:
            cls._instance = cls(dsn)
        return cls._instance

    @contextmanager
    def conn(self):
        conn = self._pool.getconn()
        try:
            conn.autocommit = False
            yield conn
        finally:
            self._pool.putconn(conn)

    def execute(self, query: str, params: tuple = ()) -> Optional[List[tuple]]:
        with self.conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    result = cur.fetchall()
                    conn.commit()
                    return result
                conn.commit()
                return None

    def close(self):
        if self._pool:
            self._pool.closeall()


db: Optional[Database] = None


def setup_database() -> None:
    """Setup database with migration support for old schema"""
    if not db:
        return
    
    with db.conn() as conn:
        with conn.cursor() as cur:
            # Create new tables
            tables = """
            CREATE TABLE IF NOT EXISTS allowed_chats (
                chat_id BIGINT PRIMARY KEY,
                chat_title TEXT,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS allowed_channels (
                channel_id BIGINT PRIMARY KEY,
                channel_title TEXT,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS forward_whitelist (
                user_id BIGINT PRIMARY KEY,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS trusted_users (
                user_id BIGINT PRIMARY KEY,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS banned_users (
                user_id BIGINT,
                chat_id BIGINT,
                reason TEXT,
                banned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, chat_id)
            );
            
            CREATE TABLE IF NOT EXISTS spam_log (
                id SERIAL PRIMARY KEY,
                user_id BIGINT,
                chat_id BIGINT,
                message_text TEXT,
                ai_verdict TEXT,
                action_taken TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            for statement in tables.split(';'):
                if statement.strip():
                    try:
                        cur.execute(statement)
                    except Exception as e:
                        logger.debug(f"Table exists: {e}")
            
            # Handle bot_settings table with migration
            try:
                # Check if old table exists with old column names
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'bot_settings' AND column_name = 'setting_key';
                """)
                old_schema = cur.fetchone() is not None
                
                if old_schema:
                    # Migrate old schema to new
                    logger.info("🔄 Migrating old bot_settings schema...")
                    cur.execute("ALTER TABLE bot_settings RENAME COLUMN setting_key TO key;")
                    cur.execute("ALTER TABLE bot_settings RENAME COLUMN setting_value TO value;")
                    
                    # Drop old columns if they exist
                    try:
                        cur.execute("ALTER TABLE bot_settings DROP COLUMN IF EXISTS updated_by;")
                        cur.execute("ALTER TABLE bot_settings DROP COLUMN IF EXISTS updated_at;")
                    except:
                        pass
                    
                    # Add updated_at if not exists
                    cur.execute("ALTER TABLE bot_settings ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")
                    logger.info("✅ Migration complete!")
                else:
                    # Check if table exists at all
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'bot_settings'
                        );
                    """)
                    table_exists = cur.fetchone()[0]
                    
                    if not table_exists:
                        # Create new table
                        cur.execute("""
                            CREATE TABLE bot_settings (
                                key TEXT PRIMARY KEY,
                                value TEXT,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );
                        """)
            except Exception as e:
                logger.warning(f"bot_settings setup: {e}")
                # Fallback: drop and recreate
                try:
                    cur.execute("DROP TABLE IF EXISTS bot_settings;")
                    cur.execute("""
                        CREATE TABLE bot_settings (
                            key TEXT PRIMARY KEY,
                            value TEXT,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                except Exception as e2:
                    logger.error(f"Failed to recreate bot_settings: {e2}")
            
            conn.commit()
    
    logger.info("✅ Database ready")


# ============================================================
# Bot State (Simple & Clean)
# ============================================================

@dataclass
class BotState:
    # Caches
    allowed_chats: Set[int] = field(default_factory=set)
    allowed_channels: Set[int] = field(default_factory=set)
    forward_whitelist: Set[int] = field(default_factory=set)
    trusted_users: Set[int] = field(default_factory=set)
    
    # Warnings
    user_warnings: Dict[int, int] = field(default_factory=dict)
    
    # Flood control
    user_messages: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    user_last_text: Dict[int, str] = field(default_factory=dict)
    user_repeat_count: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Stats
    messages_processed: int = 0
    spam_blocked: int = 0
    users_banned: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    # Settings
    max_warnings: int = 5


state = BotState()


def load_all_data(settings: Settings) -> None:
    """Load all data from database"""
    if not db:
        return
    
    # Load allowed chats
    try:
        rows = db.execute("SELECT chat_id FROM allowed_chats;")
        state.allowed_chats = {r[0] for r in (rows or [])}
    except Exception as e:
        logger.warning(f"Could not load allowed_chats: {e}")
        state.allowed_chats = set()
    
    # Load allowed channels (always include owner channel)
    state.allowed_channels = {OWNER_CHANNEL_ID}
    if settings.channel_id:
        state.allowed_channels.add(settings.channel_id)
    try:
        rows = db.execute("SELECT channel_id FROM allowed_channels;")
        state.allowed_channels |= {r[0] for r in (rows or [])}
    except Exception as e:
        logger.warning(f"Could not load allowed_channels: {e}")
    
    # Load forward whitelist
    try:
        rows = db.execute("SELECT user_id FROM forward_whitelist;")
        state.forward_whitelist = {r[0] for r in (rows or [])}
    except Exception as e:
        logger.warning(f"Could not load forward_whitelist: {e}")
        state.forward_whitelist = set()
    
    # Load trusted users
    try:
        rows = db.execute("SELECT user_id FROM trusted_users;")
        state.trusted_users = {r[0] for r in (rows or [])}
    except Exception as e:
        logger.warning(f"Could not load trusted_users: {e}")
        state.trusted_users = set()
    
    # Load max warnings - with fallback
    state.max_warnings = settings.max_warnings
    try:
        rows = db.execute("SELECT value FROM bot_settings WHERE key = 'max_warnings';")
        if rows and rows[0][0]:
            state.max_warnings = int(rows[0][0])
    except Exception as e:
        logger.debug(f"Using default max_warnings: {e}")
    
    logger.info(f"✅ Loaded: {len(state.allowed_chats)} chats, {len(state.allowed_channels)} channels, {len(state.trusted_users)} trusted users")


# ============================================================
# AI Spam Detection (The Brain)
# ============================================================

ai_model = None


def init_ai(settings: Settings) -> None:
    global ai_model
    
    if not settings.ai_enabled:
        logger.warning("⚠️ AI disabled - GEMINI_API_KEY not set")
        return
    
    try:
        genai.configure(api_key=settings.gemini_api_key)
        ai_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=AI_SYSTEM_PROMPT,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 10,
            }
        )
        logger.info("🤖 AI Model Ready (Gemini 2.0 Flash)")
    except Exception as e:
        logger.error(f"❌ AI init failed: {e}")
        ai_model = None


async def ai_analyze(text: str, timeout: float) -> str:
    """
    Analyze message with AI
    Returns: "OK", "SPAM", "ILLEGAL", or "ERROR"
    """
    if not ai_model or not text.strip():
        return "OK"
    
    # Skip very short messages (greetings, etc.)
    if len(text.strip()) < 5:
        return "OK"
    
    try:
        prompt = f"Analyze this message:\n\n{text[:1500]}"
        
        response = await asyncio.wait_for(
            ai_model.generate_content_async(prompt),
            timeout=timeout
        )
        
        result = (response.text or "").strip().upper()
        
        if "ILLEGAL" in result:
            return "ILLEGAL"
        elif "SPAM" in result:
            return "SPAM"
        else:
            return "OK"
    
    except asyncio.TimeoutError:
        logger.debug("AI timeout - allowing message")
        return "OK"
    except Exception as e:
        logger.debug(f"AI error: {e}")
        return "OK"


# ============================================================
# Flood Detection (Fast, No AI needed)
# ============================================================

def check_flood(user_id: int, text: str, settings: Settings) -> Tuple[bool, str]:
    """Check if user is flooding"""
    now = time.time()
    
    # Clean old timestamps
    cutoff = now - settings.flood_time_window
    state.user_messages[user_id] = [
        t for t in state.user_messages[user_id] if t > cutoff
    ]
    
    # Add current
    state.user_messages[user_id].append(now)
    
    # Check message count
    if len(state.user_messages[user_id]) > settings.flood_messages_limit:
        return True, f"Flooding: {len(state.user_messages[user_id])} messages in {settings.flood_time_window}s"
    
    # Check duplicate messages
    text_hash = hashlib.md5(text.lower().encode()).hexdigest()[:16]
    
    if state.user_last_text.get(user_id) == text_hash:
        state.user_repeat_count[user_id] += 1
        if state.user_repeat_count[user_id] >= 3:
            return True, "Duplicate message spam"
    else:
        state.user_last_text[user_id] = text_hash
        state.user_repeat_count[user_id] = 1
    
    return False, ""


# ============================================================
# Helper Functions
# ============================================================

def is_admin(user_id: int, settings: Settings) -> bool:
    return user_id in settings.admin_user_ids


def is_trusted(user_id: int) -> bool:
    return user_id in state.trusted_users


def is_allowed_channel(channel_id: int) -> bool:
    return channel_id in state.allowed_channels


async def safe_delete(context: ContextTypes.DEFAULT_TYPE, chat_id: int, msg_id: int) -> bool:
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
        return True
    except (BadRequest, Forbidden):
        return False
    except Exception as e:
        logger.debug(f"Delete error: {e}")
        return False


async def delete_later(context: ContextTypes.DEFAULT_TYPE, chat_id: int, msg_id: int, delay: int):
    await asyncio.sleep(delay)
    await safe_delete(context, chat_id, msg_id)


async def send_temp_message(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    delay: int
) -> None:
    """Send a message that auto-deletes"""
    try:
        msg = await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        asyncio.create_task(delete_later(context, chat_id, msg.message_id, delay))
    except Exception as e:
        logger.debug(f"Send error: {e}")


# ============================================================
# Main Message Handler
# ============================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    # ===== CHANNEL POSTS =====
    if update.channel_post:
        channel = update.channel_post.chat
        if channel and is_allowed_channel(channel.id):
            return  # Allow
        return  # Ignore other channels
    
    msg = update.effective_message
    if not msg:
        return
    
    user = msg.from_user
    chat = update.effective_chat
    
    if not user or not chat:
        return
    
    chat_id = chat.id
    user_id = user.id
    
    # ===== QUICK BYPASSES =====
    
    # Only process allowed chats
    if chat_id not in state.allowed_chats:
        return
    
    # Ignore bots
    if user.is_bot:
        return
    
    # Admin bypass
    if is_admin(user_id, settings):
        return
    
    # Trusted user bypass
    if is_trusted(user_id):
        return
    
    state.messages_processed += 1
    
    # ===== FORWARD CHECK =====
    forward_origin = getattr(msg, "forward_origin", None)
    
    if forward_origin:
        # Check if from allowed channel
        forward_chat = getattr(forward_origin, "chat", None) or getattr(forward_origin, "sender_chat", None)
        
        if forward_chat and is_allowed_channel(forward_chat.id):
            return  # Allow forwards from allowed channels
        
        # Check if user has forward permission
        if user_id in state.forward_whitelist:
            return  # Allow
        
        # Block unauthorized forward
        await safe_delete(context, chat_id, msg.message_id)
        state.spam_blocked += 1
        
        await send_temp_message(
            context, chat_id,
            f"⚠️ <a href='tg://user?id={user_id}'>User</a>, forwarding not allowed.\n\n{settings.promotion_text}",
            settings.deletion_delay_sec
        )
        return
    
    # ===== GET TEXT =====
    text = msg.text or msg.caption or ""
    
    if not text:
        return  # Allow media without caption
    
    # ===== FLOOD CHECK (Fast, no AI) =====
    is_flood, flood_reason = check_flood(user_id, text, settings)
    
    if is_flood:
        await take_action(context, settings, chat_id, user_id, msg, "SPAM", flood_reason)
        return
    
    # ===== AI ANALYSIS =====
    verdict = await ai_analyze(text, settings.ai_timeout_sec)
    
    if verdict in ("SPAM", "ILLEGAL"):
        await take_action(context, settings, chat_id, user_id, msg, verdict, "AI Detection")
        
        # Log to database
        if db:
            try:
                db.execute(
                    "INSERT INTO spam_log (user_id, chat_id, message_text, ai_verdict, action_taken) VALUES (%s, %s, %s, %s, %s);",
                    (user_id, chat_id, text[:500], verdict, "DELETE" if verdict == "SPAM" else "BAN")
                )
            except:
                pass


async def take_action(
    context: ContextTypes.DEFAULT_TYPE,
    settings: Settings,
    chat_id: int,
    user_id: int,
    msg: Message,
    verdict: str,
    reason: str
) -> None:
    """Take action based on verdict"""
    
    # Delete message
    await safe_delete(context, chat_id, msg.message_id)
    state.spam_blocked += 1
    
    if verdict == "ILLEGAL":
        # Instant ban
        try:
            await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
            state.users_banned += 1
            
            if db:
                try:
                    db.execute(
                        "INSERT INTO banned_users (user_id, chat_id, reason) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                        (user_id, chat_id, reason)
                    )
                except:
                    pass
        except Exception as e:
            logger.warning(f"Ban failed: {e}")
        
        await send_temp_message(
            context, chat_id,
            f"🚫 <a href='tg://user?id={user_id}'>User</a> <b>BANNED</b>\nReason: Illegal content detected",
            settings.deletion_delay_sec
        )
        
        state.user_warnings.pop(user_id, None)
        logger.info(f"🚫 BANNED {user_id} | Reason: {reason}")
        return
    
    # SPAM - Warning system
    state.user_warnings[user_id] = state.user_warnings.get(user_id, 0) + 1
    warnings = state.user_warnings[user_id]
    
    if warnings >= state.max_warnings:
        # Ban after max warnings
        try:
            await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
            state.users_banned += 1
            
            if db:
                try:
                    db.execute(
                        "INSERT INTO banned_users (user_id, chat_id, reason) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                        (user_id, chat_id, f"Max warnings reached: {reason}")
                    )
                except:
                    pass
        except Exception as e:
            logger.warning(f"Ban failed: {e}")
        
        await send_temp_message(
            context, chat_id,
            f"🚫 <a href='tg://user?id={user_id}'>User</a> <b>BANNED</b>\n{state.max_warnings} warnings reached.\n\n{settings.promotion_text}",
            settings.deletion_delay_sec
        )
        
        state.user_warnings.pop(user_id, None)
        logger.info(f"🚫 BANNED {user_id} | Warnings exceeded")
    else:
        # Warning
        await send_temp_message(
            context, chat_id,
            f"⚠️ <b>Warning {warnings}/{state.max_warnings}</b>\n\n<a href='tg://user?id={user_id}'>User</a>, message removed.\nReason: {reason}\n\n{settings.promotion_text}",
            settings.deletion_delay_sec
        )
        
        logger.info(f"⚠️ Warning {warnings} to {user_id} | {reason}")


# ============================================================
# Command Handlers
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"""🛡️ <b>Guardian Bot v{BOT_VERSION}</b>

AI-Powered Protection for Telegram Groups

<b>Features:</b>
• 🤖 Smart AI spam detection
• 🌍 All languages supported
• 🎬 Movie requests allowed
• 💬 Normal chat protected
• 🚫 Spam/Scam blocked
• ⛔ Illegal content = instant ban

Use /help for commands.""",
        parse_mode=ParseMode.HTML
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    await update.message.reply_text(
        f"""🛡️ <b>Admin Commands</b>

<b>💬 Chat Management:</b>
/allowchat &lt;id&gt; - Allow a chat
/allowthischat - Allow current chat
/removechat &lt;id&gt; - Remove chat
/listchats - List allowed chats

<b>📢 Channel Management:</b>
/allowchannel &lt;id&gt; - Allow channel
/removechannel &lt;id&gt; - Remove channel
/listchannels - List channels

<b>↗️ Forward Control:</b>
/allowforward &lt;user_id&gt; - Allow user to forward
/revokeforward &lt;user_id&gt; - Revoke forward
/listforwarders - List forwarders

<b>👤 User Management:</b>
/trust &lt;user_id&gt; - Trust user (bypass all)
/untrust &lt;user_id&gt; - Remove trust
/listtrusted - List trusted users
/warn &lt;user_id&gt; - Warn user
/ban &lt;user_id&gt; - Ban user
/unban &lt;user_id&gt; - Unban user
/resetwarnings &lt;user_id&gt; - Reset warnings

<b>⚙️ Settings:</b>
/setwarnings &lt;n&gt; - Set max warnings
/reload - Reload data

<b>📊 Info:</b>
/stats - View statistics
/status - Bot status""",
        parse_mode=ParseMode.HTML
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text(f"❌ Admins only.\nYour ID: <code>{update.effective_user.id}</code>", parse_mode=ParseMode.HTML)
        return
    
    uptime = str(datetime.now() - state.start_time).split('.')[0]
    
    await update.message.reply_text(
        f"""📊 <b>Guardian Bot Stats</b>

<b>Activity:</b>
• Messages: {state.messages_processed:,}
• Spam blocked: {state.spam_blocked:,}
• Users banned: {state.users_banned:,}
• Active warnings: {len(state.user_warnings)}

<b>Config:</b>
• Chats: {len(state.allowed_chats)}
• Channels: {len(state.allowed_channels)}
• Trusted: {len(state.trusted_users)}
• Max warnings: {state.max_warnings}

<b>System:</b>
• Version: {BOT_VERSION}
• AI: {'🟢 ON' if ai_model else '🔴 OFF'}
• Uptime: {uptime}""",
        parse_mode=ParseMode.HTML
    )


async def cmd_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"✅ Guardian Bot v{BOT_VERSION}\nAI: {'🟢' if ai_model else '🔴'}\nSpam blocked: {state.spam_blocked:,}",
        parse_mode=ParseMode.HTML
    )


async def cmd_allowthischat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    chat_id = update.effective_chat.id
    chat_title = update.effective_chat.title or "Unknown"
    
    try:
        db.execute(
            "INSERT INTO allowed_chats (chat_id, chat_title, added_by) VALUES (%s, %s, %s) ON CONFLICT (chat_id) DO UPDATE SET chat_title = %s;",
            (chat_id, chat_title, update.effective_user.id, chat_title)
        )
        state.allowed_chats.add(chat_id)
        await update.message.reply_text(f"✅ Chat protected!\nID: <code>{chat_id}</code>", parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_allowchat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /allowchat <chat_id>")
        return
    
    try:
        chat_id = int(context.args[0])
        db.execute(
            "INSERT INTO allowed_chats (chat_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (chat_id, update.effective_user.id)
        )
        state.allowed_chats.add(chat_id)
        await update.message.reply_text(f"✅ Chat {chat_id} allowed")
    except ValueError:
        await update.message.reply_text("❌ Invalid chat ID")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_removechat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /removechat <chat_id>")
        return
    
    try:
        chat_id = int(context.args[0])
        db.execute("DELETE FROM allowed_chats WHERE chat_id = %s;", (chat_id,))
        state.allowed_chats.discard(chat_id)
        await update.message.reply_text(f"✅ Chat {chat_id} removed")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listchats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    if not state.allowed_chats:
        await update.message.reply_text("No allowed chats.")
        return
    
    chats = "\n".join(f"• <code>{c}</code>" for c in state.allowed_chats)
    await update.message.reply_text(f"💬 <b>Allowed Chats:</b>\n\n{chats}", parse_mode=ParseMode.HTML)


async def cmd_allowchannel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /allowchannel <channel_id>")
        return
    
    try:
        channel_id = int(context.args[0])
        db.execute(
            "INSERT INTO allowed_channels (channel_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (channel_id, update.effective_user.id)
        )
        state.allowed_channels.add(channel_id)
        await update.message.reply_text(f"✅ Channel {channel_id} allowed")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_removechannel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /removechannel <channel_id>")
        return
    
    try:
        channel_id = int(context.args[0])
        if channel_id == OWNER_CHANNEL_ID:
            await update.message.reply_text("❌ Cannot remove owner channel")
            return
        db.execute("DELETE FROM allowed_channels WHERE channel_id = %s;", (channel_id,))
        state.allowed_channels.discard(channel_id)
        await update.message.reply_text(f"✅ Channel removed")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listchannels(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    channels = []
    for c in state.allowed_channels:
        if c == OWNER_CHANNEL_ID:
            channels.append(f"• <code>{c}</code> (Owner)")
        else:
            channels.append(f"• <code>{c}</code>")
    
    await update.message.reply_text(f"📢 <b>Allowed Channels:</b>\n\n" + "\n".join(channels), parse_mode=ParseMode.HTML)


async def cmd_allowforward(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    target = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target = int(context.args[0])
        except:
            pass
    
    if not target:
        await update.message.reply_text("Reply to user or /allowforward <user_id>")
        return
    
    try:
        db.execute("INSERT INTO forward_whitelist (user_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;", (target, update.effective_user.id))
        state.forward_whitelist.add(target)
        await update.message.reply_text(f"✅ User {target} can forward now")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_revokeforward(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    target = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target = int(context.args[0])
        except:
            pass
    
    if not target:
        await update.message.reply_text("Reply to user or /revokeforward <user_id>")
        return
    
    try:
        db.execute("DELETE FROM forward_whitelist WHERE user_id = %s;", (target,))
        state.forward_whitelist.discard(target)
        await update.message.reply_text(f"✅ Forward revoked for {target}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listforwarders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    if not state.forward_whitelist:
        await update.message.reply_text("No forwarders.")
        return
    
    users = "\n".join(f"• <code>{u}</code>" for u in state.forward_whitelist)
    await update.message.reply_text(f"↗️ <b>Forward Whitelist:</b>\n\n{users}", parse_mode=ParseMode.HTML)


async def cmd_trust(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    target = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target = int(context.args[0])
        except:
            pass
    
    if not target:
        await update.message.reply_text("Reply to user or /trust <user_id>")
        return
    
    try:
        db.execute("INSERT INTO trusted_users (user_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;", (target, update.effective_user.id))
        state.trusted_users.add(target)
        await update.message.reply_text(f"✅ User {target} is now trusted (bypasses all checks)")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_untrust(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    target = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target = int(context.args[0])
        except:
            pass
    
    if not target:
        await update.message.reply_text("Reply to user or /untrust <user_id>")
        return
    
    try:
        db.execute("DELETE FROM trusted_users WHERE user_id = %s;", (target,))
        state.trusted_users.discard(target)
        await update.message.reply_text(f"✅ User {target} removed from trusted")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listtrusted(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    if not state.trusted_users:
        await update.message.reply_text("No trusted users.")
        return
    
    users = "\n".join(f"• <code>{u}</code>" for u in state.trusted_users)
    await update.message.reply_text(f"👤 <b>Trusted Users:</b>\n\n{users}", parse_mode=ParseMode.HTML)


async def cmd_warn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    target = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target = int(context.args[0])
        except:
            pass
    
    if not target:
        await update.message.reply_text("Reply to user or /warn <user_id>")
        return
    
    state.user_warnings[target] = state.user_warnings.get(target, 0) + 1
    count = state.user_warnings[target]
    
    await update.message.reply_text(f"⚠️ User <code>{target}</code> warned.\nWarnings: {count}/{state.max_warnings}", parse_mode=ParseMode.HTML)


async def cmd_ban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    target = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target = int(context.args[0])
        except:
            pass
    
    if not target:
        await update.message.reply_text("Reply to user or /ban <user_id>")
        return
    
    try:
        await context.bot.ban_chat_member(chat_id=update.effective_chat.id, user_id=target)
        state.users_banned += 1
        await update.message.reply_text(f"🚫 User <code>{target}</code> banned", parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed: {e}")


async def cmd_unban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /unban <user_id>")
        return
    
    try:
        target = int(context.args[0])
        await context.bot.unban_chat_member(chat_id=update.effective_chat.id, user_id=target, only_if_banned=True)
        db.execute("DELETE FROM banned_users WHERE user_id = %s AND chat_id = %s;", (target, update.effective_chat.id))
        await update.message.reply_text(f"✅ User <code>{target}</code> unbanned", parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed: {e}")


async def cmd_resetwarnings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    target = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            target = int(context.args[0])
        except:
            pass
    
    if not target:
        await update.message.reply_text("Reply to user or /resetwarnings <user_id>")
        return
    
    state.user_warnings.pop(target, None)
    await update.message.reply_text(f"✅ Warnings reset for <code>{target}</code>", parse_mode=ParseMode.HTML)


async def cmd_setwarnings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    if not context.args:
        await update.message.reply_text(f"Current: {state.max_warnings}\nUsage: /setwarnings <number>")
        return
    
    try:
        n = max(1, int(context.args[0]))
        db.execute(
            "INSERT INTO bot_settings (key, value) VALUES ('max_warnings', %s) ON CONFLICT (key) DO UPDATE SET value = %s;",
            (str(n), str(n))
        )
        state.max_warnings = n
        await update.message.reply_text(f"✅ Max warnings set to {n}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await update.message.reply_text("❌ Admins only.")
        return
    
    load_all_data(settings)
    await update.message.reply_text("✅ All data reloaded!")


# ============================================================
# Error Handler
# ============================================================

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(context.error, (BadRequest, Forbidden)):
        return
    logger.error("Error:", exc_info=context.error)


# ============================================================
# Flask Keep-Alive
# ============================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return jsonify({
        "bot": "Guardian Bot",
        "version": BOT_VERSION,
        "status": "running",
        "ai": "enabled" if ai_model else "disabled",
        "spam_blocked": state.spam_blocked
    })


@flask_app.route("/health")
def health():
    return jsonify({"status": "healthy"})


def run_flask(port: int):
    serve(flask_app, host="0.0.0.0", port=port, _quiet=True)


# ============================================================
# Build Application
# ============================================================

def build_app(settings: Settings) -> Application:
    app = Application.builder().token(settings.telegram_bot_token).build()
    
    app.add_error_handler(error_handler)
    
    commands = [
        ("start", cmd_start),
        ("help", cmd_help),
        ("stats", cmd_stats),
        ("status", cmd_status),
        ("allowchat", cmd_allowchat),
        ("allowthischat", cmd_allowthischat),
        ("removechat", cmd_removechat),
        ("listchats", cmd_listchats),
        ("allowchannel", cmd_allowchannel),
        ("removechannel", cmd_removechannel),
        ("listchannels", cmd_listchannels),
        ("allowforward", cmd_allowforward),
        ("revokeforward", cmd_revokeforward),
        ("listforwarders", cmd_listforwarders),
        ("trust", cmd_trust),
        ("untrust", cmd_untrust),
        ("listtrusted", cmd_listtrusted),
        ("warn", cmd_warn),
        ("ban", cmd_ban),
        ("unban", cmd_unban),
        ("resetwarnings", cmd_resetwarnings),
        ("setwarnings", cmd_setwarnings),
        ("reload", cmd_reload),
    ]
    
    for name, handler in commands:
        app.add_handler(CommandHandler(name, handler))
    
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))
    
    app.bot_data["settings"] = settings
    
    return app


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║      🛡️  GUARDIAN BOT v{BOT_VERSION} - AI EDITION  🛡️       ║
║         All Languages • Smart Detection • Fast            ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    settings = Settings.from_env()
    
    if not settings.telegram_bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not set!")
        sys.exit(1)
    
    if not settings.database_url:
        print("❌ DATABASE_URL not set!")
        sys.exit(1)
    
    # Database
    global db
    db = Database.get_instance(settings.database_url)
    setup_database()
    load_all_data(settings)
    
    # AI
    init_ai(settings)
    
    # Flask
    flask_thread = threading.Thread(target=run_flask, args=(settings.port,), daemon=True)
    flask_thread.start()
    
    # Build app
    app = build_app(settings)
    
    # Shutdown handler
    def shutdown(sig, frame):
        logger.info("Shutting down...")
        if db:
            db.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    logger.info(f"""
🛡️ Guardian Bot Started!
├── Version: {BOT_VERSION}
├── AI: {'✅ Enabled' if ai_model else '❌ Disabled (Set GEMINI_API_KEY)'}
├── Chats: {len(state.allowed_chats)}
├── Channels: {len(state.allowed_channels)}
└── Port: {settings.port}
""")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
