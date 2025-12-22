#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🛡️ Guardian Bot v4.0 - Ultra Powerful Edition
- Fixed all bugs
- Added owner channel protection
- Enhanced spam detection
- Better performance
- More admin controls
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
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.errors import UniqueViolation

from flask import Flask, jsonify
from waitress import serve

from telegram import (
    Chat,
    ChatMember,
    ChatMemberUpdated,
    ChatPermissions,
    Message,
    Update,
    User,
)
from telegram.constants import ChatMemberStatus, ChatType, ParseMode
from telegram.ext import (
    Application,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.error import BadRequest, Forbidden, TelegramError

# Optional AI (Gemini) for final-stage spam classification
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# ============================================================
# OWNER CHANNEL - YE CHANNEL KABHI DELETE NAHI HOGA
# ============================================================
OWNER_CHANNEL_ID = -1003330141433  # Aapka channel ID - Posts kabhi delete nahi honge

# Bot version
BOT_VERSION = "4.0.0"

# ============================================================
# Settings & Configuration
# ============================================================

@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    database_url: str
    admin_user_ids: Tuple[int, ...]
    port: int
    ai_enabled: bool
    gemini_api_key: Optional[str]
    max_warnings_default: int
    channel_id: Optional[int]
    promotion_text: str
    spam_ai_model: str
    ai_timeout_sec: float
    deletion_delay_sec: int
    min_seconds_between_msgs: float
    min_seconds_between_same_user_msgs: float
    flood_messages_limit: int
    flood_time_window: int
    new_user_restriction_hours: int
    enable_captcha: bool
    max_message_length: int
    max_emojis_allowed: int
    log_channel_id: Optional[int]

    @staticmethod
    def from_env() -> "Settings":
        admin_ids = (6946322342,)
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        db_url = os.environ.get("DATABASE_URL", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", None)
        port = int(os.environ.get("PORT", 8080))
        channel_id_str = os.environ.get("CHANNEL_ID", "").strip()
        channel_id = int(channel_id_str) if channel_id_str else None
        log_channel_str = os.environ.get("LOG_CHANNEL_ID", "").strip()
        log_channel_id = int(log_channel_str) if log_channel_str else None

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
            spam_ai_model=os.environ.get("SPAM_AI_MODEL", "gemini-2.0-flash"),
            ai_timeout_sec=float(os.environ.get("AI_TIMEOUT_SEC", "7.0")),
            deletion_delay_sec=int(os.environ.get("DELETION_DELAY_SEC", "10")),
            min_seconds_between_msgs=float(os.environ.get("MIN_SECONDS_BETWEEN_MSGS", "1.5")),
            min_seconds_between_same_user_msgs=float(os.environ.get("MIN_SECONDS_BETWEEN_SAME_USER_MSGS", "1.0")),
            flood_messages_limit=int(os.environ.get("FLOOD_MESSAGES_LIMIT", "5")),
            flood_time_window=int(os.environ.get("FLOOD_TIME_WINDOW", "10")),
            new_user_restriction_hours=int(os.environ.get("NEW_USER_RESTRICTION_HOURS", "24")),
            enable_captcha=os.environ.get("ENABLE_CAPTCHA", "false").lower() in ("1", "true", "yes"),
            max_message_length=int(os.environ.get("MAX_MESSAGE_LENGTH", "4096")),
            max_emojis_allowed=int(os.environ.get("MAX_EMOJIS_ALLOWED", "15")),
            log_channel_id=log_channel_id,
        )


def validate_settings(settings: Settings) -> None:
    missing: List[str] = []
    if not settings.telegram_bot_token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not settings.database_url:
        missing.append("DATABASE_URL")
    if missing:
        raise SystemExit(f"❌ Missing required environment variables: {', '.join(missing)}")


# ============================================================
# Logging Setup
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("guardian-bot")

# Reduce noise from httpx and telegram
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


# ============================================================
# Database Pool & Helpers
# ============================================================

class Database:
    _instance: Optional["Database"] = None
    _lock = threading.Lock()

    def __init__(self, dsn: str):
        self._pool: pg_pool.ThreadedConnectionPool = pg_pool.ThreadedConnectionPool(
            minconn=2, maxconn=20, dsn=dsn
        )
        self._dsn = dsn

    @classmethod
    def get_instance(cls, dsn: Optional[str] = None) -> "Database":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if dsn is None:
                        raise ValueError("DSN required for first initialization")
                    cls._instance = cls(dsn)
        return cls._instance

    @contextmanager
    def conn(self):
        conn = None
        try:
            conn = self._pool.getconn()
            conn.autocommit = False
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    def execute(self, query: str, params: tuple = ()) -> Optional[List[tuple]]:
        with self.conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall()
                conn.commit()
                return None

    def execute_many(self, query: str, params_list: List[tuple]) -> None:
        with self.conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
                conn.commit()

    def close(self):
        if self._pool:
            self._pool.closeall()


db: Optional[Database] = None


def setup_database() -> None:
    assert db is not None
    
    tables = [
        """
        CREATE TABLE IF NOT EXISTS blacklist (
            id SERIAL PRIMARY KEY,
            word TEXT NOT NULL UNIQUE,
            severity INTEGER DEFAULT 1,
            added_by BIGINT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS allowed_chats (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL UNIQUE,
            chat_title TEXT,
            added_by BIGINT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS custom_commands (
            id SERIAL PRIMARY KEY,
            command TEXT NOT NULL UNIQUE,
            response TEXT NOT NULL,
            added_by BIGINT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS reported_spam (
            id SERIAL PRIMARY KEY,
            message TEXT NOT NULL,
            message_hash TEXT,
            reported_by BIGINT,
            chat_id BIGINT,
            reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS forward_whitelist (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL UNIQUE,
            username TEXT,
            added_by BIGINT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS allowed_channels (
            id SERIAL PRIMARY KEY,
            channel_id BIGINT NOT NULL UNIQUE,
            channel_title TEXT,
            added_by BIGINT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS trusted_users (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL UNIQUE,
            username TEXT,
            trust_level INTEGER DEFAULT 1,
            added_by BIGINT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS user_stats (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL UNIQUE,
            messages_count INTEGER DEFAULT 0,
            warnings_count INTEGER DEFAULT 0,
            spam_count INTEGER DEFAULT 0,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS banned_users (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            chat_id BIGINT NOT NULL,
            reason TEXT,
            banned_by BIGINT,
            banned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, chat_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS bot_settings (
            id SERIAL PRIMARY KEY,
            setting_key TEXT NOT NULL UNIQUE,
            setting_value TEXT NOT NULL,
            updated_by BIGINT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS spam_patterns (
            id SERIAL PRIMARY KEY,
            pattern TEXT NOT NULL UNIQUE,
            description TEXT,
            severity INTEGER DEFAULT 1,
            is_regex BOOLEAN DEFAULT FALSE,
            added_by BIGINT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS action_logs (
            id SERIAL PRIMARY KEY,
            action_type TEXT NOT NULL,
            user_id BIGINT,
            chat_id BIGINT,
            details JSONB,
            performed_by BIGINT,
            performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_blacklist_word ON blacklist(word);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_user_stats_user_id ON user_stats(user_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_action_logs_performed_at ON action_logs(performed_at);
        """,
    ]
    
    with db.conn() as conn:
        with conn.cursor() as cur:
            for table_sql in tables:
                try:
                    cur.execute(table_sql)
                except Exception as e:
                    logger.warning(f"Table creation warning: {e}")
        conn.commit()
    
    logger.info("✅ Database tables initialized")


# ============================================================
# Bot State & Caches
# ============================================================

@dataclass
class SpamScore:
    score: int = 0
    reasons: List[str] = field(default_factory=list)
    
    def add(self, points: int, reason: str):
        self.score += points
        self.reasons.append(reason)
    
    def is_spam(self, threshold: int = 50) -> bool:
        return self.score >= threshold


@dataclass
class UserFloodData:
    message_times: List[float] = field(default_factory=list)
    last_message_hash: str = ""
    duplicate_count: int = 0


@dataclass
class BotState:
    # In-memory caches
    user_warnings: Dict[int, int] = field(default_factory=dict)
    user_last_message_time: Dict[int, datetime] = field(default_factory=dict)
    user_flood_data: Dict[int, UserFloodData] = field(default_factory=lambda: defaultdict(UserFloodData))
    
    # Allowlists and blocklists
    blacklist_words: Set[str] = field(default_factory=set)
    blacklist_high_severity: Set[str] = field(default_factory=set)
    allowed_chats: Set[int] = field(default_factory=set)
    forward_whitelist_users: Set[int] = field(default_factory=set)
    allowed_channels: Set[int] = field(default_factory=set)
    trusted_users: Set[int] = field(default_factory=set)
    
    # Custom patterns from DB
    custom_spam_patterns: List[re.Pattern] = field(default_factory=list)
    
    # Settings
    max_warnings: int = 5
    
    # Statistics
    messages_processed: int = 0
    spam_detected: int = 0
    users_banned: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    # Rate limiting
    message_rate_limit_ts: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # Cleanup tracking
    last_cleanup: datetime = field(default_factory=datetime.now)


state = BotState()


# ============================================================
# Data Loading Functions
# ============================================================

def load_blacklist_and_seed_critical(settings: Settings) -> None:
    assert db is not None
    
    # Critical words that should always be blocked
    critical_words = {
        # Illegal content
        ("cp", 10), ("child porn", 10), ("pedo", 10), ("minor", 5),
        ("child abuse", 10), ("underage", 8), ("jailbait", 10),
        # Spam/Commerce
        ("premium", 3), ("collection", 2), ("price", 2), ("payment", 3),
        ("purchase", 2), ("dm for", 3), ("lowest price", 5),
        # Regional spam
        ("desi", 2), ("indian cp", 10), ("foreign cp", 10), ("tamil", 2),
        ("chinese", 1), ("arabian", 1), ("bro-sis", 5), ("dad-daughter", 8),
        # Adult content
        ("nude", 3), ("xxx", 4), ("porn", 5), ("sex video", 5),
        ("onlyfans", 3), ("fansly", 3), ("18+", 2),
        # Scam terms
        ("earn money", 3), ("work from home", 2), ("crypto investment", 4),
        ("double your", 5), ("guaranteed profit", 5), ("whatsapp", 2),
    }
    
    with db.conn() as conn:
        with conn.cursor() as cur:
            # Insert critical words
            for word, severity in critical_words:
                try:
                    cur.execute(
                        """INSERT INTO blacklist (word, severity, added_by) 
                           VALUES (%s, %s, %s) ON CONFLICT (word) DO NOTHING;""",
                        (word.lower(), severity, settings.admin_user_ids[0] if settings.admin_user_ids else 0)
                    )
                except Exception:
                    pass
            
            conn.commit()
            
            # Load all words
            cur.execute("SELECT word, severity FROM blacklist;")
            rows = cur.fetchall()
            
            state.blacklist_words = {r[0].lower() for r in rows}
            state.blacklist_high_severity = {r[0].lower() for r in rows if r[1] >= 5}
    
    logger.info(f"✅ Loaded {len(state.blacklist_words)} blacklist words ({len(state.blacklist_high_severity)} high severity)")


def load_allowed_chats() -> None:
    assert db is not None
    rows = db.execute("SELECT chat_id FROM allowed_chats;")
    state.allowed_chats = {r[0] for r in (rows or [])}
    logger.info(f"✅ Loaded {len(state.allowed_chats)} allowed chats")


def load_forward_whitelist() -> None:
    assert db is not None
    rows = db.execute("SELECT user_id FROM forward_whitelist;")
    state.forward_whitelist_users = {r[0] for r in (rows or [])}
    logger.info(f"✅ Loaded {len(state.forward_whitelist_users)} forward-whitelisted users")


def load_allowed_channels(settings: Settings) -> None:
    assert db is not None
    
    # Start with owner channel - YE HAMESHA ALLOWED RAHEGA
    ids: Set[int] = {OWNER_CHANNEL_ID}
    
    # Add configured channel
    if settings.channel_id:
        ids.add(settings.channel_id)
    
    # Add channels from DB
    rows = db.execute("SELECT channel_id FROM allowed_channels;")
    ids |= {r[0] for r in (rows or [])}
    
    state.allowed_channels = ids
    logger.info(f"✅ Loaded {len(state.allowed_channels)} allowed channels (including owner channel)")


def load_trusted_users() -> None:
    assert db is not None
    rows = db.execute("SELECT user_id FROM trusted_users;")
    state.trusted_users = {r[0] for r in (rows or [])}
    logger.info(f"✅ Loaded {len(state.trusted_users)} trusted users")


def load_custom_spam_patterns() -> None:
    assert db is not None
    rows = db.execute("SELECT pattern, is_regex FROM spam_patterns;")
    patterns = []
    for pattern, is_regex in (rows or []):
        try:
            if is_regex:
                patterns.append(re.compile(pattern, re.IGNORECASE))
            else:
                patterns.append(re.compile(re.escape(pattern), re.IGNORECASE))
        except re.error as e:
            logger.warning(f"Invalid pattern '{pattern}': {e}")
    state.custom_spam_patterns = patterns
    logger.info(f"✅ Loaded {len(patterns)} custom spam patterns")


def load_bot_settings(settings: Settings) -> None:
    assert db is not None
    state.max_warnings = settings.max_warnings_default
    
    rows = db.execute("SELECT setting_key, setting_value FROM bot_settings;")
    for key, value in (rows or []):
        if key == "max_warnings":
            try:
                state.max_warnings = max(1, int(value))
            except ValueError:
                pass
    
    logger.info(f"✅ Settings loaded - Max warnings: {state.max_warnings}")


def reload_all_caches(settings: Settings) -> None:
    """Reload all caches from database"""
    load_blacklist_and_seed_critical(settings)
    load_allowed_chats()
    load_forward_whitelist()
    load_allowed_channels(settings)
    load_trusted_users()
    load_custom_spam_patterns()
    load_bot_settings(settings)


# ============================================================
# AI Spam Detection
# ============================================================

SPAM_DETECTION_PROMPT = """You are an expert spam detection AI for Telegram groups. Your job is to protect users from:

1. SPAM: Promotional messages, advertisements, unsolicited offers
2. SCAM: Cryptocurrency scams, investment frauds, phishing attempts  
3. ILLEGAL: Child exploitation (CSAM), drug sales, weapons trafficking
4. ADULT: Pornography, escort services, adult content sales

ANALYSIS RULES:
- Consider obfuscation techniques (l33t speak, unicode tricks, spacing)
- Detect hidden links and disguised URLs
- Identify payment/commerce language patterns
- Check for urgency tactics and too-good-to-be-true offers
- Consider cultural context (Hindi, English, mixed languages)

RESPOND WITH ONLY:
- "SPAM" if the message violates any rule
- "OK" if the message is legitimate

Be strict but avoid false positives on normal conversation.
"""

ai_model = None


def ai_init(settings: Settings) -> None:
    global ai_model
    if not settings.ai_enabled or not GEMINI_AVAILABLE:
        logger.info("🤖 AI spam detection: DISABLED")
        return
    
    try:
        genai.configure(api_key=settings.gemini_api_key)
        ai_model = genai.GenerativeModel(
            model_name=settings.spam_ai_model,
            system_instruction=SPAM_DETECTION_PROMPT,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 10,
            }
        )
        logger.info(f"🤖 AI spam detection: ENABLED ({settings.spam_ai_model})")
    except Exception as e:
        logger.warning(f"⚠️ Failed to init AI model: {e}")
        ai_model = None


async def ai_check_spam(text: str, normalized: str, timeout: float) -> Tuple[Optional[bool], str]:
    """Check if text is spam using AI. Returns (is_spam, reason)"""
    if ai_model is None:
        return None, ""
    
    try:
        prompt = f"""Analyze this message:
---
Original: {text[:1000]}
Normalized: {normalized[:500]}
---
Is this SPAM or OK?"""
        
        response = await asyncio.wait_for(
            ai_model.generate_content_async(prompt),
            timeout=timeout
        )
        
        decision = (response.text or "").strip().upper()
        
        if "SPAM" in decision:
            return True, "AI detected spam content"
        elif "OK" in decision:
            return False, ""
        
    except asyncio.TimeoutError:
        logger.debug("AI timeout")
    except Exception as e:
        logger.debug(f"AI error: {e}")
    
    return None, ""


# ============================================================
# Spam Detection Patterns (Precompiled)
# ============================================================

SPAM_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    # High severity - Illegal content
    (re.compile(r"\b(child|kid|minor)\s*(porn|sex|nude|cp)\b", re.I), 100, "Illegal content"),
    (re.compile(r"\bpedo(phile)?\b", re.I), 100, "Illegal content"),
    (re.compile(r"\b(csam|cp)\b", re.I), 100, "Illegal content"),
    (re.compile(r"\bunderage\b", re.I), 80, "Potential illegal content"),
    
    # Medium-High severity - Adult/Explicit
    (re.compile(r"\b(desi|indian|tamil|chinese|arabian)\s*cp\b", re.I), 100, "Illegal content"),
    (re.compile(r"\b(bro-?sis|dad-?daughter|mom-?son)\s*(cp|porn|video)\b", re.I), 100, "Illegal content"),
    (re.compile(r"\bpremium\s*(cp|collection|content)\b", re.I), 90, "Illegal spam"),
    
    # Medium severity - Commerce/Promotion
    (re.compile(r"\blowest\s*price\b", re.I), 60, "Commercial spam"),
    (re.compile(r"\bdm\s*(for|me)\s*(purchase|price|deal)\b", re.I), 70, "Commercial spam"),
    (re.compile(r"\bpayment\s*(method|available)\b", re.I), 50, "Commercial spam"),
    (re.compile(r"\b(upi|paypal|crypto|bitcoin|usdt)\s*(payment|accepted|only)\b", re.I), 60, "Payment spam"),
    (re.compile(r"\b(combo|collection)\s*(available|price)\b", re.I), 55, "Commercial spam"),
    (re.compile(r"\bprice\s*(available|list|dm)\b", re.I), 50, "Commercial spam"),
    
    # Scam patterns
    (re.compile(r"\b(earn|make)\s*(money|income|₹|\$)\s*(online|daily|weekly)\b", re.I), 65, "Scam"),
    (re.compile(r"\bguaranteed\s*(profit|return|income)\b", re.I), 70, "Investment scam"),
    (re.compile(r"\bdouble\s*your\s*(money|investment|crypto)\b", re.I), 80, "Investment scam"),
    (re.compile(r"\bwork\s*from\s*home\s*(job|income|opportunity)\b", re.I), 45, "Potential scam"),
    (re.compile(r"\b(join|contact)\s*(whatsapp|telegram)\s*(group|channel|link)\b", re.I), 40, "Promotion"),
    
    # Adult services
    (re.compile(r"\b(escort|call\s*girl|sex\s*service)\b", re.I), 75, "Adult service spam"),
    (re.compile(r"\bvideo\s*call\s*(available|service)\b", re.I), 55, "Adult service spam"),
    (re.compile(r"\b(nude|naked)\s*(pic|photo|video|content)\b", re.I), 60, "Adult content"),
    (re.compile(r"\bonlyfans\.com\b", re.I), 50, "Adult platform"),
    
    # Crypto scams
    (re.compile(r"\b(airdrop|giveaway)\s*(crypto|btc|eth|token)\b", re.I), 55, "Crypto scam"),
    (re.compile(r"\bfree\s*(crypto|bitcoin|ethereum|token)\b", re.I), 60, "Crypto scam"),
    (re.compile(r"\b(invest|deposit)\s*(now|today)\s*(and|to)\s*(get|earn)\b", re.I), 65, "Investment scam"),
]

# Link/URL detection
URL_PATTERN = re.compile(
    r'(?:https?://|www\.)[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*',
    re.IGNORECASE
)

HIDDEN_URL_PATTERNS = [
    re.compile(r'\b\w+\s*\.\s*\w+\s*/\s*\w+', re.I),  # spaced out URLs
    re.compile(r'\b\w+\[?\.\]?\w+\[?\.\]?(?:com|org|net|io|xyz|me)\b', re.I),  # obfuscated TLDs
    re.compile(r't\s*\.\s*me\s*/\s*\w+', re.I),  # Telegram links
    re.compile(r'bit\s*\.\s*ly', re.I),  # URL shorteners
]

# Telegram entities that indicate links/mentions
URL_ENTITY_TYPES = {"url", "text_link"}
MENTION_ENTITY_TYPES = {"mention", "text_mention"}

# Emojis commonly used in spam
SPAM_EMOJIS = {'💰', '🤑', '💵', '💸', '🔥', '💯', '⬇️', '👇', '📣', '🚀', '💎', '🟢', '🔴', '⚡', '✅', '❌'}

# Normalization patterns
NORMALIZE_PATTERNS = [
    (re.compile(r'[^\w\s@.\-$₹€£]'), ' '),  # Remove special chars
    (re.compile(r'\s+'), ' '),  # Collapse whitespace
]

LEET_SPEAK_MAP = {
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's',
    '7': 't', '@': 'a', '$': 's', '!': 'i', '|': 'l',
}


# ============================================================
# Utility Functions
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text for spam detection"""
    result = text.lower()
    
    # Convert leet speak
    for leet, normal in LEET_SPEAK_MAP.items():
        result = result.replace(leet, normal)
    
    # Apply normalization patterns
    for pattern, replacement in NORMALIZE_PATTERNS:
        result = pattern.sub(replacement, result)
    
    return result.strip()


def get_message_hash(text: str) -> str:
    """Get hash of message for duplicate detection"""
    normalized = normalize_text(text)
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def count_emojis(text: str) -> int:
    """Count emojis in text"""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return len(emoji_pattern.findall(text))


def count_spam_emojis(text: str) -> int:
    """Count spam-associated emojis"""
    return sum(1 for char in text if char in SPAM_EMOJIS)


def is_admin(user_id: int, settings: Settings) -> bool:
    """Check if user is admin"""
    return user_id in settings.admin_user_ids


def is_trusted(user_id: int) -> bool:
    """Check if user is trusted"""
    return user_id in state.trusted_users


def is_owner_channel(chat_id: int) -> bool:
    """Check if this is the owner's channel"""
    return chat_id == OWNER_CHANNEL_ID


def is_allowed_channel(channel_id: int) -> bool:
    """Check if channel is in allowed list"""
    return channel_id in state.allowed_channels or channel_id == OWNER_CHANNEL_ID


# ============================================================
# Message Deletion & Actions
# ============================================================

async def safe_delete_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int) -> bool:
    """Safely delete a message, handling errors gracefully"""
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        return True
    except BadRequest as e:
        if "message to delete not found" in str(e).lower():
            return True  # Already deleted
        if "message can't be deleted" in str(e).lower():
            logger.debug(f"Cannot delete message {message_id} in {chat_id}")
        return False
    except Forbidden:
        logger.warning(f"No permission to delete in chat {chat_id}")
        return False
    except Exception as e:
        logger.error(f"Delete error for {message_id} in {chat_id}: {e}")
        return False


async def delete_after_delay(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int, delay: int) -> None:
    """Delete message after a delay"""
    await asyncio.sleep(delay)
    await safe_delete_message(context, chat_id, message_id)


async def send_ephemeral(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    settings: Settings,
    include_promo: bool = False
) -> Optional[Message]:
    """Send a message that auto-deletes after delay"""
    if not update.effective_message:
        return None
    
    final_text = text
    if include_promo and settings.promotion_text:
        final_text = f"{text}\n\n{settings.promotion_text}"
    
    try:
        msg = await update.effective_message.reply_text(
            final_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        context.application.create_task(
            delete_after_delay(context, update.effective_chat.id, msg.message_id, settings.deletion_delay_sec)
        )
        return msg
    except Exception as e:
        logger.error(f"Failed to send ephemeral: {e}")
        return None


async def log_action(
    context: ContextTypes.DEFAULT_TYPE,
    action_type: str,
    user_id: int,
    chat_id: int,
    details: Dict[str, Any],
    settings: Settings
) -> None:
    """Log action to database and optionally to log channel"""
    assert db is not None
    
    try:
        db.execute(
            """INSERT INTO action_logs (action_type, user_id, chat_id, details, performed_by)
               VALUES (%s, %s, %s, %s, %s);""",
            (action_type, user_id, chat_id, json.dumps(details), 0)
        )
    except Exception as e:
        logger.error(f"Failed to log action: {e}")
    
    # Send to log channel if configured
    if settings.log_channel_id:
        try:
            log_text = f"""📋 <b>Action Log</b>
Type: {action_type}
User: <code>{user_id}</code>
Chat: <code>{chat_id}</code>
Details: {json.dumps(details, indent=2)[:500]}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            
            await context.bot.send_message(
                chat_id=settings.log_channel_id,
                text=log_text,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.debug(f"Failed to send to log channel: {e}")


# ============================================================
# Spam Detection Engine
# ============================================================

async def analyze_message(
    msg: Message,
    text: str,
    settings: Settings
) -> SpamScore:
    """Comprehensive spam analysis returning a score"""
    score = SpamScore()
    
    if not text:
        return score
    
    text_lower = text.lower()
    normalized = normalize_text(text)
    
    entities = list(msg.entities or []) + list(msg.caption_entities or [])
    
    # ===== Level 1: Entity-based checks =====
    
    # Check for URLs (not in admin messages)
    url_count = sum(1 for e in entities if e.type in URL_ENTITY_TYPES)
    if url_count > 0:
        score.add(30 * url_count, f"Contains {url_count} link(s)")
    
    # Check for mentions
    mention_count = sum(1 for e in entities if e.type in MENTION_ENTITY_TYPES)
    if mention_count > 2:
        score.add(20, f"Multiple mentions ({mention_count})")
    
    # ===== Level 2: Pattern matching =====
    
    # Check hardcoded patterns
    for pattern, points, reason in SPAM_PATTERNS:
        if pattern.search(text_lower) or pattern.search(normalized):
            score.add(points, reason)
            if points >= 80:  # High severity - stop checking
                return score
    
    # Check custom patterns from DB
    for pattern in state.custom_spam_patterns:
        if pattern.search(text_lower) or pattern.search(normalized):
            score.add(40, "Matched custom pattern")
    
    # ===== Level 3: Blacklist check =====
    
    # High severity blacklist
    for word in state.blacklist_high_severity:
        if word in text_lower or word in normalized:
            score.add(70, f"High-severity blacklisted term: {word[:20]}")
            return score
    
    # Regular blacklist
    blacklist_matches = sum(1 for w in state.blacklist_words if w in text_lower or w in normalized)
    if blacklist_matches > 0:
        score.add(15 * blacklist_matches, f"Blacklisted terms: {blacklist_matches}")
    
    # ===== Level 4: Heuristic checks =====
    
    # Hidden URLs
    for pattern in HIDDEN_URL_PATTERNS:
        if pattern.search(text):
            score.add(25, "Hidden/obfuscated URL detected")
            break
    
    # Excessive emojis
    emoji_count = count_emojis(text)
    if emoji_count > settings.max_emojis_allowed:
        score.add(15, f"Excessive emojis ({emoji_count})")
    
    # Spam emojis
    spam_emoji_count = count_spam_emojis(text)
    if spam_emoji_count > 3:
        score.add(20, f"Spam emojis ({spam_emoji_count})")
    
    # Very long messages
    if len(text) > settings.max_message_length:
        score.add(10, "Excessively long message")
    
    # ALL CAPS (more than 70% uppercase for messages > 20 chars)
    if len(text) > 20:
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.7:
                score.add(15, "Excessive CAPS")
    
    # Repeated characters (like "hiiiiiiii")
    if re.search(r'(.)\1{4,}', text):
        score.add(10, "Character repetition")
    
    return score


async def check_flood(user_id: int, text: str, settings: Settings) -> Tuple[bool, str]:
    """Check if user is flooding"""
    now = time.time()
    flood_data = state.user_flood_data[user_id]
    
    # Clean old timestamps
    cutoff = now - settings.flood_time_window
    flood_data.message_times = [t for t in flood_data.message_times if t > cutoff]
    
    # Add current timestamp
    flood_data.message_times.append(now)
    
    # Check flood by message count
    if len(flood_data.message_times) > settings.flood_messages_limit:
        return True, f"Flooding ({len(flood_data.message_times)} messages in {settings.flood_time_window}s)"
    
    # Check duplicate messages
    msg_hash = get_message_hash(text)
    if msg_hash == flood_data.last_message_hash:
        flood_data.duplicate_count += 1
        if flood_data.duplicate_count >= 3:
            return True, f"Duplicate message flood ({flood_data.duplicate_count} times)"
    else:
        flood_data.last_message_hash = msg_hash
        flood_data.duplicate_count = 1
    
    return False, ""


# ============================================================
# Main Message Handler
# ============================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main message handler with comprehensive spam detection"""
    settings: Settings = context.bot_data["settings"]
    
    # ===== CHANNEL POST HANDLING =====
    # Check for channel posts FIRST - Owner channel posts are NEVER deleted
    if update.channel_post:
        channel = update.channel_post.chat
        if channel and is_owner_channel(channel.id):
            logger.debug(f"✅ Owner channel post allowed: {channel.id}")
            return
        if channel and is_allowed_channel(channel.id):
            logger.debug(f"✅ Allowed channel post: {channel.id}")
            return
        # Unknown channel - ignore (don't process as spam)
        return
    
    msg = update.effective_message
    if not msg:
        return
    
    user = msg.from_user
    chat = update.effective_chat
    
    if not user or not chat:
        return
    
    chat_id = chat.id
    user_id = user.id
    
    # ===== ALLOWLIST CHECKS =====
    
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
    
    # Update stats
    state.messages_processed += 1
    
    # ===== FORWARD HANDLING =====
    # Check forwarded messages
    forward_from_chat = msg.forward_from_chat
    forward_from_user = msg.forward_from
    
    is_forward = forward_from_chat is not None or forward_from_user is not None
    
    if is_forward:
        # Allow forwards from owner channel
        if forward_from_chat and is_owner_channel(forward_from_chat.id):
            logger.debug(f"✅ Forward from owner channel allowed")
            return
        
        # Allow forwards from allowed channels
        if forward_from_chat and is_allowed_channel(forward_from_chat.id):
            logger.debug(f"✅ Forward from allowed channel: {forward_from_chat.id}")
            return
        
        # Allow forwards from whitelisted users
        if user_id in state.forward_whitelist_users:
            logger.debug(f"✅ Forward allowed for whitelisted user: {user_id}")
            return
        
        # Block unauthorized forwards
        await enforce_spam(
            context, chat_id, user_id, msg,
            "Forwarding is not allowed without permission",
            50, settings
        )
        return
    
    # ===== RATE LIMITING =====
    now_ts = time.time()
    rate_key = (user_id, chat_id)
    last_ts = state.message_rate_limit_ts.get(rate_key, 0.0)
    
    if now_ts - last_ts < settings.min_seconds_between_msgs:
        await safe_delete_message(context, chat_id, msg.message_id)
        return
    
    state.message_rate_limit_ts[rate_key] = now_ts
    
    # ===== GET TEXT CONTENT =====
    text = msg.text or msg.caption or ""
    
    # ===== FLOOD CHECK =====
    if text:
        is_flood, flood_reason = await check_flood(user_id, text, settings)
        if is_flood:
            await enforce_spam(context, chat_id, user_id, msg, flood_reason, 60, settings)
            return
    
    # ===== SPAM ANALYSIS =====
    score = await analyze_message(msg, text, settings)
    
    # ===== AI CHECK (Final stage for borderline cases) =====
    if settings.ai_enabled and text and 20 <= score.score < 50:
        ai_result, ai_reason = await ai_check_spam(text, normalize_text(text), settings.ai_timeout_sec)
        if ai_result is True:
            score.add(40, ai_reason)
    
    # ===== ENFORCE IF SPAM =====
    if score.is_spam():
        reason = "; ".join(score.reasons[:3])  # Top 3 reasons
        await enforce_spam(context, chat_id, user_id, msg, reason, score.score, settings)
        
        # Log the action
        await log_action(context, "SPAM_DETECTED", user_id, chat_id, {
            "score": score.score,
            "reasons": score.reasons,
            "text_preview": text[:100] if text else ""
        }, settings)


async def enforce_spam(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    user_id: int,
    msg: Message,
    reason: str,
    severity: int,
    settings: Settings
) -> None:
    """Enforce spam action - delete message and warn/ban user"""
    
    # Delete the spam message
    await safe_delete_message(context, chat_id, msg.message_id)
    
    # Update stats
    state.spam_detected += 1
    
    # Increment warnings
    state.user_warnings[user_id] = state.user_warnings.get(user_id, 0) + 1
    warning_count = state.user_warnings[user_id]
    
    # High severity = immediate action
    if severity >= 80:
        warning_count = state.max_warnings  # Force ban
    
    if warning_count >= state.max_warnings:
        # BAN USER
        try:
            await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
            state.users_banned += 1
            
            # Store in DB
            if db:
                try:
                    db.execute(
                        """INSERT INTO banned_users (user_id, chat_id, reason, banned_by)
                           VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;""",
                        (user_id, chat_id, reason[:500], 0)
                    )
                except Exception:
                    pass
            
        except Exception as e:
            logger.warning(f"Ban failed for {user_id}: {e}")
        
        warning_text = f"""🚫 <b>User Banned</b>

<a href='tg://user?id={user_id}'>User</a> has been banned after {state.max_warnings} warnings.

Reason: {reason[:200]}

{settings.promotion_text}"""
        
        state.user_warnings.pop(user_id, None)
        logger.info(f"🚫 Banned user {user_id} | Reason: {reason}")
    else:
        # WARNING
        warning_text = f"""⚠️ <b>Warning {warning_count}/{state.max_warnings}</b>

<a href='tg://user?id={user_id}'>User</a>, your message was removed.

Reason: {reason[:200]}

{settings.promotion_text}"""
        
        logger.info(f"⚠️ Warning {warning_count} to {user_id} | Reason: {reason}")
    
    try:
        sent = await context.bot.send_message(
            chat_id=chat_id,
            text=warning_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        context.application.create_task(
            delete_after_delay(context, chat_id, sent.message_id, settings.deletion_delay_sec)
        )
    except Exception as e:
        logger.error(f"Failed to send warning: {e}")


# ============================================================
# Command Handlers
# ============================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start command handler"""
    await update.message.reply_text(
        f"""🛡️ <b>Guardian Bot v{BOT_VERSION}</b>

I protect Telegram groups from spam, scams, and unwanted content.

<b>Features:</b>
• Multi-layer spam detection
• AI-powered analysis
• Flood protection
• Forward control
• Customizable blacklist
• Warning system

Use /help for admin commands.""",
        parse_mode=ParseMode.HTML
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Help command for admins"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ This command is for admins only.", settings)
        return
    
    help_text = f"""🛡️ <b>Guardian Bot Admin Commands</b>

<b>📝 Blacklist Management:</b>
/addword &lt;word1&gt; &lt;word2&gt; ... - Add blacklist words
/removeword &lt;word&gt; - Remove word from blacklist
/listwords - Show all blacklisted words

<b>💬 Chat Management:</b>
/allowchat &lt;chat_id&gt; - Allow a chat
/allowthischat - Allow current chat
/removechat &lt;chat_id&gt; - Remove allowed chat
/listchats - List all allowed chats

<b>📢 Channel Management:</b>
/allowchannel &lt;channel_id&gt; - Allow channel to post/forward
/removechannel &lt;channel_id&gt; - Remove allowed channel
/listchannels - List allowed channels

<b>↗️ Forward Control:</b>
/allowforward &lt;user_id&gt; - Allow user to forward
/revokeforward &lt;user_id&gt; - Revoke forward permission
/listforwarders - List users with forward permission

<b>👤 User Management:</b>
/trust &lt;user_id&gt; - Add trusted user (bypasses all checks)
/untrust &lt;user_id&gt; - Remove trusted user
/listtrused - List trusted users
/warn &lt;user_id&gt; - Manually warn a user
/ban &lt;user_id&gt; - Ban a user
/unban &lt;user_id&gt; - Unban a user
/resetwarnings &lt;user_id&gt; - Reset user warnings

<b>⚙️ Settings:</b>
/setmaxwarnings &lt;n&gt; - Set warning threshold (Current: {state.max_warnings})
/addpattern &lt;pattern&gt; - Add custom spam pattern
/removepattern &lt;pattern&gt; - Remove spam pattern

<b>📊 Info:</b>
/stats - View bot statistics
/status - Bot status
/reload - Reload all caches
/botversion - Version info

<b>Owner Channel:</b> <code>{OWNER_CHANNEL_ID}</code> (always allowed)"""
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot statistics"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    uptime = datetime.now() - state.start_time
    uptime_str = str(uptime).split('.')[0]  # Remove microseconds
    
    stats_text = f"""📊 <b>Guardian Bot Statistics</b>

<b>📈 Activity:</b>
• Messages processed: {state.messages_processed:,}
• Spam detected: {state.spam_detected:,}
• Users banned: {state.users_banned:,}
• Active warnings: {len(state.user_warnings)}

<b>⚙️ Configuration:</b>
• Blacklist words: {len(state.blacklist_words)}
• High severity words: {len(state.blacklist_high_severity)}
• Allowed chats: {len(state.allowed_chats)}
• Allowed channels: {len(state.allowed_channels)}
• Forward whitelist: {len(state.forward_whitelist_users)}
• Trusted users: {len(state.trusted_users)}
• Custom patterns: {len(state.custom_spam_patterns)}

<b>🤖 System:</b>
• AI Enabled: {'✅ Yes' if settings.ai_enabled else '❌ No'}
• Max warnings: {state.max_warnings}
• Uptime: {uptime_str}
• Owner Channel: <code>{OWNER_CHANNEL_ID}</code>"""
    
    await update.message.reply_text(stats_text, parse_mode=ParseMode.HTML)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick status check"""
    settings: Settings = context.bot_data["settings"]
    
    await update.message.reply_text(
        f"""✅ <b>Guardian Bot is Active</b>

Version: {BOT_VERSION}
AI: {'🟢 ON' if settings.ai_enabled else '🔴 OFF'}
Processed: {state.messages_processed:,} messages
Spam blocked: {state.spam_detected:,}""",
        parse_mode=ParseMode.HTML
    )


async def cmd_botversion(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot version"""
    await update.message.reply_text(
        f"🛡️ Guardian Bot v{BOT_VERSION}\n\nUltra Powerful Edition with AI spam detection",
        parse_mode=ParseMode.HTML
    )


async def cmd_addword(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add words to blacklist"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, "Usage: /addword word1 word2 ...", settings)
        return
    
    words = {w.lower().strip() for w in context.args if w.strip()}
    added = 0
    
    for word in words:
        try:
            db.execute(
                "INSERT INTO blacklist (word, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
                (word, update.effective_user.id)
            )
            added += 1
            state.blacklist_words.add(word)
        except Exception as e:
            logger.error(f"Failed to add word '{word}': {e}")
    
    await update.message.reply_text(f"✅ Added {added} word(s) to blacklist")
    logger.info(f"Admin {update.effective_user.id} added {added} blacklist words")


async def cmd_removeword(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove word from blacklist"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, "Usage: /removeword word", settings)
        return
    
    word = context.args[0].lower().strip()
    
    try:
        db.execute("DELETE FROM blacklist WHERE word = %s;", (word,))
        state.blacklist_words.discard(word)
        state.blacklist_high_severity.discard(word)
        await update.message.reply_text(f"✅ Removed '{word}' from blacklist")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listwords(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List blacklisted words"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not state.blacklist_words:
        await update.message.reply_text("No blacklisted words.")
        return
    
    words_list = sorted(state.blacklist_words)[:100]  # Limit to 100
    words_text = ", ".join(words_list)
    
    if len(state.blacklist_words) > 100:
        words_text += f"\n\n... and {len(state.blacklist_words) - 100} more"
    
    await update.message.reply_text(f"📝 <b>Blacklisted Words ({len(state.blacklist_words)}):</b>\n\n{words_text}", parse_mode=ParseMode.HTML)


async def cmd_allowchat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Allow a chat by ID"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, f"❌ Admins only. Your ID: {update.effective_user.id}", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, "Usage: /allowchat <chat_id>", settings)
        return
    
    try:
        chat_id = int(context.args[0])
    except ValueError:
        await send_ephemeral(update, context, "❌ Invalid chat ID", settings)
        return
    
    try:
        db.execute(
            "INSERT INTO allowed_chats (chat_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (chat_id, update.effective_user.id)
        )
        state.allowed_chats.add(chat_id)
        await update.message.reply_text(f"✅ Chat {chat_id} is now allowed")
        logger.info(f"Chat {chat_id} allowed by admin {update.effective_user.id}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_allowthischat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Allow current chat"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    chat_id = update.effective_chat.id
    chat_title = update.effective_chat.title or "Unknown"
    
    try:
        db.execute(
            "INSERT INTO allowed_chats (chat_id, chat_title, added_by) VALUES (%s, %s, %s) ON CONFLICT (chat_id) DO UPDATE SET chat_title = %s;",
            (chat_id, chat_title, update.effective_user.id, chat_title)
        )
        state.allowed_chats.add(chat_id)
        await update.message.reply_text(f"✅ This chat is now protected!\n\nChat ID: <code>{chat_id}</code>", parse_mode=ParseMode.HTML)
        logger.info(f"Chat {chat_id} ({chat_title}) allowed by admin {update.effective_user.id}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_removechat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove chat from allowed list"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, "Usage: /removechat <chat_id>", settings)
        return
    
    try:
        chat_id = int(context.args[0])
    except ValueError:
        await send_ephemeral(update, context, "❌ Invalid chat ID", settings)
        return
    
    try:
        db.execute("DELETE FROM allowed_chats WHERE chat_id = %s;", (chat_id,))
        state.allowed_chats.discard(chat_id)
        await update.message.reply_text(f"✅ Chat {chat_id} removed from protection")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listchats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List allowed chats"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not state.allowed_chats:
        await update.message.reply_text("No chats are currently allowed.")
        return
    
    chats_text = "\n".join(f"• <code>{cid}</code>" for cid in sorted(state.allowed_chats))
    await update.message.reply_text(f"💬 <b>Allowed Chats ({len(state.allowed_chats)}):</b>\n\n{chats_text}", parse_mode=ParseMode.HTML)


async def cmd_allowchannel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Allow a channel"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, "Usage: /allowchannel <channel_id>", settings)
        return
    
    try:
        channel_id = int(context.args[0])
    except ValueError:
        await send_ephemeral(update, context, "❌ Invalid channel ID", settings)
        return
    
    try:
        db.execute(
            "INSERT INTO allowed_channels (channel_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (channel_id, update.effective_user.id)
        )
        state.allowed_channels.add(channel_id)
        await update.message.reply_text(f"✅ Channel {channel_id} is now allowed")
        logger.info(f"Channel {channel_id} allowed by admin {update.effective_user.id}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_removechannel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove channel from allowed list"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, "Usage: /removechannel <channel_id>", settings)
        return
    
    try:
        channel_id = int(context.args[0])
    except ValueError:
        await send_ephemeral(update, context, "❌ Invalid channel ID", settings)
        return
    
    # Don't allow removing owner channel
    if channel_id == OWNER_CHANNEL_ID:
        await update.message.reply_text("❌ Cannot remove owner channel from allowed list")
        return
    
    try:
        db.execute("DELETE FROM allowed_channels WHERE channel_id = %s;", (channel_id,))
        state.allowed_channels.discard(channel_id)
        await update.message.reply_text(f"✅ Channel {channel_id} removed")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listchannels(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List allowed channels"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    channels_list = []
    for cid in sorted(state.allowed_channels):
        if cid == OWNER_CHANNEL_ID:
            channels_list.append(f"• <code>{cid}</code> (Owner - Protected)")
        else:
            channels_list.append(f"• <code>{cid}</code>")
    
    channels_text = "\n".join(channels_list) if channels_list else "No channels allowed"
    await update.message.reply_text(f"📢 <b>Allowed Channels ({len(state.allowed_channels)}):</b>\n\n{channels_text}", parse_mode=ParseMode.HTML)


async def cmd_allowforward(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Allow user to forward messages"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
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
        await send_ephemeral(update, context, "Usage: Reply to a user or /allowforward <user_id>", settings)
        return
    
    try:
        db.execute(
            "INSERT INTO forward_whitelist (user_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (target_user_id, update.effective_user.id)
        )
        state.forward_whitelist_users.add(target_user_id)
        await update.message.reply_text(f"✅ User {target_user_id} can now forward messages")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_revokeforward(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Revoke forward permission"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
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
        await send_ephemeral(update, context, "Usage: Reply to a user or /revokeforward <user_id>", settings)
        return
    
    try:
        db.execute("DELETE FROM forward_whitelist WHERE user_id = %s;", (target_user_id,))
        state.forward_whitelist_users.discard(target_user_id)
        await update.message.reply_text(f"❌ User {target_user_id} can no longer forward messages")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listforwarders(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List users with forward permission"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not state.forward_whitelist_users:
        await update.message.reply_text("No users have forward permission.")
        return
    
    users_text = "\n".join(f"• <code>{uid}</code>" for uid in sorted(state.forward_whitelist_users))
    await update.message.reply_text(f"↗️ <b>Forward Whitelist ({len(state.forward_whitelist_users)}):</b>\n\n{users_text}", parse_mode=ParseMode.HTML)


async def cmd_trust(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add trusted user"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
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
        await send_ephemeral(update, context, "Usage: Reply to a user or /trust <user_id>", settings)
        return
    
    try:
        db.execute(
            "INSERT INTO trusted_users (user_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING;",
            (target_user_id, update.effective_user.id)
        )
        state.trusted_users.add(target_user_id)
        await update.message.reply_text(f"✅ User {target_user_id} is now trusted (bypasses all checks)")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_untrust(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove trusted user"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
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
        await send_ephemeral(update, context, "Usage: Reply to a user or /untrust <user_id>", settings)
        return
    
    try:
        db.execute("DELETE FROM trusted_users WHERE user_id = %s;", (target_user_id,))
        state.trusted_users.discard(target_user_id)
        await update.message.reply_text(f"❌ User {target_user_id} is no longer trusted")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_listtrusted(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List trusted users"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not state.trusted_users:
        await update.message.reply_text("No trusted users.")
        return
    
    users_text = "\n".join(f"• <code>{uid}</code>" for uid in sorted(state.trusted_users))
    await update.message.reply_text(f"👤 <b>Trusted Users ({len(state.trusted_users)}):</b>\n\n{users_text}", parse_mode=ParseMode.HTML)


async def cmd_warn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually warn a user"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
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
        await send_ephemeral(update, context, "Usage: Reply to a user or /warn <user_id>", settings)
        return
    
    state.user_warnings[target_user_id] = state.user_warnings.get(target_user_id, 0) + 1
    count = state.user_warnings[target_user_id]
    
    await update.message.reply_text(
        f"⚠️ User <code>{target_user_id}</code> warned by admin.\nWarnings: {count}/{state.max_warnings}",
        parse_mode=ParseMode.HTML
    )


async def cmd_ban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ban a user"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
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
        await send_ephemeral(update, context, "Usage: Reply to a user or /ban <user_id>", settings)
        return
    
    try:
        await context.bot.ban_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user_id
        )
        state.users_banned += 1
        await update.message.reply_text(f"🚫 User <code>{target_user_id}</code> has been banned.", parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to ban: {e}")


async def cmd_unban(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unban a user"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, "Usage: /unban <user_id>", settings)
        return
    
    try:
        target_user_id = int(context.args[0])
    except ValueError:
        await send_ephemeral(update, context, "❌ Invalid user ID", settings)
        return
    
    try:
        await context.bot.unban_chat_member(
            chat_id=update.effective_chat.id,
            user_id=target_user_id,
            only_if_banned=True
        )
        db.execute("DELETE FROM banned_users WHERE user_id = %s AND chat_id = %s;", (target_user_id, update.effective_chat.id))
        await update.message.reply_text(f"✅ User <code>{target_user_id}</code> has been unbanned.", parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to unban: {e}")


async def cmd_resetwarnings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset user warnings"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
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
        await send_ephemeral(update, context, "Usage: Reply to a user or /resetwarnings <user_id>", settings)
        return
    
    state.user_warnings.pop(target_user_id, None)
    await update.message.reply_text(f"✅ Warnings reset for user <code>{target_user_id}</code>", parse_mode=ParseMode.HTML)


async def cmd_setmaxwarnings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set max warnings before ban"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, f"Current: {state.max_warnings}\nUsage: /setmaxwarnings <number>", settings)
        return
    
    try:
        new_max = max(1, int(context.args[0]))
    except ValueError:
        await send_ephemeral(update, context, "❌ Please provide a valid number", settings)
        return
    
    try:
        db.execute(
            """INSERT INTO bot_settings (setting_key, setting_value, updated_by)
               VALUES ('max_warnings', %s, %s)
               ON CONFLICT (setting_key) DO UPDATE SET setting_value = %s, updated_at = CURRENT_TIMESTAMP;""",
            (str(new_max), update.effective_user.id, str(new_max))
        )
        state.max_warnings = new_max
        await update.message.reply_text(f"✅ Max warnings set to {new_max}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_addpattern(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add custom spam pattern"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, "Usage: /addpattern <pattern>", settings)
        return
    
    pattern = " ".join(context.args)
    
    try:
        # Test if it's a valid regex
        re.compile(pattern, re.IGNORECASE)
        
        db.execute(
            "INSERT INTO spam_patterns (pattern, is_regex, added_by) VALUES (%s, TRUE, %s) ON CONFLICT DO NOTHING;",
            (pattern, update.effective_user.id)
        )
        load_custom_spam_patterns()
        await update.message.reply_text(f"✅ Pattern added: <code>{pattern[:100]}</code>", parse_mode=ParseMode.HTML)
    except re.error as e:
        await update.message.reply_text(f"❌ Invalid regex pattern: {e}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_removepattern(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove spam pattern"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    if not context.args:
        await send_ephemeral(update, context, "Usage: /removepattern <pattern>", settings)
        return
    
    pattern = " ".join(context.args)
    
    try:
        db.execute("DELETE FROM spam_patterns WHERE pattern = %s;", (pattern,))
        load_custom_spam_patterns()
        await update.message.reply_text(f"✅ Pattern removed")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reload all caches"""
    settings: Settings = context.bot_data["settings"]
    
    if not is_admin(update.effective_user.id, settings):
        await send_ephemeral(update, context, "❌ Admins only.", settings)
        return
    
    reload_all_caches(settings)
    await update.message.reply_text("✅ All caches reloaded!")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Report a spam message"""
    settings: Settings = context.bot_data["settings"]
    
    if not update.message.reply_to_message:
        await send_ephemeral(update, context, "❌ Reply to a spam message to report it.", settings)
        return
    
    spam_msg = update.message.reply_to_message
    spam_text = spam_msg.text or spam_msg.caption or "[No text content]"
    
    try:
        db.execute(
            """INSERT INTO reported_spam (message, message_hash, reported_by, chat_id)
               VALUES (%s, %s, %s, %s);""",
            (spam_text[:2000], get_message_hash(spam_text), update.effective_user.id, update.effective_chat.id)
        )
        await update.message.reply_text("✅ Spam reported. Thank you for helping keep the group clean!")
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to report: {e}")


# ============================================================
# Error Handler
# ============================================================

async def error_handler(update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors gracefully"""
    error = context.error
    
    # Ignore common non-critical errors
    if isinstance(error, (BadRequest, Forbidden)):
        logger.debug(f"Telegram error: {error}")
        return
    
    logger.error("Exception while handling update:", exc_info=error)
    
    # Try to notify user
    if update and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ An error occurred. Please try again later."
            )
        except Exception:
            pass


# ============================================================
# Background Tasks
# ============================================================

async def cleanup_old_data(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic cleanup of old data"""
    now = datetime.now()
    
    # Clean rate limit data older than 1 hour
    cutoff_ts = time.time() - 3600
    state.message_rate_limit_ts = {
        k: v for k, v in state.message_rate_limit_ts.items()
        if v > cutoff_ts
    }
    
    # Clean flood data older than 1 hour
    for user_id in list(state.user_flood_data.keys()):
        flood_data = state.user_flood_data[user_id]
        flood_data.message_times = [t for t in flood_data.message_times if t > cutoff_ts]
        if not flood_data.message_times:
            del state.user_flood_data[user_id]
    
    # Clean old message times
    cutoff_dt = now - timedelta(hours=24)
    state.user_last_message_time = {
        k: v for k, v in state.user_last_message_time.items()
        if v > cutoff_dt
    }
    
    state.last_cleanup = now
    logger.debug("Cleanup completed")


# ============================================================
# Flask Keep-Alive Server
# ============================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():
    return jsonify({
        "status": "running",
        "bot": "Guardian Bot",
        "version": BOT_VERSION,
        "messages_processed": state.messages_processed,
        "spam_detected": state.spam_detected,
        "users_banned": state.users_banned,
    })


@flask_app.route("/health")
def health():
    return jsonify({"status": "healthy"})


def run_flask(port: int):
    serve(flask_app, host="0.0.0.0", port=port, _quiet=True)


# ============================================================
# Application Builder
# ============================================================

def build_application(settings: Settings) -> Application:
    """Build the Telegram bot application"""
    application = Application.builder().token(settings.telegram_bot_token).build()
    
    # Error handler
    application.add_error_handler(error_handler)
    
    # Command handlers
    commands = [
        ("start", cmd_start),
        ("help", cmd_help),
        ("stats", cmd_stats),
        ("status", cmd_status),
        ("botversion", cmd_botversion),
        ("addword", cmd_addword),
        ("removeword", cmd_removeword),
        ("listwords", cmd_listwords),
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
        ("setmaxwarnings", cmd_setmaxwarnings),
        ("addpattern", cmd_addpattern),
        ("removepattern", cmd_removepattern),
        ("reload", cmd_reload),
        ("report", cmd_report),
    ]
    
    for cmd_name, handler in commands:
        application.add_handler(CommandHandler(cmd_name, handler))
    
    # Message handler for all non-command messages
    application.add_handler(MessageHandler(
        filters.ALL & ~filters.COMMAND,
        handle_message
    ))
    
    # Store settings
    application.bot_data["settings"] = settings
    
    # Add job for cleanup
    application.job_queue.run_repeating(
        cleanup_old_data,
        interval=timedelta(minutes=30),
        first=timedelta(minutes=5)
    )
    
    return application


# ============================================================
# Main Entry Point
# ============================================================

def main() -> None:
    """Main entry point"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           🛡️  GUARDIAN BOT v{BOT_VERSION}  🛡️                     ║
║                Ultra Powerful Edition                        ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Load settings
    settings = Settings.from_env()
    validate_settings(settings)
    
    # Initialize database
    global db
    db = Database.get_instance(settings.database_url)
    setup_database()
    
    # Load all caches
    reload_all_caches(settings)
    
    # Initialize AI
    ai_init(settings)
    
    # Start Flask server in background
    flask_thread = threading.Thread(
        target=run_flask,
        args=(settings.port,),
        daemon=True
    )
    flask_thread.start()
    logger.info(f"🌐 Flask server started on port {settings.port}")
    
    # Build application
    app = build_application(settings)
    
    # Graceful shutdown handler
    def shutdown_handler(signum, frame):
        logger.info("⏹️ Shutting down...")
        if db:
            db.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Start bot
    logger.info(f"""
🛡️ Guardian Bot Started!
├── Version: {BOT_VERSION}
├── AI: {'✅ Enabled' if settings.ai_enabled else '❌ Disabled'}
├── Owner Channel: {OWNER_CHANNEL_ID}
├── Max Warnings: {state.max_warnings}
├── Allowed Chats: {len(state.allowed_chats)}
├── Allowed Channels: {len(state.allowed_channels)}
├── Blacklist Words: {len(state.blacklist_words)}
└── Port: {settings.port}
""")
    
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
