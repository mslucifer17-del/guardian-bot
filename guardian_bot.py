"""
Guardian Bot v3.0 - Advanced Telegram Spam Protection System
Enhanced with modular architecture, advanced AI, and intelligent caching
"""

import os
import asyncio
import re
import logging
import time
import json
import hashlib
from typing import Optional, Dict, List, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache, wraps
from contextlib import asynccontextmanager
import threading
from collections import defaultdict, deque
import pickle

import aiohttp
import asyncpg
from flask import Flask
import google.generativeai as genai
from telegram import Update, Message, User
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from telegram.error import TelegramError
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# ============= Configuration Management =============
@dataclass
class BotConfig:
    """Centralized configuration management"""
    telegram_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))
    admin_ids: List[int] = field(default_factory=list)
    channel_id: int = -1002533091260
    promotion_text: str = "For promotions, join: https://t.me/+scHqQ2SR0J45NjQ1"
    max_warnings: int = 10
    port: int = 8080
    
    # Advanced settings
    ai_confidence_threshold: float = 0.75
    rate_limit_window: int = 60  # seconds
    rate_limit_max_messages: int = 10
    cache_ttl: int = 300  # seconds
    max_message_length: int = 4096
    auto_delete_delay: int = 10
    db_pool_min: int = 2
    db_pool_max: int = 10
    
    def __post_init__(self):
        if admin_ids_str := os.environ.get("ADMIN_USER_IDS", ""):
            self.admin_ids = [int(id.strip()) for id in admin_ids_str.split(",") if id.strip()]
        self.port = int(os.environ.get('PORT', 8080))
        self.max_warnings = int(os.environ.get("MAX_WARNINGS", 10))


# ============= Enhanced Logging =============
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for better visibility"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging():
    """Enhanced logging setup"""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    ))
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Reduce noise from external libraries
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


# ============= Database Management =============
class DatabasePool:
    """Advanced database connection pool manager"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=self.config.db_pool_min,
                max_size=self.config.db_pool_max,
                command_timeout=10,
                max_queries=50000,
                max_cached_statement_lifetime=300
            )
            await self._setup_tables()
            self.logger.info("Database pool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        async with self.pool.acquire() as connection:
            yield connection
    
    async def _setup_tables(self):
        """Setup all required database tables with optimizations"""
        async with self.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS blacklist (
                    id SERIAL PRIMARY KEY,
                    word TEXT NOT NULL UNIQUE,
                    severity INTEGER DEFAULT 1,
                    added_by BIGINT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_blacklist_word ON blacklist(word);
                CREATE INDEX IF NOT EXISTS idx_blacklist_severity ON blacklist(severity);
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id BIGINT PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    reputation_score INTEGER DEFAULT 0,
                    total_messages INTEGER DEFAULT 0,
                    spam_messages INTEGER DEFAULT 0,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_trusted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_user_reputation ON user_profiles(reputation_score);
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS spam_patterns (
                    id SERIAL PRIMARY KEY,
                    pattern TEXT NOT NULL UNIQUE,
                    pattern_type TEXT NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    false_positives INTEGER DEFAULT 0,
                    true_positives INTEGER DEFAULT 0,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS message_history (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    chat_id BIGINT NOT NULL,
                    message_text TEXT,
                    is_spam BOOLEAN DEFAULT FALSE,
                    spam_score FLOAT DEFAULT 0,
                    detected_reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_message_history_user ON message_history(user_id);
                CREATE INDEX IF NOT EXISTS idx_message_history_timestamp ON message_history(timestamp);
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_training_data (
                    id SERIAL PRIMARY KEY,
                    message_text TEXT NOT NULL,
                    is_spam BOOLEAN NOT NULL,
                    confidence FLOAT,
                    verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Keep existing tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS allowed_chats (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT NOT NULL UNIQUE,
                    chat_name TEXT,
                    added_by BIGINT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value JSONB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()


# ============= Cache Management =============
class IntelligentCache:
    """Advanced caching system with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.access_order = deque(maxlen=max_size)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    self.hit_count += 1
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    return value
                else:
                    del self.cache[key]
            
            self.miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        async with self.lock:
            ttl = ttl or self.default_ttl
            expiry = time.time() + ttl
            
            if len(self.cache) >= self.max_size and key not in self.cache:
                # LRU eviction
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = (value, expiry)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    async def clear(self):
        """Clear all cache"""
        async with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": f"{hit_rate:.2f}%",
            "size": len(self.cache),
            "max_size": self.max_size
        }


# ============= AI Enhancement Layer =============
class AdvancedAIDetector:
    """Enhanced AI-based spam detection with multiple models"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini
        genai.configure(api_key=config.gemini_api_key)
        self.gemini_model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=self._get_enhanced_prompt()
        )
        
        # ML components
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.is_trained = False
        
        # Pattern matching
        self.compiled_patterns = self._compile_patterns()
        
    def _get_enhanced_prompt(self) -> str:
        return """You are an advanced AI spam detection system. Analyze messages with these priorities:
        
        CRITICAL VIOLATIONS (immediate SPAM):
        - Child exploitation content (CP, CSAM)
        - Adult/sexual content sales
        - Drug dealing or illegal substances
        - Financial scams and frauds
        - Phishing attempts
        
        HIGH RISK INDICATORS:
        - Multiple payment methods mentioned
        - Suspicious URLs or hidden links
        - Excessive emojis in promotional context
        - Urgency tactics ("limited time", "act now")
        - Impersonation attempts
        
        ANALYSIS OUTPUT:
        Return JSON: {"is_spam": boolean, "confidence": 0-1, "reason": "string", "severity": "low/medium/high/critical"}
        
        Consider context, language variations, and obfuscation attempts."""
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for efficiency"""
        patterns = {
            'critical': [
                r'\b(?:child|kid|minor).*(?:porn|nude|sex)\b',
                r'\b(?:cp|csam|pthc)\b',
                r'\b(?:sell|buy).*(?:drug|cocaine|meth|heroin)\b'
            ],
            'high': [
                r'(?:upi|paypal|crypto|bitcoin).*(?:payment|transfer)',
                r'(?:click|visit).*(?:link|url|website).*(?:now|fast|quick)',
                r'(?:earn|make).*(?:\$|\€|\£).*(?:day|hour|week)'
            ],
            'medium': [
                r'(?:discount|offer|sale).*(?:\d+%|free)',
                r'(?:join|subscribe).*(?:channel|group).*(?:now|today)'
            ]
        }
        
        return {
            category: [re.compile(pattern, re.IGNORECASE) 
                      for pattern in pattern_list]
            for category, pattern_list in patterns.items()
        }
    
    async def analyze_message(
        self, 
        text: str, 
        user_profile: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Comprehensive message analysis"""
        
        # Quick pattern check
        pattern_result = self._check_patterns(text)
        if pattern_result['severity'] == 'critical':
            return {
                'is_spam': True,
                'confidence': 1.0,
                'reason': pattern_result['reason'],
                'severity': 'critical',
                'method': 'pattern'
            }
        
        # ML classification if trained
        ml_result = await self._ml_classify(text) if self.is_trained else None
        
        # Gemini analysis for complex cases
        gemini_result = await self._gemini_analyze(text)
        
        # Combine results with weighted scoring
        final_result = self._combine_results(pattern_result, ml_result, gemini_result)
        
        # Consider user reputation
        if user_profile:
            final_result = self._adjust_for_reputation(final_result, user_profile)
        
        return final_result
    
    def _check_patterns(self, text: str) -> Dict[str, Any]:
        """Check text against compiled patterns"""
        for severity, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return {
                        'severity': severity,
                        'reason': f'Matched {severity} risk pattern',
                        'confidence': 0.9 if severity == 'critical' else 0.7
                    }
        
        return {'severity': 'low', 'reason': None, 'confidence': 0}
    
    async def _ml_classify(self, text: str) -> Optional[Dict[str, Any]]:
        """Machine learning classification"""
        try:
            features = self.vectorizer.transform([text])
            prediction = self.classifier.predict(features)[0]
            probability = self.classifier.predict_proba(features)[0].max()
            
            return {
                'is_spam': bool(prediction),
                'confidence': float(probability),
                'method': 'ml'
            }
        except Exception as e:
            self.logger.error(f"ML classification error: {e}")
            return None
    
    async def _gemini_analyze(self, text: str) -> Dict[str, Any]:
        """Gemini AI analysis with timeout and error handling"""
        try:
            response = await asyncio.wait_for(
                self.gemini_model.generate_content_async(
                    f"Analyze for spam/scam:\n{text[:500]}"
                ),
                timeout=5.0
            )
            
            # Parse JSON response
            result_text = response.text.strip()
            if result_text.startswith('{'):
                return json.loads(result_text)
            
            # Fallback parsing
            is_spam = 'SPAM' in result_text.upper()
            return {
                'is_spam': is_spam,
                'confidence': 0.8 if is_spam else 0.2,
                'reason': 'AI detected suspicious content' if is_spam else 'Clean',
                'method': 'gemini'
            }
            
        except asyncio.TimeoutError:
            self.logger.warning("Gemini timeout")
            return {'is_spam': False, 'confidence': 0, 'method': 'timeout'}
        except Exception as e:
            self.logger.error(f"Gemini error: {e}")
            return {'is_spam': False, 'confidence': 0, 'method': 'error'}
    
    def _combine_results(self, *results) -> Dict[str, Any]:
        """Intelligently combine multiple detection results"""
        valid_results = [r for r in results if r and r.get('confidence', 0) > 0]
        
        if not valid_results:
            return {
                'is_spam': False,
                'confidence': 0,
                'reason': 'No detection triggered',
                'severity': 'low'
            }
        
        # Weighted average based on method reliability
        weights = {'pattern': 1.0, 'ml': 0.8, 'gemini': 0.9}
        total_weight = 0
        weighted_confidence = 0
        
        for result in valid_results:
            method = result.get('method', 'unknown')
            weight = weights.get(method, 0.5)
            confidence = result.get('confidence', 0)
            
            weighted_confidence += confidence * weight
            total_weight += weight
        
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        
        # Determine if spam based on threshold
        is_spam = final_confidence >= self.config.ai_confidence_threshold
        
        # Get the highest severity
        severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        max_severity = 'low'
        for result in valid_results:
            result_severity = result.get('severity', 'low')
            if severity_order.get(result_severity, 1) > severity_order.get(max_severity, 1):
                max_severity = result_severity
        
        return {
            'is_spam': is_spam,
            'confidence': final_confidence,
            'reason': valid_results[0].get('reason', 'Multiple indicators'),
            'severity': max_severity,
            'methods_used': [r.get('method') for r in valid_results]
        }
    
    def _adjust_for_reputation(self, result: Dict, user_profile: Dict) -> Dict:
        """Adjust detection based on user reputation"""
        reputation = user_profile.get('reputation_score', 0)
        is_trusted = user_profile.get('is_trusted', False)
        
        if is_trusted:
            result['confidence'] *= 0.5  # Reduce false positives for trusted users
        elif reputation < -10:
            result['confidence'] *= 1.2  # Increase sensitivity for bad actors
        
        result['confidence'] = min(1.0, result['confidence'])
        result['is_spam'] = result['confidence'] >= self.config.ai_confidence_threshold
        
        return result
    
    async def train_model(self, training_data: List[Tuple[str, bool]]):
        """Train the ML model with new data"""
        if len(training_data) < 100:
            self.logger.warning("Insufficient training data")
            return
        
        texts, labels = zip(*training_data)
        
        try:
            self.vectorizer.fit(texts)
            features = self.vectorizer.transform(texts)
            self.classifier.fit(features, labels)
            self.is_trained = True
            self.logger.info(f"ML model trained with {len(training_data)} samples")
        except Exception as e:
            self.logger.error(f"Training error: {e}")


# ============= User Management =============
class UserManager:
    """Advanced user profiling and reputation system"""
    
    def __init__(self, db_pool: DatabasePool, cache: IntelligentCache):
        self.db = db_pool
        self.cache = cache
        self.logger = logging.getLogger(__name__)
        
    async def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get or create user profile with caching"""
        cache_key = f"user_profile_{user_id}"
        
        # Check cache first
        profile = await self.cache.get(cache_key)
        if profile:
            return profile
        
        # Fetch from database
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE user_id = $1",
                user_id
            )
            
            if not row:
                # Create new profile
                row = await conn.fetchrow("""
                    INSERT INTO user_profiles (user_id) 
                    VALUES ($1) 
                    RETURNING *
                """, user_id)
            
            profile = dict(row)
            
        # Cache the profile
        await self.cache.set(cache_key, profile, ttl=600)
        return profile
    
    async def update_reputation(
        self, 
        user_id: int, 
        delta: int, 
        reason: str
    ):
        """Update user reputation score"""
        async with self.db.acquire() as conn:
            await conn.execute("""
                UPDATE user_profiles 
                SET reputation_score = reputation_score + $2,
                    last_seen = CURRENT_TIMESTAMP
                WHERE user_id = $1
            """, user_id, delta)
        
        # Invalidate cache
        await self.cache.set(f"user_profile_{user_id}", None, ttl=1)
        
        self.logger.info(f"User {user_id} reputation changed by {delta}: {reason}")
    
    async def record_message(
        self, 
        user_id: int, 
        chat_id: int,
        message_text: str,
        is_spam: bool,
        spam_score: float,
        reason: Optional[str] = None
    ):
        """Record message in history for analysis"""
        async with self.db.acquire() as conn:
            await conn.execute("""
                INSERT INTO message_history 
                (user_id, chat_id, message_text, is_spam, spam_score, detected_reason)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, user_id, chat_id, message_text[:1000], is_spam, spam_score, reason)
            
            # Update user stats
            if is_spam:
                await conn.execute("""
                    UPDATE user_profiles 
                    SET spam_messages = spam_messages + 1,
                        total_messages = total_messages + 1
                    WHERE user_id = $1
                """, user_id)
            else:
                await conn.execute("""
                    UPDATE user_profiles 
                    SET total_messages = total_messages + 1
                    WHERE user_id = $1
                """, user_id)


# ============= Rate Limiter =============
class AdvancedRateLimiter:
    """Token bucket rate limiter with burst support"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.buckets: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
        
    async def check_rate_limit(
        self, 
        key: str, 
        max_tokens: int = 10,
        refill_rate: float = 1.0,
        burst_size: int = 5
    ) -> Tuple[bool, Optional[float]]:
        """Check if action is rate limited"""
        async with self.lock:
            now = time.time()
            
            if key not in self.buckets:
                self.buckets[key] = {
                    'tokens': max_tokens,
                    'last_refill': now,
                    'burst_tokens': burst_size
                }
            
            bucket = self.buckets[key]
            
            # Refill tokens
            time_passed = now - bucket['last_refill']
            tokens_to_add = time_passed * refill_rate
            bucket['tokens'] = min(max_tokens, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now
            
            # Check if we can consume a token
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True, None
            elif bucket['burst_tokens'] > 0:
                # Use burst token
                bucket['burst_tokens'] -= 1
                return True, None
            else:
                # Calculate wait time
                wait_time = (1 - bucket['tokens']) / refill_rate
                return False, wait_time
    
    async def cleanup_old_buckets(self):
        """Remove old unused buckets"""
        async with self.lock:
            now = time.time()
            expired_keys = [
                key for key, bucket in self.buckets.items()
                if now - bucket['last_refill'] > 3600  # 1 hour
            ]
            for key in expired_keys:
                del self.buckets[key]


# ============= Guardian Bot Main Class =============
class GuardianBot:
    """Main bot class with all enhanced features"""
    
    def __init__(self):
        self.config = BotConfig()
        self.logger = setup_logging()
        self.db = DatabasePool(self.config)
        self.cache = IntelligentCache()
        self.ai_detector = AdvancedAIDetector(self.config)
        self.rate_limiter = AdvancedRateLimiter(self.config)
        self.user_manager = None  # Initialized after DB
        
        # State management
        self.warnings: Dict[int, int] = defaultdict(int)
        self.blacklist: Set[str] = set()
        self.allowed_chats: Set[int] = set()
        self.forward_whitelist: Set[int] = set()
        self.dynamic_commands: Dict[str, Dict] = {}
        
        # Performance metrics
        self.metrics = {
            'messages_processed': 0,
            'spam_detected': 0,
            'users_banned': 0,
            'false_positives': 0,
            'start_time': datetime.now()
        }
        
        # Flask app for health checks
        self.flask_app = self._create_flask_app()
        
    def _create_flask_app(self) -> Flask:
        """Create Flask app for health checks and metrics"""
        app = Flask(__name__)
        
        @app.route('/')
        def health():
            uptime = datetime.now() - self.metrics['start_time']
            return {
                'status': 'healthy',
                'version': '3.0',
                'uptime': str(uptime),
                'metrics': self.metrics,
                'cache_stats': self.cache.get_stats()
            }
        
        @app.route('/metrics')
        def metrics():
            return self.metrics
        
        return app
    
    async def initialize(self):
        """Initialize all bot components"""
        self.logger.info("🚀 Initializing Guardian Bot v3.0...")
        
        # Initialize database
        await self.db.initialize()
        
        # Initialize user manager
        self.user_manager = UserManager(self.db, self.cache)
        
        # Load configurations
        await self._load_configurations()
        
        # Start background tasks
        asyncio.create_task(self._periodic_cleanup())
        asyncio.create_task(self._train_ml_model())
        
        self.logger.info("✅ Guardian Bot initialized successfully!")
    
    async def _load_configurations(self):
        """Load all configurations from database"""
        async with self.db.acquire() as conn:
            # Load blacklist
            rows = await conn.fetch("SELECT word FROM blacklist")
            self.blacklist = {row['word'].lower() for row in rows}
            
            # Load allowed chats
            rows = await conn.fetch("SELECT chat_id FROM allowed_chats")
            self.allowed_chats = {row['chat_id'] for row in rows}
            
            # Load forward whitelist
            rows = await conn.fetch("SELECT user_id FROM forward_whitelist")
            self.forward_whitelist = {row['user_id'] for row in rows}
            
            # Load settings
            rows = await conn.fetch("SELECT setting_key, setting_value FROM bot_settings")
            for row in rows:
                if row['setting_key'] == 'max_warnings':
                    self.config.max_warnings = int(row['setting_value'].get('value', 10))
        
        self.logger.info(f"Loaded: {len(self.blacklist)} blacklist words, "
                        f"{len(self.allowed_chats)} allowed chats")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup tasks"""
        while True:
            await asyncio.sleep(3600)  # Every hour
            try:
                # Cleanup rate limiter
                await self.rate_limiter.cleanup_old_buckets()
                
                # Clear old cache entries
                await self.cache.clear()
                
                # Cleanup old warnings
                now = datetime.now()
                expired_users = [
                    user_id for user_id, timestamp in self.warnings.items()
                    if isinstance(timestamp, datetime) and (now - timestamp).days > 7
                ]
                for user_id in expired_users:
                    del self.warnings[user_id]
                
                self.logger.info("Periodic cleanup completed")
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    async def _train_ml_model(self):
        """Periodically retrain ML model with new data"""
        while True:
            await asyncio.sleep(86400)  # Daily
            try:
                async with self.db.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT message_text, is_spam 
                        FROM ml_training_data 
                        WHERE verified = true
                        ORDER BY created_at DESC
                        LIMIT 10000
                    """)
                    
                    if len(rows) >= 100:
                        training_data = [(row['message_text'], row['is_spam']) for row in rows]
                        await self.ai_detector.train_model(training_data)
                        self.logger.info(f"ML model retrained with {len(training_data)} samples")
                
            except Exception as e:
                self.logger.error(f"ML training error: {e}")
    
    # ============= Message Handling =============
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced message handling with AI and caching"""
        if not update.message or not update.message.from_user:
            return
        
        user = update.message.from_user
        chat_id = update.effective_chat.id
        message = update.message
        
        # Skip bot messages
        if user.is_bot:
            asyncio.create_task(self._delete_message_after_delay(
                chat_id, message.message_id, context
            ))
            return
        
        # Check if chat is allowed
        if chat_id not in self.allowed_chats:
            return
        
        # Admin check
        if user.id in self.config.admin_ids:
            return
        
        # Rate limiting
        rate_limit_key = f"user_{user.id}_{chat_id}"
        allowed, wait_time = await self.rate_limiter.check_rate_limit(
            rate_limit_key,
            max_tokens=self.config.rate_limit_max_messages,
            refill_rate=self.config.rate_limit_max_messages / self.config.rate_limit_window
        )
        
        if not allowed:
            await self._handle_rate_limit_violation(update, context, wait_time)
            return
        
        # Get user profile
        user_profile = await self.user_manager.get_user_profile(user.id)
        
        # Extract and analyze message
        text = message.text or message.caption or ""
        if not text and not message.forward_from:
            return
        
        # Check forwarding permissions
        if (message.forward_from or message.forward_from_chat) and user.id not in self.forward_whitelist:
            await self._handle_spam_detection(
                update, context, user, chat_id, message,
                reason="Unauthorized forwarding",
                severity="medium"
            )
            return
        
        # Analyze message
        analysis_result = await self.ai_detector.analyze_message(
            text, 
            user_profile=user_profile
        )
        
        # Record message
        await self.user_manager.record_message(
            user.id, chat_id, text,
            analysis_result['is_spam'],
            analysis_result['confidence'],
            analysis_result.get('reason')
        )
        
        # Handle detection result
        if analysis_result['is_spam']:
            await self._handle_spam_detection(
                update, context, user, chat_id, message,
                reason=analysis_result['reason'],
                severity=analysis_result.get('severity', 'medium'),
                confidence=analysis_result['confidence']
            )
        else:
            # Update positive reputation for clean messages
            if user_profile['total_messages'] % 50 == 0:  # Every 50 messages
                await self.user_manager.update_reputation(
                    user.id, 1, "Consistent clean messaging"
                )
        
        # Update metrics
        self.metrics['messages_processed'] += 1
    
    async def _handle_spam_detection(
        self, 
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        user: User,
        chat_id: int,
        message: Message,
        reason: str,
        severity: str = "medium",
        confidence: float = 1.0
    ):
        """Handle detected spam with appropriate actions"""
        try:
            # Delete the spam message
            await message.delete()
            
            # Update metrics
            self.metrics['spam_detected'] += 1
            
            # Update user reputation
            reputation_penalty = {
                'critical': -10,
                'high': -5,
                'medium': -2,
                'low': -1
            }.get(severity, -2)
            
            await self.user_manager.update_reputation(
                user.id, reputation_penalty, f"Spam detected: {reason}"
            )
            
            # Update warnings
            self.warnings[user.id] += 1
            warning_count = self.warnings[user.id]
            
            # Determine action based on severity and warnings
            if severity == 'critical' or warning_count >= self.config.max_warnings:
                # Ban user
                await context.bot.ban_chat_member(chat_id=chat_id, user_id=user.id)
                
                ban_message = (
                    f"⛔ {user.mention_html()} has been permanently banned.\n"
                    f"Reason: {reason}\n"
                    f"Severity: {severity.upper()}\n\n"
                    f"{self.config.promotion_text}"
                )
                
                sent_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    text=ban_message,
                    parse_mode=ParseMode.HTML
                )
                
                # Schedule deletion
                asyncio.create_task(self._delete_message_after_delay(
                    chat_id, sent_msg.message_id, context
                ))
                
                # Clear warnings
                del self.warnings[user.id]
                
                # Update metrics
                self.metrics['users_banned'] += 1
                
                self.logger.warning(f"User {user.id} banned for: {reason} (severity: {severity})")
                
            elif warning_count >= self.config.max_warnings - 2:
                # Final warning
                warning_msg = (
                    f"⚠️ FINAL WARNING {user.mention_html()}!\n"
                    f"Reason: {reason}\n"
                    f"Warnings: {warning_count}/{self.config.max_warnings}\n"
                    f"Next violation will result in a permanent ban.\n\n"
                    f"{self.config.promotion_text}"
                )
                
                sent_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    text=warning_msg,
                    parse_mode=ParseMode.HTML
                )
                
                asyncio.create_task(self._delete_message_after_delay(
                    chat_id, sent_msg.message_id, context, delay=15
                ))
                
            else:
                # Regular warning
                warning_msg = (
                    f"⚠️ {user.mention_html()}, your message was removed.\n"
                    f"Reason: {reason}\n"
                    f"Warning {warning_count}/{self.config.max_warnings}\n\n"
                    f"{self.config.promotion_text}"
                )
                
                sent_msg = await context.bot.send_message(
                    chat_id=chat_id,
                    text=warning_msg,
                    parse_mode=ParseMode.HTML
                )
                
                asyncio.create_task(self._delete_message_after_delay(
                    chat_id, sent_msg.message_id, context
                ))
                
            self.logger.info(f"Spam from {user.id}: {reason} (confidence: {confidence:.2f})")
            
        except TelegramError as e:
            self.logger.error(f"Telegram error handling spam: {e}")
        except Exception as e:
            self.logger.error(f"Error handling spam detection: {e}")
    
    async def _handle_rate_limit_violation(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        wait_time: float
    ):
        """Handle rate limit violations"""
        try:
            await update.message.delete()
            
            msg = await update.message.reply_text(
                f"⏳ Slow down! Please wait {wait_time:.1f} seconds before sending another message."
            )
            
            asyncio.create_task(self._delete_message_after_delay(
                update.effective_chat.id, msg.message_id, context, delay=5
            ))
            
        except Exception as e:
            self.logger.error(f"Error handling rate limit: {e}")
    
    async def _delete_message_after_delay(
        self,
        chat_id: int,
        message_id: int,
        context: ContextTypes.DEFAULT_TYPE,
        delay: int = 10
    ):
        """Delete message after delay with error handling"""
        await asyncio.sleep(delay)
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        except TelegramError as e:
            if "message to delete not found" not in str(e).lower():
                self.logger.debug(f"Could not delete message: {e}")
    
    # ============= Command Handlers =============
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command"""
        welcome_text = (
            "🛡️ **Guardian Bot v3.0 - Advanced Protection System**\n\n"
            "I'm an AI-powered bot that protects your group from:\n"
            "• Spam and scam messages\n"
            "• Inappropriate content\n"
            "• Suspicious links and phishing\n"
            "• Rate limit violations\n\n"
            "Features:\n"
            "✅ Multi-layer AI detection\n"
            "✅ Machine learning adaptation\n"
            "✅ User reputation system\n"
            "✅ Intelligent caching\n"
            "✅ Real-time pattern matching\n\n"
            "Use /help to see available commands."
        )
        await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced help command"""
        user_id = update.effective_user.id
        
        if user_id in self.config.admin_ids:
            help_text = (
                "📋 **Admin Commands:**\n\n"
                "**Protection:**\n"
                "/allowchat <id> - Allow bot in chat\n"
                "/allowthischat - Allow current chat\n"
                "/block <words> - Block words\n"
                "/unblock <words> - Unblock words\n"
                "/setmaxwarnings <n> - Set warning threshold\n\n"
                "**User Management:**\n"
                "/allowforward <user> - Allow forwarding\n"
                "/revokeforward <user> - Revoke forwarding\n"
                "/trust <user> - Mark user as trusted\n"
                "/untrust <user> - Remove trusted status\n"
                "/checkuser <user> - Check user profile\n\n"
                "**Analytics:**\n"
                "/stats - View statistics\n"
                "/metrics - Detailed metrics\n"
                "/cache - Cache statistics\n"
                "/mlstats - ML model stats\n\n"
                "**Configuration:**\n"
                "/export - Export configuration\n"
                "/import - Import configuration\n"
                "/reset - Reset to defaults\n"
            )
        else:
            help_text = (
                "📋 **User Commands:**\n\n"
                "/start - Welcome message\n"
                "/help - Show this help\n"
                "/report - Report spam (reply to message)\n"
            )
        
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced statistics command"""
        if update.effective_user.id not in self.config.admin_ids:
            await update.message.reply_text("❌ Admin only command")
            return
        
        uptime = datetime.now() - self.metrics['start_time']
        cache_stats = self.cache.get_stats()
        
        # Get database stats
        async with self.db.acquire() as conn:
            total_users = await conn.fetchval("SELECT COUNT(*) FROM user_profiles")
            spam_users = await conn.fetchval(
                "SELECT COUNT(*) FROM user_profiles WHERE spam_messages > 0"
            )
            trusted_users = await conn.fetchval(
                "SELECT COUNT(*) FROM user_profiles WHERE is_trusted = true"
            )
        
        stats_text = f"""
📊 **Guardian Bot Statistics**

**System:**
• Version: 3.0
• Uptime: {uptime}
• Memory Cache: {cache_stats['size']}/{cache_stats['max_size']}
• Cache Hit Rate: {cache_stats['hit_rate']}

**Protection:**
• Messages Processed: {self.metrics['messages_processed']:,}
• Spam Detected: {self.metrics['spam_detected']:,}
• Users Banned: {self.metrics['users_banned']:,}
• Detection Rate: {(self.metrics['spam_detected'] / max(1, self.metrics['messages_processed']) * 100):.2f}%

**Users:**
• Total Users: {total_users:,}
• Spam Users: {spam_users:,}
• Trusted Users: {trusted_users:,}
• Active Warnings: {len(self.warnings)}

**Configuration:**
• Blacklist Words: {len(self.blacklist)}
• Allowed Chats: {len(self.allowed_chats)}
• Max Warnings: {self.config.max_warnings}
• AI Threshold: {self.config.ai_confidence_threshold}

**AI Model:**
• ML Trained: {'Yes' if self.ai_detector.is_trained else 'No'}
• Pattern Rules: {sum(len(p) for p in self.ai_detector.compiled_patterns.values())}
        """
        
        await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)
    
    async def cmd_allowthischat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Allow current chat"""
        if update.effective_user.id not in self.config.admin_ids:
            await update.message.reply_text("❌ Admin only command")
            return
        
        chat_id = update.effective_chat.id
        chat_name = update.effective_chat.title or "Private Chat"
        
        async with self.db.acquire() as conn:
            await conn.execute("""
                INSERT INTO allowed_chats (chat_id, chat_name, added_by)
                VALUES ($1, $2, $3)
                ON CONFLICT (chat_id) DO UPDATE
                SET chat_name = $2
            """, chat_id, chat_name, update.effective_user.id)
        
        self.allowed_chats.add(chat_id)
        
        await update.message.reply_text(
            f"✅ Bot activated in this chat!\n"
            f"Chat ID: `{chat_id}`\n"
            f"Chat Name: {chat_name}",
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def cmd_block(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Block words with severity levels"""
        if update.effective_user.id not in self.config.admin_ids:
            await update.message.reply_text("❌ Admin only command")
            return
        
        if len(context.args) < 1:
            await update.message.reply_text(
                "Usage: /block <word1> <word2> ...\n"
                "Or: /block high <word> - for high severity"
            )
            return
        
        severity = 1
        words_to_add = context.args
        
        # Check for severity level
        if context.args[0] in ['low', 'medium', 'high', 'critical']:
            severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            severity = severity_map[context.args[0]]
            words_to_add = context.args[1:]
        
        if not words_to_add:
            await update.message.reply_text("Please provide words to block")
            return
        
        added_count = 0
        async with self.db.acquire() as conn:
            for word in words_to_add:
                word_lower = word.lower()
                try:
                    await conn.execute("""
                        INSERT INTO blacklist (word, severity, added_by)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (word) DO UPDATE
                        SET severity = $2
                    """, word_lower, severity, update.effective_user.id)
                    self.blacklist.add(word_lower)
                    added_count += 1
                except Exception as e:
                    self.logger.error(f"Error adding word {word}: {e}")
        
        severity_names = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}
        await update.message.reply_text(
            f"✅ Added {added_count} word(s) to blacklist\n"
            f"Severity: {severity_names[severity]}"
        )
    
    def run(self):
        """Run the bot with enhanced error handling"""
        # Start Flask in background
        flask_thread = threading.Thread(
            target=lambda: self.flask_app.run(
                host='0.0.0.0', 
                port=self.config.port,
                debug=False
            ),
            daemon=True
        )
        flask_thread.start()
        
        # Setup Telegram bot
        application = Application.builder().token(self.config.telegram_token).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", self.cmd_start))
        application.add_handler(CommandHandler("help", self.cmd_help))
        application.add_handler(CommandHandler("stats", self.cmd_stats))
        application.add_handler(CommandHandler("allowthischat", self.cmd_allowthischat))
        application.add_handler(CommandHandler("block", self.cmd_block))
        
        # Add message handler
        application.add_handler(MessageHandler(
            filters.ALL & ~filters.COMMAND,
            self.handle_message
        ))
        
        # Initialize bot components
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.initialize())
        
        # Start polling
        self.logger.info("🛡️ Guardian Bot v3.0 is now running!")
        application.run_polling(
            timeout=30,
            pool_timeout=30,
            read_timeout=20,
            write_timeout=20,
            connect_timeout=20,
            allowed_updates=Update.ALL_TYPES
        )


# ============= Entry Point =============
def main():
    """Main entry point"""
    try:
        bot = GuardianBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Guardian Bot shutting down gracefully...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
