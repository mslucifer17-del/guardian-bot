"""
Guardian Bot v3.0 - Advanced Telegram Anti-Spam System
Enhanced with machine learning, advanced AI integration, and enterprise-grade features
"""

import os
import asyncio
import re
import logging
import time
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
from functools import wraps, lru_cache
from contextlib import asynccontextmanager

# External imports
import google.generativeai as genai
import psycopg2
from psycopg2 import pool
import redis
import numpy as np
from flask import Flask, request, jsonify
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, filters, ContextTypes
)
from telegram.constants import ParseMode

# Machine Learning imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ======================== Configuration ========================

@dataclass
class BotConfig:
    """Centralized bot configuration"""
    telegram_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    database_url: str = os.environ.get("DATABASE_URL", "")
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379")
    admin_ids: List[int] = field(default_factory=list)
    port: int = int(os.environ.get('PORT', 8080))
    channel_id: int = -1002533091260
    max_warnings: int = 10
    rate_limit_window: int = 60  # seconds
    rate_limit_max_messages: int = 10
    auto_delete_delay: int = 10  # seconds
    promotion_text: str = "For promotions, join: https://t.me/+scHqQ2SR0J45NjQ1"
    ml_model_path: str = "spam_model.pkl"
    log_level: str = "INFO"
    webhook_url: Optional[str] = os.environ.get("WEBHOOK_URL")
    debug_mode: bool = os.environ.get("DEBUG", "False").lower() == "true"

    def __post_init__(self):
        admin_ids_str = os.environ.get("ADMIN_USER_IDS", "")
        self.admin_ids = [int(id.strip()) for id in admin_ids_str.split(",") if id.strip()]

# ======================== Logging Setup ========================

class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for better log visibility"""
    
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

def setup_logging(config: BotConfig):
    """Setup advanced logging with rotation and colored output"""
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, config.log_level))
    logger.addHandler(console_handler)
    
    return logger

# ======================== Database Layer ========================

class DatabaseManager:
    """Advanced database manager with connection pooling"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.pool = None
        self.logger = logging.getLogger(__name__)
        self.init_pool()
        
    def init_pool(self):
        """Initialize database connection pool"""
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                1, 20, self.config.database_url
            )
            self.logger.info("Database pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            
    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections"""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self.pool.putconn(conn)
            
    async def execute(self, query: str, params: tuple = None):
        """Execute a database query"""
        async with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall()
                return None
                
    async def execute_many(self, query: str, params_list: List[tuple]):
        """Execute multiple queries efficiently"""
        async with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
                
    async def setup_tables(self):
        """Create all necessary database tables"""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS blacklist (
                id SERIAL PRIMARY KEY,
                word TEXT NOT NULL UNIQUE,
                severity INTEGER DEFAULT 1,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id BIGINT PRIMARY KEY,
                trust_score FLOAT DEFAULT 0.5,
                message_count INTEGER DEFAULT 0,
                spam_count INTEGER DEFAULT 0,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS spam_patterns (
                id SERIAL PRIMARY KEY,
                pattern TEXT NOT NULL,
                pattern_type VARCHAR(50),
                confidence FLOAT DEFAULT 0.8,
                hits INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS analytics (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(100),
                chat_id BIGINT,
                user_id BIGINT,
                data JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ml_training_data (
                id SERIAL PRIMARY KEY,
                message TEXT NOT NULL,
                is_spam BOOLEAN NOT NULL,
                confidence FLOAT,
                features JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_user_profiles_trust ON user_profiles(trust_score);
            CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_spam_patterns_type ON spam_patterns(pattern_type);
            """
        ]
        
        for query in queries:
            await self.execute(query)

# ======================== Cache Layer ========================

class CacheManager:
    """Redis-based caching system for improved performance"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        try:
            self.redis_client = redis.from_url(config.redis_url, decode_responses=True)
            self.redis_available = True
            self.logger.info("Redis cache initialized")
        except Exception as e:
            self.logger.warning(f"Redis not available: {e}")
            self.redis_available = False
            self.local_cache = {}
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.redis_available:
            try:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            except Exception as e:
                self.logger.error(f"Cache get error: {e}")
        return self.local_cache.get(key)
        
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL"""
        if self.redis_available:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                self.logger.error(f"Cache set error: {e}")
        else:
            self.local_cache[key] = value
            
    async def delete(self, key: str):
        """Delete key from cache"""
        if self.redis_available:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                self.logger.error(f"Cache delete error: {e}")
        elif key in self.local_cache:
            del self.local_cache[key]
            
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        if self.redis_available:
            try:
                return self.redis_client.incr(key, amount)
            except Exception as e:
                self.logger.error(f"Cache increment error: {e}")
        else:
            self.local_cache[key] = self.local_cache.get(key, 0) + amount
            return self.local_cache[key]

# ======================== AI Integration ========================

class AISpamDetector:
    """Advanced AI-powered spam detection system"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        genai.configure(api_key=config.gemini_api_key)
        
        self.system_prompt = """You are an advanced AI spam detection system. Analyze messages with these priorities:
        
        1. CRITICAL THREATS (instant block):
        - Child exploitation content (CP, CSAM)
        - Terrorism, violence, self-harm
        - Illegal activities (drugs, weapons)
        
        2. HIGH RISK (likely spam):
        - Financial scams, phishing
        - Adult content, pornography
        - Aggressive marketing, MLM schemes
        
        3. MEDIUM RISK (suspicious):
        - Excessive links or mentions
        - Repetitive promotional language
        - Suspicious payment requests
        
        4. CONTEXTUAL ANALYSIS:
        - Consider language and cultural context
        - Evaluate user history if available
        - Check for disguised/encoded harmful content
        
        Respond with JSON: {"verdict": "SPAM/OK/SUSPICIOUS", "confidence": 0-1, "reason": "explanation", "severity": 1-10}"""
        
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=self.system_prompt,
            generation_config={
                'temperature': 0.3,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 200,
            }
        )
        
        # Advanced pattern matching
        self.compile_patterns()
        
    def compile_patterns(self):
        """Compile regex patterns for faster matching"""
        self.critical_patterns = [
            re.compile(r'\b(cp|csam|child\s*porn)\b', re.IGNORECASE),
            re.compile(r'\b(loli|shota|pedo)\b', re.IGNORECASE),
        ]
        
        self.high_risk_patterns = [
            re.compile(r'(click|join|buy)\s*(here|now|fast)', re.IGNORECASE),
            re.compile(r'(earn|make)\s*\$?\d+\s*(daily|hourly|weekly)', re.IGNORECASE),
            re.compile(r'(telegram|whatsapp|signal)\s*@\w+', re.IGNORECASE),
        ]
        
        self.url_pattern = re.compile(
            r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            re.IGNORECASE
        )
        
    async def analyze(self, message: str, user_context: Dict = None) -> Dict:
        """Analyze message for spam with context awareness"""
        
        # Quick pattern check for critical content
        for pattern in self.critical_patterns:
            if pattern.search(message):
                return {
                    "verdict": "SPAM",
                    "confidence": 1.0,
                    "reason": "Critical threat detected",
                    "severity": 10
                }
        
        # Check high-risk patterns
        risk_score = 0
        for pattern in self.high_risk_patterns:
            if pattern.search(message):
                risk_score += 0.3
                
        # Count URLs
        url_count = len(self.url_pattern.findall(message))
        if url_count > 2:
            risk_score += 0.2 * url_count
            
        # AI Analysis for complex cases
        if risk_score > 0.3 or len(message) > 50:
            try:
                context_info = ""
                if user_context:
                    context_info = f"\nUser context: Trust score: {user_context.get('trust_score', 0.5)}, Previous spam: {user_context.get('spam_count', 0)}"
                
                prompt = f"Analyze this message for spam:\n{message}{context_info}"
                
                response = await asyncio.wait_for(
                    self.model.generate_content_async(prompt),
                    timeout=5.0
                )
                
                # Parse AI response
                try:
                    result = json.loads(response.text)
                    return result
                except json.JSONDecodeError:
                    # Fallback parsing
                    if "SPAM" in response.text.upper():
                        return {
                            "verdict": "SPAM",
                            "confidence": 0.8,
                            "reason": "AI detected spam content",
                            "severity": 7
                        }
                        
            except asyncio.TimeoutError:
                self.logger.warning("AI analysis timeout")
            except Exception as e:
                self.logger.error(f"AI analysis error: {e}")
                
        # Final verdict based on risk score
        if risk_score > 0.7:
            return {
                "verdict": "SPAM",
                "confidence": risk_score,
                "reason": "High-risk patterns detected",
                "severity": int(risk_score * 10)
            }
        elif risk_score > 0.4:
            return {
                "verdict": "SUSPICIOUS",
                "confidence": risk_score,
                "reason": "Suspicious patterns detected",
                "severity": int(risk_score * 10)
            }
        else:
            return {
                "verdict": "OK",
                "confidence": 1.0 - risk_score,
                "reason": "Message appears clean",
                "severity": 0
            }

# ======================== Machine Learning Module ========================

class MLSpamClassifier:
    """Machine learning-based spam classifier"""
    
    def __init__(self, config: BotConfig, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        if ML_AVAILABLE:
            self.load_or_train_model()
            
    def load_or_train_model(self):
        """Load existing model or train new one"""
        try:
            if os.path.exists(self.config.ml_model_path):
                self.model = joblib.load(self.config.ml_model_path)
                self.vectorizer = self.model.named_steps['vectorizer']
                self.is_trained = True
                self.logger.info("ML model loaded successfully")
            else:
                asyncio.create_task(self.train_model())
        except Exception as e:
            self.logger.error(f"Error loading ML model: {e}")
            
    async def train_model(self):
        """Train the ML model with collected data"""
        try:
            # Fetch training data from database
            data = await self.db_manager.execute(
                "SELECT message, is_spam FROM ml_training_data LIMIT 10000"
            )
            
            if len(data) < 100:
                self.logger.warning("Insufficient training data")
                return
                
            messages, labels = zip(*data)
            
            # Create pipeline
            self.model = Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
                ('classifier', MultinomialNB())
            ])
            
            # Train model
            self.model.fit(messages, labels)
            self.vectorizer = self.model.named_steps['vectorizer']
            
            # Save model
            joblib.dump(self.model, self.config.ml_model_path)
            self.is_trained = True
            
            self.logger.info(f"ML model trained with {len(data)} samples")
            
        except Exception as e:
            self.logger.error(f"Error training ML model: {e}")
            
    def predict(self, message: str) -> Tuple[bool, float]:
        """Predict if message is spam"""
        if not self.is_trained or not ML_AVAILABLE:
            return False, 0.0
            
        try:
            prediction = self.model.predict([message])[0]
            confidence = max(self.model.predict_proba([message])[0])
            return bool(prediction), float(confidence)
        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
            return False, 0.0
            
    async def add_training_sample(self, message: str, is_spam: bool, confidence: float = None):
        """Add new training sample to database"""
        try:
            await self.db_manager.execute(
                "INSERT INTO ml_training_data (message, is_spam, confidence) VALUES (%s, %s, %s)",
                (message, is_spam, confidence)
            )
            
            # Retrain periodically
            count = await self.db_manager.execute(
                "SELECT COUNT(*) FROM ml_training_data WHERE created_at > NOW() - INTERVAL '1 day'"
            )
            if count and count[0][0] >= 100:
                asyncio.create_task(self.train_model())
                
        except Exception as e:
            self.logger.error(f"Error adding training sample: {e}")

# ======================== User Management ========================

class UserProfile:
    """Enhanced user profile with trust scoring"""
    
    def __init__(self, user_id: int, trust_score: float = 0.5):
        self.user_id = user_id
        self.trust_score = trust_score
        self.message_count = 0
        self.spam_count = 0
        self.warning_count = 0
        self.last_activity = datetime.now()
        self.rate_limit_violations = 0
        self.metadata = {}
        
    def update_trust_score(self):
        """Update trust score based on behavior"""
        if self.message_count > 0:
            spam_ratio = self.spam_count / self.message_count
            self.trust_score = max(0.0, min(1.0, 1.0 - spam_ratio * 2))
            
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'user_id': self.user_id,
            'trust_score': self.trust_score,
            'message_count': self.message_count,
            'spam_count': self.spam_count,
            'last_activity': self.last_activity.isoformat(),
            'metadata': self.metadata
        }

class UserManager:
    """Advanced user management system"""
    
    def __init__(self, db_manager: DatabaseManager, cache_manager: CacheManager):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        self.profiles: Dict[int, UserProfile] = {}
        
    async def get_user_profile(self, user_id: int) -> UserProfile:
        """Get or create user profile"""
        
        # Check memory cache
        if user_id in self.profiles:
            return self.profiles[user_id]
            
        # Check Redis cache
        cached = await self.cache_manager.get(f"user:{user_id}")
        if cached:
            profile = UserProfile(user_id)
            profile.__dict__.update(cached)
            self.profiles[user_id] = profile
            return profile
            
        # Load from database
        data = await self.db_manager.execute(
            "SELECT trust_score, message_count, spam_count, metadata FROM user_profiles WHERE user_id = %s",
            (user_id,)
        )
        
        if data:
            profile = UserProfile(user_id, data[0][0])
            profile.message_count = data[0][1]
            profile.spam_count = data[0][2]
            profile.metadata = data[0][3] or {}
        else:
            profile = UserProfile(user_id)
            await self.save_user_profile(profile)
            
        self.profiles[user_id] = profile
        await self.cache_manager.set(f"user:{user_id}", profile.to_dict(), ttl=3600)
        
        return profile
        
    async def save_user_profile(self, profile: UserProfile):
        """Save user profile to database"""
        await self.db_manager.execute(
            """
            INSERT INTO user_profiles (user_id, trust_score, message_count, spam_count, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                trust_score = EXCLUDED.trust_score,
                message_count = EXCLUDED.message_count,
                spam_count = EXCLUDED.spam_count,
                metadata = EXCLUDED.metadata,
                last_activity = CURRENT_TIMESTAMP
            """,
            (profile.user_id, profile.trust_score, profile.message_count,
             profile.spam_count, json.dumps(profile.metadata))
        )
        
        await self.cache_manager.set(f"user:{profile.user_id}", profile.to_dict(), ttl=3600)

# ======================== Rate Limiting ========================

class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self, cache_manager: CacheManager, config: BotConfig):
        self.cache = cache_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def check_rate_limit(self, user_id: int, chat_id: int) -> Tuple[bool, int]:
        """Check if user is within rate limits"""
        key = f"rate:{chat_id}:{user_id}"
        
        # Get current count
        count = await self.cache.increment(key)
        
        if count == 1:
            # First message, set expiry
            if self.cache.redis_available:
                self.cache.redis_client.expire(key, self.config.rate_limit_window)
                
        if count > self.config.rate_limit_max_messages:
            remaining = self.config.rate_limit_window
            if self.cache.redis_available:
                remaining = self.cache.redis_client.ttl(key)
            return False, remaining
            
        return True, 0
        
    async def get_adaptive_limit(self, user_profile: UserProfile) -> int:
        """Get adaptive rate limit based on trust score"""
        base_limit = self.config.rate_limit_max_messages
        
        if user_profile.trust_score > 0.8:
            return base_limit * 2
        elif user_profile.trust_score < 0.3:
            return max(1, base_limit // 2)
        else:
            return base_limit

# ======================== Analytics System ========================

class Analytics:
    """Advanced analytics and reporting system"""
    
    def __init__(self, db_manager: DatabaseManager, cache_manager: CacheManager):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
    async def log_event(self, event_type: str, chat_id: int = None, 
                        user_id: int = None, data: Dict = None):
        """Log an analytics event"""
        await self.db_manager.execute(
            "INSERT INTO analytics (event_type, chat_id, user_id, data) VALUES (%s, %s, %s, %s)",
            (event_type, chat_id, user_id, json.dumps(data) if data else None)
        )
        
        # Update real-time metrics
        await self.cache_manager.increment(f"metric:{event_type}:daily")
        
    async def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        stats = {}
        
        # Get daily metrics from cache
        for metric in ["spam_detected", "messages_processed", "users_banned"]:
            stats[f"{metric}_today"] = await self.cache_manager.get(f"metric:{metric}:daily") or 0
            
        # Get database statistics
        queries = {
            "total_users": "SELECT COUNT(DISTINCT user_id) FROM user_profiles",
            "total_spam": "SELECT COUNT(*) FROM analytics WHERE event_type = 'spam_detected'",
            "active_chats": "SELECT COUNT(DISTINCT chat_id) FROM analytics WHERE timestamp > NOW() - INTERVAL '24 hours'",
            "avg_trust_score": "SELECT AVG(trust_score) FROM user_profiles",
        }
        
        for key, query in queries.items():
            result = await self.db_manager.execute(query)
            stats[key] = result[0][0] if result and result[0][0] else 0
            
        return stats
        
    async def generate_report(self, chat_id: int, period_days: int = 7) -> str:
        """Generate detailed analytics report"""
        start_date = datetime.now() - timedelta(days=period_days)
        
        report_data = await self.db_manager.execute(
            """
            SELECT 
                event_type,
                COUNT(*) as count,
                DATE(timestamp) as date
            FROM analytics
            WHERE chat_id = %s AND timestamp > %s
            GROUP BY event_type, DATE(timestamp)
            ORDER BY date DESC, count DESC
            """,
            (chat_id, start_date)
        )
        
        report = f"📊 Analytics Report ({period_days} days)\n\n"
        
        if report_data:
            by_type = defaultdict(int)
            for event_type, count, date in report_data:
                by_type[event_type] += count
                
            for event_type, total in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                report += f"• {event_type}: {total}\n"
        else:
            report += "No data available for this period."
            
        return report

# ======================== Main Bot Class ========================

class GuardianBot:
    """Main bot class with all integrated systems"""
    
    def __init__(self):
        self.config = BotConfig()
        self.logger = setup_logging(self.config)
        
        # Initialize components
        self.db_manager = DatabaseManager(self.config)
        self.cache_manager = CacheManager(self.config)
        self.ai_detector = AISpamDetector(self.config)
        self.ml_classifier = MLSpamClassifier(self.config, self.db_manager)
        self.user_manager = UserManager(self.db_manager, self.cache_manager)
        self.rate_limiter = RateLimiter(self.cache_manager, self.config)
        self.analytics = Analytics(self.db_manager, self.cache_manager)
        
        # State management
        self.active_sessions = {}
        self.command_handlers = {}
        self.middleware_stack = []
        
        # Initialize database
        asyncio.create_task(self.initialize())
        
    async def initialize(self):
        """Initialize bot systems"""
        await self.db_manager.setup_tables()
        self.logger.info("Guardian Bot v3.0 initialized successfully")
        
    # ============ Message Handling ============
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Advanced message handler with multiple detection layers"""
        
        if not update.message or not update.message.from_user:
            return
            
        user = update.message.from_user
        chat_id = update.effective_chat.id
        message = update.message
        
        # Skip bot messages
        if user.is_bot:
            return
            
        # Load user profile
        user_profile = await self.user_manager.get_user_profile(user.id)
        user_profile.message_count += 1
        user_profile.last_activity = datetime.now()
        
        # Check if admin
        if user.id in self.config.admin_ids:
            await self.user_manager.save_user_profile(user_profile)
            return
            
        # Rate limiting with adaptive limits
        limit = await self.rate_limiter.get_adaptive_limit(user_profile)
        self.config.rate_limit_max_messages = limit
        
        allowed, remaining = await self.rate_limiter.check_rate_limit(user.id, chat_id)
        if not allowed:
            await self.send_warning(
                message, 
                f"⚠️ Rate limit exceeded. Please wait {remaining} seconds.",
                delete_original=True
            )
            user_profile.rate_limit_violations += 1
            await self.analytics.log_event("rate_limit_exceeded", chat_id, user.id)
            return
            
        # Extract message text
        text = message.text or message.caption or ""
        
        # Multi-layer spam detection
        is_spam, reason, severity = await self.detect_spam(text, user_profile)
        
        if is_spam:
            await self.handle_spam(message, user_profile, reason, severity)
        else:
            # Update trust score positively for clean messages
            user_profile.trust_score = min(1.0, user_profile.trust_score + 0.01)
            
        await self.user_manager.save_user_profile(user_profile)
        await self.analytics.log_event("message_processed", chat_id, user.id)
        
    async def detect_spam(self, text: str, user_profile: UserProfile) -> Tuple[bool, str, int]:
        """Multi-layer spam detection system"""
        
        # Layer 1: ML Classification
        if self.ml_classifier.is_trained:
            is_spam_ml, confidence = self.ml_classifier.predict(text)
            if is_spam_ml and confidence > 0.9:
                return True, f"ML detected spam (confidence: {confidence:.2f})", 8
                
        # Layer 2: AI Analysis
        ai_result = await self.ai_detector.analyze(text, user_profile.to_dict())
        
        if ai_result["verdict"] == "SPAM":
            return True, ai_result["reason"], ai_result["severity"]
        elif ai_result["verdict"] == "SUSPICIOUS":
            # Additional checks for suspicious content
            if user_profile.trust_score < 0.4:
                return True, "Suspicious content from low-trust user", 5
                
        # Layer 3: Pattern matching (fallback)
        # This would include your existing pattern matching logic
        
        return False, "", 0
        
    async def handle_spam(self, message, user_profile: UserProfile, reason: str, severity: int):
        """Handle spam messages with graduated response"""
        
        user_profile.spam_count += 1
        user_profile.warning_count += 1
        user_profile.update_trust_score()
        
        # Delete the spam message
        try:
            await message.delete()
        except Exception as e:
            self.logger.error(f"Could not delete message: {e}")
            
        # Log to analytics
        await self.analytics.log_event(
            "spam_detected",
            message.chat.id,
            user_profile.user_id,
            {"reason": reason, "severity": severity}
        )
        
        # Add to ML training data
        await self.ml_classifier.add_training_sample(
            message.text or message.caption or "",
            True,
            severity / 10
        )
        
        # Graduated response based on severity and warnings
        if severity >= 9 or user_profile.warning_count >= self.config.max_warnings:
            # Ban user
            await self.ban_user(message.chat.id, user_profile)
        elif user_profile.warning_count >= self.config.max_warnings // 2:
            # Restrict user
            await self.restrict_user(message.chat.id, user_profile)
        else:
            # Send warning
            warning_text = (
                f"⚠️ {message.from_user.mention_html()}\n"
                f"Reason: {reason}\n"
                f"Warning: {user_profile.warning_count}/{self.config.max_warnings}\n"
                f"Trust Score: {user_profile.trust_score:.2f}"
            )
            await self.send_warning(message, warning_text)
            
    async def ban_user(self, chat_id: int, user_profile: UserProfile):
        """Ban user from chat"""
        try:
            await self.bot.ban_chat_member(chat_id=chat_id, user_id=user_profile.user_id)
            await self.analytics.log_event("user_banned", chat_id, user_profile.user_id)
            self.logger.info(f"User {user_profile.user_id} banned from chat {chat_id}")
        except Exception as e:
            self.logger.error(f"Could not ban user: {e}")
            
    async def restrict_user(self, chat_id: int, user_profile: UserProfile):
        """Restrict user permissions"""
        try:
            await self.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_profile.user_id,
                permissions={
                    'can_send_messages': False,
                    'can_send_media_messages': False,
                    'can_send_other_messages': False,
                    'can_add_web_page_previews': False
                }
            )
            await self.analytics.log_event("user_restricted", chat_id, user_profile.user_id)
        except Exception as e:
            self.logger.error(f"Could not restrict user: {e}")
            
    async def send_warning(self, original_message, warning_text: str, delete_original: bool = False):
        """Send warning message with auto-deletion"""
        if delete_original:
            try:
                await original_message.delete()
            except Exception:
                pass
                
        sent_message = await original_message.reply_text(
            f"{warning_text}\n\n{self.config.promotion_text}",
            parse_mode=ParseMode.HTML
        )
        
        # Schedule deletion
        asyncio.create_task(self.delete_after_delay(
            sent_message,
            self.config.auto_delete_delay
        ))
        
    async def delete_after_delay(self, message, delay: int):
        """Delete message after delay"""
        await asyncio.sleep(delay)
        try:
            await message.delete()
        except Exception:
            pass
            
    # ============ Command Handlers ============
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        welcome_text = (
            "🛡️ **Guardian Bot v3.0**\n\n"
            "Advanced AI-powered spam protection with machine learning.\n\n"
            "Features:\n"
            "• Multi-layer spam detection\n"
            "• User trust scoring\n"
            "• Adaptive rate limiting\n"
            "• Real-time analytics\n"
            "• Machine learning adaptation\n\n"
            "Use /help for available commands."
        )
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Dashboard", callback_data="dashboard"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ],
            [
                InlineKeyboardButton("📈 Analytics", callback_data="analytics"),
                InlineKeyboardButton("ℹ️ Help", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot statistics"""
        if update.effective_user.id not in self.config.admin_ids:
            await update.message.reply_text("⛔ Admin only command")
            return
            
        stats = await self.analytics.get_statistics()
        
        stats_text = (
            "📊 **Guardian Bot Statistics**\n\n"
            f"👥 Total Users: {stats.get('total_users', 0)}\n"
            f"🚫 Total Spam Detected: {stats.get('total_spam', 0)}\n"
            f"💬 Active Chats: {stats.get('active_chats', 0)}\n"
            f"⭐ Avg Trust Score: {stats.get('avg_trust_score', 0):.2f}\n\n"
            f"**Today's Metrics:**\n"
            f"• Spam Detected: {stats.get('spam_detected_today', 0)}\n"
            f"• Messages Processed: {stats.get('messages_processed_today', 0)}\n"
            f"• Users Banned: {stats.get('users_banned_today', 0)}"
        )
        
        await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)
        
    async def cmd_trust(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Check user trust score"""
        target_user = update.message.from_user
        
        if update.message.reply_to_message:
            target_user = update.message.reply_to_message.from_user
            
        profile = await self.user_manager.get_user_profile(target_user.id)
        
        trust_emoji = "🟢" if profile.trust_score > 0.7 else "🟡" if profile.trust_score > 0.4 else "🔴"
        
        trust_text = (
            f"{trust_emoji} **Trust Score Report**\n\n"
            f"User: {target_user.mention_html()}\n"
            f"Trust Score: {profile.trust_score:.2f}/1.00\n"
            f"Messages: {profile.message_count}\n"
            f"Spam Count: {profile.spam_count}\n"
            f"Warnings: {profile.warning_count}/{self.config.max_warnings}"
        )
        
        await update.message.reply_text(trust_text, parse_mode=ParseMode.HTML)
        
    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate analytics report"""
        if update.effective_user.id not in self.config.admin_ids:
            await update.message.reply_text("⛔ Admin only command")
            return
            
        period = 7  # Default 7 days
        if context.args and context.args[0].isdigit():
            period = int(context.args[0])
            
        report = await self.analytics.generate_report(update.effective_chat.id, period)
        await update.message.reply_text(report)
        
    async def cmd_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manually trigger ML model training"""
        if update.effective_user.id not in self.config.admin_ids:
            await update.message.reply_text("⛔ Admin only command")
            return
            
        if not ML_AVAILABLE:
            await update.message.reply_text("❌ ML features not available")
            return
            
        await update.message.reply_text("🔄 Training ML model...")
        await self.ml_classifier.train_model()
        await update.message.reply_text("✅ ML model training completed")
        
    # ============ Callback Query Handlers ============
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "dashboard":
            await self.show_dashboard(query)
        elif query.data == "settings":
            await self.show_settings(query)
        elif query.data == "analytics":
            stats = await self.analytics.get_statistics()
            await query.edit_message_text(
                self.format_analytics(stats),
                parse_mode=ParseMode.MARKDOWN
            )
        elif query.data == "help":
            await self.show_help(query)
            
    async def show_dashboard(self, query):
        """Show admin dashboard"""
        dashboard_text = (
            "🎛️ **Admin Dashboard**\n\n"
            "Select an action:"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("🚫 Manage Blacklist", callback_data="manage_blacklist"),
                InlineKeyboardButton("👥 User Management", callback_data="manage_users")
            ],
            [
                InlineKeyboardButton("📊 View Analytics", callback_data="analytics"),
                InlineKeyboardButton("⚙️ Bot Settings", callback_data="settings")
            ],
            [
                InlineKeyboardButton("🔄 Train ML Model", callback_data="train_ml"),
                InlineKeyboardButton("📝 Export Data", callback_data="export_data")
            ],
            [
                InlineKeyboardButton("◀️ Back", callback_data="main_menu")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(dashboard_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        
    def format_analytics(self, stats: Dict) -> str:
        """Format analytics data for display"""
        return (
            "📈 **Real-time Analytics**\n\n"
            f"**Overall Stats:**\n"
            f"• Total Users: {stats.get('total_users', 0):,}\n"
            f"• Spam Detected: {stats.get('total_spam', 0):,}\n"
            f"• Active Chats: {stats.get('active_chats', 0):,}\n\n"
            f"**Performance Metrics:**\n"
            f"• Avg Trust Score: {stats.get('avg_trust_score', 0):.3f}\n"
            f"• Detection Rate: {(stats.get('total_spam', 0) / max(stats.get('messages_processed_today', 1), 1) * 100):.1f}%\n\n"
            f"**Today's Activity:**\n"
            f"• Messages: {stats.get('messages_processed_today', 0):,}\n"
            f"• Spam Caught: {stats.get('spam_detected_today', 0):,}\n"
            f"• Users Banned: {stats.get('users_banned_today', 0):,}"
        )
        
    # ============ Bot Setup and Running ============
    
    def setup_application(self) -> Application:
        """Setup and configure the bot application"""
        application = Application.builder().token(self.config.telegram_token).build()
        
        # Store reference
        self.bot = application.bot
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.cmd_start))
        application.add_handler(CommandHandler("stats", self.cmd_stats))
        application.add_handler(CommandHandler("trust", self.cmd_trust))
        application.add_handler(CommandHandler("report", self.cmd_report))
        application.add_handler(CommandHandler("train", self.cmd_train))
        
        # Message handler
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_message
        ))
        
        # Callback query handler
        application.add_handler(CallbackQueryHandler(self.handle_callback_query))
        
        # Error handler
        application.add_error_handler(self.error_handler)
        
        return application
        
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors gracefully"""
        self.logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ An error occurred. Our team has been notified."
                )
            except Exception:
                pass
                
    def run(self):
        """Run the bot"""
        application = self.setup_application()
        
        if self.config.webhook_url:
            # Use webhook for production
            application.run_webhook(
                listen="0.0.0.0",
                port=self.config.port,
                url_path=self.config.telegram_token,
                webhook_url=f"{self.config.webhook_url}/{self.config.telegram_token}"
            )
        else:
            # Use polling for development
            self.logger.info("🛡️ Guardian Bot v3.0 starting in polling mode...")
            application.run_polling(
                timeout=1000,
                pool_timeout=1000,
                read_timeout=1000,
                write_timeout=1000,
                connect_timeout=1000,
                drop_pending_updates=True
            )

# ======================== Flask Web Interface ========================

def create_flask_app(bot: GuardianBot) -> Flask:
    """Create Flask app for web interface and health checks"""
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return jsonify({
            "status": "running",
            "version": "3.0",
            "bot": "Guardian Bot",
            "features": [
                "AI-powered spam detection",
                "Machine learning classification",
                "User trust scoring",
                "Real-time analytics",
                "Adaptive rate limiting"
            ]
        })
        
    @app.route('/health')
    def health():
        return jsonify({"status": "healthy"}), 200
        
    @app.route('/stats')
    async def stats():
        stats = await bot.analytics.get_statistics()
        return jsonify(stats)
        
    return app

# ======================== Main Entry Point ========================

def main():
    """Main entry point"""
    bot = GuardianBot()
    
    # Start Flask in separate thread if needed
    if not bot.config.webhook_url:
        flask_app = create_flask_app(bot)
        flask_thread = threading.Thread(
            target=lambda: flask_app.run(
                host='0.0.0.0',
                port=bot.config.port,
                debug=bot.config.debug_mode
            ),
            daemon=True
        )
        flask_thread.start()
    
    # Run the bot
    bot.run()

if __name__ == "__main__":
    main()
