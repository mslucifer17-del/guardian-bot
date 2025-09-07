import os
import threading
import asyncio
import re
import logging
import time
from flask import Flask
import google.generativeai as genai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from collections import defaultdict
import psycopg2
from datetime import datetime

# Configuration
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ADMIN_USER_ID = int(os.environ.get("ADMIN_USER_ID", "0").strip())
DATABASE_URL = os.environ.get("DATABASE_URL")
PORT = int(os.environ.get('PORT', 8080))
# Add your channel ID here
CHANNEL_ID = -1002533091260  # Replace with your actual channel ID

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# AI Setup for Spam Detection
SPAM_DETECTION_PROMPT = """You are a vigilant spam detection AI for Telegram. Analyze messages and:
1. Reply with "SPAM" if it contains promotions, scams, ads, or suspicious links
2. Reply with "OK" for normal messages
3. Consider cultural context and multiple languages
4. Pay special attention to CP (child porn), adult content, selling, and illegal activities"""
genai.configure(api_key=GEMINI_API_KEY)
spam_model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=SPAM_DETECTION_PROMPT)

# In-memory stores
user_warnings = defaultdict(int)
user_last_message = defaultdict(datetime)
blacklist_words = set()
allowed_chats = set()
forward_whitelist_users = set()
message_rate_limit = {}

# Advanced spam patterns
spam_patterns = [
    r"lowest price", r"premium collection", r"dm for purchase", r"payment method",
    r"combo no\d", r"latest updated", r"desi cp", r"indian cp", r"foreign cp",
    r"tamil cp", r"chinese cp", r"arabians cp", r"bro-sis cp", r"dad-daughter cp",
    r"pedo mom-son cp", r"premium cp", r"content collection", r"price available",
    r"payment method", r"upi|paypal|crypto|gift card"
]
payment_terms = ["upi", "paypal", "crypto", "gift card", "payment", "purchase", "price", "💰", "📣", "🟢"]

# Auto-delete functionality for bot messages
async def delete_message_after_delay(chat_id: int, message_id: int, context: ContextTypes.DEFAULT_TYPE, delay: int = 10):
    """Delete a message after a specified delay"""
    await asyncio.sleep(delay)
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception as e:
        # Message might have already been deleted or bot doesn't have permissions
        if "message to delete not found" not in str(e).lower():
            logger.error(f"Could not delete message {message_id} in chat {chat_id}: {e}")

async def send_auto_delete_message(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str, delay: int = 10):
    """Send a message that will be automatically deleted after delay"""
    message = await update.message.reply_text(text)
    # Schedule this message for deletion
    asyncio.create_task(delete_message_after_delay(update.effective_chat.id, message.message_id, context, delay))
    return message

# Database Functions
def db_connect():
    try:
        return psycopg2.connect(DATABASE_URL, connect_timeout=5)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def setup_database():
    conn = db_connect()
    if not conn:
        logger.error("Failed to connect to database during setup")
        return
        
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS blacklist (id SERIAL PRIMARY KEY, word TEXT NOT NULL UNIQUE, added_by BIGINT, added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        cur.execute("CREATE TABLE IF NOT EXISTS allowed_chats (id SERIAL PRIMARY KEY, chat_id BIGINT NOT NULL UNIQUE, added_by BIGINT, added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        cur.execute("CREATE TABLE IF NOT EXISTS custom_commands (id SERIAL PRIMARY KEY, command TEXT NOT NULL UNIQUE, response TEXT NOT NULL, added_by BIGINT, added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        cur.execute("CREATE TABLE IF NOT EXISTS reported_spam (id SERIAL PRIMARY KEY, message TEXT NOT NULL, reported_by BIGINT, reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forward_whitelist (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL UNIQUE,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Add table for allowed channels
        cur.execute("""
            CREATE TABLE IF NOT EXISTS allowed_channels (
                id SERIAL PRIMARY KEY,
                channel_id BIGINT NOT NULL UNIQUE,
                added_by BIGINT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    conn.close()

def load_blacklist():
    global blacklist_words
    conn = db_connect()
    if not conn:
        logger.error("Failed to load blacklist from database")
        return
        
    with conn.cursor() as cur:
        cur.execute("SELECT word FROM blacklist")
        blacklist_words = {row[0].lower() for row in cur.fetchall()}
    critical_words = ["cp", "child", "porn", "premium", "collection", "price", "payment", "purchase", "desi", "indian", "foreign", "tamil", "chinese", "arabian", "bro-sis", "dad-daughter", "pedo"]
    for word in critical_words:
        if word not in blacklist_words:
            try:
                with db_connect() as conn:
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute("INSERT INTO blacklist (word, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING", (word, ADMIN_USER_ID))
                            conn.commit()
            except Exception as e:
                logger.error(f"Error adding critical word {word}: {e}")
    logger.info(f"Loaded {len(blacklist_words)} words from blacklist")

def load_allowed_chats():
    global allowed_chats
    conn = db_connect()
    if not conn:
        logger.error("Failed to load allowed chats from database")
        return
        
    with conn.cursor() as cur:
        cur.execute("SELECT chat_id FROM allowed_chats")
        allowed_chats = {row[0] for row in cur.fetchall()}
    conn.close()
    logger.info(f"Loaded {len(allowed_chats)} allowed chats")

def load_forward_whitelist():
    global forward_whitelist_users
    conn = db_connect()
    if not conn:
        logger.error("Failed to load forward whitelist from database")
        return
        
    with conn.cursor() as cur:
        cur.execute("SELECT user_id FROM forward_whitelist")
        forward_whitelist_users = {row[0] for row in cur.fetchall()}
    conn.close()
    logger.info(f"Loaded {len(forward_whitelist_users)} users from forward whitelist")

# Advanced detection functions
def contains_hidden_links(text):
    patterns = [r'[🟢💰📣⬇️➖➗]+\s*[\w\s]+\s*[🟢💰📣⬇️➖➗]+', r'[\w\.-]+\.[a-zA-Z]{2,}', r'@[\w]+\s*[\w]*\s*(offer|deal|price|sale)', r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+']
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

def detect_spam_patterns(text):
    for pattern in spam_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, f"Spam pattern detected: {pattern}"
    return False, ""

def normalize_text(text):
    normalized = re.sub(r'[^\w\s@\.\-$]', ' ', text)
    normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
    return normalized

# Custom command system
async def handle_custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    command = update.message.text.split()[0][1:].lower()
    conn = db_connect()
    if not conn:
        await update.message.reply_text("❌ Database connection error")
        return
        
    with conn.cursor() as cur:
        cur.execute("SELECT response FROM custom_commands WHERE command = %s", (command,))
        result = cur.fetchone()
    conn.close()
    if result:
        await update.message.reply_text(result[0])

# Admin commands
async def botversion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) == str(ADMIN_USER_ID):
        await update.message.reply_text("✅ Guardian Bot v2.1 - Channel Post Fix Applied.")

async def addcommand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can add commands", 10)
        return
    if len(context.args) < 2:
        await send_auto_delete_message(update, context, "Usage: /addcommand <name> <response>", 10)
        return
    command = context.args[0].lower()
    response = " ".join(context.args[1:])
    conn = db_connect()
    if not conn:
        await update.message.reply_text("❌ Database connection error")
        return
        
    with conn.cursor() as cur:
        try:
            cur.execute("INSERT INTO custom_commands (command, response, added_by) VALUES (%s, %s, %s)", (command, response, update.effective_user.id))
            conn.commit()
            await update.message.reply_text(f"✅ Command /{command} added successfully!")
        except psycopg2.IntegrityError:
            await update.message.reply_text(f"❌ Command /{command} already exists")
    conn.close()

async def report_spam(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message:
        await send_auto_delete_message(update, context, "❌ Please reply to a spam message to report it.", 10)
        return
    spam_message = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
    conn = db_connect()
    if not conn:
        await update.message.reply_text("❌ Database connection error")
        return
        
    with conn.cursor() as cur:
        cur.execute("INSERT INTO reported_spam (message, reported_by) VALUES (%s, %s)", (spam_message, update.effective_user.id))
    conn.commit()
    conn.close()
    await update.message.reply_text("✅ Spam reported! Thank you for helping improve our protection.")
    logger.info(f"Spam reported by user {update.effective_user.id}: {spam_message[:50]}...")

async def allowchat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, f"❌ Only admin can allow chats (Your ID: {update.effective_user.id}, Admin ID: {ADMIN_USER_ID})", 10)
        return
    if not context.args:
        await send_auto_delete_message(update, context, "Usage: /allowchat <chat_id>", 10)
        return
    try:
        chat_id = int(context.args[0])
        conn = db_connect()
        if not conn:
            await update.message.reply_text("❌ Database connection error")
            return
            
        with conn.cursor() as cur:
            cur.execute("INSERT INTO allowed_chats (chat_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING", (chat_id, update.effective_user.id))
        conn.commit()
        conn.close()
        allowed_chats.add(chat_id)
        await update.message.reply_text(f"✅ Chat {chat_id} added to allowed list")
        logger.info(f"Chat {chat_id} allowed by admin {update.effective_user.id}")
    except ValueError:
        await send_auto_delete_message(update, context, "❌ Invalid chat ID. Must be a number.", 10)

async def allowthischat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can use this command.", 10)
        return
    chat_id = update.effective_chat.id
    conn = db_connect()
    if not conn:
        await update.message.reply_text("❌ Database connection error")
        return
        
    with conn.cursor() as cur:
        cur.execute("INSERT INTO allowed_chats (chat_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING", (chat_id, update.effective_user.id))
    conn.commit()
    conn.close()
    allowed_chats.add(chat_id)
    await update.message.reply_text(f"✅ Okay! I will now be active in this chat (ID: {chat_id}).")

async def listchats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can view allowed chats", 10)
        return
    if not allowed_chats:
        await update.message.reply_text("No chats are currently allowed.")
        return
    chats_list = "\n".join(str(chat_id) for chat_id in allowed_chats)
    await update.message.reply_text(f"Allowed chats:\n{chats_list}")

async def allowforward(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can use this command.", 10)
        return

    user_to_allow = None
    if update.message.reply_to_message:
        user_to_allow = update.message.reply_to_message.from_user
    elif context.args:
        try:
            user_id = int(context.args[0])
            user_to_allow = await context.bot.get_chat(user_id)
        except (ValueError, IndexError):
            await send_auto_delete_message(update, context, "Usage: Reply to a user or provide their User ID.", 10)
            return
        except Exception as e:
            await update.message.reply_text(f"Could not find user. Error: {e}")
            return
    else:
        await send_auto_delete_message(update, context, "Usage: Reply to a user's message or use /allowforward <user_id>", 10)
        return

    if user_to_allow:
        conn = db_connect()
        if not conn:
            await update.message.reply_text("❌ Database connection error")
            return
            
        with conn.cursor() as cur:
            cur.execute("INSERT INTO forward_whitelist (user_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING", (user_to_allow.id, update.effective_user.id))
        conn.commit()
        conn.close()
        forward_whitelist_users.add(user_to_allow.id)
        await update.message.reply_text(f"✅ User {user_to_allow.first_name} ({user_to_allow.id}) can now forward messages.")

async def revokeforward(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can use this command.", 10)
        return

    user_id_to_revoke = None
    if update.message.reply_to_message:
        user_id_to_revoke = update.message.reply_to_message.from_user.id
    elif context.args:
        try:
            user_id_to_revoke = int(context.args[0])
        except (ValueError, IndexError):
            await send_auto_delete_message(update, context, "Usage: Reply to a user or provide their User ID.", 10)
            return
    else:
        await send_auto_delete_message(update, context, "Usage: Reply to a user's message or use /revokeforward <user_id>", 10)
        return
        
    if user_id_to_revoke:
        conn = db_connect()
        if not conn:
            await update.message.reply_text("❌ Database connection error")
            return
            
        with conn.cursor() as cur:
            cur.execute("DELETE FROM forward_whitelist WHERE user_id = %s", (user_id_to_revoke,))
        conn.commit()
        conn.close()
        forward_whitelist_users.discard(user_id_to_revoke)
        await update.message.reply_text(f"❌ User {user_id_to_revoke} can no longer forward messages.")

async def listforwarders(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can view this list.", 10)
        return
    if not forward_whitelist_users:
        await update.message.reply_text("No users are currently allowed to forward messages.")
        return
    
    users_list = "\n".join(str(user_id) for user_id in forward_whitelist_users)
    await update.message.reply_text(f"Users allowed to forward:\n{users_list}")

# New function to allow channels
async def allowchannel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can allow channels", 10)
        return
    if not context.args:
        await send_auto_delete_message(update, context, "Usage: /allowchannel <channel_id>", 10)
        return
    try:
        channel_id = int(context.args[0])
        conn = db_connect()
        if not conn:
            await update.message.reply_text("❌ Database connection error")
            return
            
        with conn.cursor() as cur:
            cur.execute("INSERT INTO allowed_channels (channel_id, added_by) VALUES (%s, %s) ON CONFLICT DO NOTHING", (channel_id, update.effective_user.id))
        conn.commit()
        conn.close()
        await update.message.reply_text(f"✅ Channel {channel_id} added to allowed channels")
        logger.info(f"Channel {channel_id} allowed by admin {update.effective_user.id}")
    except ValueError:
        await send_auto_delete_message(update, context, "❌ Invalid channel ID. Must be a number.", 10)

# Error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the telegram bot."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    
    # Try to notify the user about the error if possible
    if update and update.effective_message:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ Sorry, an error occurred while processing your request. Please try again later."
            )
        except Exception as e:
            logger.error(f"Error while sending error message: {e}")

# Flask App for Keep-Alive
flask_app = Flask(__name__)
@flask_app.route('/')
def home():
    return "🛡️ Guardian Bot is running and vigilant!"
def run_flask():
    from waitress import serve
    serve(flask_app, host='0.0.0.0', port=PORT)

# Telegram Bot Logic
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🛡️ Hello! I'm Guardian Bot\n\nI protect groups from spam and scams with AI-powered detection.\nUse /help to see available commands.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can use this command", 10)
        return
    help_text = """
    🛡️ *Admin Commands:*
    /addword <words> - Add words to blacklist
    /addcommand <name> <response> - Add custom command
    /allowchat <chat_id> - Allow a chat by ID
    /allowthischat - Allow the current chat
    /listchats - List allowed chats
    /allowforward <user_id> - Allow a user to forward (reply or use ID)
    /revokeforward <user_id> - Revoke forward permission
    /listforwarders - List users who can forward
    /allowchannel <channel_id> - Allow a channel to post without restrictions
    /stats - Show protection statistics
    /botversion - Check bot version
    
    👥 *User Commands:*
    /report - Reply to a spam message to report it
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def addword(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can add words", 10)
        return
    words_to_add = {word.lower() for word in context.args}
    if not words_to_add:
        await send_auto_delete_message(update, context, "Usage: /addword <word1> <word2>...", 10)
        return
    conn = db_connect()
    if not conn:
        await update.message.reply_text("❌ Database connection error")
        return
        
    with conn.cursor() as cur:
        added_count = 0
        for word in words_to_add:
            try:
                cur.execute("INSERT INTO blacklist (word, added_by) VALUES (%s, %s)",(word, update.effective_user.id))
                added_count += 1
            except psycopg2.IntegrityError:
                continue
    conn.commit()
    conn.close()
    load_blacklist()
    await update.message.reply_text(f"✅ Added {added_count} word(s) to blacklist")

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(ADMIN_USER_ID):
        await send_auto_delete_message(update, context, "❌ Only admin can view stats", 10)
        return
    stats_text = f"""
    📊 *Guardian Bot Statistics*
    
    • Blacklisted words: {len(blacklist_words)}
    • Allowed chats: {len(allowed_chats)}
    • Allowed forwarders: {len(forward_whitelist_users)}
    • Active warnings: {len(user_warnings)}
    • AI Model: Gemini 1.5 Flash
    """
    await update.message.reply_text(stats_text, parse_mode='Markdown')

# Message handling with advanced protection
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if this is a channel post from your channel
    if update.channel_post and update.channel_post.chat.id == CHANNEL_ID:
        logger.info(f"Allowed channel post from {CHANNEL_ID}")
        return
        
    if not update.message or not update.message.from_user: 
        return
    
    user = update.message.from_user
    chat_id = update.effective_chat.id
    message = update.message
    
    # Check if message is from any bot and schedule for deletion
    if user.is_bot:
        # Schedule deletion after 10 seconds for all bot messages
        asyncio.create_task(
            delete_message_after_delay(chat_id, message.message_id, context, 10)
        )
        # Skip further processing for bot messages
        return
    
    # Rate limiting
    current_time = time.time()
    user_key = f"{user.id}_{chat_id}"
    if user_key in message_rate_limit and current_time - message_rate_limit[user_key] < 2:
        return
    message_rate_limit[user_key] = current_time
    
    # Check if this is a forwarded message from your channel
    if message.forward_from_chat and message.forward_from_chat.id == CHANNEL_ID:
        logger.info(f"Allowed forwarded message from channel {CHANNEL_ID}")
        return
        
    if chat_id not in allowed_chats: 
        return
    
    now = datetime.now()
    if user.id in user_last_message and (now - user_last_message[user.id]).seconds < 2:
        try: 
            await message.delete()
        except Exception as e:
            if "message to delete not found" not in str(e).lower():
                logger.error(f"Failed to delete message: {e}")
        return
    user_last_message[user.id] = now
    
    is_admin = False
    if chat_id > 0:
        if str(user.id) == str(ADMIN_USER_ID): 
            is_admin = True
    else:
        try:
            chat_member = await context.bot.get_chat_member(chat_id, user.id)
            if chat_member.status in ['administrator', 'creator']:
                is_admin = True
        except Exception as e:
            logger.error(f"Error checking user status: {e}")
            if str(user.id) == str(ADMIN_USER_ID): 
                is_admin = True
    
    if is_admin: 
        return
            
    text = message.text or message.caption or ""
    text_lower = text.lower()
    normalized_text = normalize_text(text)
    is_spam = False
    reason = ""

    logger.info(f"Message from {user.id}: {text[:100]}...")

    # Strict Rules
    if (message.forward_from or message.forward_from_chat) and (user.id not in forward_whitelist_users):
        is_spam, reason = True, "You do not have permission to forward messages"

    if not is_spam and any(entity.type in ['url', 'text_link'] for entity in message.entities or []): 
        is_spam, reason = True, "Links are not allowed"
    if not is_spam and contains_hidden_links(text): 
        is_spam, reason = True, "Hidden links detected"
    if not is_spam and '@' in text and not text.startswith('/'): 
        is_spam, reason = True, "Mentions are not allowed"
    if not is_spam and (any(word in text_lower for word in blacklist_words) or any(word in normalized_text for word in blacklist_words)): 
        is_spam, reason = True, "Blacklisted word detected"
    if not is_spam and any(term in text_lower for term in payment_terms): 
        is_spam, reason = True, "Payment terms detected"
    if not is_spam:
        pattern_detected, pattern_reason = detect_spam_patterns(text_lower)
        if pattern_detected: 
            is_spam, reason = True, pattern_reason
        
    if not is_spam and text:
        try:
            analysis_text = f"Original: {text}\nNormalized: {normalized_text}"
            response = await asyncio.wait_for(spam_model.generate_content_async(analysis_text), timeout=7.0)
            logger.info(f"AI Response: {response.text}")
            if "SPAM" in response.text.upper(): 
                is_spam, reason = True, "AI detected spam content"
        except asyncio.TimeoutError:
            logger.warning("Gemini AI timeout, skipping analysis")
        except Exception as e:
            logger.error(f"Gemini error: {e}")

    if is_spam:
        try:
            await message.delete()
            user_warnings[user.id] += 1
            warning_count = user_warnings[user.id]
            if warning_count >= 3:
                await context.bot.ban_chat_member(chat_id=chat_id, user_id=user.id)
                warning_msg = f"⚠️ {user.mention_html()} has been banned after 3 warnings."
                sent_message = await context.bot.send_message(chat_id=chat_id, text=warning_msg, parse_mode='HTML')
                # Schedule the warning message for deletion
                asyncio.create_task(delete_message_after_delay(chat_id, sent_message.message_id, context, 10))
                del user_warnings[user.id]
                logger.info(f"User {user.id} banned for spam: {reason}")
            else:
                warning_msg = f"⚠️ {user.mention_html()}, {reason}. Warning {warning_count}/3"
                sent_message = await context.bot.send_message(chat_id=chat_id, text=warning_msg, parse_mode='HTML')
                # Schedule the warning message for deletion
                asyncio.create_task(delete_message_after_delay(chat_id, sent_message.message_id, context, 10))
                logger.info(f"Spam detected from user {user.id}: {reason}")
        except Exception as e:
            if "message to delete not found" not in str(e).lower():
                logger.error(f"Action error: {e}")

def main():
    setup_database()
    load_blacklist()
    load_allowed_chats()
    load_forward_whitelist()
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add error handler
    application.add_error_handler(error_handler)

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("addword", addword))
    application.add_handler(CommandHandler("addcommand", addcommand))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("report", report_spam))
    application.add_handler(CommandHandler("allowchat", allowchat))
    application.add_handler(CommandHandler("allowthischat", allowthischat))
    application.add_handler(CommandHandler("listchats", listchats))
    application.add_handler(CommandHandler("allowforward", allowforward))
    application.add_handler(CommandHandler("revokeforward", revokeforward))
    application.add_handler(CommandHandler("listforwarders", listforwarders))
    application.add_handler(CommandHandler("botversion", botversion))
    application.add_handler(CommandHandler("allowchannel", allowchannel))  # New command
    
    command_list = r'^/(start|help|addword|addcommand|stats|report|allowchat|allowthischat|listchats|allowforward|revokeforward|listforwarders|botversion|allowchannel)'
    application.add_handler(MessageHandler(filters.COMMAND & ~filters.Regex(command_list), handle_custom_command))
    
    application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))

    logger.info("🛡️ Guardian Bot is now running with enhanced detection...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    main()
