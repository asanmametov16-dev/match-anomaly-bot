"""Root conftest: set dummy env vars so src.config.Settings() can instantiate
without a real .env file during tests."""
import os

os.environ.setdefault("ODDS_API_KEY", "test_key")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:test")
os.environ.setdefault("TELEGRAM_CHAT_ID", "999999")
