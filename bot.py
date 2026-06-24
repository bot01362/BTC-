"""
BTC/USDT Telegram Signal Bot
-----------------------------
Price source: CoinGecko /simple/price endpoint (no API key required)
Signal logic: MA5 / MA10 / MA30 crossover + RSI(14)
Features: 3-hour rolling BUY/SELL/HOLD breakdown, inline Refresh/Snooze buttons,
          /status and /price commands, Flask self-ping server for Render free tier.
"""

import os
import sys
import time
import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from threading import Thread

import requests
from flask import Flask
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("btc_bot")

# ---------------------------------------------------------------------------
# Startup environment variable validation
# ---------------------------------------------------------------------------
REQUIRED_ENV_VARS = ["BOT_TOKEN", "CHAT_ID"]

missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
if missing:
    logger.error(f"Missing required environment variable(s): {', '.join(missing)}")
    logger.error("Set these in Render's Environment tab before redeploying.")
    sys.exit(1)

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_PARAMS = {"ids": "bitcoin", "vs_currencies": "usd"}

POLL_INTERVAL_SECONDS = 30
HISTORY_MAXLEN = 60
RSI_PERIOD = 14
MA_SHORT = 5
MA_MED = 10
MA_LONG = 30
ROLLING_WINDOW_HOURS = 3

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
price_history = deque(maxlen=HISTORY_MAXLEN)
signal_log = deque()
snoozed_until = None

# ---------------------------------------------------------------------------
# Flask keep-alive server (Render free tier)
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return "BTC/USDT bot is alive.", 200


def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------
def fetch_price():
    """Fetch the latest BTC/USD price from CoinGecko. Returns float or None."""
    try:
        resp = requests.get(COINGECKO_URL, params=COINGECKO_PARAMS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "bitcoin" not in data or "usd" not in data.get("bitcoin", {}):
            logger.warning(f"Unexpected CoinGecko response: {data}")
            return None

        return float(data["bitcoin"]["usd"])

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching price: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"Error parsing price data: {e}")
        return None


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def moving_average(data, period):
    if len(data) < period:
        return None
    return sum(list(data)[-period:]) / period


def calculate_rsi(data, period=RSI_PERIOD):
    """Returns RSI value or None if insufficient data."""
    if len(data) < period + 1:
        return None

    prices = list(data)[-(period + 1):]
    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def generate_signal():
    """
    Returns one of 'BUY', 'SELL', 'HOLD' based on MA5/MA10/MA30 crossover + RSI(14).
    Returns None if not enough data yet.
    """
    ma5 = moving_average(price_history, MA_SHORT)
    ma10 = moving_average(price_history, MA_MED)
    ma30 = moving_average(price_history, MA_LONG)
    rsi = calculate_rsi(price_history)

    if ma5 is None or ma10 is None or ma30 is None:
        return None

    bullish_alignment = ma5 > ma10 > ma30
    bearish_alignment = ma5 < ma10 < ma30

    if bullish_alignment and (rsi is None or rsi < 70):
        return "BUY"
    elif bearish_alignment and (rsi is None or rsi > 30):
        return "SELL"
    else:
        return "HOLD"


def record_signal(signal):
    now = datetime.utcnow()
    signal_log.append((now, signal))
    cutoff = now - timedelta(hours=ROLLING_WINDOW_HOURS)
    while signal_log and signal_log[0][0] < cutoff:
        signal_log.popleft()


def get_rolling_breakdown():
    """Returns dict with BUY/SELL/HOLD percentages over the rolling window."""
    if not signal_log:
        return {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

    total = len(signal_log)
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for _, sig in signal_log:
        counts[sig] += 1

    return {k: round((v / total) * 100, 1) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------
def build_status_message(price, signal, rsi):
    ma5 = moving_average(price_history, MA_SHORT)
    ma10 = moving_average(price_history, MA_MED)
    ma30 = moving_average(price_history, MA_LONG)
    breakdown = get_rolling_breakdown()

    rsi_display = f"{rsi}" if rsi is not None else "N/A (gathering data)"
    ma5_display = f"{ma5:.2f}" if ma5 is not None else "N/A"
    ma10_display = f"{ma10:.2f}" if ma10 is not None else "N/A"
    ma30_display = f"{ma30:.2f}" if ma30 is not None else "N/A"
    signal_display = signal if signal is not None else "Gathering data..."

    msg = (
        f"₿ *BTC/USDT Signal*\n\n"
        f"💰 Price: ${price:,.2f}\n"
        f"📊 Signal: *{signal_display}*\n\n"
        f"MA5: {ma5_display} | MA10: {ma10_display} | MA30: {ma30_display}\n"
        f"RSI(14): {rsi_display}\n\n"
        f"📈 Last {ROLLING_WINDOW_HOURS}h breakdown:\n"
        f"  BUY: {breakdown['BUY']}% | SELL: {breakdown['SELL']}% | HOLD: {breakdown['HOLD']}%\n\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    return msg


def build_keyboard():
    keyboard = [
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh"),
            InlineKeyboardButton("🔕 Snooze 30m", callback_data="snooze"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    price = fetch_price()

    if price is not None:
        price_history.append(price)
        signal = generate_signal()
        if signal:
            record_signal(signal)
        rsi = calculate_rsi(price_history)
        msg = build_status_message(price, signal, rsi)
        await update.message.reply_text(
            msg, parse_mode="Markdown", reply_markup=build_keyboard()
        )
    else:
        await update.message.reply_text(
            "⚠️ Couldn't fetch the current BTC price. Try again shortly."
        )


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    price = fetch_price()
    if price is not None:
        await update.message.reply_text(f"₿ BTC/USDT: ${price:,.2f}")
    else:
        await update.message.reply_text("⚠️ Couldn't fetch the current BTC price.")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global snoozed_until
    query = update.callback_query
    await query.answer()

    if query.data == "refresh":
        price = fetch_price()
        if price is not None:
            price_history.append(price)
            signal = generate_signal()
            if signal:
                record_signal(signal)
            rsi = calculate_rsi(price_history)
            msg = build_status_message(price, signal, rsi)
            await query.edit_message_text(
                msg, parse_mode="Markdown", reply_markup=build_keyboard()
            )
        else:
            await query.edit_message_text("⚠️ Couldn't fetch the current BTC price.")

    elif query.data == "snooze":
        snoozed_until = datetime.utcnow() + timedelta(minutes=30)
        await query.edit_message_text(
            f"🔕 Snoozed until {snoozed_until.strftime('%H:%M:%S')} UTC.\n"
            f"Use /status anytime to check manually."
        )


# ---------------------------------------------------------------------------
# Background polling loop (sends proactive signal updates)
# ---------------------------------------------------------------------------
async def poll_and_alert(context: ContextTypes.DEFAULT_TYPE):
    global snoozed_until

    if snoozed_until and datetime.utcnow() < snoozed_until:
        return

    price = fetch_price()
    if price is None:
        logger.warning("Skipping this poll cycle — no price returned.")
        return

    price_history.append(price)
    signal = generate_signal()

    if signal:
        record_signal(signal)

    if signal in ("BUY", "SELL"):
        rsi = calculate_rsi(price_history)
        msg = build_status_message(price, signal, rsi)
        try:
            await context.bot.send_message(
                chat_id=CHAT_ID,
                text=msg,
                parse_mode="Markdown",
                reply_markup=build_keyboard(),
            )
        except Exception as e:
            logger.error(f"Failed to send proactive alert: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CallbackQueryHandler(button_callback))

    application.job_queue.run_repeating(
        poll_and_alert, interval=POLL_INTERVAL_SECONDS, first=10
    )

    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    logger.info("BTC/USDT bot starting...")

    # Python 3.13+ no longer auto-creates an event loop on the main thread.
    # python-telegram-bot 21.4's run_polling() expects one to already exist,
    # so we create and set it explicitly here before handing off control.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
