"""
BTC/USDT Telegram Signal Bot
-----------------------------
Price source: CoinCap v3 /assets/bitcoin endpoint (requires free API key)
Signal logic: MA5 / MA10 / MA30 crossover + RSI(14)
Features: 3-hour rolling BUY/SELL/HOLD breakdown, inline Refresh/Snooze buttons,
          /status and /price commands, Flask self-ping server for Render free tier.

CHANGES FROM PREVIOUS VERSION (fixes noisy / contradicting alerts):
  1. POLL_INTERVAL_SECONDS: 30 -> 300 (5 min). MA30 now spans 2.5 hours instead
     of 15 minutes, so it reflects an actual trend instead of second-to-second noise.
  2. Alerts only fire when the confirmed signal CHANGES (no more repeat BUY/BUY/BUY spam).
  3. PRICE_REVERSAL_PCT: 0.3 -> 1.2. BTC moves 0.3% in minutes as pure noise;
     1.2% is a meaningful move against your entry.
  4. Added REVERSAL_GRACE_PERIOD_MINUTES: reversal checks are skipped for the
     first 15 minutes after a position is opened, so a signal can't immediately
     contradict itself.
  5. CONFIRMATION_STREAK: 2 -> 3, since each cycle now represents 5 minutes
     instead of 30 seconds, 3 cycles = 15 minutes of sustained trend.
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
REQUIRED_ENV_VARS = ["BOT_TOKEN", "CHAT_ID", "COINCAP_API_KEY"]

missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
if missing:
    logger.error(f"Missing required environment variable(s): {', '.join(missing)}")
    logger.error("Set these in Render's Environment tab before redeploying.")
    sys.exit(1)

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
COINCAP_API_KEY = os.environ["COINCAP_API_KEY"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COINCAP_URL = "https://rest.coincap.io/v3/assets/bitcoin"

POLL_INTERVAL_SECONDS = 300   # 5 minutes — was 30s, far too fast for BTC's MA windows
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
last_alerted_signal = None    # tracks last signal actually sent, to avoid repeats

# --- Signal confidence filtering ---
MIN_MA_SPREAD_PCT = 0.08
RSI_BUY_CEILING = 60
RSI_SELL_FLOOR = 40
CONFIRMATION_STREAK = 3        # was 2 — now 3 cycles of 5 min = 15 min sustained trend
recent_raw_signals = deque(maxlen=CONFIRMATION_STREAK)

# --- Active position tracking (for exit/reversal warnings) ---
active_position = None        # "BUY", "SELL", or None
entry_price = None
entry_time = None             # NEW — used for grace period
PRICE_REVERSAL_PCT = 1.2      # was 0.3 — too tight, tripped by normal BTC noise
RSI_REVERSAL_BUY = 70
RSI_REVERSAL_SELL = 30
REVERSAL_GRACE_PERIOD_MINUTES = 15   # NEW — no reversal checks right after entry

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
    """Fetch the latest BTC/USD price from CoinCap v3. Returns float or None."""
    params = {"apiKey": COINCAP_API_KEY}
    try:
        resp = requests.get(COINCAP_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        asset = data.get("data")
        if not asset or "priceUsd" not in asset:
            logger.warning(f"Unexpected CoinCap response: {data}")
            return None

        return float(asset["priceUsd"])

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
    Returns one of 'BUY', 'SELL', 'HOLD' based on MA5/MA10/MA30 crossover + RSI(14),
    filtered for confidence: requires meaningful MA separation and RSI clearly
    past the midline. Returns None if not enough data yet.
    """
    ma5 = moving_average(price_history, MA_SHORT)
    ma10 = moving_average(price_history, MA_MED)
    ma30 = moving_average(price_history, MA_LONG)
    rsi = calculate_rsi(price_history)

    if ma5 is None or ma10 is None or ma30 is None:
        return None

    bullish_alignment = ma5 > ma10 > ma30
    bearish_alignment = ma5 < ma10 < ma30

    ma_spread_pct = abs(ma5 - ma30) / ma30 * 100 if ma30 else 0
    strong_spread = ma_spread_pct >= MIN_MA_SPREAD_PCT

    if bullish_alignment and strong_spread and (rsi is None or rsi < RSI_BUY_CEILING):
        return "BUY"
    elif bearish_alignment and strong_spread and (rsi is None or rsi > RSI_SELL_FLOOR):
        return "SELL"
    else:
        return "HOLD"


def confirmed_signal(raw_signal):
    """
    Tracks the last few raw signals and only returns BUY/SELL once the same
    signal has held for CONFIRMATION_STREAK consecutive cycles. Otherwise
    returns 'HOLD' so a single noisy tick can't trigger a false alert.
    """
    recent_raw_signals.append(raw_signal)

    if len(recent_raw_signals) < CONFIRMATION_STREAK:
        return "HOLD"

    if all(s == "BUY" for s in recent_raw_signals):
        return "BUY"
    elif all(s == "SELL" for s in recent_raw_signals):
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


def check_for_reversal(price, ma5, ma10, rsi):
    """
    Checks whether the current active position shows signs of reversing.
    Returns a warning message string if any condition trips, else None.
    Skips entirely during the grace period right after entry, so a fresh
    signal can't immediately contradict itself on normal short-term noise.
    """
    global active_position, entry_price, entry_time

    if active_position is None or entry_price is None or entry_time is None:
        return None

    minutes_since_entry = (datetime.utcnow() - entry_time).total_seconds() / 60
    if minutes_since_entry < REVERSAL_GRACE_PERIOD_MINUTES:
        return None

    reasons = []

    if ma5 is not None and ma10 is not None:
        if active_position == "BUY" and ma5 < ma10:
            reasons.append("MA5 has crossed back below MA10")
        elif active_position == "SELL" and ma5 > ma10:
            reasons.append("MA5 has crossed back above MA10")

    if rsi is not None:
        if active_position == "BUY" and rsi > RSI_REVERSAL_BUY:
            reasons.append(f"RSI spiked to {rsi} (overbought)")
        elif active_position == "SELL" and rsi < RSI_REVERSAL_SELL:
            reasons.append(f"RSI dropped to {rsi} (oversold)")

    price_change_pct = (price - entry_price) / entry_price * 100
    if active_position == "BUY" and price_change_pct <= -PRICE_REVERSAL_PCT:
        reasons.append(f"price is down {abs(price_change_pct):.2f}% from entry (${entry_price:,.2f})")
    elif active_position == "SELL" and price_change_pct >= PRICE_REVERSAL_PCT:
        reasons.append(f"price is up {price_change_pct:.2f}% from entry (${entry_price:,.2f})")

    if reasons:
        warning = (
            f"⚠️ *Possible Reversal — Consider HOLD/EXIT*\n\n"
            f"Your last signal was *{active_position}* at ${entry_price:,.2f}.\n"
            f"Current price: ${price:,.2f}\n\n"
            f"Reason(s):\n" + "\n".join(f"• {r}" for r in reasons)
        )
        active_position = None
        entry_price = None
        entry_time = None
        return warning

    return None


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
    global snoozed_until, active_position, entry_price, entry_time, last_alerted_signal

    if snoozed_until and datetime.utcnow() < snoozed_until:
        return

    price = fetch_price()
    if price is None:
        logger.warning("Skipping this poll cycle — no price returned.")
        return

    price_history.append(price)
    raw_signal = generate_signal()
    signal = confirmed_signal(raw_signal) if raw_signal is not None else None

    if signal:
        record_signal(signal)

    ma5 = moving_average(price_history, MA_SHORT)
    ma10 = moving_average(price_history, MA_MED)
    rsi = calculate_rsi(price_history)

    reversal_warning = check_for_reversal(price, ma5, ma10, rsi)
    if reversal_warning:
        try:
            await context.bot.send_message(
                chat_id=CHAT_ID,
                text=reversal_warning,
                parse_mode="Markdown",
            )
            last_alerted_signal = None  # position closed, allow a fresh alert next time
        except Exception as e:
            logger.error(f"Failed to send reversal warning: {e}")

    # Only alert when the confirmed signal is different from the last one we
    # actually sent — stops repeat BUY/BUY/BUY spam every single cycle.
    if signal in ("BUY", "SELL") and signal != last_alerted_signal:
        msg = build_status_message(price, signal, rsi)
        try:
            await context.bot.send_message(
                chat_id=CHAT_ID,
                text=msg,
                parse_mode="Markdown",
                reply_markup=build_keyboard(),
            )
            active_position = signal
            entry_price = price
            entry_time = datetime.utcnow()
            last_alerted_signal = signal
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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
