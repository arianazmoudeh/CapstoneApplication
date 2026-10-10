import logging
import hashlib
from datetime import datetime, timedelta, timezone

import pandas as pd

from .eulerpool import EulerpoolError as MarketError, price_frame
from .models import PriceSnapshot

CRYPTO = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "USDT-USD": "Tether",
    "BNB-USD": "BNB",
    "XRP-USD": "XRP",
    "SOL-USD": "Solana",
    "USDC-USD": "USD Coin",
    "TRX-USD": "TRON",
    "DOGE-USD": "Dogecoin",
    "ADA-USD": "Cardano",
    "BCH-USD": "Bitcoin Cash",
    "LINK-USD": "Chainlink",
    "XLM-USD": "Stellar",
    "LTC-USD": "Litecoin",
    "AVAX-USD": "Avalanche",
    "SHIB-USD": "Shiba Inu",
    "DOT-USD": "Polkadot",
    "ETC-USD": "Ethereum Classic",
    "ATOM-USD": "Cosmos",
    "NEAR-USD": "NEAR Protocol",
}
STOCKS = {
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "AVGO": "Broadcom",
    "TSLA": "Tesla",
    "BRK-B": "Berkshire Hathaway",
    "WMT": "Walmart",
    "LLY": "Eli Lilly",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "ORCL": "Oracle",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "MA": "Mastercard",
    "COST": "Costco",
    "NFLX": "Netflix",
    "PG": "Procter & Gamble",
}
ASSETS = {**CRYPTO, **STOCKS}
TICKERS = {symbol: symbol.removesuffix("-USD") for symbol in ASSETS}
CACHE_KEY = "ep-market-v1"
logger = logging.getLogger(__name__)


def download_prices():
    today = datetime.now(timezone.utc).date()
    return price_frame(ASSETS, today - timedelta(days=365), today, strict=False)


def encode_prices(prices):
    return {
        "source": "Eulerpool", "dates": prices.index.strftime("%Y-%m-%d").tolist(),
        "prices": prices.astype(object).where(prices.notna(), None).to_dict(orient="list"),
        "errors": prices.attrs.get("errors", {}),
    }


def decode_prices(data):
    prices = pd.DataFrame(data["prices"], index=pd.to_datetime(data["dates"]), dtype=float).reindex(columns=list(ASSETS))
    prices.attrs["errors"] = data.get("errors", {})
    return prices


def align_portfolio_prices(prices):
    prices = prices.reindex(pd.date_range(prices.index[0], prices.index[-1], freq="D"))
    for symbol in prices:
        if not symbol.endswith("-USD"):
            prices[symbol] = prices[symbol].ffill(limit=4)
    prices = prices.dropna()
    if len(prices) < 60:
        raise MarketError("There are not enough shared daily prices for these assets yet.")
    return prices


def download_symbol_prices(symbols):
    today = datetime.now(timezone.utc).date()
    prices = price_frame(sorted(set(symbols)), today - timedelta(days=365), today)
    if any(prices[symbol].notna().sum() < 60 for symbol in symbols):
        raise MarketError("There are not enough daily prices for this asset yet.")
    return prices


def get_symbol_prices(symbols, align=True):
    def prepare(prices):
        return align_portfolio_prices(prices) if align else prices.dropna()

    symbols = sorted(set(symbols))
    market = PriceSnapshot.objects.filter(key=CACHE_KEY).first()
    if market and market.refresh_after > datetime.now(timezone.utc) and set(symbols).issubset(ASSETS):
        prices = decode_prices(market.data)[symbols]
        if all(prices[symbol].notna().sum() >= 60 for symbol in symbols):
            return prepare(prices), market.fetched_at, datetime.now(timezone.utc) - market.fetched_at > timedelta(hours=1)
    cache_id = hashlib.sha256(",".join(symbols).encode()).hexdigest()[:18]
    key = f"ep-p-{cache_id}"
    now = datetime.now(timezone.utc)
    saved = PriceSnapshot.objects.filter(key=key).first()
    if saved and saved.refresh_after > now:
        prices = pd.DataFrame(saved.data["prices"], index=pd.to_datetime(saved.data["dates"])).reindex(columns=symbols)
        return prepare(prices), saved.fetched_at, now - saved.fetched_at > timedelta(hours=1)
    try:
        prices = download_symbol_prices(symbols)
    except MarketError:
        logger.warning("Could not load portfolio prices from Eulerpool.")
        if not saved:
            if market and set(symbols).issubset(ASSETS):
                previous = decode_prices(market.data)[symbols]
                if all(previous[symbol].notna().sum() >= 60 for symbol in symbols):
                    return prepare(previous), market.fetched_at, True
            raise
        prices = pd.DataFrame(saved.data["prices"], index=pd.to_datetime(saved.data["dates"])).reindex(columns=symbols)
        return prepare(prices), saved.fetched_at, True
    PriceSnapshot.objects.update_or_create(key=key, defaults={
        "data": encode_prices(prices), "fetched_at": now,
        "refresh_after": now + timedelta(hours=1),
    })
    return prepare(prices), now, False


def get_prices(force=False):
    now = datetime.now(timezone.utc)
    saved = PriceSnapshot.objects.filter(key=CACHE_KEY).first()
    if saved and (saved.refresh_after > now and not force or now - saved.fetched_at < timedelta(minutes=1)):
        prices = decode_prices(saved.data)
        stale = now - saved.fetched_at > timedelta(hours=1) or prices.index[-1].date() < now.date() - timedelta(days=1)
        return prices, saved.fetched_at, stale
    try:
        prices = download_prices()
    except MarketError:
        logger.warning("Could not refresh Eulerpool prices.")
        if not saved:
            raise
        saved.refresh_after = now + timedelta(minutes=5)
        saved.save(update_fields=["refresh_after"])
        return decode_prices(saved.data), saved.fetched_at, True
    PriceSnapshot.objects.update_or_create(key=CACHE_KEY, defaults={
        "data": encode_prices(prices), "fetched_at": now,
        "refresh_after": now + timedelta(hours=1),
    })
    return prices, now, prices.index[-1].date() < now.date() - timedelta(days=1)
