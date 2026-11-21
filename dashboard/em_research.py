from datetime import datetime, timedelta, timezone

import pandas as pd

from .eulerpool import EulerpoolError, price_frame
from .models import PriceSnapshot
from .research import analyze

MARKETS = {
    "brazil": {
        "name": "Brazil",
        "currency": "R$",
        "universe": {
            "VALE3.SA": "Vale",
            "PETR4.SA": "Petrobras",
            "ITUB4.SA": "Itau Unibanco",
            "ABEV3.SA": "Ambev",
            "WEGE3.SA": "WEG",
        },
    },
    "india": {
        "name": "India",
        "currency": "₹",
        "universe": {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "HDFCBANK.NS": "HDFC Bank",
            "INFY.NS": "Infosys",
            "ICICIBANK.NS": "ICICI Bank",
        },
    },
    "mexico": {
        "name": "Mexico",
        "currency": "MX$",
        "universe": {
            "AMXB.MX": "America Movil",
            "WALMEX.MX": "Walmart de Mexico",
            "FEMSAUBD.MX": "FEMSA",
            "GMEXICOB.MX": "Grupo Mexico",
            "GFNORTEO.MX": "Banorte",
        },
    },
}


class ResearchDataError(ValueError):
    pass


def download_em_prices(market_code):
    market = MARKETS[market_code]
    universe = market["universe"]
    today = datetime.now(timezone.utc).date()
    currency = {"brazil": "BRL", "india": "INR", "mexico": "MXN"}[market_code]
    try:
        close = price_frame(universe, today - timedelta(days=730), today, currency=currency)
    except EulerpoolError as exc:
        raise ResearchDataError(str(exc)) from None
    coverage = []
    for symbol, name in universe.items():
        available = close[symbol].dropna()
        if available.empty:
            raise ResearchDataError(f"No price history was found for {symbol}.")
        coverage.append({
            "symbol": symbol,
            "name": name,
            "start": available.index[0],
            "end": available.index[-1],
            "observations": len(available),
            "missing": float(close[symbol].isna().mean()) * 100,
        })
    shared = close.dropna()
    if len(shared) < 180:
        raise ResearchDataError("There is not enough shared price history for the backtest.")
    return shared, coverage


def encode_research(prices, coverage):
    return {
        "source": "Eulerpool", "return_type": "price",
        "dates": prices.index.strftime("%Y-%m-%d").tolist(),
        "prices": prices.to_dict(orient="list"),
        "coverage": [
            {**row, "start": row["start"].strftime("%Y-%m-%d"), "end": row["end"].strftime("%Y-%m-%d")}
            for row in coverage
        ],
    }


def decode_research(data, universe):
    prices = pd.DataFrame(data["prices"], index=pd.to_datetime(data["dates"])).reindex(columns=list(universe))
    coverage = [
        {**row, "start": pd.Timestamp(row["start"]), "end": pd.Timestamp(row["end"])}
        for row in data["coverage"]
    ]
    return prices, coverage


def get_em_research(market_code, force=False):
    if market_code not in MARKETS:
        market_code = "brazil"
    market = MARKETS[market_code]
    universe = market["universe"]
    cache_key = f"ep-em-{market_code}"
    now = datetime.now(timezone.utc)
    saved = PriceSnapshot.objects.filter(key=cache_key).first()
    if saved and (saved.refresh_after > now and not force or now - saved.fetched_at < timedelta(minutes=1)):
        prices, coverage = decode_research(saved.data, universe)
        return build_result(prices, coverage, market_code), saved.fetched_at, now - saved.fetched_at > timedelta(hours=1)
    try:
        prices, coverage = download_em_prices(market_code)
    except ResearchDataError:
        if not saved:
            raise
        prices, coverage = decode_research(saved.data, universe)
        return build_result(prices, coverage, market_code), saved.fetched_at, True
    PriceSnapshot.objects.update_or_create(key=cache_key, defaults={
        "data": encode_research(prices, coverage),
        "fetched_at": now,
        "refresh_after": now + timedelta(hours=1),
    })
    return build_result(prices, coverage, market_code), now, False


def build_result(prices, coverage, market_code):
    market = MARKETS[market_code]
    result = analyze(
        prices,
        market["universe"],
        {"rows": int(prices.size), "dropped_dates": 0},
        lookback=60,
        top_n=3,
    )
    result["coverage"] = coverage
    result["market_code"] = market_code
    result["market_name"] = market["name"]
    result["currency"] = market["currency"]
    return result
