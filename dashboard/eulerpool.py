from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from django.conf import settings


class EulerpoolError(ValueError):
    pass


def request_data(path, params=None):
    key = settings.EULERPOOL_API_KEY
    if not key:
        raise EulerpoolError("Eulerpool is not connected yet. Set EULERPOOL_API_KEY in the application settings.")
    try:
        response = requests.get(
            f"https://api.eulerpool.com/api/1/{path}", params=params,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            timeout=(3, 8), allow_redirects=False,
        )
    except requests.RequestException:
        raise EulerpoolError("Eulerpool could not be reached. Please try again shortly.") from None
    if response.status_code in (401, 403):
        raise EulerpoolError("Eulerpool rejected this request. Check the API key and data access.")
    if response.status_code == 404:
        raise EulerpoolError("This asset or its price history was not found on Eulerpool.")
    if response.status_code == 429:
        raise EulerpoolError("The Eulerpool request limit has been reached. Please use the saved data or try again later.")
    if response.status_code != 200:
        raise EulerpoolError("Eulerpool is unavailable right now. Please try again shortly.")
    try:
        return response.json()
    except ValueError:
        raise EulerpoolError("Eulerpool returned a response that could not be read.") from None


def daily_prices(symbol, start, end, currency="USD"):
    crypto = symbol.endswith("-USD")
    identifier = symbol.removesuffix("-USD") if crypto else symbol
    identifier = quote(identifier, safe="")
    category = "crypto" if crypto else "equity"
    profile = request_data(f"{category}/profile/{identifier}")
    if not isinstance(profile, dict) or not isinstance(profile.get("currency"), str) or not profile["currency"]:
        raise EulerpoolError(f"Eulerpool did not identify the currency for {symbol}.")
    if profile["currency"].upper() != currency:
        raise EulerpoolError(f"{symbol} is priced in {profile['currency']}. This view requires {currency} prices.")
    start = pd.Timestamp(start).normalize().tz_localize("UTC")
    end = min(pd.Timestamp(end).normalize().tz_localize("UTC"), pd.Timestamp(datetime.now(timezone.utc)).normalize())
    if crypto:
        raw = request_data(f"crypto/quotes/{identifier}", {
            "startdate": int(start.timestamp() * 1000), "enddate": int(end.timestamp() * 1000),
        })
        if not isinstance(raw, list) or not raw:
            raise EulerpoolError(f"No price history was found for {symbol}.")
        try:
            timestamps = [row["timestamp"] for row in raw]
            values = [row["price"] for row in raw]
        except (KeyError, TypeError):
            raise EulerpoolError(f"Eulerpool returned unreadable prices for {symbol}.") from None
        unit = "ms"
    else:
        raw = request_data(f"charting/ohlcv/{identifier}", {
            "resolution": "D", "from": int(start.timestamp()), "to": int(end.timestamp()),
        })
        if not isinstance(raw, dict) or not isinstance(raw.get("t"), list) or not isinstance(raw.get("c"), list):
            raise EulerpoolError(f"Eulerpool returned unreadable prices for {symbol}.")
        timestamps, values, unit = raw["t"], raw["c"], "s"
    if not timestamps or len(timestamps) != len(values):
        raise EulerpoolError(f"No complete price history was found for {symbol}.")
    try:
        dates = pd.to_datetime(timestamps, unit=unit, utc=True, errors="raise")
        if dates.hasnans:
            raise ValueError("Missing dates")
        series = pd.Series(pd.to_numeric(values, errors="raise"), index=dates, name=symbol, dtype=float).sort_index()
    except (ValueError, TypeError, OverflowError):
        raise EulerpoolError(f"Eulerpool returned invalid prices for {symbol}.") from None
    series = series.loc[(series.index >= start) & (series.index < end)]
    if series.empty or series.index.hasnans or not np.isfinite(series).all() or series.le(0).any():
        raise EulerpoolError(f"Eulerpool returned missing or invalid prices for {symbol}.")
    series.index = series.index.tz_localize(None).normalize()
    series = series.groupby(level=0).last()
    if not crypto:
        series = series.loc[series.index.dayofweek < 5]
    if series.empty:
        raise EulerpoolError(f"No daily prices were found for {symbol}.")
    return series


def price_frame(symbols, start, end, currency="USD", strict=True):
    symbols = list(dict.fromkeys(symbols))

    def load(symbol):
        try:
            return symbol, daily_prices(symbol, start, end, currency), None
        except EulerpoolError as exc:
            return symbol, None, str(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(load, symbols))
    errors = {symbol: error for symbol, _, error in results if error}
    if errors and (strict or len(errors) == len(symbols)):
        raise EulerpoolError(next(iter(errors.values())))
    frame = pd.concat([series for _, series, error in results if not error], axis=1).sort_index().reindex(columns=symbols)
    frame.attrs["errors"] = errors
    return frame
