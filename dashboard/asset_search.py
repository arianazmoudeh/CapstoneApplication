import hashlib
import re

from django.core.cache import cache

from .eulerpool import EulerpoolError, request_data
from .market import CRYPTO, STOCKS


def search_assets(query):
    query = " ".join(query.split()).casefold()
    if len(query) < 2:
        return {"results": []}
    if len(query) > 100:
        raise ValueError("Search using up to 100 characters.")
    key = "asset-search-v1-" + hashlib.sha256(query.encode()).hexdigest()
    saved = cache.get(key)
    if saved is not None:
        return saved
    matches = {
        symbol: {"symbol": symbol, "name": name, "type": "Crypto", "currency": "USD"}
        for symbol, name in CRYPTO.items()
        if query in symbol.casefold() or query in name.casefold()
    }
    try:
        data = request_data("equity/search", {"q": query})
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise EulerpoolError("Search results could not be read.")
    except EulerpoolError:
        if matches:
            return {"results": list(matches.values())[:8], "message": "Stock search is unavailable. Showing matching crypto."}
        raise
    for row in data["results"]:
        if not isinstance(row, dict) or row.get("type") != "stock" or row.get("currency") != "USD":
            continue
        symbol, name = row.get("ticker"), row.get("name")
        if not isinstance(symbol, str) or not isinstance(name, str) or not name.strip():
            continue
        symbol = symbol.strip().upper()
        if not re.fullmatch(r"[A-Z0-9.\-]{1,20}", symbol) or symbol.endswith("-USD"):
            continue
        matches.setdefault(symbol, {"symbol": symbol, "name": name.strip()[:120], "type": "Stock", "currency": "USD"})

    def relevance(asset):
        symbol, name = asset["symbol"].casefold(), asset["name"].casefold()
        familiar_name = STOCKS.get(asset["symbol"], "").casefold()
        return (
            0 if query in (symbol, symbol.removesuffix("-usd"), name, familiar_name) else
            1 if symbol.startswith(query) or name.startswith(query) else 2
        )

    result = {"results": sorted(matches.values(), key=relevance)[:8]}
    cache.set(key, result, 900)
    return result
