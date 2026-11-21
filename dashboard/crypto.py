import math

import pandas as pd

from .market import ASSETS, CRYPTO, STOCKS, TICKERS
from .research import chart_data, max_drawdown


def market_summary(prices):
    def points(series):
        series = series.tail(60)
        low, high = float(series.min()), float(series.max())
        spread = high - low or 1
        return " ".join(
            f"{(date - series.index[0]).days / max((series.index[-1] - series.index[0]).days, 1) * 130:.1f},{34 - (float(value) - low) / spread * 32:.1f}"
            for date, value in series.items()
        )

    def rows(assets, weekdays=False):
        output = []
        for rank, (symbol, name) in enumerate(assets.items(), 1):
            series = prices[symbol].dropna()
            if weekdays:
                series = series[series.index.dayofweek < 5]
            if len(series) < 31:
                continue
            price = float(series.iloc[-1])
            if price < 0.01:
                price_display = f"{price:.8f}".rstrip("0").rstrip(".")
            elif price < 1:
                price_display = f"{price:.4f}"
            else:
                price_display = f"{price:,.2f}"
            month = series.loc[series.index <= series.index[-1] - pd.Timedelta(days=30)]
            if month.empty:
                continue
            output.append({
                "rank": rank,
                "symbol": symbol,
                "ticker": TICKERS[symbol],
                "name": name,
                "price": price,
                "price_display": price_display,
                "change": float(series.iloc[-1] / series.iloc[-2] - 1) * 100,
                "momentum": float(series.iloc[-1] / month.iloc[-1] - 1) * 100,
                "date": series.index[-1], "previous_date": series.index[-2],
                "points": points(series),
            })
        return output

    return {
        "crypto_rows": rows(CRYPTO),
        "stock_rows": rows(STOCKS, weekdays=True),
        "last_date": prices.index[-1],
        "date_count": len(prices),
        "unavailable": [symbol for symbol in ASSETS if prices[symbol].notna().sum() < 31],
        "gap_days": len(pd.date_range(prices.index[0], prices.index[-1])) - len(prices),
    }


def portfolio_summary(portfolio, prices):
    quantities = pd.Series({symbol: float(quantity) for symbol, quantity in portfolio.quantities.items() if float(quantity) > 0})
    if quantities.empty:
        return None
    selected = prices[quantities.index]
    history = selected.mul(quantities, axis="columns").sum(axis=1)
    changes = history.pct_change(fill_method=None).dropna()
    latest_prices = selected.iloc[-1]
    values = latest_prices * quantities
    total_value = float(values.sum())
    return {
        "holdings": [{
            "symbol": symbol, "name": ASSETS.get(symbol, symbol), "ticker": TICKERS.get(symbol, symbol),
            "quantity": quantity, "price": float(latest_prices[symbol]),
            "value": float(values[symbol]), "weight": float(values[symbol] / total_value) * 100,
        } for symbol, quantity in quantities.items()],
        "chart": chart_data(history.to_frame("My portfolio (USD)"), currency=True),
        "start": prices.index[0], "end": prices.index[-1],
        "total_return": float(history.iloc[-1] / history.iloc[0] - 1) * 100,
        "volatility": float(changes.std(ddof=1)) * math.sqrt(365) * 100,
        "drawdown": max_drawdown(history) * 100,
        "start_value": float(history.iloc[0]),
        "end_value": total_value,
        "value_change": total_value - float(history.iloc[0]),
    }
