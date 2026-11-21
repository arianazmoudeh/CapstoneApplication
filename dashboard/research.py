import math
import re
from io import BytesIO

import numpy as np
import pandas as pd

TRADING_DAYS = 252
COST_RATE = 0.001
REBALANCE_DAYS = 20
MAX_FILE_SIZE = 2_000_000
COLORS = ["#2863a6", "#df8934", "#377d62", "#9063a6", "#c45564", "#738394"]


class DataError(ValueError):
    pass


def load_prices(raw):
    if len(raw) > MAX_FILE_SIZE:
        raise DataError("Please use a CSV smaller than 2 MB.")
    try:
        frame = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False)
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataError("This file could not be read. Use a UTF-8 CSV file.") from exc
    required = {"date", "symbol", "adjusted_close"}
    if not required.issubset(frame.columns):
        raise DataError("The CSV needs these columns: date, symbol, adjusted_close.")
    if frame.empty or len(frame) > 30_000:
        raise DataError("Use a CSV with between 1 and 30,000 price rows.")

    frame["symbol"] = frame["symbol"].str.strip().str.upper()
    if not frame["symbol"].map(lambda s: bool(re.fullmatch(r"[A-Z0-9.^_-]{1,20}", s))).all():
        raise DataError("Symbols should contain letters, numbers, dots, dashes or underscores.")
    try:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
        frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataError("Use dates like 2025-01-31 and numeric adjusted closing prices.") from exc
    if frame["date"].isna().any():
        raise DataError("Every price row needs a date like 2025-01-31.")
    if not np.isfinite(frame["adjusted_close"]).all() or (frame["adjusted_close"] <= 0).any():
        raise DataError("All adjusted closing prices must be positive, finite numbers.")
    if frame.duplicated(["date", "symbol"]).any():
        raise DataError("Each stock can have only one price per date. Remove duplicate rows.")
    count = frame["symbol"].nunique()
    if not 2 <= count <= 30:
        raise DataError("Please include between 2 and 30 stocks.")

    wide = frame.pivot(index="date", columns="symbol", values="adjusted_close").sort_index()
    prices = wide.dropna()
    dropped_dates = len(wide) - len(prices)
    names = {}
    for symbol, group in frame.groupby("symbol"):
        names[symbol] = str(group.iloc[0].get("name", symbol))[:100] or symbol
    return prices, names, {"rows": len(frame), "dropped_dates": dropped_dates}


def max_drawdown(values):
    return float((values / values.cummax() - 1).min())


def portfolio_test(prices, lookback, top_n, cost_rate=COST_RATE):
    if not 1 <= top_n <= prices.shape[1]:
        raise DataError("Portfolio size cannot exceed the number of stocks.")
    start = lookback + 2
    if len(prices) < start + 20:
        raise DataError(f"Please provide at least {start + 20} shared trading dates for this lookback.")
    daily_returns = prices.pct_change(fill_method=None)
    strategies = {"Momentum": [], "Equal weight": []}
    values = {key: 100.0 for key in strategies}
    weights = {key: np.zeros(prices.shape[1]) for key in strategies}
    turnovers = {key: 0.0 for key in strategies}
    last_target = None
    dates = [prices.index[start - 1]]
    for key in strategies:
        strategies[key].append(100.0)

    for t in range(start, len(prices)):
        rebalance = (t - start) % REBALANCE_DAYS == 0
        if rebalance:
            signal = prices.iloc[t - 2] / prices.iloc[t - 2 - lookback] - 1
            chosen = signal.sort_values(ascending=False, kind="mergesort").head(top_n).index
            target = pd.Series(0.0, index=prices.columns)
            target.loc[chosen] = 1 / top_n
            last_target = target
        day_returns = daily_returns.iloc[t].to_numpy(dtype=float)
        for key in strategies:
            trade_cost = 0.0
            if rebalance:
                new_weights = target.to_numpy() if key == "Momentum" else np.full(prices.shape[1], 1 / prices.shape[1])
                traded = float(np.abs(new_weights - weights[key]).sum())
                turnovers[key] += traded
                trade_cost = traded * cost_rate
                weights[key] = new_weights.copy()
            gross_return = float(weights[key] @ day_returns)
            values[key] *= (1 - trade_cost) * (1 + gross_return)
            weights[key] = weights[key] * (1 + day_returns) / (1 + gross_return)
            strategies[key].append(values[key])
        dates.append(prices.index[t])

    curves = pd.DataFrame(strategies, index=pd.DatetimeIndex(dates))
    metrics = []
    for key in strategies:
        series = curves[key]
        changes = series.pct_change().dropna()
        std = float(changes.std(ddof=1))
        metrics.append({
            "name": key,
            "total_return": float(series.iloc[-1] / 100 - 1) * 100,
            "volatility": std * math.sqrt(TRADING_DAYS) * 100,
            "sharpe": float(changes.mean() / std * math.sqrt(TRADING_DAYS)) if std else 0.0,
            "drawdown": max_drawdown(series) * 100,
            "turnover": turnovers[key],
        })
    return curves, metrics, last_target


def chart_data(frame, currency=False):
    left, right, top, bottom = 58, 875, 20, 235
    low, high = float(frame.min().min()), float(frame.max().max())
    padding = max((high - low) * 0.1, 1)
    low, high = max(0, low - padding), high + padding
    indices = np.unique(np.linspace(0, len(frame) - 1, min(len(frame), 400)).astype(int))
    lines = []
    for i, name in enumerate(frame.columns):
        points = []
        for j in indices:
            x = left + j / max(len(frame) - 1, 1) * (right - left)
            y = bottom - (float(frame.iloc[j, i]) - low) / (high - low) * (bottom - top)
            points.append(f"{x:.2f},{y:.2f}")
        lines.append({"name": name, "color": COLORS[i % len(COLORS)], "points": " ".join(points)})
    ticks = []
    for i in range(5):
        value = low + i / 4 * (high - low)
        if currency and high < 10:
            label = f"${value:,.2f}"
        elif currency and high < 100:
            label = f"${value:,.1f}"
        elif currency:
            label = f"${value:,.0f}"
        else:
            label = f"{value:,.0f}"
        ticks.append({"y": bottom - i / 4 * (bottom - top), "label": label})
    labels = []
    for j in np.unique(np.linspace(0, len(frame) - 1, 4).astype(int)):
        labels.append({"x": left + j / max(len(frame) - 1, 1) * (right - left), "label": frame.index[j].strftime("%b %Y")})
    return {"lines": lines, "ticks": ticks, "labels": labels}


def analyze(prices, names, quality, lookback=60, top_n=3):
    curves, metrics, target = portfolio_test(prices, lookback, top_n)
    returns = prices.pct_change(fill_method=None)
    momentum = (prices.iloc[-1] / prices.iloc[-1 - lookback] - 1).sort_values(ascending=False, kind="mergesort")
    rankings = []
    for rank, (symbol, score) in enumerate(momentum.items(), 1):
        rankings.append({
            "rank": rank, "symbol": symbol, "name": names.get(symbol, symbol),
            "price": float(prices[symbol].iloc[-1]), "momentum": float(score) * 100,
            "volatility": float(returns[symbol].tail(20).std(ddof=1)) * math.sqrt(TRADING_DAYS) * 100,
            "drawdown": max_drawdown(prices[symbol]) * 100,
        })
    normalized = prices / prices.iloc[0] * 100
    chart_columns = list(momentum.index[:5])
    correlations = returns.tail(60).corr()
    upper = correlations.to_numpy()[np.triu_indices_from(correlations, k=1)]
    finite_correlations = upper[np.isfinite(upper)]
    average_correlation = float(finite_correlations.mean()) if len(finite_correlations) else None
    return {
        "stock_count": prices.shape[1], "date_count": len(prices), **quality,
        "start": prices.index[0].strftime("%d %b %Y"), "end": prices.index[-1].strftime("%d %b %Y"),
        "test_start": curves.index[0].strftime("%d %b %Y"),
        "test_end": curves.index[-1].strftime("%d %b %Y"),
        "rankings": rankings, "metrics": metrics,
        "price_chart": chart_data(normalized[chart_columns]),
        "portfolio_chart": chart_data(curves),
        "average_correlation": average_correlation,
        "holdings": [{"symbol": key, "weight": float(value) * 100} for key, value in target.items() if value > 0],
        "chart_subset": len(chart_columns) < prices.shape[1],
    }
