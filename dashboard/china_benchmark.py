import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .research import chart_data


RESULTS = Path(__file__).resolve().parent / "china_benchmark_results"


class BenchmarkError(ValueError):
    pass


def equal_weight_basket(observations, calendar, codes, start, end, initial_value=100000.0):
    codes = list(codes)
    if not codes or len(set(codes)) != len(codes):
        raise BenchmarkError("The basket needs unique stock codes.")
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise BenchmarkError("The starting value must be positive.")
    data = observations.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data.loc[data.code.isin(codes)]
    data["adjusted_close"] = pd.to_numeric(data["adjusted_close"], errors="raise")
    if data.duplicated(["date", "code"]).any():
        raise BenchmarkError("A stock has duplicate prices on the same date.")
    if not np.isfinite(data.adjusted_close).all() or data.adjusted_close.le(0).any():
        raise BenchmarkError("Adjusted prices must be positive and finite.")
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    before = calendar[calendar < start]
    dates = calendar[(calendar >= start) & (calendar <= end)]
    if not len(before) or not len(dates):
        raise BenchmarkError("The calendar needs a starting valuation and test dates.")
    base_date = before[-1]
    dates = pd.DatetimeIndex([base_date]).append(dates)
    raw_prices = data.pivot(index="date", columns="code", values="adjusted_close").reindex(columns=codes)
    all_dates = raw_prices.index.union(dates).sort_values()
    prices = raw_prices.reindex(all_dates).ffill().reindex(dates)
    if prices.isna().any().any():
        missing = prices.columns[prices.isna().any()].tolist()
        raise BenchmarkError("Prices are missing before the start for: " + ", ".join(missing))
    observed = raw_prices.reindex(dates).notna()
    carried = ~observed
    target = np.full(len(codes), 1.0 / len(codes))
    weights = target.copy()
    value = float(initial_value)
    values = [value]
    weight_rows = [weights.copy()]
    rebalance_rows = [{"date": base_date, "stock_count": len(codes), "carried_quotes": int(carried.iloc[0].sum())}]
    returns = prices.pct_change(fill_method=None)
    for position in range(1, len(dates)):
        changes = returns.iloc[position].to_numpy(dtype=float)
        portfolio_return = float(weights @ changes)
        value *= 1 + portfolio_return
        weights *= 1 + changes
        weights /= 1 + portfolio_return
        if position < len(dates) - 1 and dates[position].to_period("M") != dates[position + 1].to_period("M"):
            weights = target.copy()
            rebalance_rows.append({"date": dates[position], "stock_count": len(codes), "carried_quotes": int(carried.iloc[position].sum())})
        values.append(value)
        weight_rows.append(weights.copy())
    curve = pd.DataFrame({"benchmark_value": values}, index=dates).rename_axis("date")
    holdings = pd.DataFrame(weight_rows, index=dates, columns=codes).rename_axis("date")
    gaps = []
    for code in codes:
        streak, longest = 0, 0
        for missing in carried[code]:
            streak = streak + 1 if missing else 0
            longest = max(longest, streak)
        stock = data.loc[data.code.eq(code)].sort_values("date")
        gaps.append({
            "code": code,
            "first_quote": stock.date.min().date().isoformat(),
            "last_quote": stock.loc[stock.date.le(end), "date"].max().date().isoformat(),
            "observed_dates": int(observed[code].sum()),
            "carried_dates": int(carried[code].sum()),
            "longest_gap": longest,
        })
    return curve, holdings, pd.DataFrame(rebalance_rows), pd.DataFrame(gaps)


def basket_metrics(curve):
    values = curve["benchmark_value"]
    changes = values.pct_change(fill_method=None).dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    if years <= 0 or len(changes) < 2:
        raise BenchmarkError("The benchmark needs at least two return observations.")
    return {
        "initial_value": float(values.iloc[0]),
        "final_value": float(values.iloc[-1]),
        "total_return": float((values.iloc[-1] / values.iloc[0] - 1) * 100),
        "annual_return": float(((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1) * 100),
        "volatility": float(changes.std(ddof=1) * np.sqrt(252) * 100),
        "drawdown": float((values / values.cummax() - 1).min() * 100),
    }


def benchmark_context():
    try:
        summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        curve = pd.read_csv(RESULTS / "portfolio_values.csv", parse_dates=["date"]).set_index("date")
        holdings = pd.read_csv(RESULTS / "holdings.csv", dtype={"code": str}, keep_default_na=False)
        excluded = pd.read_csv(RESULTS / "excluded.csv", dtype={"code": str}, keep_default_na=False)
        if not curve.index.is_unique or not curve.index.is_monotonic_increasing:
            raise ValueError("Invalid benchmark dates")
        if not np.isfinite(curve.to_numpy()).all() or not curve.gt(0).all().all():
            raise ValueError("Invalid benchmark values")
        if len(holdings) != summary["stock_count"] or len(holdings) + len(excluded) != summary["candidate_count"]:
            raise ValueError("Invalid benchmark membership")
        if not holdings.code.is_unique or set(holdings.code) & set(excluded.code):
            raise ValueError("Duplicate benchmark members")
        if not np.isclose(holdings.weight.sum(), 100) or not np.isclose(holdings.latest_weight.sum(), 100):
            raise ValueError("Invalid benchmark weights")
        for frame in (holdings, excluded):
            frame["listing_date"] = frame["listing_date"].map(lambda value: date.fromisoformat(value) if value else None)
        metrics = basket_metrics(curve)
        for key, value in metrics.items():
            if not np.isclose(summary[key], value):
                raise ValueError("Benchmark summary does not match its prices")
        for key in ("base_date", "start", "end"):
            summary[key] = date.fromisoformat(summary[key])
        if curve.index[0].date() != summary["base_date"] or curve.index[-1].date() != summary["end"]:
            raise ValueError("Benchmark timeframe does not match its prices")
        chart = chart_data(curve.rename(columns={"benchmark_value": "Equal-weight A-shares"}))
        summary.update(chart=chart, holdings=holdings.to_dict("records"), excluded=excluded.to_dict("records"))
        return summary
    except (OSError, KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise BenchmarkError("The benchmark results could not be loaded. Please try again later.") from exc
