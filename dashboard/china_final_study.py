import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .research import COLORS


RESULTS = Path(__file__).resolve().parent / "china_final_results"
FACTORS = ["Value", "Profitability", "Momentum"]


class FinalStudyError(ValueError):
    pass


def _read(name, columns, dates=()):
    frame = pd.read_csv(RESULTS / f"{name}.csv")
    if frame.empty or not set(columns).issubset(frame.columns):
        raise ValueError(f"Incomplete {name}")
    for column in dates:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError(f"Missing {column}")
    return frame


def _finite(frame, columns):
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite result")


def _chart(frame, unit="", step=False):
    low, high = float(frame.min().min()), float(frame.max().max())
    negative_only = high <= 0
    positive_only = low >= 0
    if unit == "%":
        low, high = min(low, 0), max(high, 0)
    padding = max((high - low) * 0.08, 0.5 if unit == "%" else 1)
    low, high = low - padding, high + padding
    if unit == "%" and negative_only:
        high = 0.0
    elif unit == "%" and positive_only:
        low = 0.0
    left, right, top, bottom = 58, 875, 20, 235
    width = max((frame.index[-1] - frame.index[0]).total_seconds(), 1)
    x = left + (frame.index - frame.index[0]).total_seconds().to_numpy() / width * (right - left)
    lines = []
    mobile_lines = []
    for i, name in enumerate(frame.columns):
        y = bottom - (frame[name].to_numpy() - low) / (high - low) * (bottom - top)
        points = []
        mobile_points = []
        for j in range(len(frame)):
            mobile_x = 50 + (x[j] - left) / (right - left) * 280
            if step and j:
                points.append(f"{x[j]:.2f},{y[j - 1]:.2f}")
                mobile_points.append(f"{mobile_x:.2f},{y[j - 1]:.2f}")
            points.append(f"{x[j]:.2f},{y[j]:.2f}")
            mobile_points.append(f"{mobile_x:.2f},{y[j]:.2f}")
        lines.append({"name": name, "points": " ".join(points), "color": COLORS[i % len(COLORS)], "dash": ["", "7 3", "2 3", "10 3 2 3"][i % 4]})
        mobile_lines.append({**lines[-1], "points": " ".join(mobile_points)})
    ticks = []
    for i in range(5):
        value = low + i * (high - low) / 4
        ticks.append({"y": bottom - i * (bottom - top) / 4, "label": f"{value:.1f}%" if unit == "%" else f"{value:,.0f}"})
    labels = [{"x": float(x[j]), "label": frame.index[j].strftime("%b %Y")} for j in np.unique(np.linspace(0, len(frame) - 1, 4).astype(int))]
    mobile_labels = [{"x": 50 + float(x[j] - left) / (right - left) * 280, "label": frame.index[j].strftime("%b %Y")} for j in np.unique(np.linspace(0, len(frame) - 1, 3).astype(int))]
    return {"lines": lines, "ticks": ticks, "labels": labels, "mobile_lines": mobile_lines, "mobile_labels": mobile_labels}


def final_study_context():
    try:
        summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status", "complete") != "complete":
            raise ValueError("Incomplete run")
        start, end = pd.Timestamp(summary["evaluation_start"]), pd.Timestamp(summary["evaluation_end"])
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError("Invalid evaluation dates")
        period_count = int(summary["period_count"])
        initial_value = float(summary["initial_value"])
        models = summary["models"]
        if period_count < 1 or initial_value <= 0 or not np.isfinite(initial_value):
            raise ValueError("Invalid study size")
        if len(models) != int(summary["model_count"]) or len(models) != len(set(models)):
            raise ValueError("Invalid model list")
        primary = summary["primary_strategy"]
        comparisons = summary["comparison_strategies"]
        if primary not in comparisons or "Equal factors" not in comparisons or len(comparisons) != len(set(comparisons)):
            raise ValueError("Missing controls")
        notebook = summary["notebook_filename"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.ipynb", notebook):
            raise ValueError("Invalid notebook name")
        for field in ["findings", "limitations"]:
            if not isinstance(summary[field], list) or not summary[field] or not all(isinstance(item, str) and item.strip() for item in summary[field]):
                raise ValueError("Missing interpretation")
        metric_columns = ["annual_return", "volatility", "sharpe", "drawdown", "final_value", "trading_costs", "borrow_costs", "average_monthly_turnover"]
        performance = _read("performance", ["strategy", *metric_columns])
        _finite(performance, metric_columns)
        if not performance.strategy.is_unique or not set([*comparisons, *models]).issubset(performance.strategy):
            raise ValueError("Incomplete portfolio comparison")
        if (performance.volatility < 0).any() or (performance.drawdown > 1e-9).any() or (performance.final_value <= 0).any():
            raise ValueError("Invalid portfolio measures")
        performance = performance.set_index("strategy")
        performance["monthly_turnover_percent"] = performance.average_monthly_turnover * 100
        values = _read("portfolio_values", ["date", *comparisons], ["date"]).set_index("date")
        _finite(values, comparisons)
        if len(values) < 2 or not values.index.is_unique or not values.index.is_monotonic_increasing or (values[comparisons] <= 0).any().any():
            raise ValueError("Invalid portfolio history")
        if values.index[0] > start or values.index[-1] != end:
            raise ValueError("Dates do not match")
        if not np.allclose(values.iloc[0][comparisons], initial_value, atol=0.01, rtol=1e-9):
            raise ValueError("Starting values do not match")
        if not np.allclose(values.iloc[-1][comparisons], performance.loc[comparisons, "final_value"], atol=0.02, rtol=1e-9):
            raise ValueError("Final values do not match")
        selection = _read("selection", ["selection_date", "execution_date", "model", "score", "lookback_count"], ["selection_date", "execution_date"])
        _finite(selection, ["score", "lookback_count"])
        if not selection.selection_date.is_unique or not selection.selection_date.is_monotonic_increasing or not selection.execution_date.gt(selection.selection_date).all():
            raise ValueError("Invalid selection dates")
        if not selection.lookback_count.eq(18).all() or not set(selection.model).issubset(models):
            raise ValueError("Invalid selection rule")
        selection["score_pp"] = selection.score * 100
        weights = _read("factor_weights", ["signal", "strategy", "selected_model", *FACTORS], ["signal"])
        weights = weights.loc[weights.strategy.eq(primary)].sort_values("signal").copy()
        _finite(weights, FACTORS)
        if len(weights) != period_count or not weights.signal.is_unique or not set(weights.selected_model).issubset(models):
            raise ValueError("Incomplete monthly allocation")
        if not np.allclose(np.sort(weights[FACTORS].to_numpy(), axis=1), [0.2, 0.3, 0.5]):
            raise ValueError("Invalid factor mix")
        for row in weights.itertuples():
            available = selection.loc[selection.selection_date.le(row.signal)]
            if available.empty or available.iloc[-1].model != row.selected_model:
                raise ValueError("Selected model does not match monthly weights")
        uncertainty = _read("uncertainty", ["block_length", "mean_difference", "lower", "upper"])
        _finite(uncertainty, ["block_length", "mean_difference", "lower", "upper"])
        if set(uncertainty.block_length) != {2, 3, 6} or not uncertainty.block_length.is_unique or (uncertainty.lower > uncertainty.upper).any():
            raise ValueError("Incomplete uncertainty checks")
        seeds = _read("seed_results", ["seed", "annual_return", "sharpe", "drawdown"])
        _finite(seeds, ["seed", "annual_return", "sharpe", "drawdown"])
        if set(seeds.seed) != {7, 21, 42, 84, 123} or not seeds.seed.is_unique:
            raise ValueError("Incomplete seed checks")
        costs = _read("cost_sensitivity", ["strategy", "kind", "trading_cost_rate", "annual_borrow_cost_rate", "annual_return", "sharpe", "drawdown"])
        costs = costs.loc[costs.strategy.eq(primary)].copy()
        _finite(costs, ["trading_cost_rate", "annual_borrow_cost_rate", "annual_return", "sharpe", "drawdown"])
        if costs.empty or (costs[["trading_cost_rate", "annual_borrow_cost_rate"]] < 0).any().any():
            raise ValueError("Missing cost checks")
        costs["trading_percent"] = costs.trading_cost_rate * 100
        costs["borrow_percent"] = costs.annual_borrow_cost_rate * 100
        costs["kind_order"] = costs.kind.map({"Trading cost": 0, "Borrow cost": 1}).fillna(2)
        costs = costs.sort_values(["kind_order", "trading_cost_rate", "annual_borrow_cost_rate"])
        forecast_rows = []
        if (RESULTS / "forecast_metrics.csv").exists():
            forecast = _read("forecast_metrics", ["model", "split", "months", "mae_pp", "direction_accuracy_pct", "best_factor_hit_rate_pct"])
            forecast = forecast.loc[forecast.split.eq("evaluation")].copy()
            _finite(forecast, ["months", "mae_pp", "direction_accuracy_pct", "best_factor_hit_rate_pct"])
            if forecast.empty or not forecast.model.is_unique or not forecast.months.eq(period_count).all():
                raise ValueError("Invalid forecast comparison")
            if (forecast.mae_pp < 0).any() or not forecast.direction_accuracy_pct.between(0, 100).all() or not forecast.best_factor_hit_rate_pct.between(0, 100).all():
                raise ValueError("Invalid forecast measures")
            forecast_rows = forecast.to_dict("records")
        plot_names = list(dict.fromkeys([primary, "Equal factors", *comparisons]))[:4]
        plot = values[plot_names]
        drawdown = (plot / plot.cummax() - 1) * 100
        monthly = weights.set_index("signal")[FACTORS] * 100
        monthly.loc[end] = monthly.iloc[-1]
        estimate = uncertainty.loc[uncertainty.block_length.eq(3)].iloc[0].to_dict()
        return {
            "summary": summary, "start": start.date(), "end": end.date(), "primary": primary,
            "initial_value": initial_value, "period_count": period_count, "models": models,
            "notebook_url": "https://github.com/arianazmoudeh/CapstoneApplication/blob/main/experiments/" + notebook,
            "metrics": [{"strategy": name, **performance.loc[name].to_dict()} for name in comparisons],
            "all_metrics": performance.reset_index().to_dict("records"),
            "primary_metrics": performance.loc[primary].to_dict(),
            "annual_gap": float(performance.loc[primary, "annual_return"] - performance.loc["Equal factors", "annual_return"]),
            "selection": selection.to_dict("records"), "weights": weights.to_dict("records"),
            "value_chart": _chart(plot), "drawdown_chart": _chart(drawdown, "%"),
            "weight_chart": _chart(monthly, "%", step=True),
            "estimate": estimate, "uncertainty": uncertainty.sort_values("block_length").to_dict("records"),
            "seeds": seeds.sort_values("seed").to_dict("records"), "costs": costs.to_dict("records"),
            "seed_median": float(seeds.annual_return.median()), "seed_min": float(seeds.annual_return.min()), "seed_max": float(seeds.annual_return.max()),
            "forecast_rows": forecast_rows,
        }
    except (OSError, KeyError, TypeError, ValueError, OverflowError, AttributeError, pd.errors.ParserError) as exc:
        raise FinalStudyError("The final research results are not available right now. Please try again later.") from exc
