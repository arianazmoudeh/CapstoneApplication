from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .research import chart_data


RESULTS = Path(__file__).resolve().parent / "research_results"
CHINA_RESULTS = Path(__file__).resolve().parent / "china_research_results"
MODELS = ("Ridge", "Elastic Net", "KNN", "SVR", "Random Forest", "Extra Trees", "XGBoost", "MLP", "LSTM", "CNN")
CHINA_MODELS = MODELS + ("Prophet",)
FACTORS = ("Value", "Profitability", "Momentum")
BASELINES = ("Equal factors", "Equal stocks", "Historical mean")
INITIAL_VALUE = 100000.0
NOTEBOOK_URL = "https://github.com/arianazmoudeh/CapstoneApplication/blob/main/experiments/US_Stock_Expanded_Experiment.ipynb"
CHINA_NOTEBOOK_URL = "https://github.com/arianazmoudeh/CapstoneApplication/blob/main/experiments/China_A_Share_Monthly_Factor_Experiment.ipynb"


class FactorResearchError(ValueError):
    pass


def _require(condition):
    if not condition:
        raise FactorResearchError("The saved research results could not be loaded. Please try again later.")


@lru_cache(maxsize=2)
def _load_results(market="us"):
    try:
        _require(market in ("us", "china"))
        directory = CHINA_RESULTS if market == "china" else RESULTS
        models = CHINA_MODELS if market == "china" else MODELS
        currency = "CNY" if market == "china" else "USD"
        country = "CN" if market == "china" else "US"
        cost_column = "cost_cny" if market == "china" else "cost_usd"
        frames = {
            name: pd.read_csv(directory / f"{name}.csv", float_precision="round_trip")
            for name in ("test_schedule", "stocks", "factor_scores", "predictions", "factor_weights", "stock_weights", "portfolio_values", "performance", "forecast_accuracy", "trading_costs")
        }
        for frame in frames.values():
            _require(not frame.empty and not frame.isna().any().any())
            for column in ("signal", "entry", "exit", "date", "statement_period", "available_date"):
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise")
            numeric = frame.select_dtypes(include="number")
            _require(np.isfinite(numeric.to_numpy()).all())

        schedule = frames["test_schedule"].set_index("signal")
        _require(schedule.index.is_unique and schedule.index.is_monotonic_increasing)
        _require((schedule.entry > schedule.index).all() and (schedule.exit > schedule.entry).all())
        _require(np.array_equal(schedule.exit.iloc[:-1].to_numpy(), schedule.entry.iloc[1:].to_numpy()))
        stocks = frames["stocks"].set_index("ticker")
        _require(stocks.index.is_unique and len(stocks) == 20)
        _require(stocks.currency.eq(currency).all() and stocks.country.eq(country).all())
        scores = frames["factor_scores"].set_index(["signal", "stock"])
        _require(scores.index.is_unique)
        _require(set(scores.index) == set(pd.MultiIndex.from_product([schedule.index, stocks.index])))
        _require((scores.available_date <= scores.index.get_level_values("signal")).all())
        scores = scores.reindex(pd.MultiIndex.from_product([schedule.index, stocks.index], names=["signal", "stock"]))
        predictions = frames["predictions"].set_index(["signal", "model", "factor"]).sort_index()
        names = ("Historical mean",) + models
        expected_predictions = pd.MultiIndex.from_product([schedule.index, names, FACTORS])
        _require(predictions.index.is_unique and set(predictions.index) == set(expected_predictions))
        _require(predictions.completed_test.eq(True).all())
        for signal, row in schedule.iterrows():
            part = predictions.loc[signal]
            _require(part.entry.eq(row.entry).all() and part.exit.eq(row.exit).all())
        _require((predictions.trend.to_numpy() == np.where(predictions.predicted_return > 0, "Up", "Down")).all())

        factor_weights = frames["factor_weights"].set_index(["signal", "strategy"])
        stock_weights = frames["stock_weights"].set_index(["signal", "strategy"])
        strategies = ("Equal stocks", "Equal factors") + names
        for frame, columns, expected_names in ((factor_weights, list(FACTORS), names), (stock_weights, stocks.index.tolist(), strategies)):
            _require(frame.index.is_unique and set(frame.index) == set(pd.MultiIndex.from_product([schedule.index, expected_names])))
            _require((frame[columns] >= 0).all().all() and np.allclose(frame[columns].sum(axis=1), 1))

        for signal in schedule.index:
            buckets = pd.DataFrame(0.0, index=stocks.index, columns=FACTORS)
            for factor in FACTORS:
                selected = scores.loc[signal, factor].sort_values(ascending=False, kind="stable").head(5).index
                buckets.loc[selected, factor] = 0.2
            _require(np.allclose(stock_weights.loc[(signal, "Equal stocks")], 1 / len(stocks)))
            _require(np.allclose(stock_weights.loc[(signal, "Equal factors")], buckets.mean(axis=1)))
            for name in names:
                weights = factor_weights.loc[(signal, name), list(FACTORS)]
                ranking = predictions.loc[(signal, name)].predicted_return.reindex(FACTORS).sort_values(ascending=False, kind="stable").index
                _require(np.allclose(weights.loc[ranking], [0.5, 0.3, 0.2]))
                _require(np.allclose(stock_weights.loc[(signal, name), stocks.index], buckets.dot(weights)))

        values = frames["portfolio_values"].set_index("date")
        _require(values.index.is_unique and values.index.is_monotonic_increasing and (values > 0).all().all())
        _require(set(values.columns) == set(strategies) and np.allclose(values.iloc[0], INITIAL_VALUE))
        _require(values.index[0] == schedule.index[0] and values.index[-1] == schedule.exit.iloc[-1])
        _require(schedule.entry.isin(values.index).all() and schedule.exit.isin(values.index).all())
        performance = frames["performance"].set_index("Strategy")
        _require(performance.index.is_unique and set(performance.index) == set(strategies))
        _require(np.allclose(performance.loc[values.columns, f"Final value {currency}"], values.iloc[-1]))
        _require(np.allclose(performance.loc[values.columns, "Cumulative return %"], (values.iloc[-1] / INITIAL_VALUE - 1) * 100))
        costs = frames["trading_costs"].set_index(["signal", "strategy"])
        _require(costs.index.is_unique and set(costs.index) == set(stock_weights.index))
        _require((costs[cost_column] >= 0).all())
        totals = costs.groupby("strategy")[cost_column].sum()
        _require(np.allclose(performance.loc[totals.index, f"Trading costs {currency}"], totals))
        for signal, row in schedule.iterrows():
            part = costs.loc[signal]
            _require(part.entry.eq(row.entry).all())
            if market == "china":
                _require(part.exit.eq(row.exit).all())
                _require((part.post_trade_value_cny > 0).all())
                _require(np.allclose(part.pre_trade_value_cny - part.cost_cny, part.post_trade_value_cny))
                _require(np.allclose(part.post_trade_value_cny, values.loc[row.entry, part.index]))
        accuracy = frames["forecast_accuracy"].set_index("Model")
        _require(accuracy.index.is_unique and set(accuracy.index) == set(names))
        _require(accuracy.Forecasts.eq(len(schedule) * len(FACTORS)).all())
        return {
            "schedule": schedule, "stocks": stocks, "scores": scores,
            "predictions": predictions, "factor_weights": factor_weights,
            "stock_weights": stock_weights, "values": values,
            "performance": performance, "accuracy": accuracy, "costs": costs,
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise FactorResearchError("The saved research results could not be loaded. Please try again later.") from exc


def research_options(market="us"):
    schedule = _load_results(market)["schedule"]
    models = CHINA_MODELS if market == "china" else MODELS
    return {
        "models": [{"value": model, "label": model} for model in models],
        "dates": [{"value": signal.strftime("%Y-%m-%d"), "label": signal.strftime("%b %Y" if market == "china" else "%d %b %Y")} for signal in reversed(schedule.index)],
        "default_date": schedule.index[-1].strftime("%Y-%m-%d"),
    }


def build_research(model, signal, amount, market="us"):
    data = _load_results(market)
    models = CHINA_MODELS if market == "china" else MODELS
    currency = "CNY" if market == "china" else "USD"
    currency_symbol = "¥" if market == "china" else "$"
    try:
        starting_amount = Decimal(str(amount))
        _require(starting_amount.is_finite() and Decimal("1") <= starting_amount <= Decimal("1000000000"))
        _require(starting_amount == starting_amount.quantize(Decimal("0.01")))
        signal_date = pd.Timestamp(signal)
        _require(model in models and signal_date in data["schedule"].index)
    except (ValueError, TypeError, InvalidOperation) as exc:
        raise ValueError(f"Choose a listed model and date, and enter an investment between {currency_symbol}1 and {currency_symbol}1,000,000,000 with up to two decimal places.") from exc

    amount = float(starting_amount)
    scale = amount / INITIAL_VALUE
    schedule = data["schedule"].loc[signal_date]
    snapshot_value = float(data["values"].loc[schedule.entry, model]) * scale
    if market == "china":
        snapshot_value = float(data["costs"].loc[(signal_date, model), "post_trade_value_cny"]) * scale
    scores = data["scores"].loc[signal_date]
    ranks = scores[list(FACTORS)].rank(ascending=False, method="first")
    forecasts = data["predictions"].loc[(signal_date, model)].reindex(FACTORS)
    factor_weights = data["factor_weights"].loc[(signal_date, model)]
    factor_rows = []
    for factor in FACTORS:
        forecast = forecasts.loc[factor]
        weight = float(factor_weights[factor])
        selected = ranks[factor].sort_values().head(5).index.tolist()
        factor_rows.append({
            "name": factor, "predicted_return": float(forecast.predicted_return) * 100,
            "actual_return": float(forecast.actual_return) * 100, "trend": forecast.trend,
            "weight": weight * 100, "amount": weight * snapshot_value, "selected": selected,
        })

    rankings = []
    for symbol in ranks.Value.sort_values().index:
        stock = data["stocks"].loc[symbol]
        row = {"symbol": symbol, "name": stock["name"], "sector": stock.sector}
        for factor in FACTORS:
            field = factor.lower()
            row[field] = float(scores.loc[symbol, factor]) * (1 if factor == "Momentum" else 100)
            row[f"{field}_rank"] = int(ranks.loc[symbol, factor])
            row[f"{field}_selected"] = bool(ranks.loc[symbol, factor] <= 5)
        rankings.append(row)

    holdings = []
    weights = data["stock_weights"].loc[(signal_date, model)].sort_values(ascending=False, kind="stable")
    for symbol, weight in weights.items():
        if weight > 0:
            stock = data["stocks"].loc[symbol]
            holdings.append({
                "symbol": symbol, "name": stock["name"], "sector": stock.sector,
                "weight": float(weight) * 100, "amount": float(weight) * snapshot_value,
            })

    comparison = [model, *BASELINES]
    metrics = []
    for name in comparison:
        result = data["performance"].loc[name]
        metrics.append({
            "name": name, "annual_return": float(result["Annual return %"]),
            "total_return": float(result["Cumulative return %"]),
            "volatility": float(result["Volatility %"]), "drawdown": float(result["Max drawdown %"]),
            "sharpe": float(result["Sharpe (0% reference)"]),
            "final_value": float(result[f"Final value {currency}"]) * scale,
            "costs": float(result[f"Trading costs {currency}"]) * scale,
        })

    accuracy = data["accuracy"].loc[model]
    chart = chart_data(data["values"][comparison] * scale, currency=True)
    for tick in chart["ticks"]:
        value = float(tick["label"].replace("$", "").replace(",", ""))
        if value >= 1_000_000_000:
            tick["label"] = f"{currency_symbol}{value / 1_000_000_000:.1f}bn"
        elif value >= 1_000_000:
            tick["label"] = f"{currency_symbol}{value / 1_000_000:.1f}m"
        else:
            tick["label"] = tick["label"].replace("$", currency_symbol)
    return {
        "model": model, "signal": signal_date.date(), "entry": schedule.entry.date(), "exit": schedule.exit.date(),
        "start": data["values"].index[0].date(), "end": data["values"].index[-1].date(),
        "amount": amount, "snapshot_value": snapshot_value,
        "chart": chart,
        "metrics": metrics, "selected_metrics": metrics[0], "factor_rows": factor_rows,
        "holdings": holdings, "rankings": rankings,
        "accuracy": {
            "direction_accuracy": float(accuracy["Direction accuracy %"]),
            "balanced_accuracy": float(accuracy["Balanced accuracy %"]),
            "down_recall": float(accuracy["Down recall %"]),
            "predicted_up": float(accuracy["Predicted Up %"]),
            "forecasts": int(accuracy["Forecasts"]), "mae": float(accuracy["MAE percentage points"]),
        },
        "stock_count": len(data["stocks"]), "model_count": len(models), "period_count": len(data["schedule"]),
        "cost_rate": 0.1, "notebook_url": CHINA_NOTEBOOK_URL if market == "china" else NOTEBOOK_URL,
        "market": market, "currency": currency, "currency_symbol": currency_symbol,
        "period_label": "monthly" if market == "china" else "four-week",
        "prediction_label": "Predicted next-month return" if market == "china" else "Predicted four-week return",
        "definitions": [
            {"name": "Value", "formula": "Earnings yield: annual net income divided by estimated market value."},
            {"name": "Profitability", "formula": "Net profit margin: annual net income divided by revenue."},
            {"name": "Momentum", "formula": "Return from 52 weeks ago to 4 weeks ago, divided by annualized 52-week volatility."},
        ],
    }
