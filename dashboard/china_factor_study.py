from pathlib import Path

import numpy as np
import pandas as pd

from .research import chart_data


RESULTS = Path(__file__).resolve().parent / "china_factor_results"
FACTORS = ["Value", "Profitability", "Momentum"]
BASELINES = ["Broad equal stocks", "Eligible equal stocks", "Equal factors", "Historical mean"]


class FactorStudyError(ValueError):
    pass


class StudySelectionError(ValueError):
    pass


def _read(name, required, dates=()):
    frame = pd.read_csv(RESULTS / f"{name}.csv", dtype={"code": str}, keep_default_na=False)
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError(f"Invalid {name}")
    for column in dates:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError(f"Missing dates in {name}")
    return frame


def _finite(frame, columns):
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError("Invalid numeric results")


def study_context(model=None, signal=None):
    try:
        summary = _read("data_summary", ["item", "value"])
        if not summary.item.is_unique:
            raise ValueError("Duplicate summary fields")
        summary = dict(zip(summary.item, summary.value))
        performance = _read("performance", ["strategy", "annual_return", "volatility", "sharpe", "drawdown", "total_return", "final_value", "costs"])
        _finite(performance, ["annual_return", "volatility", "sharpe", "drawdown", "total_return", "final_value", "costs"])
        if not performance.strategy.is_unique or not set(BASELINES).issubset(performance.strategy):
            raise ValueError("Missing comparison portfolios")
        selection = _read("model_selection", ["model", "validation_mae", "selected"])
        _finite(selection, ["validation_mae"])
        selection["selected"] = selection.selected.astype(str).str.lower().isin(["true", "1"])
        if not selection.model.is_unique or selection.selected.sum() != 1 or not set(selection.model).issubset(performance.strategy):
            raise ValueError("Invalid model selection")
        selected_model = selection.loc[selection.selected, "model"].iloc[0]
        selection = selection.loc[~selection.model.isin(BASELINES)].copy()
        if not selection.selected.any():
            raise ValueError("Missing selected prediction model")
        models = selection.model.tolist()
        if "model_count" in summary and int(summary["model_count"]) != len(models):
            raise ValueError("Model count does not match")
        model = selected_model if model is None else model
        if model not in models:
            raise StudySelectionError("Please choose one of the models in this study.")
        predictions = _read("predictions", ["signal", "entry", "exit", "model", "factor", "predicted_return", "actual_return"], ["signal", "entry", "exit"])
        if "split" in predictions:
            predictions = predictions.loc[predictions.split.eq("evaluation")].copy()
        _finite(predictions, ["predicted_return", "actual_return"])
        if predictions.duplicated(["signal", "model", "factor"]).any():
            raise ValueError("Duplicate predictions")
        model_predictions = predictions.loc[predictions.model.eq(model)]
        dates = model_predictions.signal.sort_values().unique()
        if not len(dates):
            raise ValueError("Missing predictions")
        if "selection_end" in summary and pd.Timestamp(summary["selection_end"]) > pd.Timestamp(dates[0]):
            raise ValueError("Model selection extends into the test period")
        signal = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d") if signal is None else signal
        date_options = [{"value": pd.Timestamp(value).strftime("%Y-%m-%d"), "label": pd.Timestamp(value).strftime("%B %Y")} for value in reversed(dates)]
        if signal not in [row["value"] for row in date_options]:
            raise StudySelectionError("Please choose one of the completed months in this study.")
        month = model_predictions.loc[model_predictions.signal.eq(pd.Timestamp(signal))].set_index("factor")
        if set(month.index) != set(FACTORS) or month.entry.nunique() != 1 or month.exit.nunique() != 1:
            raise ValueError("Incomplete monthly forecasts")
        if not (month.entry.gt(month.signal) & month.exit.ge(month.entry)).all():
            raise ValueError("Invalid trading dates")
        weights = _read("factor_weights", ["signal", "model", *FACTORS], ["signal"])
        weights = weights.loc[weights.model.eq(model) & weights.signal.eq(pd.Timestamp(signal))]
        _finite(weights, FACTORS)
        if len(weights) != 1 or not np.allclose(sorted(weights.iloc[0][FACTORS].astype(float)), [0.2, 0.3, 0.5]):
            raise ValueError("Invalid factor allocation")
        order = month.loc[FACTORS, "predicted_return"].sort_values(ascending=False, kind="stable").index
        if not np.allclose(weights.iloc[0][order].astype(float), [0.5, 0.3, 0.2]):
            raise ValueError("Factor weights do not match the forecast order")
        holdings = _read("stock_weights", ["signal", "model", "code", "weight"], ["signal"])
        holdings = holdings.loc[holdings.model.eq(model) & holdings.signal.eq(pd.Timestamp(signal))].copy()
        _finite(holdings, ["weight"])
        if holdings.empty or holdings.code.duplicated().any() or holdings.weight.lt(0).any() or not np.isclose(holdings.weight.sum(), 1):
            raise ValueError("Invalid stock allocation")
        coverage = _read("coverage", ["code"])
        coverage["code"] = coverage.code.str.zfill(6)
        for name in ["name_en", "name_zh", "exchange", "status", "eligible_months"]:
            if name not in coverage:
                coverage[name] = ""
        if not coverage.code.is_unique:
            raise ValueError("Duplicate companies")
        holdings["code"] = holdings.code.str.zfill(6)
        if not set(holdings.code).issubset(coverage.code):
            raise ValueError("Unknown holding")
        coverage_rows = coverage.to_dict("records")
        company_lookup = {row["code"]: row for row in coverage_rows}
        holdings = holdings.loc[holdings.weight.gt(0)].sort_values(["weight", "code"], ascending=[False, True])
        holding_rows = [{**company_lookup[row.code], "weight": float(row.weight) * 100} for row in holdings.itertuples()]
        values = _read("portfolio_values", ["date", *BASELINES, *models], ["date"]).set_index("date")
        _finite(values, [*BASELINES, *models])
        if len(values) < 2 or not values.index.is_unique or not values.index.is_monotonic_increasing or not values[[*BASELINES, *models]].gt(0).all().all():
            raise ValueError("Invalid portfolio values")
        if not np.allclose(values.iloc[0][[*BASELINES, *models]], values.iloc[0][model], rtol=1e-9, atol=0.01):
            raise ValueError("Portfolio starting values do not match")
        metrics = performance.set_index("strategy")
        for name in [*BASELINES, *models]:
            if not np.isclose(metrics.loc[name, "final_value"], values.iloc[-1][name], rtol=1e-9, atol=0.02):
                raise ValueError("Final values do not match")
        comparisons = ["Eligible equal stocks", "Equal factors", model]
        scores = _read("factor_scores", ["signal", "code", *FACTORS], ["signal"])
        current_scores = scores.loc[scores.signal.eq(pd.Timestamp(signal))].copy()
        month_reasons = dict(zip(current_scores.code.str.zfill(6), current_scores.reason)) if "reason" in current_scores else {}
        if "eligible" in current_scores:
            current_scores = current_scores.loc[current_scores.eligible.astype(str).str.lower().isin(["true", "1"])].copy()
        _finite(current_scores, FACTORS)
        if current_scores.empty or current_scores.code.duplicated().any():
            raise ValueError("Invalid monthly stock scores")
        eligible_codes = set(current_scores.code.str.zfill(6))
        for row in coverage_rows:
            row["status"] = month_reasons.get(row["code"], "Available" if row["code"] in eligible_codes else "Not eligible")
        metric_rows = [{"name": name, **metrics.loc[name].to_dict()} for name in comparisons]
        factor_difference = metrics.loc["Equal factors", "annual_return"] - metrics.loc["Eligible equal stocks", "annual_return"]
        model_difference = metrics.loc[model, "annual_return"] - metrics.loc["Equal factors", "annual_return"]
        factor_rows = [{"name": name, "predicted_return": float(month.loc[name, "predicted_return"]) * 100, "actual_return": float(month.loc[name, "actual_return"]) * 100, "weight": float(weights.iloc[0][name]) * 100} for name in FACTORS]
        model_rows = [{"name": row.model, "selected": row.selected, "validation_mae": row.validation_mae * 100, **metrics.loc[row.model].to_dict()} for row in selection.itertuples()]
        accuracy = _read("forecast_accuracy", ["model", "accuracy", "forecasts", "mae"])
        _finite(accuracy, ["accuracy", "forecasts"])
        if not accuracy.model.is_unique or not set([*models, "Historical mean", "Always Up"]).issubset(accuracy.model):
            raise ValueError("Incomplete forecast comparison")
        accuracy = accuracy.set_index("model")
        forecast_mae = accuracy.loc[[*models, "Historical mean"], ["mae"]].copy()
        _finite(forecast_mae, ["mae"])
        if accuracy.accuracy.lt(0).any() or accuracy.accuracy.gt(100).any() or accuracy.forecasts.le(0).any():
            raise ValueError("Invalid forecast accuracy")
        forecast_result = {"accuracy": float(accuracy.loc[model, "accuracy"]),
            "always_up_accuracy": float(accuracy.loc["Always Up", "accuracy"]),
            "mae": float(forecast_mae.loc[model, "mae"]) * 100,
            "mean_mae": float(forecast_mae.loc["Historical mean", "mae"]) * 100,
            "count": int(accuracy.loc[model, "forecasts"])}
        fixed_control = None
        fixed_name = "Fixed starting factor weights"
        if fixed_name in metrics.index:
            fixed_weights = [float(summary[f"fixed_{factor.lower()}_weight"]) for factor in FACTORS]
            if not np.isclose(sum(fixed_weights), 1) or min(fixed_weights) < 0:
                raise ValueError("Invalid fixed factor weights")
            gap = float(metrics.loc[selected_model, "annual_return"] - metrics.loc[fixed_name, "annual_return"])
            fixed_control = {"model": selected_model, "dynamic": metrics.loc[selected_model].to_dict(),
                "fixed": metrics.loc[fixed_name].to_dict(), "gap": gap, "gap_abs": abs(gap),
                "different_months": int(summary["different_from_start_months"]),
                "weights": [{"name": factor, "weight": weight * 100} for factor, weight in zip(FACTORS, fixed_weights)]}
        definitions = _read("factor_comparisons", ["factor", "measure", "alternative", "annual_return", "drawdown"])
        _finite(definitions, ["annual_return", "drawdown"])
        definition_rows = []
        for factor in FACTORS:
            pair = definitions.loc[definitions.factor.eq(factor)].set_index("measure")
            if not pair.index.is_unique or set(pair.index) != {"Main measure", "Alternative measure"}:
                raise ValueError("Incomplete factor definition comparison")
            main, alternative = pair.loc["Main measure"], pair.loc["Alternative measure"]
            definition_rows.append({"factor": factor, "alternative": alternative.alternative,
                "main_return": main.annual_return, "alternative_return": alternative.annual_return,
                "main_drawdown": main.drawdown, "alternative_drawdown": alternative.drawdown})
        candidate_count = int(summary.get("candidate_count", len(coverage)))
        if len(coverage) != candidate_count:
            raise ValueError("Company count does not match")
        return {
            "summary": summary, "candidate_count": candidate_count, "model_count": len(models),
            "eligible_count": len(current_scores),
            "model": model, "selected_model": selected_model, "is_selected_model": model == selected_model,
            "models": models, "dates": date_options, "signal": pd.Timestamp(signal).date(),
            "signal_value": signal, "entry": month.entry.iloc[0].date(), "exit": month.exit.iloc[0].date(),
            "start": values.index[0].date(), "end": values.index[-1].date(), "period_count": len(dates),
            "initial_value": float(values.iloc[0][model]), "cost_rate": float(summary.get("cost_rate", 0.001)) * 100,
            "chart": chart_data(values[comparisons]), "metrics": metric_rows,
            "factor_difference": float(factor_difference), "model_difference": float(model_difference),
            "factor_difference_abs": abs(float(factor_difference)), "model_difference_abs": abs(float(model_difference)),
            "selected_metrics": metrics.loc[model].to_dict(), "broad_metrics": metrics.loc["Broad equal stocks"].to_dict(),
            "equal_factor_metrics": metrics.loc["Equal factors"].to_dict(),
            "historical_metrics": metrics.loc["Historical mean"].to_dict(), "factor_rows": factor_rows,
            "model_rows": model_rows, "holdings": holding_rows, "coverage": coverage_rows,
            "definition_rows": definition_rows,
            "forecast_result": forecast_result,
            "fixed_control": fixed_control,
            "validation_start": pd.Timestamp(summary["validation_start"]).date() if "validation_start" in summary else None,
            "validation_end": pd.Timestamp(summary["selection_end"]).date() if "selection_end" in summary else None,
        }
    except StudySelectionError:
        raise
    except (OSError, KeyError, TypeError, ValueError, OverflowError, AttributeError, pd.errors.ParserError) as exc:
        raise FactorStudyError("The saved study results could not be loaded. Please try again later.") from exc
