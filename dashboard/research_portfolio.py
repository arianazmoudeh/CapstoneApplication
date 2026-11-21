import math
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_FLOOR

import numpy as np
import pandas as pd

from .china_factor_study import FACTORS, RESULTS, FactorStudyError


def portfolio_snapshot(study):
    try:
        signal = study["signal"].isoformat()
        scores = pd.read_csv(RESULTS / "factor_scores.csv", dtype={"code": str}, keep_default_na=False)
        scores = scores.loc[scores.signal.eq(signal) & scores.eligible.astype(str).str.lower().isin(["true", "1"])].copy()
        scores["code"] = scores.code.str.zfill(6)
        scores[FACTORS] = scores[FACTORS].apply(pd.to_numeric, errors="raise")
        if scores.empty or scores.code.duplicated().any() or not np.isfinite(scores[FACTORS].to_numpy()).all():
            raise ValueError("Invalid factor scores")
        if len(scores) != study["eligible_count"]:
            raise ValueError("Eligible company counts do not match")
        top_fraction = float(study["summary"]["top_fraction"])
        if not 0 < top_fraction <= 1:
            raise ValueError("Invalid factor selection size")
        count = math.ceil(len(scores) * top_fraction)
        factor_rows = [{"name": row["name"], "weight": float(row["weight"]), "predicted_return": float(row["predicted_return"])} for row in study["factor_rows"]]
        weights = {row["name"]: row["weight"] for row in factor_rows}
        if len(factor_rows) != len(FACTORS) or set(weights) != set(FACTORS):
            raise ValueError("Incomplete factor weights")
        if any(not math.isfinite(row["weight"]) or row["weight"] <= 0 or not math.isfinite(row["predicted_return"]) for row in factor_rows):
            raise ValueError("Invalid factor weights")
        if not math.isclose(sum(weights.values()), 100, abs_tol=1e-8):
            raise ValueError("Factor weights do not add to one hundred")
        members = {}
        expected = {}
        for factor in FACTORS:
            codes = scores.sort_values(factor, ascending=False, kind="stable").head(count).code
            for code in codes:
                members.setdefault(code, []).append(factor)
                expected[code] = expected.get(code, 0) + weights[factor] / count
        holdings = []
        for row in study["holdings"]:
            code = str(row["code"]).zfill(6)
            weight = float(row["weight"])
            if code not in expected or not math.isfinite(weight) or not math.isclose(weight, expected[code], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("Factor membership does not match the saved stock weights")
            holdings.append({"code": code, "exchange": str(row["exchange"]), "name_en": str(row["name_en"]), "name_zh": str(row["name_zh"]), "weight": weight, "factors": members[code]})
        codes = [row["code"] for row in holdings]
        if len(codes) != len(set(codes)) or set(codes) != set(expected):
            raise ValueError("Incomplete stock weights")
        snapshot = {
            "model": str(study["model"]), "selected_model": str(study["selected_model"]),
            "signal": signal, "entry": study["entry"].isoformat(), "exit": study["exit"].isoformat(),
            "factor_rows": factor_rows, "holdings": holdings, "eligible_count": int(study["eligible_count"]),
        }
        if not date.fromisoformat(snapshot["signal"]) < date.fromisoformat(snapshot["entry"]) <= date.fromisoformat(snapshot["exit"]):
            raise ValueError("Invalid portfolio dates")
        return snapshot
    except (OSError, KeyError, TypeError, ValueError, OverflowError, AttributeError, pd.errors.ParserError) as exc:
        raise FactorStudyError("The portfolio could not be created from the saved research results.") from exc


def _allocate_money(rows, budget):
    weights = [Decimal(str(row["weight"])) for row in rows]
    if not weights or any(not weight.is_finite() or weight <= 0 for weight in weights):
        raise ValueError("Invalid portfolio weights")
    total_weight = sum(weights)
    if abs(total_weight - Decimal(100)) > Decimal("0.000001"):
        raise ValueError("Portfolio weights do not add to one hundred")
    total_cents = int(budget * 100)
    exact = [Decimal(total_cents) * weight / total_weight for weight in weights]
    cents = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in exact]
    order = sorted(range(len(rows)), key=lambda i: exact[i] - cents[i], reverse=True)
    for index in order[:total_cents - sum(cents)]:
        cents[index] += 1
    for row, amount in zip(rows, cents):
        row["amount"] = Decimal(amount).scaleb(-2)


def portfolio_context(snapshot, budget):
    try:
        budget = Decimal(budget)
        if not budget.is_finite() or budget <= 0 or budget != budget.quantize(Decimal("0.01")):
            raise ValueError("Enter a positive budget with no more than two decimal places")
        result = deepcopy(snapshot)
        for key in ["signal", "entry", "exit"]:
            result[key] = date.fromisoformat(result[key])
        if not result["signal"] < result["entry"] <= result["exit"]:
            raise ValueError("Invalid portfolio dates")
        factors = [row["name"] for row in result["factor_rows"]]
        codes = [row["code"] for row in result["holdings"]]
        if len(factors) != len(FACTORS) or set(factors) != set(FACTORS) or len(codes) != len(set(codes)):
            raise ValueError("Invalid saved portfolio")
        for row in result["holdings"]:
            if not row["factors"] or len(set(row["factors"])) != len(row["factors"]) or not set(row["factors"]).issubset(FACTORS):
                raise ValueError("Invalid saved factor membership")
        _allocate_money(result["factor_rows"], budget)
        _allocate_money(result["holdings"], budget)
        result["budget"] = budget.quantize(Decimal("0.01"))
        result["company_count"] = len(result["holdings"])
        return result
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError) as exc:
        raise FactorStudyError("The saved research portfolio could not be loaded.") from exc
