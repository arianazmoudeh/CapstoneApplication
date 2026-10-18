import json
from copy import deepcopy
from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from .china_factor_study import FACTORS, RESULTS, FactorStudyError, study_context
from .research_portfolio import portfolio_context, portfolio_snapshot


class ResearchPortfolioDataTests(SimpleTestCase):
    def fixture(self):
        scores = pd.DataFrame({
            "signal": ["2025-01-31"] * 4,
            "code": ["000003", "000001", "000002", "000004"],
            "eligible": [True, True, True, False],
            "Value": [10, 10, 0, 100],
            "Profitability": [1, 3, 2, 100],
            "Momentum": [3, 2, 1, 100],
        })
        study = {
            "model": "CNN", "selected_model": "CNN", "signal": date(2025, 1, 31),
            "entry": date(2025, 2, 3), "exit": date(2025, 3, 3),
            "eligible_count": 3, "summary": {"top_fraction": "0.5"},
            "factor_rows": [{"name": factor, "weight": weight, "predicted_return": weight / 100} for factor, weight in zip(FACTORS, [20, 30, 50])],
            "holdings": [{"code": code, "exchange": "SZ", "name_en": code, "name_zh": code, "weight": weight} for code, weight in [("000001", 50), ("000003", 35), ("000002", 15)]],
        }
        return scores, study

    def test_ceil_selection_and_overlapping_factor_contributions(self):
        scores, study = self.fixture()
        with patch("dashboard.research_portfolio.pd.read_csv", return_value=scores):
            snapshot = portfolio_snapshot(study)
        self.assertEqual(snapshot["holdings"][0]["factors"], FACTORS)
        self.assertEqual(snapshot["holdings"][1]["factors"], ["Value", "Momentum"])
        self.assertEqual(snapshot["holdings"][2]["factors"], ["Profitability"])
        self.assertNotIn("000004", [row["code"] for row in snapshot["holdings"]])
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)

    def test_ties_follow_the_saved_company_order(self):
        scores, study = self.fixture()
        study["summary"]["top_fraction"] = "0.3333333333333333"
        study["holdings"] = [dict(study["holdings"][1], weight=70), dict(study["holdings"][0], weight=30)]
        with patch("dashboard.research_portfolio.pd.read_csv", return_value=scores):
            snapshot = portfolio_snapshot(study)
        self.assertEqual(snapshot["holdings"][0]["code"], "000003")
        self.assertEqual(snapshot["holdings"][0]["factors"], ["Value", "Momentum"])

    def test_mismatch_and_missing_holdings_are_rejected(self):
        scores, study = self.fixture()
        for mode in ["weight", "missing", "duplicate", "eligible"]:
            changed = deepcopy(study)
            if mode == "weight":
                changed["holdings"][0]["weight"] = 49
            elif mode == "missing":
                changed["holdings"].pop()
            elif mode == "duplicate":
                changed["holdings"].append(changed["holdings"][0])
            else:
                changed["eligible_count"] = 4
            with self.subTest(mode=mode), patch("dashboard.research_portfolio.pd.read_csv", return_value=scores), self.assertRaises(FactorStudyError):
                portfolio_snapshot(changed)

    def test_rounding_keeps_factor_and_company_amounts_equal_to_budget(self):
        scores, study = self.fixture()
        with patch("dashboard.research_portfolio.pd.read_csv", return_value=scores):
            snapshot = portfolio_snapshot(study)
        unchanged = deepcopy(snapshot)
        for budget in [Decimal("0.01"), Decimal("1.03"), Decimal("100000.07")]:
            with self.subTest(budget=budget):
                result = portfolio_context(snapshot, budget)
                self.assertEqual(sum(row["amount"] for row in result["holdings"]), budget)
                self.assertEqual(sum(row["amount"] for row in result["factor_rows"]), budget)
                for row in result["holdings"] + result["factor_rows"]:
                    self.assertEqual(row["amount"], row["amount"].quantize(Decimal("0.01")))
                    self.assertLess(abs(row["amount"] - budget * Decimal(str(row["weight"])) / 100), Decimal("0.01"))
                self.assertEqual(result["signal"], study["signal"])
                self.assertEqual(result["company_count"], 3)
        self.assertEqual(snapshot, unchanged)

    def test_invalid_budgets_and_saved_weights_are_rejected(self):
        scores, study = self.fixture()
        with patch("dashboard.research_portfolio.pd.read_csv", return_value=scores):
            snapshot = portfolio_snapshot(study)
        for budget in [Decimal("0"), Decimal("-1"), Decimal("1.001"), Decimal("NaN"), Decimal("Infinity")]:
            with self.subTest(budget=budget), self.assertRaises(FactorStudyError):
                portfolio_context(snapshot, budget)
        for weight in [-1, 0, 30, float("inf")]:
            changed = deepcopy(snapshot)
            changed["holdings"][0]["weight"] = weight
            with self.subTest(weight=weight), self.assertRaises(FactorStudyError):
                portfolio_context(changed, Decimal("100"))

    def test_every_saved_model_and_month_matches_reconstructed_membership(self):
        initial = study_context()
        factor_weights = pd.read_csv(RESULTS / "factor_weights.csv", keep_default_na=False)
        holdings = pd.read_csv(RESULTS / "stock_weights.csv", dtype={"code": str}, keep_default_na=False)
        predictions = pd.read_csv(RESULTS / "predictions.csv", keep_default_na=False)
        predictions = predictions.loc[predictions.split.eq("evaluation")]
        scores = pd.read_csv(RESULTS / "factor_scores.csv", dtype={"code": str}, keep_default_na=False)
        coverage = {row["code"]: row for row in initial["coverage"]}
        cases = 0
        for model in initial["models"]:
            for month in initial["dates"]:
                signal = month["value"]
                forecast = predictions.loc[predictions.model.eq(model) & predictions.signal.eq(signal)].set_index("factor")
                weights = factor_weights.loc[factor_weights.model.eq(model) & factor_weights.signal.eq(signal)].iloc[0]
                selected = holdings.loc[holdings.model.eq(model) & holdings.signal.eq(signal)]
                eligible = scores.loc[scores.signal.eq(signal) & scores.eligible.astype(str).str.lower().eq("true")]
                study = {
                    **initial, "model": model, "signal": date.fromisoformat(signal),
                    "entry": date.fromisoformat(forecast.entry.iloc[0]), "exit": date.fromisoformat(forecast.exit.iloc[0]),
                    "eligible_count": len(eligible),
                    "factor_rows": [{"name": factor, "weight": float(weights[factor]) * 100, "predicted_return": float(forecast.loc[factor, "predicted_return"]) * 100} for factor in FACTORS],
                    "holdings": [{**coverage[row.code], "weight": float(row.weight) * 100} for row in selected.itertuples()],
                }
                with self.subTest(model=model, signal=signal):
                    snapshot = portfolio_snapshot(study)
                    result = portfolio_context(snapshot, Decimal("100000.07"))
                    self.assertEqual(len(snapshot["holdings"]), len(selected))
                    self.assertTrue(any(len(row["factors"]) > 1 for row in snapshot["holdings"]))
                    self.assertEqual(sum(row["amount"] for row in result["holdings"]), Decimal("100000.07"))
                    self.assertEqual(sum(row["amount"] for row in result["factor_rows"]), Decimal("100000.07"))
                cases += 1
        self.assertEqual(cases, 231)
