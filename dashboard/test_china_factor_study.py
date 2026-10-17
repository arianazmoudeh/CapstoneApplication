from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .china_factor_study import BASELINES, FACTORS, FactorStudyError, StudySelectionError, study_context
from .models import Portfolio


class ChinaFactorStudyTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username="factor-study-user", password="test")
        Portfolio.objects.create(user=cls.user, budget=0, weights={}, quantities={"AAPL": "2"})

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.patch = patch("dashboard.china_factor_study.RESULTS", self.path)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.models = ["Ridge", "Random Forest"]
        strategies = [*BASELINES, *self.models]
        self.save("data_summary", [{"item": "candidate_count", "value": 2}, {"item": "cost_rate", "value": 0.001}])
        self.save("performance", [{"strategy": name, "annual_return": i + 10, "volatility": 15, "sharpe": 0.8, "drawdown": -10, "total_return": i + 1, "final_value": 101000 + i * 1000, "costs": 120} for i, name in enumerate(strategies)])
        self.save("portfolio_values", [{"date": "2024-12-31", **{name: 100000 for name in strategies}}, {"date": "2025-01-31", **{name: 101000 + i * 1000 for i, name in enumerate(strategies)}}])
        self.save("model_selection", [{"model": "Ridge", "validation_mae": 0.03, "selected": True}, {"model": "Random Forest", "validation_mae": 0.04, "selected": False}])
        self.save("predictions", [{"signal": "2024-12-31", "entry": "2025-01-02", "exit": "2025-01-31", "model": model, "factor": factor, "predicted_return": 0.03 - i * 0.01, "actual_return": 0.02 + i * 0.01} for model in self.models for i, factor in enumerate(FACTORS)])
        self.save("factor_weights", [{"signal": "2024-12-31", "model": model, "Value": 0.5, "Profitability": 0.3, "Momentum": 0.2} for model in self.models])
        self.save("stock_weights", [{"signal": "2024-12-31", "model": model, "code": code, "weight": 0.6 if code == "000001" else 0.4} for model in self.models for code in ["000001", "600001"]])
        self.save("factor_scores", [{"signal": "2024-12-31", "code": code, "Value": 0.1, "Profitability": 0.2, "Momentum": 0.3} for code in ["000001", "600001"]])
        self.save("coverage", [{"code": "000001", "name_en": "Company A", "name_zh": "公司甲", "eligible_months": 1, "status": "Eligible"}, {"code": "600001", "name_en": "Company B", "name_zh": "公司乙", "eligible_months": 1, "status": "Eligible"}])
        self.save("factor_comparisons", [{"factor": factor, "measure": measure, "alternative": alternative, "annual_return": 12 if measure == "Main measure" else 11, "drawdown": -10} for factor, alternative in zip(FACTORS, ["Book to market", "EBIT to book equity", "Ordinary momentum"]) for measure in ["Main measure", "Alternative measure"]])
        self.save("forecast_accuracy", [{"model": model, "accuracy": 50, "forecasts": 3, "mae": 0.04 if model != "Always Up" else ""} for model in [*self.models, "Historical mean", "Always Up"]])
        self.client.force_login(self.user)

    def save(self, name, rows):
        pd.DataFrame(rows).to_csv(self.path / f"{name}.csv", index=False)

    def test_comparisons_use_chosen_model_and_three_matching_curves(self):
        result = study_context()
        self.assertEqual(result["model"], "Ridge")
        self.assertEqual(result["selected_model"], "Ridge")
        self.assertEqual(result["factor_difference"], 1)
        self.assertEqual(result["model_difference"], 2)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["eligible_count"], 2)
        self.assertEqual([row["name"] for row in result["chart"]["lines"]], ["Eligible equal stocks", "Equal factors", "Ridge"])
        self.assertEqual(result["factor_rows"][0]["predicted_return"], 3)
        self.assertEqual([row["weight"] for row in result["factor_rows"]], [50, 30, 20])
        self.assertEqual(result["holdings"][0]["code"], "000001")
        self.assertEqual(sum(row["weight"] for row in result["holdings"]), 100)
        self.assertEqual(result["model_rows"][0]["validation_mae"], 3)
        self.assertEqual(len(result["definition_rows"]), 3)
        self.assertEqual(result["definition_rows"][0]["alternative_return"], 11)
        self.assertEqual(result["forecast_result"]["mae"], 4)

    def test_validation_months_and_ineligible_scores_are_not_shown_as_test_results(self):
        selection_path = self.path / "model_selection.csv"
        selection = pd.read_csv(selection_path)
        selection.loc[len(selection)] = ["Historical mean", 0.02, False]
        selection.to_csv(selection_path, index=False)
        prediction_path = self.path / "predictions.csv"
        predictions = pd.read_csv(prediction_path)
        predictions["split"] = "evaluation"
        validation = predictions.copy()
        validation["split"] = "validation"
        validation["signal"] = "2024-11-29"
        pd.concat([predictions, validation]).to_csv(prediction_path, index=False)
        scores_path = self.path / "factor_scores.csv"
        scores = pd.read_csv(scores_path, dtype={"code": str})
        scores["eligible"] = True
        scores.loc[len(scores)] = ["2024-12-31", "000003", float("nan"), float("nan"), float("nan"), False]
        scores.to_csv(scores_path, index=False)
        result = study_context()
        self.assertEqual(result["period_count"], 1)
        self.assertEqual(result["model_count"], 2)
        self.assertEqual(result["eligible_count"], 2)

    @patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected online request"))
    def test_page_is_read_only_and_has_clear_comparisons(self, request):
        before = list(Portfolio.objects.values())
        response = self.client.get(reverse("china_factor_study"))
        self.assertContains(response, "What we found")
        self.assertContains(response, "Equal stock weights")
        self.assertContains(response, "Chosen before the test")
        self.assertContains(response, "Portfolio value in CNY")
        self.assertContains(response, "Company A")
        self.assertEqual(list(Portfolio.objects.values()), before)
        request.assert_not_called()

    def test_model_selection_does_not_change_preselected_model(self):
        result = study_context("Random Forest")
        self.assertEqual(result["model"], "Random Forest")
        self.assertEqual(result["selected_model"], "Ridge")
        self.assertFalse(result["is_selected_model"])
        self.assertEqual(result["model_difference"], 3)

    def test_fixed_control_stays_with_model_chosen_before_test(self):
        path = self.path / "performance.csv"
        frame = pd.read_csv(path)
        fixed = frame.loc[frame.strategy.eq("Ridge")].copy()
        fixed["strategy"] = "Fixed starting factor weights"
        fixed["annual_return"] = 13
        pd.concat([frame, fixed]).to_csv(path, index=False)
        path = self.path / "data_summary.csv"
        summary = pd.read_csv(path)
        for item, value in [("fixed_value_weight", 0.5), ("fixed_profitability_weight", 0.3), ("fixed_momentum_weight", 0.2), ("different_from_start_months", 0)]:
            summary.loc[len(summary)] = [item, value]
        summary.to_csv(path, index=False)
        result = study_context("Random Forest")
        self.assertEqual(result["model"], "Random Forest")
        self.assertEqual(result["fixed_control"]["model"], "Ridge")
        self.assertEqual(result["fixed_control"]["dynamic"]["annual_return"], 14)
        self.assertEqual(result["fixed_control"]["gap"], 1)

    def test_invalid_model_and_month_are_rejected(self):
        for model, signal in [("../settings", None), ("Ridge", "2025-02-28"), ("Ridge", "bad-date")]:
            with self.subTest(model=model, signal=signal):
                with self.assertRaises(StudySelectionError):
                    study_context(model, signal)
        response = self.client.get(reverse("china_factor_study"), {"model": "Unknown"})
        self.assertEqual(response.status_code, 400)
        self.assertIsNone(response.context.get("study"))

    def test_invalid_saved_numbers_and_allocations_do_not_render(self):
        path = self.path / "stock_weights.csv"
        original = pd.read_csv(path, dtype={"code": str})
        broken = original.copy()
        broken.loc[0, "weight"] = 0.8
        broken.to_csv(path, index=False)
        with self.assertRaises(FactorStudyError):
            study_context()
        original.to_csv(path, index=False)
        performance = pd.read_csv(self.path / "performance.csv")
        performance.loc[performance.strategy.eq("Ridge"), "final_value"] = 1
        performance.to_csv(self.path / "performance.csv", index=False)
        with self.assertRaises(FactorStudyError):
            study_context()

    def test_missing_results_show_clear_error(self):
        (self.path / "predictions.csv").unlink()
        response = self.client.get(reverse("china_factor_study"))
        self.assertContains(response, "The saved study results could not be loaded.", status_code=503)

    def test_model_choice_cannot_use_dates_after_the_first_test_signal(self):
        path = self.path / "data_summary.csv"
        summary = pd.read_csv(path)
        summary.loc[len(summary)] = ["selection_end", "2025-01-31"]
        summary.to_csv(path, index=False)
        with self.assertRaises(FactorStudyError):
            study_context()

    def test_access_requires_login_and_get(self):
        self.assertEqual(self.client.post(reverse("china_factor_study")).status_code, 405)
        self.client.logout()
        response = self.client.get(reverse("china_factor_study"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)
