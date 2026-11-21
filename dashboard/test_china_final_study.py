import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .china_final_study import FinalStudyError, final_study_context


class FinalStudyTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username="final-study-reader", password="test")

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.patch = patch("dashboard.china_final_study.RESULTS", self.path)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        summary = {
            "evaluation_start": "2025-01-02", "evaluation_end": "2025-03-03",
            "period_count": 2, "initial_value": 100000, "models": ["Ridge", "Prophet"],
            "model_count": 2, "company_count": 100, "primary_strategy": "Six-month selector",
            "comparison_strategies": ["Six-month selector", "Equal factors", "First model", "First weights"],
            "notebook_filename": "China_A_Share_Final.ipynb",
            "findings": ["The selector earned more in this sample."],
            "limitations": ["The same evaluation history was already examined."],
        }
        (self.path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        self.strategies = [*summary["comparison_strategies"], *summary["models"]]
        self.save("performance", [
            {"strategy": name, "annual_return": 1 + i, "volatility": 10, "sharpe": 0.2,
             "drawdown": -8, "final_value": 101000 + i * 100, "trading_costs": 100,
             "borrow_costs": 50, "average_monthly_turnover": 0.3}
            for i, name in enumerate(self.strategies)
        ])
        self.save("portfolio_values", [
            {"date": "2024-12-31", **{name: 100000 for name in self.strategies}},
            {"date": "2025-02-03", **{name: 92000 for name in self.strategies}},
            {"date": "2025-03-03", **{name: 101000 + i * 100 for i, name in enumerate(self.strategies)}},
        ])
        self.save("selection", [{"selection_date": "2024-12-31", "execution_date": "2025-01-02", "model": "Ridge", "score": 0.001, "lookback_count": 18}])
        self.save("factor_weights", [{"signal": signal, "strategy": "Six-month selector", "selected_model": "Ridge", "Value": 0.5, "Profitability": 0.3, "Momentum": 0.2} for signal in ["2024-12-31", "2025-01-31"]])
        self.save("uncertainty", [{"block_length": block, "mean_difference": 0.1, "lower": -0.4, "upper": 0.6} for block in [2, 3, 6]])
        self.save("seed_results", [{"seed": seed, "annual_return": i, "sharpe": 0.2, "drawdown": -8} for i, seed in enumerate([7, 21, 42, 84, 123])])
        self.save("cost_sensitivity", [{"strategy": "Six-month selector", "kind": "Main costs", "trading_cost_rate": 0.001, "annual_borrow_cost_rate": 0.03, "annual_return": 1, "sharpe": 0.2, "drawdown": -8}])
        self.client.force_login(self.user)

    def save(self, name, rows):
        pd.DataFrame(rows).to_csv(self.path / f"{name}.csv", index=False)

    @patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected online request"))
    def test_page_reads_saved_results_and_keeps_negative_drawdowns_visible(self, request):
        result = final_study_context()
        self.assertEqual(result["selection"][0]["score_pp"], 0.1)
        self.assertEqual(result["costs"][0]["trading_percent"], 0.1)
        self.assertEqual(result["seed_median"], 2)
        self.assertTrue(any(tick["label"].startswith("-") for tick in result["drawdown_chart"]["ticks"]))
        for series in result["drawdown_chart"]["lines"]:
            for point in series["points"].split():
                self.assertTrue(20 <= float(point.split(",")[1]) <= 235)
        response = self.client.get(reverse("china_final_study"))
        self.assertContains(response, "Final factor timing study")
        self.assertContains(response, "What we found")
        self.assertContains(response, "The selector earned more in this sample.")
        self.assertContains(response, "The same evaluation history was already examined.")
        self.assertContains(response, "0.100 pp")
        request.assert_not_called()

    def test_mismatched_final_value_is_not_published(self):
        path = self.path / "performance.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "final_value"] = 200000
        frame.to_csv(path, index=False)
        with self.assertRaises(FinalStudyError):
            final_study_context()

    def test_monthly_weights_cannot_use_a_model_chosen_later(self):
        path = self.path / "factor_weights.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "selected_model"] = "Prophet"
        frame.to_csv(path, index=False)
        with self.assertRaises(FinalStudyError):
            final_study_context()

    def test_missing_seed_is_not_shown_as_complete_research(self):
        path = self.path / "seed_results.csv"
        frame = pd.read_csv(path).iloc[:-1]
        frame.to_csv(path, index=False)
        with self.assertRaises(FinalStudyError):
            final_study_context()

    def test_missing_results_show_clear_error(self):
        (self.path / "summary.json").unlink()
        response = self.client.get(reverse("china_final_study"))
        self.assertContains(response, "The final research results are not available", status_code=503)
        self.assertNotContains(response, "What we found", status_code=503)

    def test_get_only_and_login_required(self):
        self.assertEqual(self.client.post(reverse("china_final_study")).status_code, 405)
        self.client.logout()
        self.assertEqual(self.client.get(reverse("china_final_study")).status_code, 302)
