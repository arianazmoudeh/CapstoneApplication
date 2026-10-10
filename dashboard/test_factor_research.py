import csv
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from .factor_research import FactorResearchError, build_research, research_options
from .models import Portfolio, PriceSnapshot


RESULTS = Path(__file__).resolve().parent.parent / "results" / "us_stock_expanded"


def source_rows(filename):
    with (RESULTS / filename).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class FactorResearchTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username="factor-researcher", password="test")

    def setUp(self):
        self.client.force_login(self.user)

    def test_only_ten_models_and_completed_dates_can_be_selected(self):
        options = research_options()
        self.assertEqual(len(options["models"]), 10)
        self.assertEqual(len(options["dates"]), 42)
        self.assertEqual(options["default_date"], "2026-02-27")
        self.assertEqual(
            {row["value"] for row in options["dates"]},
            {row["signal"] for row in source_rows("test_schedule.csv")},
        )
        self.assertNotIn("2026-04-17", {row["value"] for row in options["dates"]})
        self.assertNotIn("Historical mean", {row["value"] for row in options["models"]})

    @patch("dashboard.eulerpool.request_data", side_effect=AssertionError("Unexpected online request"))
    def test_page_loads_saved_results_without_market_calls_or_portfolio_changes(self, request_data):
        before = (Portfolio.objects.count(), PriceSnapshot.objects.count())
        response = self.client.get("/research/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "US stock research")
        self.assertContains(response, "Factor predictions and allocation")
        self.assertContains(response, "Full test period")
        self.assertNotContains(response, "Emerging market research")
        result = response.context["result"]
        self.assertEqual(result["model"], "LSTM")
        self.assertEqual(result["signal"].isoformat(), "2026-02-27")
        self.assertEqual(result["amount"], 100000)
        self.assertEqual(result["stock_count"], 20)
        self.assertEqual((Portfolio.objects.count(), PriceSnapshot.objects.count()), before)
        request_data.assert_not_called()

    @patch("dashboard.views.research_options", side_effect=FactorResearchError("The saved research results could not be loaded. Please try again later."))
    def test_unavailable_saved_results_show_a_service_error(self, research_options_mock):
        response = self.client.get("/research/")
        self.assertContains(response, "The saved research results could not be loaded.", status_code=503)
        self.assertIsNone(response.context.get("result"))

    def test_partial_selection_uses_defaults_and_post_is_not_a_calculation(self):
        response = self.client.get("/research/", {"model": "XGBoost"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["result"]["model"], "XGBoost")
        self.assertEqual(response.context["result"]["signal"].isoformat(), "2026-02-27")
        self.assertEqual(self.client.post("/research/", {"model": "LSTM"}).status_code, 405)

    def test_invalid_controls_return_field_errors(self):
        cases = [
            ("model", "../config/local_settings.py"),
            ("model", "Unknown model"),
            ("signal", "2026-04-17"),
            ("signal", "2026-02-30"),
            ("amount", "NaN"),
            ("amount", "Infinity"),
            ("amount", "0"),
            ("amount", "-1"),
            ("amount", "100000000000000000000"),
            ("amount", ""),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                response = self.client.get("/research/", {field: value})
                self.assertEqual(response.status_code, 400)
                self.assertIn(field, response.context["form"].errors)
                self.assertIsNone(response.context.get("result"))

    def test_forecasts_and_performance_match_notebook_outputs(self):
        result = build_research("LSTM", "2026-02-27", 100000)
        forecasts = {
            row["factor"]: row
            for row in source_rows("predictions.csv")
            if row["model"] == "LSTM" and row["signal"] == "2026-02-27"
        }
        for factor in result["factor_rows"]:
            expected = forecasts[factor["name"]]
            self.assertAlmostEqual(factor["predicted_return"], float(expected["predicted_return"]) * 100)
            self.assertAlmostEqual(factor["actual_return"], float(expected["actual_return"]) * 100)
            self.assertEqual(factor["trend"], expected["trend"])
        scores = {
            row["stock"]: row for row in source_rows("factor_scores.csv")
            if row["signal"] == "2026-02-27"
        }
        for row in result["rankings"]:
            expected = scores[row["symbol"]]
            self.assertAlmostEqual(row["value"], float(expected["Value"]) * 100)
            self.assertAlmostEqual(row["profitability"], float(expected["Profitability"]) * 100)
            self.assertAlmostEqual(row["momentum"], float(expected["Momentum"]))
        for factor in result["factor_rows"]:
            selected = sorted(scores, key=lambda symbol: float(scores[symbol][factor["name"]]), reverse=True)[:5]
            self.assertEqual(factor["selected"], selected)
        source = source_rows("performance.csv")
        self.assertEqual(len(source), 13)
        expected = {row["Strategy"]: row for row in source}
        self.assertEqual(
            {row["name"] for row in result["metrics"]},
            {"LSTM", "Equal factors", "Equal stocks", "Historical mean"},
        )
        fields = {
            "annual_return": "Annual return %",
            "total_return": "Cumulative return %",
            "volatility": "Volatility %",
            "drawdown": "Max drawdown %",
            "sharpe": "Sharpe (0% reference)",
            "final_value": "Final value USD",
            "costs": "Trading costs USD",
        }
        for row in result["metrics"]:
            for field, column in fields.items():
                self.assertAlmostEqual(row[field], float(expected[row["name"]][column]))

    def test_snapshot_date_changes_allocations_but_not_full_period_performance(self):
        early = build_research("LSTM", "2023-01-06", 100000)
        late = build_research("LSTM", "2026-02-27", 100000)
        self.assertNotEqual(early["holdings"], late["holdings"])
        self.assertNotEqual(early["factor_rows"], late["factor_rows"])
        self.assertEqual(early["metrics"], late["metrics"])
        self.assertEqual(early["chart"], late["chart"])
        self.assertEqual(early["start"], late["start"])
        self.assertEqual(early["end"], late["end"])

    def test_holdings_use_post_cost_snapshot_value_and_amount_scaling(self):
        signal = "2026-02-27"
        regular = build_research("LSTM", signal, 100000)
        smaller = build_research("LSTM", signal, 25000)
        entry = regular["entry"].isoformat()
        expected_nav = float(next(row for row in source_rows("portfolio_values.csv") if row["date"] == entry)["LSTM"])
        self.assertAlmostEqual(regular["snapshot_value"], expected_nav)
        self.assertAlmostEqual(sum(row["amount"] for row in regular["holdings"]), expected_nav)
        self.assertAlmostEqual(sum(row["amount"] for row in smaller["holdings"]), expected_nav / 4)
        self.assertNotAlmostEqual(sum(row["amount"] for row in regular["holdings"]), 100000)
        for original, scaled in zip(regular["metrics"], smaller["metrics"]):
            for field in ("final_value", "costs"):
                self.assertAlmostEqual(scaled[field], original[field] / 4)
            for field in ("annual_return", "total_return", "volatility", "drawdown", "sharpe"):
                self.assertEqual(scaled[field], original[field])
        for original, scaled in zip(regular["holdings"], smaller["holdings"]):
            self.assertEqual(scaled["weight"], original["weight"])
            self.assertAlmostEqual(scaled["amount"], original["amount"] / 4)
        self.assertEqual(regular["chart"]["lines"], smaller["chart"]["lines"])
        self.assertEqual(regular["chart"]["labels"], smaller["chart"]["labels"])
        for original, scaled in zip(regular["chart"]["ticks"], smaller["chart"]["ticks"]):
            original_value = float(original["label"].replace("$", "").replace(",", ""))
            scaled_value = float(scaled["label"].replace("$", "").replace(",", ""))
            self.assertAlmostEqual(scaled_value, original_value / 4, delta=1)

    def test_all_models_have_complete_factor_and_stock_allocations(self):
        for model in research_options()["models"]:
            with self.subTest(model=model["value"]):
                result = build_research(model["value"], "2026-02-27", 100000)
                self.assertEqual(len(result["rankings"]), 20)
                self.assertEqual({row["name"] for row in result["factor_rows"]}, {"Value", "Profitability", "Momentum"})
                self.assertEqual(sorted(row["weight"] for row in result["factor_rows"]), [20, 30, 50])
                self.assertAlmostEqual(sum(row["weight"] for row in result["holdings"]), 100)
                for factor in result["factor_rows"]:
                    self.assertEqual(len(factor["selected"]), 5)
                self.assertAlmostEqual(sum(row["amount"] for row in result["factor_rows"]), result["snapshot_value"])
