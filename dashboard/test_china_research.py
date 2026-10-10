import csv
import math
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .factor_research import FactorResearchError, build_research, research_options
from .models import Portfolio, PriceSnapshot


RESULTS = Path(__file__).resolve().parent.parent / "results" / "china_stock_monthly"
MODELS = {
    "Ridge", "Elastic Net", "KNN", "SVR", "Random Forest", "Extra Trees",
    "XGBoost", "MLP", "LSTM", "CNN", "Prophet",
}
FACTORS = ("Value", "Profitability", "Momentum")


def source_rows(name):
    with (RESULTS / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ChinaResearchTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username="china-researcher", password="test")
        cls.portfolio = Portfolio.objects.create(
            user=cls.user, budget=1250, weights={}, quantities={"AAPL": "2"},
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.schedule = {row["signal"]: row for row in source_rows("monthly_schedule")}
        cls.forecasts = {
            (row["signal"], row["model"], row["factor"]): row
            for row in source_rows("predictions") if row["mode"] == "Long only"
        }
        cls.factor_weights = {
            (row["signal"], row["strategy"]): row for row in source_rows("factor_weights")
        }
        cls.stock_weights = {
            (row["signal"], row["strategy"]): row for row in source_rows("stock_weights")
        }
        cls.costs = {
            (row["signal"], row["strategy"]): row for row in source_rows("trading_costs")
        }
        cls.performance = {row["Strategy"]: row for row in source_rows("performance")}
        cls.values = source_rows("portfolio_values")
        cls.scores = {
            (row["signal"], row["stock"]): row for row in source_rows("factor_scores")
        }

    def setUp(self):
        self.client.force_login(self.user)

    def test_selectors_match_all_completed_months_and_eleven_models(self):
        options = research_options(market="china")
        self.assertEqual({row["value"] for row in options["models"]}, MODELS)
        self.assertEqual(len(options["models"]), 11)
        self.assertEqual(len(options["dates"]), 21)
        self.assertEqual({row["value"] for row in options["dates"]}, set(self.schedule))
        self.assertEqual(options["default_date"], "2026-08-31")
        self.assertEqual(options["dates"][0]["value"], "2026-08-31")
        self.assertEqual(options["dates"][-1]["value"], "2024-12-31")

    @patch("dashboard.eulerpool.request_data", side_effect=AssertionError("Unexpected online request"))
    @patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected online request"))
    def test_page_uses_saved_results_and_keeps_the_personal_portfolio(self, session_request, request_data):
        before = list(Portfolio.objects.values())
        price_count = PriceSnapshot.objects.count()
        response = self.client.get(reverse("china_research"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "China A-share research")
        self.assertContains(response, "Factor predictions and allocation")
        self.assertContains(response, "Full test period")
        self.assertContains(response, "Starting investment (CNY)")
        self.assertNotContains(response, "Predicted four-week return")
        self.assertNotContains(response, "USD")
        self.assertNotContains(response, "$")
        result = response.context["result"]
        self.assertEqual(result["model"], "Ridge")
        self.assertEqual(result["signal"].isoformat(), "2026-08-31")
        self.assertEqual(result["amount"], 100000)
        self.assertEqual(result["currency"], "CNY")
        self.assertEqual(result["currency_symbol"], "¥")
        self.assertEqual(result["period_label"], "monthly")
        self.assertEqual(result["stock_count"], 20)
        self.assertEqual(list(Portfolio.objects.values()), before)
        self.assertEqual(PriceSnapshot.objects.count(), price_count)
        session_request.assert_not_called()
        request_data.assert_not_called()

    def test_page_requires_login_and_only_accepts_get(self):
        self.assertEqual(self.client.post(reverse("china_research")).status_code, 405)
        self.client.logout()
        response = self.client.get(reverse("china_research"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    def test_partial_selection_keeps_china_defaults(self):
        response = self.client.get(reverse("china_research"), {"model": "Prophet"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["result"]["model"], "Prophet")
        self.assertEqual(response.context["result"]["signal"].isoformat(), "2026-08-31")
        self.assertEqual(response.context["result"]["amount"], 100000)

    def test_invalid_controls_show_errors_without_a_result(self):
        cases = [
            ("model", "../config/local_settings.py"), ("model", "Unknown"),
            ("signal", "2026-09-30"), ("signal", "2026-02-30"),
            ("amount", "NaN"), ("amount", "Infinity"), ("amount", "0"),
            ("amount", "-1"), ("amount", "1000000001"),
            ("amount", "1.001"), ("amount", ""),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                response = self.client.get(reverse("china_research"), {field: value})
                self.assertEqual(response.status_code, 400)
                self.assertIn(field, response.context["form"].errors)
                self.assertIsNone(response.context.get("result"))

    @patch("dashboard.views.research_options", side_effect=FactorResearchError("The saved research results could not be loaded. Please try again later."))
    def test_unavailable_results_show_a_service_error(self, options):
        response = self.client.get(reverse("china_research"))
        self.assertContains(response, "The saved research results could not be loaded.", status_code=503)
        self.assertIsNone(response.context.get("result"))
        options.assert_called_once()

    def test_all_model_month_results_match_original_notebook_files(self):
        for model in sorted(MODELS):
            strategy = f"Long only | {model}"
            for signal, schedule in self.schedule.items():
                with self.subTest(model=model, signal=signal):
                    result = build_research(model, signal, 100000, market="china")
                    self.assertEqual(result["entry"].isoformat(), schedule["entry"])
                    self.assertEqual(result["exit"].isoformat(), schedule["exit"])
                    self.assertEqual(len(result["rankings"]), 20)
                    self.assertEqual({row["name"] for row in result["factor_rows"]}, set(FACTORS))
                    self.assertEqual(sorted(row["weight"] for row in result["factor_rows"]), [20, 30, 50])
                    expected_nav = float(self.costs[signal, strategy]["post_trade_value_cny"])
                    self.assertAlmostEqual(result["snapshot_value"], expected_nav)
                    self.assertAlmostEqual(sum(row["amount"] for row in result["holdings"]), expected_nav)
                    self.assertAlmostEqual(sum(row["amount"] for row in result["factor_rows"]), expected_nav)
                    self.assertAlmostEqual(sum(row["weight"] for row in result["holdings"]), 100)
                    for factor in result["factor_rows"]:
                        source = self.forecasts[signal, model, factor["name"]]
                        self.assertAlmostEqual(factor["predicted_return"], float(source["prediction"]) * 100)
                        self.assertAlmostEqual(factor["actual_return"], float(source["actual"]) * 100)
                        self.assertEqual(factor["trend"], source["trend"])
                        self.assertAlmostEqual(factor["weight"], float(self.factor_weights[signal, strategy][factor["name"]]) * 100)
                        self.assertEqual(len(factor["selected"]), 5)
                        ranked = sorted(
                            (stock for date, stock in self.scores if date == signal),
                            key=lambda stock: float(self.scores[signal, stock][factor["name"]]),
                            reverse=True,
                        )
                        self.assertEqual(factor["selected"], ranked[:5])
                    expected_weights = {
                        stock: float(weight) for stock, weight in self.stock_weights[signal, strategy].items()
                        if stock not in {"signal", "strategy"} and float(weight) > 0
                    }
                    self.assertEqual({row["symbol"] for row in result["holdings"]}, set(expected_weights))
                    for row in result["holdings"]:
                        self.assertAlmostEqual(row["weight"], expected_weights[row["symbol"]] * 100)
                        self.assertAlmostEqual(row["amount"], expected_weights[row["symbol"]] * expected_nav)
                    for row in result["rankings"]:
                        source = self.scores[signal, row["symbol"]]
                        self.assertAlmostEqual(row["value"], float(source["Value"]) * 100)
                        self.assertAlmostEqual(row["profitability"], float(source["Profitability"]) * 100)
                        self.assertAlmostEqual(row["momentum"], float(source["Momentum"]))

    def test_performance_and_costs_match_the_long_only_baselines(self):
        for model in sorted(MODELS):
            result = build_research(model, "2026-08-31", 100000, market="china")
            self.assertEqual(
                {row["name"] for row in result["metrics"]},
                {model, "Equal factors", "Equal stocks", "Historical mean"},
            )
            for metric in result["metrics"]:
                source_name = "Monthly equal stocks" if metric["name"] == "Equal stocks" else f"Long only | {metric['name']}"
                source = self.performance[source_name]
                fields = {
                    "annual_return": "Annual return %", "volatility": "Volatility %",
                    "sharpe": "Sharpe (0% reference)", "drawdown": "Max drawdown %",
                    "final_value": "Final value CNY", "costs": "Trading costs CNY",
                }
                with self.subTest(model=model, comparison=metric["name"]):
                    for field, column in fields.items():
                        self.assertTrue(math.isfinite(metric[field]))
                        self.assertAlmostEqual(metric[field], float(source[column]))
                    self.assertAlmostEqual(metric["final_value"], float(self.values[-1][source_name]))
                    self.assertAlmostEqual(metric["total_return"], (metric["final_value"] / 100000 - 1) * 100)
                    expected_costs = sum(float(row["cost_cny"]) for (_, name), row in self.costs.items() if name == source_name)
                    self.assertAlmostEqual(metric["costs"], expected_costs)

    def test_month_changes_snapshot_but_keeps_the_full_comparison(self):
        early = build_research("Ridge", "2024-12-31", 100000, market="china")
        late = build_research("Ridge", "2026-08-31", 100000, market="china")
        self.assertNotEqual(early["holdings"], late["holdings"])
        self.assertNotEqual(early["factor_rows"], late["factor_rows"])
        self.assertEqual(early["metrics"], late["metrics"])
        self.assertEqual(early["chart"], late["chart"])
        self.assertEqual(early["start"].isoformat(), self.values[0]["date"])
        self.assertEqual(early["end"].isoformat(), self.values[-1]["date"])
        self.assertEqual((early["start"], early["end"]), (late["start"], late["end"]))
        self.assertTrue(all(row["label"].startswith("¥") for row in early["chart"]["ticks"]))

    def test_starting_amount_scales_money_without_changing_returns(self):
        original = build_research("LSTM", "2026-08-31", 100000, market="china")
        smaller = build_research("LSTM", "2026-08-31", 25000, market="china")
        self.assertAlmostEqual(smaller["snapshot_value"], original["snapshot_value"] / 4)
        for before, after in zip(original["metrics"], smaller["metrics"]):
            for field in ("final_value", "costs"):
                self.assertAlmostEqual(after[field], before[field] / 4)
            for field in ("annual_return", "total_return", "volatility", "drawdown", "sharpe"):
                self.assertEqual(after[field], before[field])
        for group in ("factor_rows", "holdings"):
            for before, after in zip(original[group], smaller[group]):
                self.assertAlmostEqual(after["amount"], before["amount"] / 4)
                self.assertEqual(after["weight"], before["weight"])
        self.assertEqual(original["accuracy"], smaller["accuracy"])
        self.assertEqual(original["chart"]["lines"], smaller["chart"]["lines"])
        self.assertEqual(original["chart"]["labels"], smaller["chart"]["labels"])

    def test_us_research_keeps_usd_and_its_original_defaults(self):
        response = self.client.get("/research/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "US stock research")
        self.assertContains(response, "Starting investment (USD)")
        result = response.context["result"]
        self.assertEqual(result["model"], "LSTM")
        self.assertEqual(result["signal"].isoformat(), "2026-02-27")
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["currency_symbol"], "$")
        self.assertEqual(result["model_count"], 10)
        self.assertEqual(result["period_count"], 42)
        self.assertTrue(all(row["label"].startswith("$") for row in result["chart"]["ticks"]))
