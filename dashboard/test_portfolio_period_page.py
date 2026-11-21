from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .china_factor_study import FactorStudyError
from .models import ResearchPortfolio


class PortfolioPeriodPageTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.owner = User.objects.create_user(username="period-owner", password="test")

    def setUp(self):
        self.client.force_login(self.owner)
        self.url = reverse("portfolio")
        self.data = {"model": "CNN", "signal": "2026-08-31", "budget": "100000"}

    @patch("requests.sessions.Session.request", side_effect=AssertionError("No online data needed"))
    def test_page_compares_same_period_and_budget_without_annualizing(self, online):
        self.assertRedirects(self.client.post(self.url, self.data), self.url)
        response = self.client.get(self.url)
        period = response.context["period"]
        self.assertEqual(period["start"].isoformat(), "2026-09-01")
        self.assertEqual(period["end"].isoformat(), "2026-10-08")
        self.assertEqual(len(period["rows"]), 3)
        self.assertTrue(all(row["initial_value"] == 100000 for row in period["rows"]))
        self.assertAlmostEqual(float(period["selected"]["final_value"]), 93532.96, places=2)
        self.assertAlmostEqual(float(period["selected"]["total_return"]), -6.4670, places=3)
        self.assertContains(response, "Period return")
        self.assertNotContains(response, "Annual return")
        self.assertNotContains(response, "over the full test period")
        self.assertContains(response, "Equal factors")
        self.assertContains(response, "Eligible equal stocks")
        online.assert_not_called()

    def test_saved_budget_scales_values_and_costs_but_not_return(self):
        self.client.post(self.url, self.data)
        first = self.client.get(self.url).context["period"]
        self.client.post(self.url, {**self.data, "budget": "200000"})
        second = self.client.get(self.url).context["period"]
        for field in ["final_value", "profit_loss", "costs"]:
            self.assertAlmostEqual(float(second["selected"][field]), float(first["selected"][field]) * 2, delta=0.02)
        self.assertAlmostEqual(second["selected"]["total_return"], first["selected"]["total_return"])
        self.assertEqual(ResearchPortfolio.objects.filter(user=self.owner).count(), 1)

    def test_unsaved_form_selection_keeps_saved_results(self):
        self.client.post(self.url, self.data)
        response = self.client.get(self.url, {"model": "Ridge", "signal": "2024-12-31"})
        self.assertEqual(response.context["form"]["model"].value(), "Ridge")
        self.assertEqual(response.context["period"]["selected"]["name"], "CNN")
        self.assertEqual(response.context["period"]["end"].isoformat(), "2026-10-08")

    def test_existing_snapshot_needs_no_database_update_to_show_results(self):
        self.client.post(self.url, self.data)
        saved = ResearchPortfolio.objects.get(user=self.owner)
        before = (saved.snapshot, saved.updated_at)
        response = self.client.get(self.url)
        self.assertIsNotNone(response.context["period"])
        saved.refresh_from_db()
        self.assertEqual((saved.snapshot, saved.updated_at), before)

    def test_missing_results_keep_allocation_and_do_not_overwrite_saved_choice(self):
        self.client.post(self.url, self.data)
        with patch("dashboard.views.period_context", side_effect=FactorStudyError("The period results could not be loaded.")):
            response = self.client.get(self.url)
            self.assertEqual(response.status_code, 503)
            self.assertIsNotNone(response.context["allocation"])
            self.assertIsNone(response.context.get("period"))
            self.assertContains(response, "The period results could not be loaded.", status_code=503)
            update = self.client.post(self.url, {**self.data, "budget": "200000"})
            self.assertEqual(update.status_code, 503)
        self.assertEqual(ResearchPortfolio.objects.get(user=self.owner).budget, Decimal("100000"))
