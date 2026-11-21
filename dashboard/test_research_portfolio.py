from copy import deepcopy
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from .china_factor_study import FactorStudyError, study_context
from .models import Portfolio, ResearchPortfolio


class ResearchPortfolioTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.owner = User.objects.create_user(username="research-owner", password="test")
        cls.other = User.objects.create_user(username="research-other", password="test")
        cls.study = study_context()

    def setUp(self):
        self.client.force_login(self.owner)
        self.url = reverse("portfolio")
        self.data = {"model": self.study["model"], "signal": self.study["signal_value"], "budget": "100000.01"}

    def test_research_link_prefills_form_without_saving(self):
        response = self.client.get(self.url, {"model": "Random Forest", "signal": self.study["dates"][1]["value"]})
        self.assertContains(response, "Research portfolio")
        self.assertNotContains(response, "Add or update an asset")
        self.assertEqual(response.context["form"]["model"].value(), "Random Forest")
        self.assertEqual(response.context["form"]["signal"].value(), self.study["dates"][1]["value"])
        self.assertFalse(ResearchPortfolio.objects.exists())
        research = self.client.get(reverse("china_factor_study"), {"model": "Random Forest"})
        self.assertContains(research, "Use this allocation")
        self.assertContains(research, "model=Random%20Forest")

    @patch("dashboard.views.get_symbol_prices", side_effect=AssertionError("Online prices are not needed"))
    def test_save_uses_research_weights_and_amounts_total_to_budget(self, prices):
        legacy = Portfolio.objects.create(user=self.owner, budget=0, quantities={"AAPL": "3"})
        response = self.client.post(self.url, {**self.data, "weights": "fake", "user": self.other.pk})
        self.assertRedirects(response, self.url)
        saved = ResearchPortfolio.objects.get(user=self.owner)
        self.assertEqual(saved.budget, Decimal("100000.01"))
        self.assertEqual(saved.snapshot["model"], self.study["model"])
        self.assertFalse(ResearchPortfolio.objects.filter(user=self.other).exists())
        page = self.client.get(self.url)
        allocation = page.context["allocation"]
        self.assertEqual(sum(row["amount"] for row in allocation["holdings"]), saved.budget)
        self.assertEqual(sum(row["amount"] for row in allocation["factor_rows"]), saved.budget)
        self.assertTrue(any(len(row["factors"]) > 1 for row in allocation["holdings"]))
        self.assertContains(page, "Selected by")
        self.assertContains(page, "before trading costs")
        self.assertContains(page, 'href="/research/china/factors/?model=CNN&amp;signal=2026-08-31"')
        legacy.refresh_from_db()
        self.assertEqual(legacy.quantities, {"AAPL": "3"})
        prices.assert_not_called()

    def test_update_one_saved_portfolio_and_keep_users_separate(self):
        self.client.post(self.url, self.data)
        first = ResearchPortfolio.objects.get(user=self.owner)
        self.client.force_login(self.other)
        self.client.post(self.url, {**self.data, "model": "LSTM", "budget": "50000"})
        other = ResearchPortfolio.objects.get(user=self.other)
        other_snapshot = deepcopy(other.snapshot)
        self.client.force_login(self.owner)
        self.client.post(self.url, {**self.data, "model": "Random Forest", "budget": "20000"})
        first.refresh_from_db()
        other.refresh_from_db()
        self.assertEqual(ResearchPortfolio.objects.filter(user=self.owner).count(), 1)
        self.assertEqual(first.model_name, "Random Forest")
        self.assertEqual(first.budget, Decimal("20000"))
        self.assertEqual(other.snapshot, other_snapshot)
        self.assertEqual(other.budget, Decimal("50000"))

    def test_invalid_input_keeps_saved_allocation(self):
        self.client.post(self.url, self.data)
        saved = ResearchPortfolio.objects.get(user=self.owner)
        original = deepcopy(saved.snapshot)
        for changes in [
            {"budget": "0"}, {"budget": "-1"}, {"budget": "NaN"}, {"budget": "Infinity"},
            {"budget": "1000000001"}, {"budget": "10.001"}, {"model": "Unknown"},
            {"signal": "2030-01-31"}, {"signal": "bad-date"},
        ]:
            with self.subTest(changes=changes):
                response = self.client.post(self.url, {**self.data, **changes})
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.context["form"].errors)
                saved.refresh_from_db()
                self.assertEqual(saved.snapshot, original)
                self.assertEqual(saved.budget, Decimal("100000.01"))

    def test_incoming_selection_does_not_replace_saved_display(self):
        self.client.post(self.url, self.data)
        response = self.client.get(self.url, {"model": "Prophet", "signal": self.study["dates"][1]["value"]})
        self.assertEqual(response.context["form"]["model"].value(), "Prophet")
        self.assertEqual(response.context["allocation"]["model"], self.data["model"])
        self.assertEqual(response.context["allocation"]["signal"].isoformat(), self.data["signal"])

    def test_saved_snapshot_is_available_when_research_files_are_unavailable(self):
        self.client.post(self.url, self.data)
        with patch("dashboard.views.study_context", side_effect=FactorStudyError("Research unavailable")):
            response = self.client.get(self.url)
        self.assertEqual(response.status_code, 503)
        self.assertIsNotNone(response.context["allocation"])
        self.assertContains(response, "Selected companies", status_code=503)

    def test_invalid_research_query_does_not_save(self):
        response = self.client.get(self.url, {"model": "Unknown", "signal": "bad-date"})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(ResearchPortfolio.objects.exists())

    def test_login_csrf_and_request_methods(self):
        self.client.logout()
        self.assertEqual(self.client.get(self.url).status_code, 302)
        secure = Client(enforce_csrf_checks=True)
        secure.force_login(self.owner)
        self.assertEqual(secure.post(self.url, self.data).status_code, 403)
        self.client.force_login(self.owner)
        self.assertEqual(self.client.delete(self.url).status_code, 405)
