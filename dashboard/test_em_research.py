from unittest.mock import patch

import numpy as np
import pandas as pd
from django.contrib.auth.models import User
from django.test import TestCase

from .em_research import MARKETS, ResearchDataError, build_result, download_em_prices
from .eulerpool import EulerpoolError


class EMResearchTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username="researcher", password="test")

    def setUp(self):
        self.client.force_login(self.user)
        self.market_code = "brazil"
        self.universe = MARKETS[self.market_code]["universe"]
        dates = pd.bdate_range("2025-01-01", periods=220)
        self.prices = pd.DataFrame({
            symbol: 50 + np.arange(len(dates)) * (i + 1) / 20
            for i, symbol in enumerate(self.universe)
        }, index=dates)
        self.coverage = [
            {
                "symbol": symbol,
                "name": name,
                "start": dates[0],
                "end": dates[-1],
                "observations": len(dates),
                "missing": 0.0,
            }
            for symbol, name in self.universe.items()
        ]

    def test_research_result_has_rankings_and_baselines(self):
        result = build_result(self.prices, self.coverage, self.market_code)
        self.assertEqual(len(result["rankings"]), 5)
        self.assertEqual({row["name"] for row in result["metrics"]}, {"Momentum", "Equal weight"})
        self.assertTrue(all("sharpe" in row for row in result["metrics"]))

    @patch("dashboard.em_research.price_frame")
    def test_research_checks_local_currency_and_keeps_only_shared_dates(self, download):
        download.return_value = self.prices.copy()
        download.return_value.iloc[10, 0] = np.nan
        prices, coverage = download_em_prices("brazil")
        self.assertEqual(download.call_args.kwargs["currency"], "BRL")
        self.assertEqual(len(prices), 219)
        self.assertEqual(coverage[0]["observations"], 219)

    @patch("dashboard.em_research.price_frame", side_effect=EulerpoolError("This view requires MXN prices."))
    def test_currency_errors_prevent_the_research_comparison(self, download):
        with self.assertRaisesRegex(ResearchDataError, "requires MXN"):
            download_em_prices("mexico")

    def test_research_page_requires_login(self):
        self.client.logout()
        response = self.client.get("/research/")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.url.startswith("/login/?next="))
