import io
import zipfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import RequestFactory, SimpleTestCase, TestCase

from .factors import CACHE_KEY, FactorDataError, download_factors, get_factor_data, read_monthly
from .market import MarketError
from .models import PriceSnapshot
from .views import index


def archive(content):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        zipped.writestr("factors.csv", content)
    return output.getvalue()


DATA = {"rows": [{"month": "2025-08-01", "value": -1.25, "profitability": 2.5, "momentum": 0.75}]}


class FactorCalculationTests(SimpleTestCase):
    def test_monthly_parser_preserves_percentages_and_excludes_annual_rows(self):
        content = archive("Monthly returns\n,HML,RMW\n202501,-1.25,2.50\n202502,-99.99,-999\n\nAnnual returns\n,HML,RMW\n2025,99,99\n")
        result = read_monthly(content, ["HML", "RMW"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0].tolist(), [-1.25, 2.5])
        self.assertTrue(result.iloc[1].isna().all())

    def test_duplicate_months_are_rejected(self):
        with self.assertRaises(FactorDataError):
            read_monthly(archive(",HML\n202501,1\n202501,2\n"), ["HML"])

    @patch("dashboard.factors.urlopen")
    def test_download_aligns_complete_months_without_filling_missing_factors(self, opened):
        current = datetime.now(timezone.utc).strftime("%Y%m")
        def response(url, timeout):
            content = (
                f",HML,RMW\n202501,-1.25,2.5\n202502,3,-999\n202503,4,5\n{current},99,99\n"
                if "5_Factors" in url else
                f",WML\n202501,0.75\n202502,2\n{current},99\n"
            )
            return io.BytesIO(archive(content))
        opened.side_effect = response
        self.assertEqual(download_factors(), {"rows": [
            {"month": "2025-01-01", "value": -1.25, "profitability": 2.5, "momentum": 0.75}
        ]})

    @patch("dashboard.views.get_factor_data")
    @patch("dashboard.views.price_context")
    def test_one_data_source_can_fail_without_hiding_the_other(self, prices, factors):
        request = RequestFactory().get("/")
        request.user = SimpleNamespace(is_authenticated=True, first_name="Arian", username="arian")
        factors.return_value = ({"rows": [{**DATA["rows"][0], "month": pd.Timestamp("2025-08-01")}]}, False)
        prices.side_effect = MarketError("Price service unavailable")
        response = index(request)
        self.assertContains(response, "-1.25%")
        self.assertContains(response, "Aug 2025")
        self.assertContains(response, "Price service unavailable")
        factors.side_effect = FactorDataError("Factor service unavailable")
        prices.side_effect = None
        prices.return_value = (None, {"market": {"stock_rows": [], "crypto_rows": []}})
        response = index(request)
        self.assertContains(response, "Factor service unavailable")
        content = response.content.decode()
        self.assertLess(content.index('id="stocks-title"'), content.index('id="factors-title"'))
        self.assertLess(content.index('id="factors-title"'), content.index('id="crypto-title"'))


class FactorCacheTests(TestCase):
    @patch("dashboard.factors.download_factors", return_value=DATA)
    def test_cached_factor_data_avoid_repeated_downloads(self, download):
        result, stale = get_factor_data()
        self.assertEqual(result["rows"][0]["month"], datetime(2025, 8, 1))
        self.assertFalse(stale)
        self.assertEqual(get_factor_data(force=True), (result, False))
        download.assert_called_once()

    @patch("dashboard.factors.download_factors", side_effect=OSError("Unavailable"))
    def test_failed_download_preserves_cached_data_and_backs_off(self, download):
        old = datetime.now(timezone.utc) - timedelta(days=2)
        PriceSnapshot.objects.create(key=CACHE_KEY, data=DATA, fetched_at=old, refresh_after=old)
        result, stale = get_factor_data()
        self.assertTrue(stale)
        self.assertEqual(result["rows"][0]["value"], -1.25)
        self.assertTrue(get_factor_data()[1])
        download.assert_called_once()
        PriceSnapshot.objects.all().delete()
        with self.assertRaises(FactorDataError):
            get_factor_data()
