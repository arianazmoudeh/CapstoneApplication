from unittest.mock import patch

import numpy as np
import pandas as pd
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from .china_benchmark import BenchmarkError, equal_weight_basket


class EqualWeightBasketTests(SimpleTestCase):
    def setUp(self):
        self.calendar = pd.to_datetime(["2015-12-31", "2016-01-04", "2016-01-29", "2016-02-01", "2016-02-02"])
        self.data = pd.DataFrame({
            "date": list(self.calendar) * 2,
            "code": ["000001"] * 5 + ["600001"] * 5,
            "adjusted_close": [100, 200, 300, 300, 300, 100, 100, 100, 200, 200],
        })

    def run_basket(self, data=None):
        return equal_weight_basket(
            self.data if data is None else data, self.calendar,
            ["000001", "600001"], "2016-01-01", "2016-02-02", initial_value=100,
        )

    def test_weights_drift_and_reset_only_after_month_end_return(self):
        curve, weights, rebalances, gaps = self.run_basket()
        np.testing.assert_allclose(curve.benchmark_value, [100, 150, 200, 300, 300])
        np.testing.assert_allclose(weights.iloc[1], [2 / 3, 1 / 3])
        np.testing.assert_allclose(weights.iloc[2], [0.5, 0.5])
        self.assertEqual(rebalances.date.dt.strftime("%Y-%m-%d").tolist(), ["2015-12-31", "2016-01-29"])
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        self.assertEqual(gaps.carried_dates.sum(), 0)

    def test_quote_gap_carries_last_price_without_dropping_a_stock(self):
        data = self.data.drop(index=1)
        curve, weights, rebalances, gaps = self.run_basket(data)
        np.testing.assert_allclose(curve.benchmark_value, [100, 100, 200, 300, 300])
        self.assertEqual(list(weights.columns), ["000001", "600001"])
        self.assertEqual(gaps.set_index("code").loc["000001", "carried_dates"], 1)

    def test_missing_starting_price_is_not_backfilled_from_the_future(self):
        with self.assertRaises(BenchmarkError):
            self.run_basket(self.data.drop(index=0))

    def test_duplicate_or_invalid_prices_are_rejected(self):
        with self.assertRaises(BenchmarkError):
            self.run_basket(pd.concat([self.data, self.data.iloc[:1]]))
        for value in [0, -1, np.nan, np.inf]:
            data = self.data.copy().astype({"adjusted_close": float})
            data.loc[0, "adjusted_close"] = value
            with self.subTest(value=value), self.assertRaises(BenchmarkError):
                self.run_basket(data)

    def test_future_prices_do_not_change_past_results(self):
        expected = self.run_basket()[0]
        data = pd.concat([self.data, pd.DataFrame({"date": [pd.Timestamp("2017-01-03")], "code": ["000001"], "adjusted_close": [10000]})])
        actual = self.run_basket(data)[0]
        pd.testing.assert_frame_equal(actual, expected)


class BenchmarkPageTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username="benchmark-reader", password="test")

    def test_requires_login(self):
        response = self.client.get(reverse("china_benchmark"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    @patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected online request"))
    def test_saved_benchmark_can_be_viewed_without_network_requests(self, request):
        self.client.force_login(self.user)
        response = self.client.get(reverse("china_benchmark"))
        self.assertEqual(response.status_code, 200)
        data = response.context["benchmark"]
        self.assertEqual(data["candidate_count"], 100)
        self.assertEqual(data["stock_count"] + data["excluded_count"], 100)
        self.assertContains(response, "China equal-weight benchmark")
        self.assertEqual(self.client.post(reverse("china_benchmark")).status_code, 405)
        request.assert_not_called()

    @patch("dashboard.views.benchmark_context", side_effect=BenchmarkError("The benchmark results could not be loaded. Please try again later."))
    def test_load_error_has_a_clear_message(self, load):
        self.client.force_login(self.user)
        response = self.client.get(reverse("china_benchmark"))
        self.assertEqual(response.status_code, 503)
        self.assertContains(response, "The benchmark results could not be loaded.", status_code=503)
