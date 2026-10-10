from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase

from .models import PriceSnapshot
from .prediction import PredictionError, completed_prices, download_prediction_prices, lstm_prediction, prediction_context, run_prediction, settle_predictions


NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
INDEX = pd.date_range("2025-10-04", "2026-10-04")
PRICES = pd.Series(np.linspace(60000, 85000, 366), index=INDEX)


class PredictionPriceTests(SimpleTestCase):
    @patch("dashboard.prediction.daily_prices")
    def test_provider_gaps_are_not_filled_for_prediction(self, download):
        download.return_value = PRICES.drop(INDEX[30])
        with self.assertRaisesRegex(PredictionError, "missing daily prices"):
            download_prediction_prices(NOW)

    @patch("dashboard.prediction.load_lstm", return_value=(None, {"data_source": "Yahoo Finance"}))
    def test_model_from_another_provider_is_not_reused(self, load):
        with self.assertRaisesRegex(PredictionError, "retrained with Eulerpool"):
            lstm_prediction(PRICES)

    def test_unfinished_day_is_removed(self):
        raw = PRICES.to_frame("Close")
        raw.loc[pd.Timestamp("2026-10-05"), "Close"] = 999999
        pd.testing.assert_series_equal(completed_prices(raw, NOW), PRICES, check_names=False)

    def test_missing_or_stale_prices_are_rejected(self):
        raw = PRICES.to_frame("Close")
        for invalid in [raw.iloc[:-1], raw.drop(INDEX[30]), raw.assign(Close=np.nan), raw * 0]:
            with self.subTest(rows=len(invalid)):
                with self.assertRaises(PredictionError):
                    completed_prices(invalid, NOW)


class DailyPredictionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="forecast-user", password="local-password")
        self.client.force_login(self.user)

    @patch("dashboard.prediction.prophet_prediction", return_value={"trend": "Up", "price": 86000, "change": 1.176})
    @patch("dashboard.prediction.lstm_prediction", return_value={"trend": "Down", "up_probability": 49, "trained_through": "2026-10-04"})
    @patch("dashboard.prediction.download_prediction_prices", return_value=PRICES)
    def test_forecast_is_saved_once_per_day(self, download, lstm, prophet):
        first = run_prediction(NOW)
        second = run_prediction(NOW + timedelta(minutes=2))
        self.assertEqual(first, second)
        self.assertEqual(first["price_date"], "2026-10-04")
        self.assertEqual(first["target_date"], "2026-10-05")
        self.assertEqual(PriceSnapshot.objects.count(), 1)
        self.assertLessEqual(len(PriceSnapshot.objects.first().key), 30)
        download.assert_called_once()
        lstm.assert_called_once()
        self.assertEqual(prophet.call_args.args[1], pd.Timestamp("2026-10-05"))

    @patch("dashboard.prediction.download_prediction_prices", side_effect=PredictionError("Prices unavailable"))
    @patch("dashboard.prediction.prophet_prediction")
    @patch("dashboard.prediction.lstm_prediction")
    def test_failed_download_does_not_create_a_forecast(self, lstm, prophet, download):
        with self.assertRaises(PredictionError):
            run_prediction(NOW)
        self.assertEqual(PriceSnapshot.objects.count(), 0)
        lstm.assert_not_called()
        prophet.assert_not_called()

    def test_finished_forecast_is_scored_against_its_recorded_close(self):
        record = PriceSnapshot.objects.create(key="ep-btc-old", fetched_at=NOW, refresh_after=NOW, data={
            "target_date": "2026-10-04", "last_close": 84000,
            "lstm": {"trend": "Up"}, "prophet": {"trend": "Down"},
        })
        settle_predictions(PRICES)
        record.refresh_from_db()
        self.assertEqual(record.data["actual_close"], 85000)
        self.assertEqual(record.data["actual_trend"], "Up")
        self.assertTrue(record.data["lstm_correct"])
        self.assertFalse(record.data["prophet_correct"])

    def test_previous_provider_predictions_are_not_displayed_or_scored(self):
        record = PriceSnapshot.objects.create(key="btc-trend-old", fetched_at=NOW, refresh_after=NOW, data={
            "target_date": "2026-10-04", "last_close": 84000,
            "lstm": {"trend": "Up"}, "prophet": {"trend": "Down"},
        })
        settle_predictions(PRICES)
        record.refresh_from_db()
        self.assertNotIn("actual_close", record.data)
        self.assertIsNone(prediction_context()["prediction"])

    @patch("dashboard.views.run_prediction")
    def test_page_load_does_not_run_models(self, run):
        self.assertContains(self.client.get("/predictions/"), "Run prediction")
        run.assert_not_called()

    @patch("dashboard.views.run_prediction", side_effect=PredictionError("Prices unavailable"))
    def test_errors_are_shown_on_page(self, run):
        self.assertContains(self.client.post("/predictions/"), "Prices unavailable")

    def test_login_and_csrf_are_required(self):
        anonymous = Client()
        self.assertEqual(anonymous.post("/predictions/").status_code, 302)
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.force_login(self.user)
        self.assertEqual(csrf_client.post("/predictions/").status_code, 403)
