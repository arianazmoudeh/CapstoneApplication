from datetime import datetime, timezone
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests
from django.test import SimpleTestCase, override_settings

from .eulerpool import EulerpoolError, daily_prices, price_frame, request_data


def response(data=None, status=200):
    result = Mock(status_code=status)
    result.json.return_value = data
    return result


def timestamp(value, unit="s"):
    seconds = pd.Timestamp(value, tz="UTC").timestamp()
    return int(seconds * (1000 if unit == "ms" else 1))


@override_settings(EULERPOOL_API_KEY="test-key")
class EulerpoolRequestTests(SimpleTestCase):
    @patch("dashboard.eulerpool.requests.get")
    def test_key_is_sent_in_header_and_redirects_are_disabled(self, get):
        get.return_value = response({"currency": "USD"})
        self.assertEqual(request_data("equity/profile/AAPL"), {"currency": "USD"})
        url = get.call_args.args[0]
        options = get.call_args.kwargs
        self.assertEqual(url, "https://api.eulerpool.com/api/1/equity/profile/AAPL")
        self.assertNotIn("test-key", url)
        self.assertEqual(options["headers"]["Authorization"], "Bearer test-key")
        self.assertFalse(options["allow_redirects"])
        self.assertTrue(options["timeout"])

    @override_settings(EULERPOOL_API_KEY="")
    @patch("dashboard.eulerpool.requests.get")
    def test_missing_key_does_not_make_a_request(self, get):
        with self.assertRaisesRegex(EulerpoolError, "Set EULERPOOL_API_KEY"):
            request_data("equity/profile/AAPL")
        get.assert_not_called()

    @patch("dashboard.eulerpool.requests.get")
    def test_http_errors_have_distinct_messages(self, get):
        for status, message in (
            (401, "API key and data access"),
            (403, "API key and data access"),
            (404, "not found"),
            (429, "request limit"),
            (500, "unavailable"),
            (302, "unavailable"),
        ):
            with self.subTest(status=status):
                get.return_value = response(status=status)
                with self.assertRaisesRegex(EulerpoolError, message):
                    request_data("equity/profile/AAPL")
                get.return_value.json.assert_not_called()

    @patch("dashboard.eulerpool.requests.get")
    def test_connection_errors_do_not_expose_request_details(self, get):
        for error in (requests.Timeout("test-key"), requests.ConnectionError("test-key")):
            with self.subTest(error=type(error).__name__):
                get.side_effect = error
                with self.assertRaisesRegex(EulerpoolError, "could not be reached") as caught:
                    request_data("equity/profile/AAPL")
                self.assertNotIn("test-key", str(caught.exception))

    @patch("dashboard.eulerpool.requests.get")
    def test_invalid_json_is_reported(self, get):
        get.return_value = response()
        get.return_value.json.side_effect = ValueError("invalid JSON")
        with self.assertRaisesRegex(EulerpoolError, "could not be read"):
            request_data("equity/profile/AAPL")


@override_settings(EULERPOOL_API_KEY="test-key")
class EulerpoolPriceTests(SimpleTestCase):
    def setUp(self):
        clock = patch("dashboard.eulerpool.datetime")
        self.clock = clock.start()
        self.clock.now.return_value = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
        self.addCleanup(clock.stop)
        network = patch("dashboard.eulerpool.requests.get")
        self.get = network.start()
        self.addCleanup(network.stop)

    def stock_history(self):
        return {
            "t": [timestamp("2026-10-02 13:30"), timestamp("2026-10-05 13:30")],
            "o": [100.0, 102.0],
            "h": [100.0, 102.0],
            "l": [100.0, 102.0],
            "c": [100.0, 102.0],
            "v": [1000, 1200],
        }

    def serve(self, history, currency="USD"):
        self.get.side_effect = [response({"currency": currency}), response(history)]

    def test_stock_charting_uses_seconds_and_only_completed_dates(self):
        history = self.stock_history()
        history["t"] += [timestamp("2026-10-06 13:30")]
        history["c"] += [999.0]
        self.serve(history)
        prices = daily_prices("AAPL", "2026-10-01", "2026-10-20")
        self.assertEqual(prices.to_dict(), {
            pd.Timestamp("2026-10-02"): 100.0,
            pd.Timestamp("2026-10-05"): 102.0,
        })
        self.assertIsNone(prices.index.tz)
        call = self.get.call_args_list[1]
        self.assertTrue(call.args[0].endswith("/charting/ohlcv/AAPL"))
        self.assertEqual(call.kwargs["params"], {
            "resolution": "D",
            "from": timestamp("2026-10-01"),
            "to": timestamp("2026-10-06"),
        })

    def test_crypto_uses_milliseconds_last_observation_and_keeps_gaps(self):
        history = [
            {"timestamp": timestamp("2026-10-04 23:00", "ms"), "price": 204.0},
            {"timestamp": timestamp("2026-10-03 21:00", "ms"), "price": 200.0},
            {"timestamp": timestamp("2026-10-04 01:00", "ms"), "price": 201.0},
            {"timestamp": timestamp("2026-10-06 02:00", "ms"), "price": 999.0},
        ]
        self.serve(history)
        prices = daily_prices("BTC-USD", "2026-10-01", "2026-10-20")
        self.assertEqual(prices.to_dict(), {
            pd.Timestamp("2026-10-03"): 200.0,
            pd.Timestamp("2026-10-04"): 204.0,
        })
        self.assertNotIn(pd.Timestamp("2026-10-05"), prices.index)
        self.assertTrue(self.get.call_args_list[0].args[0].endswith("/crypto/profile/BTC"))
        call = self.get.call_args_list[1]
        self.assertTrue(call.args[0].endswith("/crypto/quotes/BTC"))
        self.assertEqual(call.kwargs["params"], {
            "startdate": timestamp("2026-10-01", "ms"),
            "enddate": timestamp("2026-10-06", "ms"),
        })

    def test_foreign_currency_is_rejected_before_loading_usd_prices(self):
        self.serve(self.stock_history(), currency="BRL")
        with self.assertRaisesRegex(EulerpoolError, "priced in BRL.*requires USD"):
            daily_prices("VALE3.SA", "2026-10-01", "2026-10-06")
        self.assertEqual(self.get.call_count, 1)

    def test_research_can_request_the_actual_local_currency(self):
        self.serve(self.stock_history(), currency="BRL")
        prices = daily_prices("VALE3.SA", "2026-10-01", "2026-10-06", currency="BRL")
        self.assertEqual(prices.tolist(), [100.0, 102.0])

    def test_missing_profile_currency_is_rejected(self):
        for profile in ({}, {"currency": None}, []):
            with self.subTest(profile=profile):
                self.get.side_effect = [response(profile)]
                with self.assertRaisesRegex(EulerpoolError, "did not identify the currency"):
                    daily_prices("AAPL", "2026-10-01", "2026-10-06")

    def test_invalid_prices_are_not_turned_into_usable_data(self):
        for value in (0, -1, np.inf, np.nan, "not a price"):
            with self.subTest(value=value):
                history = self.stock_history()
                history["c"][1] = value
                self.serve(history)
                with self.assertRaisesRegex(EulerpoolError, "invalid prices"):
                    daily_prices("AAPL", "2026-10-01", "2026-10-06")

    def test_missing_and_malformed_history_is_rejected(self):
        cases = (
            ("AAPL", {}),
            ("AAPL", {"t": [], "c": []}),
            ("AAPL", {"t": [timestamp("2026-10-02")], "c": []}),
            ("BTC-USD", []),
            ("BTC-USD", [{"timestamp": timestamp("2026-10-02", "ms")}]),
        )
        for symbol, history in cases:
            with self.subTest(symbol=symbol, history=history):
                self.serve(history)
                with self.assertRaises(EulerpoolError):
                    daily_prices(symbol, "2026-10-01", "2026-10-06")

    def test_missing_crypto_does_not_fall_back_to_usdt_or_a_stock(self):
        self.get.side_effect = [response({"currency": "USD"}), response(status=404)]
        with self.assertRaisesRegex(EulerpoolError, "not found"):
            daily_prices("BTC-USD", "2026-10-01", "2026-10-06")
        paths = [call.args[0] for call in self.get.call_args_list]
        self.assertEqual(len(paths), 2)
        self.assertTrue(all("/crypto/" in path and "USDT" not in path for path in paths))

    def test_partial_batch_keeps_available_assets_and_reports_missing_assets(self):
        def serve_url(url, **kwargs):
            if url.endswith("/profile/MISSING"):
                return response(status=404)
            if "/profile/" in url:
                return response({"currency": "USD"})
            return response(self.stock_history())

        self.get.side_effect = serve_url
        frame = price_frame(["AAPL", "MISSING", "AAPL"], "2026-10-01", "2026-10-06", strict=False)
        self.assertEqual(list(frame.columns), ["AAPL", "MISSING"])
        self.assertEqual(frame["AAPL"].tolist(), [100.0, 102.0])
        self.assertTrue(frame["MISSING"].isna().all())
        self.assertIn("not found", frame.attrs["errors"]["MISSING"])
        self.assertEqual(self.get.call_count, 3)
        with self.assertRaisesRegex(EulerpoolError, "not found"):
            price_frame(["AAPL", "MISSING"], "2026-10-01", "2026-10-06")

    def test_batch_does_not_return_an_empty_success_when_all_assets_fail(self):
        self.get.return_value = response(status=404)
        with self.assertRaisesRegex(EulerpoolError, "not found"):
            price_frame(["MISSING"], "2026-10-01", "2026-10-06", strict=False)
