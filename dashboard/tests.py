import numpy as np
import pandas as pd
from django.test import SimpleTestCase
from django.conf import settings

from .research import DataError, analyze, load_prices, portfolio_test
DATA_FILE = settings.BASE_DIR / "data" / "prices.csv"


class ResearchTests(SimpleTestCase):
    def prices(self, length=70):
        return pd.DataFrame({"A": 100 + np.arange(length), "B": np.full(length, 100.0)}, index=pd.bdate_range("2025-01-01", periods=length))

    def test_execution_waits_one_session(self):
        prices = self.prices()
        prices.loc[prices.index[21], "B"] = 200
        curves, _, _ = portfolio_test(prices, 20, 1)
        expected = 100 * (1 - 0.001) * (122 / 121)
        self.assertAlmostEqual(curves["Momentum"].iloc[1], expected)

    def test_weights_drift_between_rebalances(self):
        prices = self.prices()
        prices.loc[:, "A"] = 100.0
        prices.loc[prices.index[22], "A"] = 200.0
        prices.loc[prices.index[23]:, "A"] = 400.0
        curves, _, _ = portfolio_test(prices, 20, 1)
        self.assertAlmostEqual(curves["Equal weight"].iloc[2], 100 * 0.999 * (0.5 * 4 + 0.5))

    def test_flat_market_only_pays_initial_cost(self):
        prices = self.prices() * 0 + 100
        curves, metrics, _ = portfolio_test(prices, 20, 1)
        self.assertAlmostEqual(curves["Momentum"].iloc[-1], 99.9)
        self.assertAlmostEqual(metrics[0]["drawdown"], -0.1)
        self.assertAlmostEqual(metrics[0]["turnover"], 1.0)

    def test_future_prices_do_not_change_earlier_returns(self):
        prices = self.prices(100)
        first, _, _ = portfolio_test(prices, 20, 1)
        prices.loc[prices.index[75]:, "B"] *= 8
        second, _, _ = portfolio_test(prices, 20, 1)
        pd.testing.assert_frame_equal(first.loc[:prices.index[74]], second.loc[:prices.index[74]])

    def test_missing_dates_are_removed_without_filling(self):
        raw = b'date,symbol,adjusted_close\n2025-01-01,A,100\n2025-01-01,B,100\n2025-01-02,A,110\n'
        prices, _, quality = load_prices(raw)
        self.assertEqual(len(prices), 1)
        self.assertEqual(quality["dropped_dates"], 1)

    def test_bad_csvs_are_rejected(self):
        for raw in [
            b'date,symbol,close\n2025-01-01,A,100\n',
            b'date,symbol,adjusted_close\n2025-01-01,A,-1\n',
            b'date,symbol,adjusted_close\n2025-01-01,A,inf\n',
            b'date,symbol,adjusted_close\n2025-01-01,A,100\n2025-01-01,A,110\n',
            b'date,symbol,adjusted_close\n,A,100\n',
        ]:
            with self.subTest(raw=raw), self.assertRaises(DataError):
                load_prices(raw)

    def test_short_history_and_large_portfolio_are_rejected(self):
        for prices, size in [(self.prices(30), 1), (self.prices(), 3)]:
            with self.subTest(size=size), self.assertRaises(DataError):
                portfolio_test(prices, 20, size)

    def test_included_data_produces_finite_results(self):
        prices, names, quality = load_prices(DATA_FILE.read_bytes())
        result = analyze(prices, names, quality)
        self.assertEqual(result["stock_count"], 5)
        self.assertEqual(result["rankings"][0]["rank"], 1)
        for row in result["metrics"]:
            self.assertTrue(np.isfinite(row["total_return"]))
            self.assertLessEqual(row["drawdown"], 0)
