import shutil
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from .china_factor_study import RESULTS, FactorStudyError, study_context
from .portfolio_periods import calculate_period, period_context
from .research_portfolio import portfolio_snapshot


class PeriodCalculationTests(SimpleTestCase):
    def test_purchase_at_entry_close_skips_unavailable_stock_and_includes_fee(self):
        dates = pd.date_range('2025-01-01', periods=5)
        prices = pd.DataFrame({'A': [1, 10, 20, 10, 15], 'B': [2, 2, 3, 4, 5]}, index=dates)
        tradable = pd.DataFrame(True, index=dates, columns=prices.columns)
        tradable.loc[dates[1], 'B'] = False
        target = pd.Series({'A': .6, 'B': .4})
        curve, result = calculate_period(prices, tradable, target, dates[1], dates[-1], 100, .1)
        self.assertEqual(curve.iloc[0], 100)
        self.assertAlmostEqual(curve.iloc[1], 100 / 1.1)
        self.assertAlmostEqual(result['final_value'], 100 / 1.1 * 1.5)
        self.assertAlmostEqual(result['costs'], 100 / 1.1 * .1)
        self.assertAlmostEqual(result['drawdown'], -50)
        self.assertEqual(result['company_count'], 1)
        self.assertEqual(result['unavailable_codes'], 'B')
        self.assertAlmostEqual(result['entry_cash'], 0)

    def test_all_unavailable_keeps_cash_without_fees(self):
        dates = pd.date_range('2025-01-01', periods=3)
        prices = pd.DataFrame({'A': [10, 20, 30]}, index=dates)
        tradable = pd.DataFrame(False, index=dates, columns=prices.columns)
        curve, result = calculate_period(prices, tradable, pd.Series({'A': 1}), dates[1], dates[-1], 100, .001)
        self.assertTrue(curve.eq(100).all())
        self.assertEqual(result['costs'], 0)
        self.assertEqual(result['entry_cash'], 100)
        self.assertEqual(result['total_return'], 0)

    def test_held_stock_without_valuation_fails(self):
        dates = pd.date_range('2025-01-01', periods=3)
        prices = pd.DataFrame({'A': [10, 20, float('nan')]}, index=dates)
        tradable = pd.DataFrame(True, index=dates, columns=prices.columns)
        with self.assertRaises(ValueError):
            calculate_period(prices, tradable, pd.Series({'A': 1}), dates[1], dates[-1], 100, .001)


class PeriodContextTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.snapshot = portfolio_snapshot(study_context('CNN', '2026-08-31'))

    def test_independently_checked_month_and_comparisons(self):
        period = period_context(self.snapshot, Decimal('100000'))
        stocks, factors, model = period['rows']
        self.assertAlmostEqual(stocks['final_value'], 94413.1320293116, places=6)
        self.assertAlmostEqual(factors['final_value'], 94177.8285991451, places=6)
        self.assertAlmostEqual(model['final_value'], 93532.9593991998, places=6)
        self.assertAlmostEqual(model['total_return'], -6.4670406008, places=8)
        self.assertAlmostEqual(model['drawdown'], -6.4670406008, places=8)
        self.assertAlmostEqual(model['costs'], 99.9000999001, places=8)
        self.assertAlmostEqual(period['gains_vs_stocks']['amount'], model['final_value'] - stocks['final_value'])
        self.assertAlmostEqual(period['gains_vs_factors']['percentage_points'], model['total_return'] - factors['total_return'])
        self.assertEqual(period['start'].isoformat(), '2026-09-01')
        self.assertEqual(period['base'].isoformat(), '2026-08-31')
        self.assertEqual(period['end'].isoformat(), '2026-10-08')
        self.assertEqual(period['chart']['labels'][0]['label'], '31 Aug')
        self.assertEqual(sum(row['amount'] for row in period['holdings']), Decimal('99900.10'))

    def test_budget_scales_money_without_changing_percentage_results(self):
        whole = period_context(self.snapshot, '100000')
        half = period_context(self.snapshot, '50000')
        for first, second in zip(whole['rows'], half['rows']):
            self.assertAlmostEqual(first['final_value'] / 2, second['final_value'])
            self.assertAlmostEqual(first['profit_loss'] / 2, second['profit_loss'])
            self.assertAlmostEqual(first['costs'] / 2, second['costs'])
            self.assertEqual(first['total_return'], second['total_return'])
            self.assertEqual(first['drawdown'], second['drawdown'])
        tiny = period_context(self.snapshot, '0.01')
        self.assertEqual(sum(row['amount'] for row in tiny['holdings']), Decimal('0.01'))
        self.assertTrue(all('.' in tick['label'] for tick in tiny['chart']['ticks']))

    def test_unavailable_entry_has_zero_purchase_and_other_weights_rescaled(self):
        snapshot = portfolio_snapshot(study_context('CNN', '2025-08-29'))
        period = period_context(snapshot, '100000')
        unavailable = next(row for row in period['holdings'] if row['code'] == '688981')
        self.assertFalse(unavailable['purchased'])
        self.assertEqual(unavailable['amount'], Decimal('0.00'))
        self.assertEqual(unavailable['weight'], 0)
        self.assertGreater(unavailable['target_weight'], 0)
        self.assertEqual(period['selected']['unavailable_target_count'], 1)
        self.assertEqual(period['selected']['company_count'], 54)
        self.assertAlmostEqual(sum(row['weight'] for row in period['holdings']), 100)
        self.assertAlmostEqual(period['selected']['final_value'], 107359.8265961856, places=6)
        self.assertEqual(sum(row['amount'] for row in period['holdings']), Decimal('99900.10'))

    def test_changed_snapshot_and_invalid_budget_are_rejected(self):
        for field in ['entry', 'exit']:
            snapshot = deepcopy(self.snapshot)
            snapshot[field] = '2030-01-01'
            with self.subTest(field=field), self.assertRaises(FactorStudyError):
                period_context(snapshot, '100000')
        snapshot = deepcopy(self.snapshot)
        snapshot['holdings'][0]['weight'] += 1
        with self.assertRaises(FactorStudyError):
            period_context(snapshot, '100000')
        for budget in ['0', '-1', 'NaN', 'Infinity', '10.001', '1000000001']:
            with self.subTest(budget=budget), self.assertRaises(FactorStudyError):
                period_context(self.snapshot, budget)

    def test_missing_or_changed_period_data_fails(self):
        files = ['data_summary', 'predictions', 'period_results', 'period_values', 'stock_weights', 'portfolio_values']
        changes = [
            ('period_results', 'costs', 1000), ('period_results', 'final_value', 1000),
            ('period_results', 'weight_signature', 'changed'), ('stock_weights', 'weight', .99),
            ('period_values', 'value', 1000),
        ]
        with TemporaryDirectory() as temporary:
            target = Path(temporary)
            for filename in files:
                shutil.copy2(RESULTS / f'{filename}.csv', target / f'{filename}.csv')
            for filename, column, value in changes:
                path = target / f'{filename}.csv'
                content = path.read_bytes()
                frame = pd.read_csv(path, dtype={'code': str}, keep_default_na=False, float_precision='round_trip')
                mask = frame.signal.eq(self.snapshot['signal'])
                if 'strategy' in frame:
                    mask &= frame.strategy.eq('CNN')
                if 'model' in frame:
                    mask &= frame.model.eq('CNN')
                frame.loc[frame.index[mask][0], column] = value
                frame.to_csv(path, index=False)
                with self.subTest(filename=filename, column=column), patch('dashboard.portfolio_periods.RESULTS', target), self.assertRaises(FactorStudyError):
                    period_context(self.snapshot, '100000')
                path.write_bytes(content)
            (target / 'period_values.csv').unlink()
            with patch('dashboard.portfolio_periods.RESULTS', target), self.assertRaises(FactorStudyError):
                period_context(self.snapshot, '100000')
