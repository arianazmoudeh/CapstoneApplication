from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
from unittest.mock import patch

import numpy as np
import pandas as pd
from django.contrib.auth.models import User
from django.test import Client, SimpleTestCase, TestCase

from .crypto import market_summary, portfolio_summary
from .forms import PortfolioForm
from .market import ASSETS, CACHE_KEY, MarketError, align_portfolio_prices, download_prices, encode_prices, get_prices, get_symbol_prices
from .models import Portfolio, PriceSnapshot


def sample_prices():
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    symbols = list(ASSETS)
    return pd.DataFrame({symbol: 100 + np.arange(100) * (i + 1) for i, symbol in enumerate(symbols)}, index=pd.date_range(end=end, periods=100))


class CryptoCalculationTests(SimpleTestCase):
    def test_market_lists_include_sparklines(self):
        prices = sample_prices()
        result = market_summary(prices)
        self.assertEqual(len(result['crypto_rows']), 20)
        self.assertEqual(len(result['stock_rows']), 20)
        self.assertEqual(result['crypto_rows'][0]['name'], 'Bitcoin')
        self.assertEqual(result['stock_rows'][0]['name'], 'NVIDIA')
        self.assertTrue(all(row['points'] for row in result['crypto_rows'] + result['stock_rows']))

    def test_portfolio_uses_saved_quantities_in_historical_view(self):
        prices = sample_prices() * 0 + 100.0
        prices.loc[prices.index[1]:, 'BTC-USD'] = 200.0
        prices.loc[prices.index[2]:, 'BTC-USD'] = 400.0
        portfolio = Portfolio(budget=Decimal('0'), weights={}, quantities={'BTC-USD': '1.25', 'ETH-USD': '5'})
        result = portfolio_summary(portfolio, prices)
        self.assertEqual(result['start_value'], 625)
        self.assertEqual(result['end_value'], 1000)
        self.assertEqual(result['value_change'], 375)
        self.assertAlmostEqual(result['total_return'], 60)
        self.assertTrue(all(tick['label'].startswith('$') for tick in result['chart']['ticks']))
        self.assertEqual(sum(row['value'] for row in result['holdings']), 1000)
        self.assertEqual(result['holdings'][0]['quantity'], 1.25)

    @patch('dashboard.market.price_frame')
    def test_download_requests_completed_days_and_allows_missing_assets(self, download):
        prices = sample_prices()
        prices['XRP-USD'] = np.nan
        prices.attrs['errors'] = {'XRP-USD': 'No price history was found.'}
        download.return_value = prices
        result = download_prices()
        today = datetime.now(timezone.utc).date()
        download.assert_called_once_with(ASSETS, today - timedelta(days=365), today, strict=False)
        summary = market_summary(result)
        self.assertEqual(len(summary['crypto_rows']), 19)
        self.assertEqual(len(summary['stock_rows']), 20)
        self.assertEqual(summary['unavailable'], ['XRP-USD'])
        self.assertNotIn('XRP-USD', [row['symbol'] for row in summary['crypto_rows']])

    @patch('dashboard.market.price_frame')
    def test_missing_crypto_observations_are_not_filled(self, download):
        prices = sample_prices().astype(float)
        missing_date = prices.index[40]
        prices.loc[missing_date, 'BTC-USD'] = np.nan
        download.return_value = prices
        result = download_prices()
        self.assertTrue(pd.isna(result.loc[missing_date, 'BTC-USD']))
        portfolio_prices = align_portfolio_prices(result[['BTC-USD', 'AAPL']])
        self.assertNotIn(missing_date, portfolio_prices.index)
        self.assertEqual(len(portfolio_prices), 99)

    def test_sparse_prices_are_serialized_without_nan(self):
        prices = sample_prices().astype(float)
        prices.loc[prices.index[5], 'BTC-USD'] = np.nan
        data = encode_prices(prices)
        self.assertIsNone(data['prices']['BTC-USD'][5])
        self.assertEqual(data['source'], 'Eulerpool')


class CryptoPageTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(username='owner', password='Owner-test-41576!')
        cls.other = User.objects.create_user(username='other', password='Other-test-93465!')

    def setUp(self):
        self.enterContext(patch('dashboard.views.get_factor_data', return_value=(None, False)))
        self.client.force_login(self.user)
        self.prices = sample_prices()[list(ASSETS)]
        now = datetime.now(timezone.utc)
        PriceSnapshot.objects.create(key=CACHE_KEY, data=encode_prices(self.prices), fetched_at=now, refresh_after=now + timedelta(hours=1))

    def holding(self, **changes):
        data = {'symbol': 'SOL-USD', 'quantity': '1'}
        data.update(changes)
        return data

    def test_dashboard_and_csv_use_crypto_data(self):
        with patch('dashboard.market.download_prices') as download:
            response = self.client.get('/')
            self.assertContains(response, 'Cryptocurrencies')
            self.assertContains(response, 'US stocks')
            self.assertContains(response, 'Bitcoin')
            self.assertContains(response, 'Berkshire Hathaway')
            self.assertContains(response, 'NVIDIA')
            self.assertContains(response, 'Ethereum')
            self.assertContains(response, 'class="sparkline"', count=40)
            self.assertNotContains(response, 'Gold')
            self.assertNotContains(response, 'HDFCBANK')
            csv = self.client.get('/data/prices.csv')
            self.assertContains(csv, 'BTC-USD')
            self.assertContains(csv, 'NVDA')
            self.assertContains(csv, 'date,symbol,close')
            download.assert_not_called()

    def test_save_and_update_only_the_current_users_portfolio(self):
        other = Portfolio.objects.create(user=self.other, name='Private portfolio', budget=0, weights={}, quantities={'SOL-USD': '2'})
        response = self.client.post('/portfolio/', self.holding(user=self.other.pk))
        self.assertRedirects(response, '/portfolio/', fetch_redirect_response=False)
        own = Portfolio.objects.get(user=self.user)
        self.assertEqual(own.quantities['SOL-USD'], '1')
        self.assertNotContains(self.client.get('/portfolio/'), 'Private portfolio')
        self.client.post('/portfolio/', self.holding(quantity='2.5'))
        self.assertEqual(Portfolio.objects.filter(user=self.user).count(), 1)
        own.refresh_from_db()
        self.assertEqual(own.quantities['SOL-USD'], '2.5')
        other.refresh_from_db()
        self.assertEqual(other.name, 'Private portfolio')
        self.assertEqual(other.quantities['SOL-USD'], '2')

    def test_bad_quantities_do_not_overwrite_a_saved_portfolio(self):
        self.client.post('/portfolio/', self.holding())
        for changes in [{'quantity': '-5'}, {'quantity': 'NaN'}, {'quantity': '1.000000001'}, {'symbol': 'bad symbol!'}]:
            with self.subTest(changes=changes):
                response = self.client.post('/portfolio/', self.holding(**changes))
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.context['form'].errors)
        self.assertEqual(Portfolio.objects.get(user=self.user).quantities['SOL-USD'], '1')

    def test_one_coin_portfolio_is_allowed(self):
        form = PortfolioForm(self.holding(quantity='0.25'))
        self.assertTrue(form.is_valid(), form.errors)

    def test_saved_asset_can_be_deleted(self):
        Portfolio.objects.create(
            user=self.user, name='My portfolio', budget=0, weights={},
            quantities={'SOL-USD': '1', 'BNB-USD': '2'},
        )
        response = self.client.post('/portfolio/', {'action': 'remove', 'symbol': 'SOL-USD'})
        self.assertRedirects(response, '/portfolio/', fetch_redirect_response=False)
        portfolio = Portfolio.objects.get(user=self.user)
        self.assertEqual(portfolio.quantities, {'BNB-USD': '2'})

    @patch('dashboard.views.get_symbol_prices', side_effect=MarketError('Prices unavailable'))
    def test_saved_assets_can_still_be_removed_when_data_is_unavailable(self, prices):
        portfolio = Portfolio.objects.create(user=self.user, budget=0, weights={}, quantities={'ETC-USD': '2'})
        response = self.client.get('/portfolio/')
        self.assertContains(response, 'ETC-USD')
        self.assertContains(response, '>Delete</button>')
        prices.reset_mock()
        response = self.client.post('/portfolio/', self.holding(symbol='ETC-USD', quantity='0'))
        self.assertRedirects(response, '/portfolio/', fetch_redirect_response=False)
        prices.assert_not_called()
        portfolio.refresh_from_db()
        self.assertEqual(portfolio.quantities, {})

    @patch('dashboard.market.download_symbol_prices')
    def test_stock_analysis_and_portfolio_dates_do_not_depend_on_market_cache(self, download):
        from .market import get_symbol_prices

        raw = self.prices[['AAPL']].loc[self.prices.index.dayofweek < 5]
        market = self.prices.copy()
        market['AAPL'] = raw['AAPL']
        PriceSnapshot.objects.filter(key=CACHE_KEY).update(data=encode_prices(market))
        cached, _, _ = get_symbol_prices(['AAPL'])
        analysis, _, _ = get_symbol_prices(['AAPL'], align=False)
        self.assertTrue((analysis.index.dayofweek < 5).all())
        PriceSnapshot.objects.filter(key=CACHE_KEY).delete()
        download.return_value = raw
        direct, _, _ = get_symbol_prices(['AAPL'])
        pd.testing.assert_frame_equal(cached, direct, check_freq=False)

    def test_saved_asset_opens_an_indicator_page(self):
        Portfolio.objects.create(
            user=self.user, name='My portfolio', budget=0, weights={},
            quantities={'SOL-USD': '1'},
        )
        response = self.client.get('/portfolio/asset/SOL-USD/')
        self.assertContains(response, 'Price and moving averages')
        self.assertContains(response, 'EMA 20')
        self.assertContains(response, 'RSI')
        self.assertContains(response, 'MACD')

    def test_asset_page_only_opens_for_a_saved_holding(self):
        response = self.client.get('/portfolio/asset/SOL-USD/')
        self.assertRedirects(response, '/portfolio/', fetch_redirect_response=False)

    @patch('dashboard.views.get_symbol_prices')
    def test_valid_asset_symbol_can_be_added(self, get_symbol_prices_mock):
        get_symbol_prices_mock.return_value = (self.prices[['SOL-USD']].rename(columns={'SOL-USD': 'AAPL'}), datetime.now(timezone.utc), False)
        response = self.client.post('/portfolio/', self.holding(symbol='aapl', quantity='3'))
        self.assertRedirects(response, '/portfolio/', fetch_redirect_response=False)
        self.assertEqual(Portfolio.objects.get(user=self.user).quantities, {'AAPL': '3'})

    @patch('dashboard.views.get_symbol_prices')
    def test_unknown_asset_symbol_is_rejected(self, get_symbol_prices_mock):
        get_symbol_prices_mock.side_effect = MarketError('Not found')
        response = self.client.post('/portfolio/', self.holding(symbol='NOT-A-REAL-SYMBOL'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('symbol', response.context['form'].errors)
        self.assertContains(response, 'Not found')
        self.assertFalse(Portfolio.objects.filter(user=self.user).exists())

    @patch('dashboard.views.get_symbol_prices')
    def test_provider_access_error_is_preserved(self, get_symbol_prices_mock):
        message = 'Eulerpool rejected this request. Check the API key and data access.'
        get_symbol_prices_mock.side_effect = MarketError(message)
        response = self.client.post('/portfolio/', self.holding())
        self.assertContains(response, message)
        self.assertFalse(Portfolio.objects.filter(user=self.user).exists())

    def test_missing_asset_is_omitted_without_hiding_other_prices(self):
        prices = self.prices.astype(float)
        prices['XRP-USD'] = np.nan
        PriceSnapshot.objects.filter(key=CACHE_KEY).update(data=encode_prices(prices))
        response = self.client.get('/')
        self.assertContains(response, 'class="sparkline"', count=39)
        self.assertContains(response, 'Bitcoin')
        self.assertContains(response, 'NVIDIA')
        self.assertNotContains(response, '<strong>XRP</strong>', html=True)
        self.assertNotContains(response, '$nan')

    def test_portfolio_requires_login_and_csrf(self):
        anonymous = Client()
        self.assertEqual(anonymous.get('/portfolio/').status_code, 302)
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.user)
        self.assertEqual(client.post('/portfolio/', self.holding()).status_code, 403)
        self.assertEqual(client.post('/').status_code, 403)

    @patch('dashboard.market.download_prices')
    def test_failed_refresh_keeps_the_previous_prices(self, download):
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        PriceSnapshot.objects.update(fetched_at=old, refresh_after=old)
        download.side_effect = MarketError('Unavailable')
        prices, fetched_at, stale = get_prices()
        pd.testing.assert_frame_equal(prices, self.prices, check_freq=False, check_dtype=False)
        self.assertEqual(fetched_at, old)
        self.assertTrue(stale)
        get_prices()
        self.assertEqual(download.call_count, 1)

    @patch('dashboard.market.download_prices')
    def test_first_visit_reports_unavailable_data(self, download):
        PriceSnapshot.objects.all().delete()
        download.side_effect = MarketError('Eulerpool is unavailable right now.')
        response = self.client.get('/')
        self.assertContains(response, 'Eulerpool is unavailable right now.')
        self.assertNotContains(response, 'SOL</strong>')

    @patch('dashboard.market.download_prices')
    def test_previous_provider_market_cache_is_not_reused(self, download):
        PriceSnapshot.objects.all().delete()
        now = datetime.now(timezone.utc)
        old_data = encode_prices(self.prices * 10)
        old_data['source'] = 'Yahoo Finance'
        PriceSnapshot.objects.create(
            key='market-lists-v1', data=old_data, fetched_at=now,
            refresh_after=now + timedelta(hours=1),
        )
        download.return_value = self.prices
        prices, _, _ = get_prices()
        download.assert_called_once()
        self.assertEqual(prices.iloc[-1, 0], self.prices.iloc[-1, 0])
        self.assertEqual(PriceSnapshot.objects.get(key=CACHE_KEY).data['source'], 'Eulerpool')

    @patch('dashboard.market.download_symbol_prices')
    def test_previous_provider_portfolio_cache_is_not_reused(self, download):
        PriceSnapshot.objects.all().delete()
        now = datetime.now(timezone.utc)
        symbol = 'AAPL'
        cache_id = hashlib.sha256(symbol.encode()).hexdigest()[:18]
        old_data = encode_prices(self.prices[[symbol]] * 10)
        old_data['source'] = 'Yahoo Finance'
        PriceSnapshot.objects.create(
            key=f'portfolio-{cache_id}', data=old_data, fetched_at=now,
            refresh_after=now + timedelta(hours=1),
        )
        download.return_value = self.prices[[symbol]]
        prices, _, _ = get_symbol_prices([symbol])
        download.assert_called_once_with([symbol])
        self.assertEqual(prices.iloc[-1, 0], self.prices[symbol].iloc[-1])
        self.assertTrue(PriceSnapshot.objects.filter(key__startswith='ep-p-').exists())

    @patch('dashboard.market.download_prices')
    def test_refresh_saves_the_new_snapshot(self, download):
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        PriceSnapshot.objects.update(fetched_at=old, refresh_after=old)
        download.return_value = self.prices * 2
        prices, _, stale = get_prices(force=True)
        self.assertFalse(stale)
        self.assertEqual(prices.iloc[-1, 0], self.prices.iloc[-1, 0] * 2)
        self.assertEqual(PriceSnapshot.objects.count(), 1)
