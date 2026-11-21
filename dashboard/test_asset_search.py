import json
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth.models import AnonymousUser
from django.core.cache import cache
from django.test import RequestFactory, SimpleTestCase, override_settings
from django.urls import resolve, reverse

from .asset_search import search_assets
from .eulerpool import EulerpoolError
from .views import asset_search


@override_settings(CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache", "LOCATION": "asset-search-tests"}})
class AssetSearchTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        provider = patch("dashboard.asset_search.request_data")
        self.provider = provider.start()
        self.provider.return_value = {"count": 0, "results": []}
        self.addCleanup(provider.stop)
        self.addCleanup(cache.clear)

    def test_live_search_shape_returns_company_name_and_ticker(self):
        self.provider.return_value = {"count": 1, "results": [{
            "type": "stock", "name": "Apple Inc", "ticker": "AAPL",
            "isin": "US0378331005", "currency": "USD",
        }]}
        self.assertEqual(search_assets("Apple"), {"results": [{
            "symbol": "AAPL", "name": "Apple Inc", "type": "Stock", "currency": "USD",
        }]})
        self.provider.assert_called_once_with("equity/search", {"q": "apple"})

    def test_only_valid_usd_stocks_are_returned_once(self):
        valid = {"type": "stock", "name": " Apple Inc ", "ticker": " aapl ", "currency": "USD"}
        self.provider.return_value = {"results": [
            valid, {**valid, "ticker": "AAPL", "name": "Duplicate"},
            {**valid, "ticker": "SAP.DE", "currency": "EUR"},
            {**valid, "ticker": "APLY", "type": "etf"},
            {**valid, "ticker": "NONE", "currency": None},
            {**valid, "ticker": "BAD/USD"},
            {**valid, "ticker": "A" * 21},
            {**valid, "ticker": "BTC-USD"},
            {**valid, "ticker": None},
            {**valid, "name": " "},
            {**valid, "name": 123},
            {}, None, "AAPL",
        ]}
        self.assertEqual(search_assets("apple")["results"], [{
            "symbol": "AAPL", "name": "Apple Inc", "type": "Stock", "currency": "USD",
        }])

    def test_equivalent_queries_share_cached_results(self):
        first = search_assets("  APPLE   INC  ")
        second = search_assets("apple inc")
        self.assertEqual(first, second)
        self.provider.assert_called_once_with("equity/search", {"q": "apple inc"})

    def test_short_queries_do_not_request_provider(self):
        for query in ("", " ", "a", " a "):
            with self.subTest(query=query):
                self.assertEqual(search_assets(query), {"results": []})
        self.provider.assert_not_called()

    def test_query_length_boundaries(self):
        self.assertEqual(search_assets("ap"), {"results": []})
        self.assertEqual(search_assets("z" * 100), {"results": []})
        with self.assertRaisesRegex(ValueError, "100 characters"):
            search_assets("z" * 101)
        self.assertEqual(self.provider.call_count, 2)

    def test_no_matches_are_cached(self):
        self.assertEqual(search_assets("unlistedcompany"), {"results": []})
        self.assertEqual(search_assets("UNLISTEDCOMPANY"), {"results": []})
        self.provider.assert_called_once()

    def test_malformed_response_is_not_cached_as_no_matches(self):
        for response in ([], {}, {"results": {}}, {"results": None}):
            with self.subTest(response=response):
                self.provider.return_value = response
                with self.assertRaises(EulerpoolError):
                    search_assets("apple")
        self.assertEqual(self.provider.call_count, 4)

    def test_crypto_uses_explicit_usd_symbol_and_ignores_btc_etf(self):
        self.provider.return_value = {"results": [{
            "type": "etf", "name": "Grayscale Bitcoin Mini Trust ETF", "ticker": "BTC", "currency": None,
        }]}
        self.assertEqual(search_assets("BTC"), {"results": [{
            "symbol": "BTC-USD", "name": "Bitcoin", "type": "Crypto", "currency": "USD",
        }]})

    def test_crypto_can_be_found_by_name(self):
        results = search_assets("bitcoin")["results"]
        self.assertEqual(results[0]["symbol"], "BTC-USD")
        self.assertEqual({asset["symbol"] for asset in results}, {"BTC-USD", "BCH-USD"})

    def test_provider_failure_keeps_crypto_and_is_not_cached(self):
        self.provider.side_effect = [EulerpoolError("private provider details"), {"results": []}]
        first = search_assets("btc")
        self.assertEqual(first["results"][0]["symbol"], "BTC-USD")
        self.assertEqual(first["message"], "Stock search is unavailable. Showing matching crypto.")
        self.assertNotIn("private", first["message"])
        second = search_assets("btc")
        self.assertNotIn("message", second)
        self.assertEqual(self.provider.call_count, 2)

    def test_exact_ticker_is_first_and_suggestions_are_limited(self):
        self.provider.return_value = {"results": [
            {"type": "stock", "name": f"Apple related {i}", "ticker": f"AP{i}", "currency": "USD"}
            for i in range(12)
        ] + [{"type": "stock", "name": "Apple Inc", "ticker": "AAPL", "currency": "USD"}]}
        results = search_assets("aapl")["results"]
        self.assertEqual(len(results), 8)
        self.assertEqual(results[0]["symbol"], "AAPL")

    def test_known_company_name_ranks_before_partial_matches(self):
        self.provider.return_value = {"results": [
            {"type": "stock", "name": "Apple Hospitality REIT Inc", "ticker": "APLE", "currency": "USD"},
            {"type": "stock", "name": "Apple Inc", "ticker": "AAPL", "currency": "USD"},
            {"type": "stock", "name": "Apple iSports Group Inc", "ticker": "AAPI", "currency": "USD"},
        ]}
        self.assertEqual([asset["symbol"] for asset in search_assets("apple")["results"]], ["AAPL", "APLE", "AAPI"])


@override_settings(CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache", "LOCATION": "asset-search-view-tests"}})
class AssetSearchViewTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.factory = RequestFactory()
        self.url = reverse("asset_search")
        provider = patch("dashboard.asset_search.request_data")
        self.provider = provider.start()
        self.provider.return_value = {"results": []}
        self.addCleanup(provider.stop)
        self.addCleanup(cache.clear)

    def request(self, method="get", query="apple", authenticated=True):
        request = getattr(self.factory, method)(self.url, {"q": query})
        request.user = SimpleNamespace(is_authenticated=True) if authenticated else AnonymousUser()
        return asset_search(request)

    def test_route_resolves_and_anonymous_search_requires_login(self):
        self.assertEqual(self.url, "/portfolio/search/")
        self.assertEqual(resolve(self.url).func, asset_search)
        response = self.request(authenticated=False)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response["Location"].startswith("/login/?next="))
        self.provider.assert_not_called()

    def test_authenticated_search_returns_json_without_browser_cache(self):
        self.provider.return_value = {"results": [{
            "type": "stock", "name": "Apple Inc", "ticker": "AAPL", "currency": "USD",
        }]}
        response = self.request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(json.loads(response.content)["results"][0]["symbol"], "AAPL")
        self.assertIn("no-store", response["Cache-Control"])

    def test_authenticated_endpoint_is_get_only(self):
        for method in ("post", "put", "delete", "head"):
            with self.subTest(method=method):
                response = self.request(method=method)
                self.assertEqual(response.status_code, 405)
                self.assertEqual(response["Allow"], "GET")
        self.provider.assert_not_called()

    def test_provider_failure_returns_generic_message(self):
        self.provider.side_effect = EulerpoolError("private provider details")
        response = self.request()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.content), {
            "results": [], "message": "Search is unavailable right now. You can still enter an asset symbol.",
        })
        self.assertNotIn(b"private", response.content)

    def test_long_query_returns_validation_error_without_provider_request(self):
        response = self.request(query="z" * 101)
        self.assertEqual(response.status_code, 400)
        self.assertIn("100 characters", json.loads(response.content)["message"])
        self.provider.assert_not_called()

    def test_missing_query_returns_empty_results(self):
        request = self.factory.get(self.url)
        request.user = SimpleNamespace(is_authenticated=True)
        response = asset_search(request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.content), {"results": []})
        self.provider.assert_not_called()
