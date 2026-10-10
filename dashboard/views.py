from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET, require_http_methods

from .asset_analysis import asset_summary
from .asset_search import search_assets
from .crypto import market_summary, portfolio_summary
from .factor_research import FactorResearchError, build_research, research_options
from .factors import FactorDataError, get_factor_data
from .forms import PortfolioForm, ResearchForm
from .market import MarketError, get_prices, get_symbol_prices
from .models import Portfolio
from .prediction import PredictionError, prediction_context, run_prediction


def price_context(force=False):
    prices, fetched_at, stale = get_prices(force=force)
    return prices, {
        "market": market_summary(prices), "fetched_at": fetched_at,
        "stale": stale,
    }


@login_required
@never_cache
@require_http_methods(["GET", "POST"])
def index(request):
    context = {}
    try:
        _, context = price_context(force=request.method == "POST")
    except MarketError as exc:
        context["error"] = str(exc)
    try:
        context["factors"], context["factors_stale"] = get_factor_data(force=request.method == "POST")
    except FactorDataError as exc:
        context["factors_error"] = str(exc)
    return render(request, "dashboard/index.html", context)


@login_required
@never_cache
@require_GET
def experiments(request):
    return render(request, "dashboard/experiments.html")


@login_required
@never_cache
@require_http_methods(["GET", "POST"])
def prediction(request):
    error = None
    if request.method == "POST":
        try:
            run_prediction()
        except PredictionError as exc:
            error = str(exc)
        else:
            return redirect("prediction")
    context = prediction_context()
    context["error"] = error
    return render(request, "dashboard/prediction.html", context)


@login_required
@never_cache
@require_http_methods(["GET", "POST"])
def my_portfolio(request):
    portfolio = Portfolio.objects.filter(user=request.user).first()
    if request.method == "POST" and request.POST.get("action") == "remove" and portfolio:
        symbol = request.POST.get("symbol", "").strip().upper()
        quantities = dict(portfolio.quantities)
        if symbol in quantities:
            quantities.pop(symbol)
            portfolio.quantities = quantities
            portfolio.save(update_fields=["quantities", "updated_at"])
            messages.success(request, f"{symbol} was removed from your portfolio.")
        return redirect("portfolio")
    form = PortfolioForm(request.POST if request.method == "POST" else None)
    if request.method == "POST" and form.is_valid():
        symbol = form.cleaned_data["symbol"]
        quantity = form.cleaned_data["quantity"]
        try:
            if quantity > 0:
                get_symbol_prices([symbol])
        except MarketError as exc:
            form.add_error("symbol", str(exc))
        else:
            quantities = dict(portfolio.quantities) if portfolio else {}
            if quantity == 0:
                quantities.pop(symbol, None)
            else:
                quantities[symbol] = str(quantity)
            Portfolio.objects.update_or_create(user=request.user, defaults={
                "budget": 0, "weights": {}, "quantities": quantities,
            })
            messages.success(request, "Your portfolio has been saved.")
            return redirect("portfolio")
    context = {"form": form, "portfolio": portfolio}
    if portfolio and portfolio.quantities:
        try:
            prices, fetched_at, stale = get_symbol_prices(portfolio.quantities)
            context.update(fetched_at=fetched_at, stale=stale)
            context["result"] = portfolio_summary(portfolio, prices)
        except MarketError as exc:
            context["error"] = str(exc)
    return render(request, "dashboard/portfolio.html", context)


@login_required
@never_cache
@require_GET
def asset_search(request):
    try:
        result = search_assets(request.GET.get("q", ""))
    except MarketError:
        return JsonResponse({"results": [], "message": "Search is unavailable right now. You can still enter an asset symbol."}, status=503)
    except ValueError as exc:
        return JsonResponse({"results": [], "message": str(exc)}, status=400)
    return JsonResponse(result)


@login_required
@never_cache
@require_GET
def asset_detail(request, symbol):
    symbol = symbol.strip().upper()
    portfolio = Portfolio.objects.filter(user=request.user).first()
    if not portfolio or symbol not in portfolio.quantities:
        messages.error(request, "This asset is not in your portfolio.")
        return redirect("portfolio")
    context = {"symbol": symbol}
    try:
        prices, fetched_at, stale = get_symbol_prices([symbol], align=False)
        context.update(
            result=asset_summary(symbol, prices),
            fetched_at=fetched_at,
            stale=stale,
        )
    except MarketError as exc:
        context["error"] = str(exc)
    return render(request, "dashboard/asset_detail.html", context)


@login_required
@never_cache
@require_GET
def research(request, market="us"):
    currency = "CNY" if market == "china" else "USD"
    context = {"market": market, "currency": currency}
    status = 200
    try:
        options = research_options(market=market)
        form = ResearchForm({
            "model": request.GET.get("model", "Ridge" if market == "china" else "LSTM"),
            "signal": request.GET.get("signal", options["default_date"]),
            "amount": request.GET.get("amount", "100000"),
        }, options=options)
        form.fields["amount"].label = f"Starting investment ({currency})"
        form.fields["signal"].label = "Test month" if market == "china" else "Research date"
        context["form"] = form
        if form.is_valid():
            context["result"] = build_research(**form.cleaned_data, market=market)
        else:
            status = 400
    except FactorResearchError as exc:
        context["error"] = str(exc)
        status = 503
    return render(request, "dashboard/research.html", context, status=status)


@login_required
@never_cache
@require_GET
def download_prices(request):
    try:
        prices, _, _ = get_prices()
    except MarketError as exc:
        return HttpResponse(str(exc), status=503, content_type="text/plain")
    rows = prices.rename_axis("date").reset_index().melt(id_vars="date", var_name="symbol", value_name="close")
    response = HttpResponse(rows.to_csv(index=False, date_format="%Y-%m-%d"), content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="crypto_prices.csv"'
    return response


@require_GET
def health(request):
    return JsonResponse({"status": "ok"})
