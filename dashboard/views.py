from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET, require_http_methods

from .asset_analysis import asset_summary
from .asset_search import search_assets
from .china_benchmark import BenchmarkError, benchmark_context
from .china_factor_study import FactorStudyError, StudySelectionError, study_context
from .china_final_study import FinalStudyError, final_study_context
from .crypto import market_summary
from .factor_research import FactorResearchError, build_research, research_options
from .factors import FactorDataError, get_factor_data
from .forms import ResearchForm, ResearchPortfolioForm
from .market import MarketError, get_prices, get_symbol_prices
from .models import Portfolio, ResearchPortfolio
from .prediction import PredictionError, prediction_context, run_prediction
from .portfolio_periods import period_context
from .research_portfolio import portfolio_context, portfolio_snapshot


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
    saved = ResearchPortfolio.objects.filter(user=request.user).first()
    coming_from_research = "model" in request.GET or "signal" in request.GET
    context = {"saved_portfolio": saved, "coming_from_research": coming_from_research}
    status = 200
    if saved:
        try:
            context["allocation"] = portfolio_context(saved.snapshot, saved.budget)
        except FactorStudyError as exc:
            context["error"] = str(exc)
            status = 503
        else:
            try:
                context["period"] = period_context(saved.snapshot, saved.budget)
            except FactorStudyError as exc:
                context["period_error"] = str(exc)
                status = 503
    try:
        study = study_context()
        initial = {
            "model": saved.model_name if saved else study["model"],
            "signal": saved.signal.isoformat() if saved else study["signal_value"],
            "budget": saved.budget if saved else "100000.00",
        }
        for field in ["model", "signal"]:
            if field in request.GET:
                initial[field] = request.GET[field]
        data = request.POST if request.method == "POST" else initial if coming_from_research else None
        form = ResearchPortfolioForm(data, initial=initial, study=study)
        context["form"] = form
        if data is not None:
            if form.is_valid():
                if request.method == "POST":
                    selected = study_context(form.cleaned_data["model"], form.cleaned_data["signal"])
                    snapshot = portfolio_snapshot(selected)
                    period_context(snapshot, form.cleaned_data["budget"])
                    ResearchPortfolio.objects.update_or_create(user=request.user, defaults={
                        "model_name": selected["model"], "signal": selected["signal"],
                        "budget": form.cleaned_data["budget"], "snapshot": snapshot,
                    })
                    messages.success(request, "Your research portfolio has been saved.")
                    return redirect("portfolio")
            else:
                status = 400
    except StudySelectionError as exc:
        context["error"] = str(exc)
        status = 400
    except FactorStudyError as exc:
        context["error"] = str(exc)
        status = 503
    return render(request, "dashboard/portfolio.html", context, status=status)


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
def china_final_study(request):
    context = {}
    status = 200
    try:
        context["study"] = final_study_context()
    except FinalStudyError as exc:
        context["error"] = str(exc)
        status = 503
    return render(request, "dashboard/china_final_study.html", context, status=status)


@login_required
@never_cache
@require_GET
def china_factor_study(request):
    context = {"currency": "CNY"}
    status = 200
    try:
        context["study"] = study_context(request.GET.get("model"), request.GET.get("signal"))
    except StudySelectionError as exc:
        context["error"] = str(exc)
        status = 400
    except FactorStudyError as exc:
        context["error"] = str(exc)
        status = 503
    return render(request, "dashboard/china_factor_study.html", context, status=status)


@login_required
@never_cache
@require_GET
def china_benchmark(request):
    context = {"currency": "CNY"}
    status = 200
    try:
        context["benchmark"] = benchmark_context()
    except BenchmarkError as exc:
        context["error"] = str(exc)
        status = 503
    return render(request, "dashboard/china_benchmark.html", context, status=status)


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
