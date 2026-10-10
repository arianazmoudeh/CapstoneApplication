from django.contrib import admin
from django.urls import path
from dashboard import views
from dashboard import accounts

urlpatterns = [
    path("admin/", admin.site.urls),
    path("login/", accounts.AccountLoginView.as_view(), name="login"),
    path("signup/", accounts.signup, name="signup"),
    path("logout/", accounts.AccountLogoutView.as_view(), name="logout"),
    path("", views.index, name="dashboard"),
    path("portfolio/", views.my_portfolio, name="portfolio"),
    path("portfolio/search/", views.asset_search, name="asset_search"),
    path("portfolio/asset/<path:symbol>/", views.asset_detail, name="asset_detail"),
    path("research/", views.research, name="research"),
    path("research/china/", views.research, {"market": "china"}, name="china_research"),
    path("experiments/", views.experiments, name="experiments"),
    path("predictions/", views.prediction, name="prediction"),
    path("data/prices.csv", views.download_prices, name="download_prices"),
    path("health/", views.health, name="health"),
]
