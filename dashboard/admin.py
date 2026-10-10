from django.contrib import admin

from .models import Portfolio, PriceSnapshot


@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "updated_at")
    search_fields = ("name", "user__username", "user__email")
    readonly_fields = ("updated_at",)


@admin.register(PriceSnapshot)
class PriceSnapshotAdmin(admin.ModelAdmin):
    list_display = ("key", "fetched_at", "refresh_after")
    readonly_fields = ("data",)
