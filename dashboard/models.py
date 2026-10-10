from django.conf import settings
from django.db import models


class Portfolio(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=80, default="My crypto portfolio")
    budget = models.DecimalField(max_digits=14, decimal_places=2)
    weights = models.JSONField(default=dict)
    quantities = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)


class PriceSnapshot(models.Model):
    key = models.CharField(max_length=30, primary_key=True)
    data = models.JSONField(default=dict)
    fetched_at = models.DateTimeField()
    refresh_after = models.DateTimeField()
