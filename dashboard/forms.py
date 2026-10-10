from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from decimal import Decimal
import re


class LoginForm(AuthenticationForm):
    error_messages = {
        "invalid_login": "The username or password is incorrect. Please try again.",
        "inactive": "This account is inactive.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update(autocomplete="username", placeholder="Your username")
        self.fields["password"].widget.attrs.update(autocomplete="current-password", placeholder="Your password")


class SignupForm(UserCreationForm):
    first_name = forms.CharField(label="First name", max_length=150, widget=forms.TextInput(attrs={"autocomplete": "given-name", "placeholder": "First name"}))
    last_name = forms.CharField(label="Family name", max_length=150, widget=forms.TextInput(attrs={"autocomplete": "family-name", "placeholder": "Family name"}))

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("first_name", "last_name", "username", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].help_text = "Letters, numbers, and @ . + - _ only."
        self.fields["username"].widget.attrs.pop("autofocus", None)
        self.fields["first_name"].widget.attrs["autofocus"] = True
        self.fields["username"].widget.attrs.update(autocomplete="username", placeholder="Choose a username")
        self.fields["password1"].help_text = ""
        self.fields["password1"].widget.attrs.update(autocomplete="new-password", placeholder="Create a password")
        self.fields["password2"].label = "Confirm password"
        self.fields["password2"].help_text = ""
        self.fields["password2"].widget.attrs.update(autocomplete="new-password", placeholder="Enter the password again")


class PortfolioForm(forms.Form):
    symbol = forms.CharField(
        label="Asset", max_length=20,
        widget=forms.TextInput(attrs={
            "placeholder": "Search Apple, AAPL or Bitcoin", "autocomplete": "off",
            "maxlength": 100, "role": "combobox", "aria-autocomplete": "list",
            "aria-expanded": "false", "aria-controls": "asset-search-results",
            "aria-describedby": "asset-search-status", "spellcheck": "false",
        }),
    )
    quantity = forms.DecimalField(
        label="Quantity", min_value=0, max_value=Decimal("999999999999.99999999"),
        max_digits=20, decimal_places=8,
        widget=forms.NumberInput(attrs={"step": "0.00000001", "inputmode": "decimal"}),
    )

    def clean_symbol(self):
        symbol = self.cleaned_data["symbol"].strip().upper()
        if not re.fullmatch(r"[A-Z0-9.\-]+", symbol):
            raise forms.ValidationError("Enter a valid asset symbol.")
        return symbol


class ResearchForm(forms.Form):
    model = forms.ChoiceField(label="Model")
    signal = forms.ChoiceField(label="Research date")
    amount = forms.DecimalField(
        label="Starting investment (USD)", min_value=Decimal("1"), max_value=Decimal("1000000000"),
        max_digits=12, decimal_places=2,
        widget=forms.NumberInput(attrs={"step": "0.01", "inputmode": "decimal"}),
    )

    def __init__(self, *args, options=None, **kwargs):
        from .factor_research import research_options

        options = options if options is not None else research_options()
        initial = {"model": "LSTM", "signal": options["default_date"], "amount": Decimal("100000")}
        initial.update(kwargs.pop("initial", {}))
        super().__init__(*args, initial=initial, **kwargs)
        self.fields["model"].choices = [(item["value"], item["label"]) for item in options["models"]]
        self.fields["signal"].choices = [(item["value"], item["label"]) for item in options["dates"]]
