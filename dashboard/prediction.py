import hashlib
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .eulerpool import EulerpoolError, daily_prices
from .models import PriceSnapshot


MODEL_PATH = Path(__file__).parent / "ml" / "eulerpool_bitcoin_lstm.onnx"
logger = logging.getLogger(__name__)


class PredictionError(ValueError):
    pass


def completed_prices(raw, now):
    if raw is None or raw.empty or "Close" not in raw:
        raise PredictionError("Bitcoin prices are unavailable. Please try again shortly.")
    close = raw["Close"].copy()
    if not isinstance(close, pd.Series):
        raise PredictionError("The price data could not be read. Please try again.")
    close.index = pd.to_datetime(close.index, utc=True).tz_localize(None).normalize()
    today = pd.Timestamp(now).tz_convert("UTC").normalize().tz_localize(None)
    close = close.loc[close.index < today]
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise PredictionError("The daily price history is incomplete. Please try again later.")
    close = pd.to_numeric(close, errors="coerce").tail(366)
    expected = pd.date_range(today - pd.Timedelta(days=366), today - pd.Timedelta(days=1))
    if not close.index.equals(expected) or not np.isfinite(close).all() or close.le(0).any():
        raise PredictionError("Eulerpool's Bitcoin history has missing daily prices. A complete year is needed before running this prediction.")
    return close


def download_prediction_prices(now):
    today = now.astimezone(timezone.utc).date()
    try:
        prices = daily_prices("BTC-USD", today - timedelta(days=400), today)
    except EulerpoolError as exc:
        raise PredictionError(str(exc)) from None
    return completed_prices(prices.to_frame("Close"), now)


@lru_cache(maxsize=1)
def load_lstm():
    if not MODEL_PATH.exists():
        raise PredictionError("The LSTM model needs to be trained with complete Eulerpool data before it can make new predictions.")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL_PATH), options, providers=["CPUExecutionProvider"])
    return session, session.get_modelmeta().custom_metadata_map


@lru_cache(maxsize=1)
def model_version():
    model = MODEL_PATH.read_bytes() if MODEL_PATH.exists() else b"not-trained"
    return hashlib.sha256(model + b"eulerpool-prophet-v1").hexdigest()[:8]


def lstm_prediction(prices):
    session, metadata = load_lstm()
    if metadata.get("data_source") != "Eulerpool":
        raise PredictionError("The LSTM model needs to be retrained with Eulerpool data before it can make new predictions.")
    if metadata["symbol"] != "BTC-USD" or pd.Timestamp(metadata["trained_through"]) > prices.index[-1]:
        raise PredictionError("The model and price dates do not match. Please refresh the daily prices later.")
    lookback = int(metadata["lookback"])
    mean, scale = float(metadata["mean"]), float(metadata["scale"])
    if not np.isfinite([mean, scale]).all() or scale <= 0:
        raise PredictionError("The saved model needs to be updated.")
    values = (prices.pct_change().dropna().to_numpy()[-lookback:] - mean) / scale
    inputs = values.astype(np.float32).reshape(1, lookback, 1)
    logit = float(session.run(["logit"], {"returns": inputs})[0].item())
    if not np.isfinite(logit):
        raise PredictionError("LSTM could not produce a prediction. Please try again later.")
    probability = float(1 / (1 + np.exp(-np.clip(logit, -50, 50))))
    return {
        "trend": "Up" if probability >= 0.5 else "Down",
        "up_probability": probability * 100,
        "trained_through": metadata["trained_through"],
    }


def prophet_prediction(prices, target):
    from prophet import Prophet

    training = pd.DataFrame({"ds": prices.index, "y": prices.to_numpy()})
    model = Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
        n_changepoints=10, changepoint_prior_scale=0.05, uncertainty_samples=0,
    )
    model.fit(training, seed=42, algorithm="Newton", timeout=12)
    forecast = float(model.predict(pd.DataFrame({"ds": [target]}))["yhat"].iloc[0])
    if not np.isfinite(forecast) or forecast <= 0:
        raise PredictionError("Prophet could not produce a valid price forecast. Please try again later.")
    change = (forecast / float(prices.iloc[-1]) - 1) * 100
    return {
        "trend": "Up" if change > 0 else "Down" if change < 0 else "Flat",
        "price": forecast, "change": change,
    }


def settle_predictions(prices):
    for saved in PriceSnapshot.objects.filter(key__startswith="ep-btc-").order_by("-fetched_at")[:100]:
        data = saved.data
        target = pd.Timestamp(data["target_date"])
        if target not in prices.index or data.get("actual_close") is not None:
            continue
        actual = float(prices.loc[target])
        direction = "Up" if actual > data["last_close"] else "Down" if actual < data["last_close"] else "Flat"
        data.update(actual_close=actual, actual_trend=direction)
        data["lstm_correct"] = data["lstm"]["trend"] == direction
        data["prophet_correct"] = data["prophet"]["trend"] == direction
        saved.save(update_fields=["data"])


def run_prediction(now=None):
    now = now or datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)
    try:
        key = f"ep-btc-{model_version()}-{now.date()}"
        saved = PriceSnapshot.objects.filter(key=key).first()
        if saved:
            return saved.data
        prices = download_prediction_prices(now)
        target = prices.index[-1] + pd.Timedelta(days=1)
        data = {
            "source": "Eulerpool", "symbol": "BTC-USD", "price_date": str(prices.index[-1].date()),
            "target_date": str(target.date()), "last_close": float(prices.iloc[-1]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lstm": lstm_prediction(prices), "prophet": prophet_prediction(prices, target),
        }
        settle_predictions(prices)
        saved, _ = PriceSnapshot.objects.get_or_create(key=key, defaults={
            "data": data, "fetched_at": now,
            "refresh_after": datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc),
        })
        return saved.data
    except PredictionError:
        raise
    except Exception:
        logger.exception("The daily prediction could not be completed.")
        raise PredictionError("The prediction could not finish. Please try again shortly.") from None


def prediction_context():
    records = list(PriceSnapshot.objects.filter(key__startswith="ep-btc-").order_by("-fetched_at")[:5])
    result = records[0].data if records else None
    return {
        "prediction": result,
        "history": [record.data for record in records],
        "current_prediction": bool(result and result["target_date"] == str(datetime.now(timezone.utc).date())),
        "generated_at": datetime.fromisoformat(result["generated_at"]) if result else None,
    }
