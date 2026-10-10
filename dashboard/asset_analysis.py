import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD

from .market import ASSETS, TICKERS
from .research import chart_data, max_drawdown


def asset_summary(symbol, prices):
    close = prices[symbol].dropna()
    month = close.loc[close.index <= close.index[-1] - pd.Timedelta(days=30)]
    ema20 = EMAIndicator(close, window=20).ema_indicator()
    ema50 = EMAIndicator(close, window=50).ema_indicator()
    rsi = RSIIndicator(close, window=14).rsi()
    macd_model = MACD(close)
    macd = macd_model.macd()
    signal = macd_model.macd_signal()
    price_chart = pd.concat([
        close.rename("Price"),
        ema20.rename("EMA 20"),
        ema50.rename("EMA 50"),
    ], axis=1).dropna()
    rsi_chart = rsi.rename("RSI").dropna().to_frame()
    macd_chart = pd.concat([
        macd.rename("MACD"),
        signal.rename("Signal"),
    ], axis=1).dropna()
    latest_rsi = float(rsi.iloc[-1])
    return {
        "symbol": symbol,
        "ticker": TICKERS.get(symbol, symbol.removesuffix("-USD")),
        "name": ASSETS.get(symbol, symbol),
        "price": float(close.iloc[-1]),
        "daily_change": float(close.iloc[-1] / close.iloc[-2] - 1) * 100,
        "month_change": float(close.iloc[-1] / month.iloc[-1] - 1) * 100,
        "year_change": float(close.iloc[-1] / close.iloc[0] - 1) * 100,
        "drawdown": max_drawdown(close) * 100,
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "rsi": latest_rsi,
        "rsi_status": "High" if latest_rsi >= 70 else "Low" if latest_rsi <= 30 else "Neutral",
        "macd": float(macd.iloc[-1]),
        "macd_signal": float(signal.iloc[-1]),
        "trend": "Above EMA 20" if close.iloc[-1] >= ema20.iloc[-1] else "Below EMA 20",
        "start": close.index[0],
        "end": close.index[-1],
        "price_chart": chart_data(price_chart, currency=True),
        "rsi_chart": chart_data(rsi_chart),
        "macd_chart": chart_data(macd_chart),
    }
