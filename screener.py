"""
Screener logic for the S&P 500 Behavioral Finance Stock Screener.

This module keeps data collection, indicators, screening rules, and scoring
separate from the Streamlit UI in app.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from io import StringIO
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


LOOKBACK_PERIOD = "18mo"
PRICE_INTERVAL = "1d"
PRICE_BATCH_SIZE = 75
FUNDAMENTAL_SLEEP_SECONDS = 0.10

RSI_PERIOD = 14
TRADING_DAYS_1M = 21
TRADING_DAYS_3M = 63
TRADING_DAYS_6M = 126
TRADING_DAYS_1Y = 252
VOLATILITY_WINDOW = 63
AVERAGE_VOLUME_WINDOW = 63

OVERSOLD_MAX_1M_RETURN = -0.03
MOMENTUM_MIN_3M_RETURN = 0.05
MOMENTUM_MIN_6M_RETURN = 0.08
QUALITY_MIN_ROE = 0.10
QUALITY_MAX_FORWARD_PE = 30
QUALITY_MAX_PRICE_TO_BOOK = 10
QUALITY_MAX_DISTANCE_FROM_HIGH = -0.08

# Weights are easy to edit for class discussion.
OVERSOLD_WEIGHTS = {
    "rsi_score": 0.35,
    "pullback_score": 0.30,
    "anchoring_score": 0.20,
    "volatility_score": 0.15,
}
MOMENTUM_WEIGHTS = {
    "rsi_score": 0.20,
    "return_3m_score": 0.30,
    "return_6m_score": 0.25,
    "trend_score": 0.15,
    "volatility_score": 0.10,
}
QUALITY_WEIGHTS = {
    "roe_score": 0.30,
    "forward_pe_score": 0.20,
    "price_to_book_score": 0.15,
    "discount_score": 0.25,
    "trend_quality_score": 0.10,
}

SCREEN_EXPLANATIONS = {
    "Oversold Rebound Candidates": (
        "Low RSI and recent weakness may indicate investor overreaction and a "
        "possible mean-reversion setup."
    ),
    "Momentum Continuation Candidates": (
        "Healthy RSI, strong returns, and positive trend can reflect "
        "underreaction, herding, and momentum persistence."
    ),
    "Quality at a Discount": (
        "Profitable companies below their 52-week highs may reflect temporary "
        "mispricing, anchoring, or excessive pessimism."
    ),
}


@dataclass
class ScreenParameters:
    screen_type: str
    min_rsi: float
    max_rsi: float
    min_market_cap: float
    min_average_volume: float
    include_valuation_filters: bool


def get_sp500_constituents() -> pd.DataFrame:
    """Download current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    constituents = pd.read_html(StringIO(response.text))[0].copy()
    constituents["Symbol"] = constituents["Symbol"].str.replace(".", "-", regex=False)
    return constituents[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]


def download_price_history(tickers: Iterable[str]) -> pd.DataFrame:
    """Download adjusted price history in batches and skip failed batches."""
    ticker_list = list(tickers)
    batches: List[pd.DataFrame] = []
    for start in range(0, len(ticker_list), PRICE_BATCH_SIZE):
        batch = ticker_list[start:start + PRICE_BATCH_SIZE]
        try:
            data = yf.download(
                tickers=batch,
                period=LOOKBACK_PERIOD,
                interval=PRICE_INTERVAL,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if not data.empty:
                batches.append(data)
        except Exception:
            continue
    if not batches:
        raise RuntimeError("No price data downloaded. Check internet access.")
    return pd.concat(batches, axis=1)


def fetch_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    """Fetch basic fundamentals where yfinance has them."""
    rows: List[Dict[str, object]] = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rows.append(
                {
                    "ticker": ticker,
                    "forward_pe": info.get("forwardPE"),
                    "price_to_book": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "market_cap": info.get("marketCap"),
                }
            )
            time.sleep(FUNDAMENTAL_SLEEP_SECONDS)
        except Exception:
            rows.append(
                {
                    "ticker": ticker,
                    "forward_pe": np.nan,
                    "price_to_book": np.nan,
                    "roe": np.nan,
                    "market_cap": np.nan,
                }
            )
    return pd.DataFrame(rows)


def get_ticker_frame(price_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(price_data.columns, pd.MultiIndex):
        if ticker not in price_data.columns.get_level_values(0):
            return pd.DataFrame()
        frame = price_data[ticker].copy()
    else:
        frame = price_data.copy()
    if frame.empty or not {"Close", "Volume"}.issubset(frame.columns):
        return pd.DataFrame()
    return frame.dropna(subset=["Close"])


def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI(14), used as a simple overreaction/momentum proxy."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return rsi.fillna(50)


def calculate_price_signals(price_data: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ticker in tickers:
        frame = get_ticker_frame(price_data, ticker)
        if len(frame) < TRADING_DAYS_6M + 5:
            continue
        try:
            close = frame["Close"].dropna()
            volume = frame["Volume"].reindex(close.index)
            price = close.iloc[-1]
            daily_returns = close.pct_change()
            high_52w = close.tail(TRADING_DAYS_1Y).max()
            rows.append(
                {
                    "ticker": ticker,
                    "price": price,
                    "rsi": calculate_rsi(close).iloc[-1],
                    "dma_50": close.rolling(50).mean().iloc[-1],
                    "dma_200": close.rolling(200).mean().iloc[-1],
                    "return_1m": close.iloc[-1] / close.iloc[-TRADING_DAYS_1M] - 1,
                    "return_3m": close.iloc[-1] / close.iloc[-TRADING_DAYS_3M] - 1,
                    "return_6m": close.iloc[-1] / close.iloc[-TRADING_DAYS_6M] - 1,
                    "volatility": daily_returns.rolling(VOLATILITY_WINDOW).std().iloc[-1] * np.sqrt(252),
                    "avg_volume": volume.rolling(AVERAGE_VOLUME_WINDOW).mean().iloc[-1],
                    "pct_from_52w_high": price / high_52w - 1,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def clean_screener_data(data: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "price", "rsi", "dma_50", "dma_200", "return_1m", "return_3m",
        "return_6m", "volatility", "avg_volume", "pct_from_52w_high",
        "forward_pe", "price_to_book", "roe", "market_cap",
    ]
    cleaned = data.copy()
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(
        subset=[
            "ticker", "company_name", "price", "rsi", "dma_50", "dma_200",
            "return_1m", "return_3m", "return_6m", "volatility",
            "avg_volume", "pct_from_52w_high",
        ]
    )
    cleaned = cleaned[(cleaned["price"] > 0) & (cleaned["avg_volume"] > 0)]
    return cleaned.reset_index(drop=True)


def build_screener_dataset() -> pd.DataFrame:
    constituents = get_sp500_constituents()
    tickers = constituents["Symbol"].tolist()
    price_data = download_price_history(tickers)
    price_signals = calculate_price_signals(price_data, tickers)
    if price_signals.empty:
        return price_signals
    fundamentals = fetch_fundamentals(price_signals["ticker"].tolist())
    combined = (
        price_signals
        .merge(fundamentals, on="ticker", how="left")
        .merge(constituents, left_on="ticker", right_on="Symbol", how="left")
        .rename(columns={"Security": "company_name", "GICS Sector": "sector"})
        .drop(columns=["Symbol", "GICS Sub-Industry"])
    )
    return clean_screener_data(combined)


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    score = series.rank(pct=True, method="average") * 100
    return score.fillna(50) if higher_is_better else (100 - score).fillna(50)


def weighted_score(data: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=data.index)
    for column, weight in weights.items():
        score += data[column] * weight
    return score


def apply_common_filters(data: pd.DataFrame, params: ScreenParameters) -> pd.DataFrame:
    filtered = data[
        (data["rsi"] >= params.min_rsi)
        & (data["rsi"] <= params.max_rsi)
        & (data["avg_volume"] >= params.min_average_volume)
    ].copy()
    if params.min_market_cap > 0:
        filtered = filtered[filtered["market_cap"].fillna(0) >= params.min_market_cap]
    return filtered


def run_screen(data: pd.DataFrame, params: ScreenParameters) -> pd.DataFrame:
    filtered = apply_common_filters(data, params)
    if params.screen_type == "Oversold Rebound Candidates":
        results = screen_oversold_rebound(filtered)
    elif params.screen_type == "Momentum Continuation Candidates":
        results = screen_momentum_continuation(filtered)
    elif params.screen_type == "Quality at a Discount":
        results = screen_quality_at_discount(filtered, params.include_valuation_filters)
    else:
        raise ValueError(f"Unknown screen type: {params.screen_type}")
    return results.sort_values("composite_score", ascending=False).reset_index(drop=True) if not results.empty else results


def screen_oversold_rebound(data: pd.DataFrame) -> pd.DataFrame:
    qualified = data[data["return_1m"] <= OVERSOLD_MAX_1M_RETURN].copy()
    if qualified.empty:
        return qualified
    qualified["rsi_score"] = percentile_score(qualified["rsi"], higher_is_better=False)
    qualified["pullback_score"] = percentile_score(qualified["return_1m"], higher_is_better=False)
    qualified["anchoring_score"] = percentile_score(qualified["pct_from_52w_high"], higher_is_better=False)
    qualified["volatility_score"] = percentile_score(qualified["volatility"], higher_is_better=False)
    qualified["composite_score"] = weighted_score(qualified, OVERSOLD_WEIGHTS)
    qualified["behavioral_label"] = "Possible Overreaction"
    return qualified


def screen_momentum_continuation(data: pd.DataFrame) -> pd.DataFrame:
    qualified = data[
        (data["return_3m"] >= MOMENTUM_MIN_3M_RETURN)
        & (data["return_6m"] >= MOMENTUM_MIN_6M_RETURN)
        & (data["price"] > data["dma_50"])
        & (data["dma_50"] > data["dma_200"])
    ].copy()
    if qualified.empty:
        return qualified
    qualified["rsi_score"] = percentile_score(qualified["rsi"], higher_is_better=True)
    qualified["return_3m_score"] = percentile_score(qualified["return_3m"], higher_is_better=True)
    qualified["return_6m_score"] = percentile_score(qualified["return_6m"], higher_is_better=True)
    qualified["trend_score"] = percentile_score(qualified["price"] / qualified["dma_50"] - 1, higher_is_better=True)
    qualified["volatility_score"] = percentile_score(qualified["volatility"], higher_is_better=False)
    qualified["composite_score"] = weighted_score(qualified, MOMENTUM_WEIGHTS)
    qualified["behavioral_label"] = "Momentum / Herding"
    return qualified


def screen_quality_at_discount(data: pd.DataFrame, include_valuation_filters: bool) -> pd.DataFrame:
    qualified = data[
        (data["roe"] >= QUALITY_MIN_ROE)
        & (data["pct_from_52w_high"] <= QUALITY_MAX_DISTANCE_FROM_HIGH)
        & (data["price"] >= data["dma_200"])
    ].copy()
    if include_valuation_filters:
        qualified = qualified[
            (qualified["forward_pe"] > 0)
            & (qualified["forward_pe"] <= QUALITY_MAX_FORWARD_PE)
            & (qualified["price_to_book"] > 0)
            & (qualified["price_to_book"] <= QUALITY_MAX_PRICE_TO_BOOK)
        ].copy()
    if qualified.empty:
        return qualified
    qualified["roe_score"] = percentile_score(qualified["roe"], higher_is_better=True)
    qualified["forward_pe_score"] = percentile_score(qualified["forward_pe"], higher_is_better=False)
    qualified["price_to_book_score"] = percentile_score(qualified["price_to_book"], higher_is_better=False)
    qualified["discount_score"] = percentile_score(qualified["pct_from_52w_high"], higher_is_better=False)
    qualified["trend_quality_score"] = percentile_score(qualified["price"] / qualified["dma_200"] - 1, higher_is_better=True)
    qualified["composite_score"] = weighted_score(qualified, QUALITY_WEIGHTS)
    qualified["behavioral_label"] = "Value + Mean Reversion"
    return qualified


def format_results_for_display(results: pd.DataFrame, top_n: int) -> pd.DataFrame:
    columns = [
        "ticker", "company_name", "composite_score", "rsi", "return_1m",
        "return_3m", "return_6m", "dma_50", "dma_200", "pct_from_52w_high",
        "volatility", "avg_volume", "forward_pe", "price_to_book", "roe",
        "market_cap", "behavioral_label",
    ]
    display = results.head(top_n).copy()
    for column in columns:
        if column not in display.columns:
            display[column] = np.nan
    return display[columns]


def download_chart_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        history = yf.download(
            ticker,
            period=period,
            interval=PRICE_INTERVAL,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)
    if history.empty or "Close" not in history.columns:
        return None
    history = history.dropna(subset=["Close"]).copy()
    history["RSI"] = calculate_rsi(history["Close"])
    return history
