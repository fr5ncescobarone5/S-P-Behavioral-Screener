"""
Reusable screening logic for the Streamlit behavioral finance stock screener.

The UI lives in app.py. This file intentionally contains the data collection,
indicator calculations, screening rules, and scoring model so the project is
easy to test, explain, and modify for a class presentation.
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


# ---------------------------------------------------------------------------
# Editable assumptions and scoring weights
# ---------------------------------------------------------------------------

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

# Baseline strategy thresholds. The Streamlit sidebar can override RSI, market
# cap, volume, and valuation usage without changing the source code.
OVERSOLD_MAX_1M_RETURN = -0.03
MOMENTUM_MIN_3M_RETURN = 0.05
MOMENTUM_MIN_6M_RETURN = 0.08
QUALITY_MIN_ROE = 0.10
QUALITY_MAX_FORWARD_PE = 30
QUALITY_MAX_PRICE_TO_BOOK = 10
QUALITY_MAX_DISTANCE_FROM_HIGH = -0.08

# Composite score weights. These sum to 1.00 inside each strategy and are
# deliberately visible here so students can discuss how changing weights changes
# the behavioral-finance interpretation.
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
        "Looks for stocks with low RSI and recent weakness. In behavioral "
        "finance terms, this screen asks whether investors may have overreacted "
        "to bad news, creating a possible mean-reversion setup."
    ),
    "Momentum Continuation Candidates": (
        "Looks for stocks with healthy RSI, strong recent returns, and positive "
        "moving-average trends. This connects to underreaction, herding, and "
        "momentum persistence."
    ),
    "Quality at a Discount": (
        "Looks for profitable companies that have pulled back from their "
        "52-week highs. This can represent temporary mispricing if investors "
        "anchor to recent losses or over-extrapolate short-term problems."
    ),
}


@dataclass
class ScreenParameters:
    """User-editable screen parameters supplied by the Streamlit sidebar."""

    screen_type: str
    min_rsi: float
    max_rsi: float
    min_market_cap: float
    min_average_volume: float
    include_valuation_filters: bool


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def get_sp500_constituents() -> pd.DataFrame:
    """Download the current S&P 500 list from Wikipedia.

    Using a current public source keeps the project academically transparent and
    avoids hard-coding stale index membership.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not download the current S&P 500 constituent list from "
            "Wikipedia. Try again in a few minutes, or check the internet "
            "connection used by Streamlit."
        ) from exc

    tables = pd.read_html(StringIO(response.text))
    constituents = tables[0].copy()
    constituents["Symbol"] = constituents["Symbol"].str.replace(".", "-", regex=False)
    return constituents[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]


def download_price_history(tickers: Iterable[str]) -> pd.DataFrame:
    """Download adjusted daily prices in batches and tolerate failed tickers."""
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
            # A failed batch should not stop a classroom demo.
            continue

    if not batches:
        raise RuntimeError("No price data downloaded. Check the internet connection.")

    return pd.concat(batches, axis=1)


def fetch_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    """Pull simple fundamentals from yfinance when they are available."""
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


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def get_ticker_frame(price_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract one ticker from yfinance's usual multi-index price format."""
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
    """Calculate RSI(14) with Wilder-style exponential smoothing."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_gain == 0) & (average_loss > 0), 0)
    return rsi.fillna(50)


def calculate_price_signals(price_data: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Calculate the technical and behavioral price signals for each stock."""
    rows: List[Dict[str, object]] = []

    for ticker in tickers:
        frame = get_ticker_frame(price_data, ticker)
        if len(frame) < TRADING_DAYS_6M + 5:
            continue

        try:
            close = frame["Close"].dropna()
            volume = frame["Volume"].reindex(close.index)
            current_price = close.iloc[-1]

            daily_returns = close.pct_change()
            high_52_week = close.tail(TRADING_DAYS_1Y).max()

            rows.append(
                {
                    "ticker": ticker,
                    "price": current_price,
                    "rsi": calculate_rsi(close).iloc[-1],
                    "dma_50": close.rolling(50).mean().iloc[-1],
                    "dma_200": close.rolling(200).mean().iloc[-1],
                    "return_1m": close.iloc[-1] / close.iloc[-TRADING_DAYS_1M] - 1,
                    "return_3m": close.iloc[-1] / close.iloc[-TRADING_DAYS_3M] - 1,
                    "return_6m": close.iloc[-1] / close.iloc[-TRADING_DAYS_6M] - 1,
                    "volatility": daily_returns.rolling(VOLATILITY_WINDOW).std().iloc[-1] * np.sqrt(252),
                    "avg_volume": volume.rolling(AVERAGE_VOLUME_WINDOW).mean().iloc[-1],
                    "pct_from_52w_high": current_price / high_52_week - 1,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def build_screener_dataset() -> pd.DataFrame:
    """Build the complete current S&P 500 screener dataset."""
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


def clean_screener_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable rows while allowing optional fundamentals to be missing."""
    cleaned = data.copy()
    numeric_columns = [
        "price", "rsi", "dma_50", "dma_200", "return_1m", "return_3m",
        "return_6m", "volatility", "avg_volume", "pct_from_52w_high",
        "forward_pe", "price_to_book", "roe", "market_cap",
    ]

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


# ---------------------------------------------------------------------------
# Screening and scoring
# ---------------------------------------------------------------------------

def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Convert a numeric factor into a 0-100 percentile score."""
    score = series.rank(pct=True, method="average") * 100
    if not higher_is_better:
        score = 100 - score
    return score.fillna(50)


def weighted_score(data: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Combine normalized factor scores into one composite score."""
    score = pd.Series(0.0, index=data.index)
    for column, weight in weights.items():
        score += data[column] * weight
    return score


def apply_common_filters(data: pd.DataFrame, params: ScreenParameters) -> pd.DataFrame:
    """Apply controls that are shared by all screens."""
    filtered = data[
        (data["rsi"] >= params.min_rsi)
        & (data["rsi"] <= params.max_rsi)
        & (data["avg_volume"] >= params.min_average_volume)
    ].copy()

    if params.min_market_cap > 0:
        filtered = filtered[filtered["market_cap"].fillna(0) >= params.min_market_cap]

    return filtered


def run_screen(data: pd.DataFrame, params: ScreenParameters) -> pd.DataFrame:
    """Run one of the three behavioral screens and return ranked results."""
    filtered = apply_common_filters(data, params)

    if params.screen_type == "Oversold Rebound Candidates":
        results = screen_oversold_rebound(filtered)
    elif params.screen_type == "Momentum Continuation Candidates":
        results = screen_momentum_continuation(filtered)
    elif params.screen_type == "Quality at a Discount":
        results = screen_quality_at_discount(filtered, params.include_valuation_filters)
    else:
        raise ValueError(f"Unknown screen type: {params.screen_type}")

    if results.empty:
        return results

    return results.sort_values("composite_score", ascending=False).reset_index(drop=True)


def screen_oversold_rebound(data: pd.DataFrame) -> pd.DataFrame:
    """Screen for possible investor overreaction and mean reversion."""
    qualified = data[data["return_1m"] <= OVERSOLD_MAX_1M_RETURN].copy()
    if qualified.empty:
        return qualified

    # Lower RSI and larger pullbacks receive higher scores because they may
    # represent excessive pessimism or overreaction.
    qualified["rsi_score"] = percentile_score(qualified["rsi"], higher_is_better=False)
    qualified["pullback_score"] = percentile_score(qualified["return_1m"], higher_is_better=False)
    qualified["anchoring_score"] = percentile_score(qualified["pct_from_52w_high"], higher_is_better=False)
    qualified["volatility_score"] = percentile_score(qualified["volatility"], higher_is_better=False)
    qualified["composite_score"] = weighted_score(qualified, OVERSOLD_WEIGHTS)
    qualified["behavioral_label"] = "Possible Overreaction"
    return qualified


def screen_momentum_continuation(data: pd.DataFrame) -> pd.DataFrame:
    """Screen for momentum, trend persistence, and possible herding."""
    qualified = data[
        (data["return_3m"] >= MOMENTUM_MIN_3M_RETURN)
        & (data["return_6m"] >= MOMENTUM_MIN_6M_RETURN)
        & (data["price"] > data["dma_50"])
        & (data["dma_50"] > data["dma_200"])
    ].copy()
    if qualified.empty:
        return qualified

    # Strong returns and trend alignment can reflect delayed reaction to news
    # and herding into stocks that are already working.
    qualified["rsi_score"] = percentile_score(qualified["rsi"], higher_is_better=True)
    qualified["return_3m_score"] = percentile_score(qualified["return_3m"], higher_is_better=True)
    qualified["return_6m_score"] = percentile_score(qualified["return_6m"], higher_is_better=True)
    qualified["trend_score"] = percentile_score(qualified["price"] / qualified["dma_50"] - 1, higher_is_better=True)
    qualified["volatility_score"] = percentile_score(qualified["volatility"], higher_is_better=False)
    qualified["composite_score"] = weighted_score(qualified, MOMENTUM_WEIGHTS)
    qualified["behavioral_label"] = "Momentum / Herding"
    return qualified


def screen_quality_at_discount(data: pd.DataFrame, include_valuation_filters: bool) -> pd.DataFrame:
    """Screen for quality companies that have pulled back from recent highs."""
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

    # High ROE proxies for business quality; cheaper valuation and pullback
    # metrics ask whether pessimism has gone too far.
    qualified["roe_score"] = percentile_score(qualified["roe"], higher_is_better=True)
    qualified["forward_pe_score"] = percentile_score(qualified["forward_pe"], higher_is_better=False)
    qualified["price_to_book_score"] = percentile_score(qualified["price_to_book"], higher_is_better=False)
    qualified["discount_score"] = percentile_score(qualified["pct_from_52w_high"], higher_is_better=False)
    qualified["trend_quality_score"] = percentile_score(qualified["price"] / qualified["dma_200"] - 1, higher_is_better=True)
    qualified["composite_score"] = weighted_score(qualified, QUALITY_WEIGHTS)
    qualified["behavioral_label"] = "Value + Mean Reversion"
    return qualified


# ---------------------------------------------------------------------------
# Formatting and chart helpers
# ---------------------------------------------------------------------------

def format_results_for_display(results: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Return clean columns in the order requested by the app."""
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
    """Download one ticker for charts and add RSI."""
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
