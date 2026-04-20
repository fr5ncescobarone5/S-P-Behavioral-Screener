"""
Streamlit frontend for the S&P 500 Behavioral Finance Stock Screener.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from screener import (
    SCREEN_EXPLANATIONS,
    ScreenParameters,
    build_screener_dataset,
    download_chart_history,
    format_results_for_display,
    run_screen,
)


SCREEN_TYPES = [
    "Oversold Rebound Candidates",
    "Momentum Continuation Candidates",
    "Quality at a Discount",
]


st.set_page_config(
    page_title="S&P 500 Behavioral Finance Stock Screener",
    layout="wide",
)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Cache the expensive data pull for one hour."""
    return build_screener_dataset()


def format_market_cap(value: float) -> str:
    if pd.isna(value):
        return ""
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def prepare_table(display_data: pd.DataFrame) -> pd.DataFrame:
    """Format numeric values for a clean Streamlit table."""
    table = display_data.copy()
    percent_columns = ["return_1m", "return_3m", "return_6m", "pct_from_52w_high", "volatility", "roe"]
    numeric_columns = ["composite_score", "rsi", "dma_50", "dma_200", "forward_pe", "price_to_book"]

    for column in percent_columns:
        table[column] = table[column].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    for column in numeric_columns:
        table[column] = table[column].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")

    table["avg_volume"] = table["avg_volume"].map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
    table["market_cap"] = table["market_cap"].map(format_market_cap)

    return table.rename(
        columns={
            "ticker": "Ticker",
            "company_name": "Company Name",
            "composite_score": "Composite Score",
            "rsi": "RSI",
            "return_1m": "1M Return",
            "return_3m": "3M Return",
            "return_6m": "6M Return",
            "dma_50": "50DMA",
            "dma_200": "200DMA",
            "pct_from_52w_high": "% From 52W High",
            "volatility": "Volatility",
            "avg_volume": "Avg Volume",
            "forward_pe": "Forward P/E",
            "price_to_book": "Price-to-Book",
            "roe": "ROE",
            "market_cap": "Market Cap",
            "behavioral_label": "Behavioral Label",
        }
    )


def sidebar_controls() -> tuple[ScreenParameters, int, bool, bool]:
    st.sidebar.header("Screen Controls")
    screen_type = st.sidebar.selectbox("Screen type", SCREEN_TYPES)

    default_min_rsi = 0 if screen_type == "Oversold Rebound Candidates" else 50
    default_max_rsi = 35 if screen_type == "Oversold Rebound Candidates" else 75
    if screen_type == "Quality at a Discount":
        default_min_rsi = 0
        default_max_rsi = 65

    min_rsi = st.sidebar.number_input("Minimum RSI", 0.0, 100.0, float(default_min_rsi), step=1.0)
    max_rsi = st.sidebar.number_input("Maximum RSI", 0.0, 100.0, float(default_max_rsi), step=1.0)
    min_market_cap_billions = st.sidebar.number_input("Minimum market cap ($B)", min_value=0.0, value=0.0, step=1.0)
    min_average_volume = st.sidebar.number_input("Minimum average volume", min_value=0, value=500_000, step=100_000)
    top_n = st.sidebar.slider("Number of top results", 5, 100, 25, step=5)
    include_valuation_filters = st.sidebar.checkbox("Include valuation filters", value=True)
    show_charts = st.sidebar.checkbox("Show charts", value=True)
    run_button = st.sidebar.button("Run Screener", type="primary")

    params = ScreenParameters(
        screen_type=screen_type,
        min_rsi=min_rsi,
        max_rsi=max_rsi,
        min_market_cap=min_market_cap_billions * 1_000_000_000,
        min_average_volume=float(min_average_volume),
        include_valuation_filters=include_valuation_filters,
    )
    return params, top_n, show_charts, run_button


def render_chart_section(results: pd.DataFrame) -> None:
    st.subheader("Charts for Top Results")
    for _, row in results.head(3).iterrows():
        ticker = row["ticker"]
        history = download_chart_history(ticker)
        if history is None or history.empty:
            st.info(f"Chart data was unavailable for {ticker}.")
            continue
        st.markdown(f"**{ticker} - {row['company_name']}**")
        left, right = st.columns([2, 1])
        with left:
            st.line_chart(history[["Close"]].rename(columns={"Close": "Price"}), height=260)
        with right:
            st.line_chart(history[["RSI"]], height=260)


def main() -> None:
    st.title("S&P 500 Behavioral Finance Stock Screener")
    st.caption("Uses RSI, momentum, trend, valuation, and quality signals to rank current S&P 500 stocks.")
    st.write(
        "Low RSI may signal overreaction and mean reversion. Strong trend plus "
        "healthy RSI may signal momentum, underreaction, and herding. Quality "
        "stocks on pullbacks may signal temporary mispricing."
    )

    params, top_n, show_charts, run_button = sidebar_controls()
    st.info(SCREEN_EXPLANATIONS[params.screen_type])

    if params.min_rsi > params.max_rsi:
        st.warning("Minimum RSI must be less than or equal to maximum RSI.")
        return
    if not run_button:
        st.write("Choose a screen in the sidebar, then click **Run Screener**.")
        return

    with st.spinner("Downloading S&P 500 data and calculating signals. First run can take a few minutes..."):
        try:
            dataset = load_dataset()
        except Exception as exc:
            st.error(f"Data download failed: {exc}")
            st.stop()

    if dataset.empty:
        st.error("No usable stock data was downloaded. Try again later.")
        st.stop()

    results = run_screen(dataset, params)
    left, right = st.columns(2)
    left.metric("Usable S&P 500 Stocks", f"{len(dataset):,}")
    right.metric("Qualified Stocks", f"{len(results):,}")

    if results.empty:
        st.warning("No stocks qualified. Try widening RSI or lowering volume/market-cap filters.")
        return

    display_data = format_results_for_display(results, top_n)
    st.subheader("Ranked Results")
    st.dataframe(prepare_table(display_data), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Results as CSV",
        data=display_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{params.screen_type.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.subheader("How to Interpret This Screen")
    st.write(SCREEN_EXPLANATIONS[params.screen_type])
    st.write("Composite scores are percentile-based comparisons within the selected screen, not investment advice.")

    if show_charts:
        render_chart_section(results)


if __name__ == "__main__":
    main()
