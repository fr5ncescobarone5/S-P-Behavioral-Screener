"""
Streamlit app for the S&P 500 Behavioral Finance Stock Screener.

Run locally:
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


st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        max-width: 1220px;
    }
    div[data-testid="stMetric"] {
        background-color: #f7f7f9;
        border: 1px solid #e6e6ea;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Cache the expensive S&P 500 data pull for one hour."""
    return build_screener_dataset()


def format_market_cap(value: float) -> str:
    """Readable market-cap formatting for sidebar examples and table display."""
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
    """Format a copy for a cleaner classroom-friendly table."""
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


def make_csv_download(display_data: pd.DataFrame) -> bytes:
    """Create CSV bytes from the unformatted numeric result table."""
    return display_data.to_csv(index=False).encode("utf-8")


def render_chart_section(results: pd.DataFrame) -> None:
    """Show simple price and RSI charts for the top three ranked stocks."""
    st.subheader("Charts for Top Results")

    for _, row in results.head(3).iterrows():
        ticker = row["ticker"]
        company = row["company_name"]
        history = download_chart_history(ticker)

        if history is None or history.empty:
            st.info(f"Chart data was unavailable for {ticker}.")
            continue

        with st.container():
            st.markdown(f"**{ticker} - {company}**")
            price_chart = history[["Close"]].rename(columns={"Close": "Price"})
            rsi_chart = history[["RSI"]]

            left, right = st.columns([2, 1])
            with left:
                st.line_chart(price_chart, height=260)
            with right:
                st.line_chart(rsi_chart, height=260)


def sidebar_controls() -> tuple[ScreenParameters, int, bool, bool]:
    """Collect sidebar controls and convert them into screen parameters."""
    st.sidebar.header("Screen Controls")

    screen_type = st.sidebar.selectbox("Screen type", SCREEN_TYPES)

    default_min_rsi = 0 if screen_type == "Oversold Rebound Candidates" else 50
    default_max_rsi = 35 if screen_type == "Oversold Rebound Candidates" else 75
    if screen_type == "Quality at a Discount":
        default_min_rsi = 0
        default_max_rsi = 65

    min_rsi = st.sidebar.number_input("Minimum RSI", min_value=0.0, max_value=100.0, value=float(default_min_rsi), step=1.0)
    max_rsi = st.sidebar.number_input("Maximum RSI", min_value=0.0, max_value=100.0, value=float(default_max_rsi), step=1.0)
    min_market_cap_billions = st.sidebar.number_input("Minimum market cap ($B)", min_value=0.0, value=0.0, step=1.0)
    min_average_volume = st.sidebar.number_input("Minimum average volume", min_value=0, value=500_000, step=100_000)
    top_n = st.sidebar.slider("Number of top results", min_value=5, max_value=100, value=25, step=5)

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


def main() -> None:
    """Render the Streamlit website."""
    st.title("S&P 500 Behavioral Finance Stock Screener")
    st.caption("Uses RSI, momentum, trend, valuation, and quality signals to rank current S&P 500 stocks.")

    st.write(
        "This tool connects common market signals to behavioral finance ideas. "
        "Low RSI may point to overreaction and mean reversion, strong trend plus "
        "healthy RSI may indicate momentum, underreaction, or herding, and quality "
        "stocks on pullbacks may suggest temporary mispricing."
    )

    params, top_n, show_charts, run_button = sidebar_controls()

    st.info(SCREEN_EXPLANATIONS[params.screen_type])

    if params.min_rsi > params.max_rsi:
        st.warning("Minimum RSI must be less than or equal to maximum RSI.")
        return

    if not run_button:
        st.write("Use the sidebar to choose a screen, then click **Run Screener**.")
        return

    with st.spinner("Downloading S&P 500 data and calculating behavioral signals..."):
        try:
            dataset = load_dataset()
        except Exception as exc:
            st.error(f"Data download failed: {exc}")
            st.stop()

    if dataset.empty:
        st.error("No usable stock data was downloaded. Try again later.")
        st.stop()

    results = run_screen(dataset, params)
    display_data = format_results_for_display(results, top_n)

    total_stocks, qualified_stocks = st.columns(2)
    total_stocks.metric("Usable S&P 500 Stocks", f"{len(dataset):,}")
    qualified_stocks.metric("Qualified Stocks", f"{len(results):,}")

    if results.empty:
        st.warning("No stocks qualified under the current settings. Try widening the RSI range or lowering the volume/market-cap filters.")
        return

    st.subheader("Ranked Results")
    st.dataframe(
        prepare_table(display_data),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download Results as CSV",
        data=make_csv_download(display_data),
        file_name=f"{params.screen_type.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.subheader("How to Interpret This Screen")
    st.write(SCREEN_EXPLANATIONS[params.screen_type])
    st.write(
        "Composite scores are percentile-based. They are useful for comparing "
        "qualified stocks within the same screen, but they are not forecasts or "
        "investment recommendations."
    )

    if show_charts:
        render_chart_section(results)


if __name__ == "__main__":
    main()
