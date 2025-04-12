from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Constants
COLOR_PALETTE = [
    "#d62728",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
MONTH_ABBR = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
DEFAULT_TICKER = "AAPL"
CACHE_TTL = 3600  # Cache data for 1 hour

# Page configuration
st.set_page_config(page_title="Yearly Performance Analyzer", layout="wide")


def main() -> None:
    """Main function to render the Yearly Performance Analyzer page"""
    st.title("📊 Yearly Performance Analyzer")

    # Get user inputs
    ticker_symbol, years_to_display = get_user_inputs()

    # Display main content
    st.header(f"Yearly Performance for {ticker_symbol.upper()}")

    # Process and display data
    if ticker_symbol:
        process_and_display_data(ticker_symbol, years_to_display)
    else:
        st.info("Please enter a ticker symbol to get started.")


def get_user_inputs() -> tuple[str, list[int]]:
    """
    Collect and process user inputs from the sidebar

    Returns:
        Tuple containing ticker symbol and list of years to display
    """
    with st.sidebar:
        st.header("Settings")

        # Ticker input
        ticker_symbol = st.text_input("Enter Ticker Symbol", value=DEFAULT_TICKER)

        # Year selection inputs
        current_year = datetime.now().year
        num_years = st.slider(
            "Number of Years to Display", min_value=1, max_value=10, value=5
        )

        # Calculate available years
        all_years = list(range(current_year, current_year - num_years, -1))

        # Let user select which years to display
        st.subheader("Select Years to Display")
        years_to_display = st.multiselect(
            "Choose years to display on the chart", options=all_years, default=all_years
        )

    return ticker_symbol, years_to_display


@st.cache_data(ttl=CACHE_TTL)
def get_asset_data(ticker: str, years: list[int]) -> pd.DataFrame | None:
    """
    Fetch and process asset data from Yahoo Finance

    Args:
        ticker: Stock ticker symbol
        years: List of years to analyze

    Returns:
        DataFrame with asset price data or None if data retrieval failed
    """
    try:
        if not years:
            st.warning("Please select at least one year to display")
            return None

        # Calculate start date (January 1st of the earliest year needed)
        earliest_year = min(years)
        start_date = f"{earliest_year - 1}-12-31"  # Get data from end of previous year
        end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch data
        asset_df = yf.download(tickers=ticker, start=start_date, end=end_date)

        if asset_df is None or asset_df.empty:
            st.error(f"No data found for ticker '{ticker}'")
            return None

        # Clean up multi-level columns if present
        if isinstance(asset_df.columns, pd.MultiIndex):
            asset_df.columns = asset_df.columns.get_level_values(0)

        return asset_df

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def process_yearly_data(
    data: pd.DataFrame, years: list[int]
) -> dict[str, dict[int, Any]] | None:
    """
    Process data to create normalized year-over-year comparison

    Args:
        data: DataFrame with price data
        years: List of years to process

    Returns:
        Dictionary with normalized data, original data and dates by year
    """
    if data is None or not years:
        return None

    yearly_data = {}  # Normalized percentage change
    yearly_original = {}  # Original price values
    yearly_dates = {}  # Original dates

    for year in years:
        # Get data for this year
        year_data = data[cast(pd.DatetimeIndex, data.index).year == year][
            "Close"
        ].copy()

        # Only process if we have enough data points
        if len(year_data) > 5:
            # Find first non-NaN value
            first_valid_idx = year_data.first_valid_index()

            if first_valid_idx is not None:
                first_value = year_data.loc[first_valid_idx]

                # Fill NaN values using forward fill then backward fill
                year_data = year_data.ffill().bfill()

                # Store original data and dates
                yearly_original[year] = year_data.copy()
                yearly_dates[year] = year_data.index

                # Normalize to percentage change from first valid value
                year_data = ((year_data / first_value) - 1) * 100

                # Create a date index with the year set to a common year for alignment
                common_year_index = [
                    pd.Timestamp(2000, date.month, date.day) for date in year_data.index
                ]
                yearly_data[year] = pd.Series(year_data.values, index=common_year_index)

    result = {
        "normalized": yearly_data,
        "original": yearly_original,
        "dates": yearly_dates,
    }

    return result


def create_performance_chart(
    yearly_results: dict[str, dict[int, Any]],
    ticker_symbol: str,
    years_to_display: list[int],
) -> go.Figure:
    """
    Create a Plotly chart comparing yearly performance

    Args:
        yearly_results: Processed data from process_yearly_data
        ticker_symbol: Stock ticker symbol
        years_to_display: Years to include in the chart

    Returns:
        Plotly figure object with the chart
    """
    yearly_data = yearly_results["normalized"]
    yearly_original = yearly_results["original"]
    yearly_dates = yearly_results["dates"]

    # Create figure
    fig = go.Figure()

    # Add traces for each year
    for i, year in enumerate(years_to_display):
        if year in yearly_data:
            year_series: pd.Series = yearly_data[year]
            original_series: pd.Series = yearly_original[year]
            actual_dates: pd.DatetimeIndex = yearly_dates[year]

            # Format dates for x-axis
            dates = cast(pd.DatetimeIndex, year_series.index)

            # Create hover text with detailed information
            hover_text = create_hover_text(actual_dates, original_series, year_series)

            # Create custom data for additional info on hover
            custom_data = [
                [d.strftime("%Y-%m-%d"), p, v]
                for d, p, v in zip(actual_dates, original_series, year_series.values)
            ]

            # Add the trace for this year
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=year_series.values,
                    mode="lines",
                    name=f"{year}",
                    line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)], width=2),
                    hoverinfo="text",
                    hovertext=hover_text if len(hover_text) == len(dates) else None,
                    customdata=custom_data,
                )
            )

    # Configure layout
    fig.update_layout(
        title=f"{ticker_symbol} - Yearly Performance Comparison (%)",
        xaxis=dict(
            title="Month",
            tickformat="%b",
            tickmode="array",
            tickvals=pd.date_range(start="2000-01-01", end="2000-12-31", freq="MS"),
            ticktext=MONTH_ABBR,
        ),
        yaxis=dict(title="Performance (%)", ticksuffix="%"),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_hover_text(
    actual_dates: pd.DatetimeIndex, original_series: pd.Series, year_series: pd.Series
) -> list[str]:
    """
    Create formatted hover text for the performance chart

    Args:
        actual_dates: Original dates for the data points
        original_series: Original price data
        year_series: Normalized performance data

    Returns:
        List of formatted hover text strings
    """
    hover_text = []

    for j, (_, value) in enumerate(year_series.items()):
        if j < len(actual_dates):
            actual_date: pd.Timestamp = actual_dates[j]
            price: np.float64 = original_series.iloc[j]
            hover_text.append(
                f"<b>Date</b>: {actual_date.strftime('%Y-%m-%d')}<br>"
                + f"<b>Day</b>: {actual_date.strftime('%A')}<br>"
                + f"<b>Price</b>: ${price:.2f}<br>"
                + f"<b>Performance</b>: {value:.2f}%"
            )

    return hover_text


def create_summary_table(
    yearly_data: dict[int, pd.Series],
    yearly_original: dict[int, pd.Series],
    years_to_display: list[int],
) -> pd.DataFrame | None:
    """
    Create summary table showing end-of-year performance

    Args:
        yearly_data: Normalized yearly performance data
        yearly_original: Original price data by year
        years_to_display: Years to include in the summary

    Returns:
        DataFrame with yearly summary or None if no data
    """
    # Calculate year-end performance for each year
    year_end_perf = {}
    year_end_price = {}

    for year in years_to_display:
        if year in yearly_data:
            year_series = yearly_data[year]
            orig_series = yearly_original[year]

            if not year_series.empty:
                year_end_perf[year] = year_series.iloc[-1]
                year_end_price[year] = orig_series.iloc[-1]

    # Create summary table
    if year_end_perf:
        perf_df = pd.DataFrame(
            {
                "Year": list(year_end_perf.keys()),
                "Year-End Price": [
                    f"${price:.2f}" for price in year_end_price.values()
                ],
                "Year Performance (%)": [
                    f"{perf:.2f}%" for perf in year_end_perf.values()
                ],
            }
        ).set_index("Year")

        return perf_df

    return None


def process_and_display_data(ticker_symbol: str, years_to_display: list[int]) -> None:
    """
    Process data and display visualizations for the given ticker and years

    Args:
        ticker_symbol: Stock ticker symbol
        years_to_display: Years to analyze and display
    """
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        # Get asset data
        asset_data = get_asset_data(ticker_symbol, years_to_display)

        if asset_data is not None and years_to_display:
            # Process data for yearly comparison
            yearly_results = process_yearly_data(
                data=asset_data, years=years_to_display
            )

            if yearly_results and yearly_results["normalized"]:
                # Create and display performance chart
                fig = create_performance_chart(
                    yearly_results, ticker_symbol, years_to_display
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show additional information
                st.subheader("Performance Summary")

                # Create summary table
                summary_df = create_summary_table(
                    yearly_results["normalized"],
                    yearly_results["original"],
                    years_to_display,
                )

                if summary_df is not None:
                    st.table(summary_df)
            else:
                st.warning(
                    f"No complete yearly data available for {ticker_symbol} in the selected years."
                )
        elif years_to_display:
            st.error(
                f"Failed to retrieve data for {ticker_symbol}. Please check the ticker symbol."
            )
        else:
            st.warning("Please select at least one year to display")


# Entry point of the application
if __name__ == "__main__":
    main()
