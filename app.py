import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="M² Portfolio Tracker", layout="centered")
st.title("📈 M² Portfolio Tracker by Horizon Capital")

# -----------------------------------------
# Initialize user-specific session state
# -----------------------------------------
if "positions" not in st.session_state:
    st.session_state.positions = []     # user's private portfolio
if "benchmark" not in st.session_state:
    st.session_state.benchmark = "IWRD.L"


# -----------------------------------------
# INPUT SECTION
# -----------------------------------------
st.subheader("Your Portfolio")

ticker = st.text_input("Add ticker (e.g., AAPL)", "")
weight = st.number_input("Weight (0–1)", min_value=0.0, max_value=1.0, value=0.1)

if st.button("Add position"):
    if ticker.strip() != "":
        st.session_state.positions.append({
            "ticker": ticker.upper(),
            "weight": weight
        })
        st.success(f"Added {ticker.upper()}")
    else:
        st.error("Please enter a ticker.")


# Show current user portfolio
if len(st.session_state.positions) > 0:
    st.write("### Current Portfolio")
    st.dataframe(pd.DataFrame(st.session_state.positions))
else:
    st.info("No tickers added yet.")


# Remove position
remove_ticker = st.text_input("Remove ticker")
if st.button("Remove"):
    st.session_state.positions = [
        p for p in st.session_state.positions
        if p["ticker"] != remove_ticker.upper()
    ]
    st.success(f"Removed {remove_ticker.upper()}")


# Benchmark input
st.subheader("Benchmark")
benchmark = st.text_input("Benchmark ticker", st.session_state.benchmark)
st.session_state.benchmark = benchmark


# Start date
start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))


# Compute button
run = st.button("Compute Performance")


# -----------------------------------------
# PERFORMANCE CALCULATION
# -----------------------------------------
if run:

    if len(st.session_state.positions) == 0:
        st.error("Add at least one position to compute performance.")
        st.stop()

    tickers = [p["ticker"] for p in st.session_state.positions]
    weights = np.array([p["weight"] for p in st.session_state.positions])

    # Normalize weights
    if weights.sum() == 0:
        st.error("Total weight cannot be zero.")
        st.stop()

    weights = weights / weights.sum()

    # Download data
    try:
        data = yf.download(tickers + [benchmark], start=start_date)["Close"]
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    if data is None or data.empty:
        st.error("No market data returned. Check ticker symbols or start date.")
        st.stop()

    rets = data.pct_change().dropna()

    if rets.empty or len(rets) < 2:
        st.error("Not enough data to compute returns. Try an earlier start date.")
        st.stop()

    # Portfolio returns
    port_ret = rets[tickers] @ weights

    # Benchmark returns
    try:
        bench_ret = rets[benchmark]
    except KeyError:
        st.error("Benchmark ticker not found in downloaded data.")
        st.stop()

    # Risk-free rate
    rf = 0.05 / 252   # 5% annual

    # Annualized stats
    Rp = port_ret.mean() * 252
    Rb = bench_ret.mean() * 252
    sig_p = port_ret.std() * np.sqrt(252)
    sig_b = bench_ret.std() * np.sqrt(252)

    if sig_p == 0:
        st.error("Portfolio volatility is zero — cannot compute M².")
        st.stop()

    sharpe = (Rp - 0.05) / sig_p
    M2 = sharpe * sig_b + 0.05

    # Display results
    st.subheader("📊 Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Return", f"{Rp:.2%}")
    col2.metric("Benchmark Return", f"{Rb:.2%}")
    col3.metric("M²", f"{M2:.2%}")

    # Plot cumulative returns
    cum_p = (1 + port_ret).cumprod()
    cum_b = (1 + bench_ret).cumprod()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum_p, label="Portfolio")
    ax.plot(cum_b, label="Benchmark")
    ax.legend()
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")

    st.pyplot(fig)


# -----------------------------------------
# Reset portfolio button
# -----------------------------------------
if st.button("Reset My Portfolio"):
    st.session_state.positions = []
    st.success("Your private portfolio has been reset!")





