import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

st.title("M² Portfolio Tracker (Auto-Saving)")

# -----------------------------------------
# Load / save portfolio helpers
# -----------------------------------------
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return []

def save_portfolio(data):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)

# -----------------------------------------
# Initialize persistent state
# -----------------------------------------
if "positions" not in st.session_state:
    st.session_state.positions = load_portfolio()

# -----------------------------------------
# INPUT SECTION
# -----------------------------------------
st.subheader("Your Saved Portfolio")

ticker = st.text_input("Add ticker (e.g., AAPL)")
weight = st.number_input("Weight (0–1)", min_value=0.0, max_value=1.0, value=0.1)

if st.button("Add position"):
    if ticker != "":
        st.session_state.positions.append({"ticker": ticker.upper(), "weight": weight})
        save_portfolio(st.session_state.positions)
        st.success(f"Added {ticker.upper()}")

st.write(pd.DataFrame(st.session_state.positions))

remove_ticker = st.text_input("Ticker to remove")
if st.button("Remove"):
    st.session_state.positions = [
        p for p in st.session_state.positions if p["ticker"] != remove_ticker.upper()
    ]
    save_portfolio(st.session_state.positions)
    st.success(f"Removed {remove_ticker.upper()}")

benchmark = st.text_input("Benchmark", value="IWRD.L")
start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))
run = st.button("Compute")

# -----------------------------------------
# COMPUTATION
# -----------------------------------------
if run:

    if len(st.session_state.positions) == 0:
        st.error("You need at least one position.")
        st.stop()

    tickers = [p["ticker"] for p in st.session_state.positions]
    weights = np.array([p["weight"] for p in st.session_state.positions])

    if weights.sum() == 0:
        st.error("Weights cannot all be zero.")
        st.stop()

    weights = weights / weights.sum()  # normalize

    data = yf.download(tickers + [benchmark], start=start_date)["Close"]
    rets = data.pct_change().dropna()
    port_ret = rets[tickers] @ weights
    bench_ret = rets[benchmark]

    rf = 0.05 / 252

    Rp = port_ret.mean() * 252
    Rb = bench_ret.mean() * 252
    sig_p = port_ret.std() * np.sqrt(252)
    sig_b = bench_ret.std() * np.sqrt(252)

    sharpe = (Rp - 0.05) / sig_p
    M2 = sharpe * sig_b + 0.05

    # -----------------------------------------
    # OUTPUT
    # -----------------------------------------
    st.subheader("Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Return", f"{Rp:.2%}")
    col2.metric("Benchmark Return", f"{Rb:.2%}")
    col3.metric("M²", f"{M2:.2%}")

    cum_p = (1 + port_ret).cumprod()
    cum_b = (1 + bench_ret).cumprod()

    fig, ax = plt.subplots(figsize=(18,10))
    ax.plot(cum_p, label="Portfolio")
    ax.plot(cum_b, label="Benchmark")
    ax.legend()
    ax.set_title("Cumulative Returns")
    st.pyplot(fig)
