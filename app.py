import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="M² Portfolio Tracker (Research Committee)", layout="centered")
st.title("📈 M² Portfolio Tracker by Horizon Capital")


# ------------------------------------------------
# URL STORAGE HELPERS
# ------------------------------------------------
def save_to_url():
    tickers = ",".join([p["ticker"] for p in st.session_state.positions])
    weights = ",".join([str(p["weight"]) for p in st.session_state.positions])
    st.experimental_set_query_params(tickers=tickers, weights=weights)


def load_from_url():
    params = st.experimental_get_query_params()
    tickers = params.get("tickers", [""])[0]
    weights = params.get("weights", [""])[0]

    positions = []

    if tickers.strip() != "":
        tick_list = tickers.split(",")
        weight_list = [float(w) for w in weights.split(",")]

        for t, w in zip(tick_list, weight_list):
            positions.append({"ticker": t.strip(), "weight": w})

    return positions


# ------------------------------------------------
# INITIALIZATION
# ------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.positions = load_from_url()
    st.session_state.initialized = True

if "benchmark" not in st.session_state:
    st.session_state.benchmark = "IWRD.L"


# ------------------------------------------------
# INPUT SECTION
# ------------------------------------------------
st.subheader("Your Portfolio")

ticker = st.text_input("Add ticker (e.g., AAPL)", "")
weight = st.number_input("Weight (0–1)", min_value=0.0, max_value=1.0, value=0.1)

if st.button("Add position"):
    if ticker.strip() != "":
        st.session_state.positions.append({
            "ticker": ticker.upper(),
            "weight": weight
        })
        save_to_url()
        st.success(f"Added {ticker.upper()}")
    else:
        st.error("Please enter a ticker.")


# Show starting vs current portfolios
initial_positions = load_from_url()

st.subheader("📦 Starting Portfolio")
if len(initial_positions) > 0:
    st.dataframe(pd.DataFrame(initial_positions))
else:
    st.info("No starting portfolio stored in link.")

st.subheader("📊 Current Portfolio")
if len(st.session_state.positions) > 0:
    st.dataframe(pd.DataFrame(st.session_state.positions))
else:
    st.info("Your current portfolio is empty.")


# Remove ticker
remove_ticker = st.text_input("Remove ticker")
if st.button("Remove"):
    st.session_state.positions = [
        p for p in st.session_state.positions if p["ticker"] != remove_ticker.upper()
    ]
    save_to_url()
    st.success(f"Removed {remove_ticker.upper()}")


# Benchmark
st.subheader("Benchmark")
benchmark = st.text_input("Benchmark ticker (IWRD.L is used in the competition)", st.session_state.benchmark)
st.session_state.benchmark = benchmark


# Starting portfolio value in €
starting_value = st.number_input(
    "💰 Starting portfolio value (€)",
    min_value=0.0,
    value=10000.0,
    step=100.0
)


# Start date
start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))


# Compute performance
run = st.button("Compute Performance")


# ------------------------------------------------
# PERFORMANCE CALCULATION
# ------------------------------------------------
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

    # Download market data
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
    rf = 0.05 / 252

    # Annualized statistics
    Rp = port_ret.mean() * 252
    Rb = bench_ret.mean() * 252
    sig_p = port_ret.std() * np.sqrt(252)
    sig_b = bench_ret.std() * np.sqrt(252)

    if sig_p == 0:
        st.error("Portfolio volatility is zero — cannot compute M².")
        st.stop()

    sharpe = (Rp - 0.05) / sig_p
    M2 = sharpe * sig_b + 0.05

    # --------------------------------------
    # Real Money Calculations
    # --------------------------------------
    cum_p = (1 + port_ret).cumprod()
    cum_b = (1 + bench_ret).cumprod()

    port_value = cum_p * starting_value
    bench_value = cum_b * starting_value

    current_value = float(port_value.iloc[-1])
    profit = current_value - starting_value
    absolute_return = (current_value / starting_value) - 1

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------
    st.subheader("💶 Portfolio Results (Real Money)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Starting Value", f"€{starting_value:,.2f}")
    col2.metric("Current Value", f"€{current_value:,.2f}", f"{absolute_return:.2%}")
    col3.metric("Profit / Loss", f"€{profit:,.2f}")

    st.subheader("📈 Risk-Adjusted Metrics")
    colA, colB, colC = st.columns(3)
    colA.metric("Portfolio Return (Annualized)", f"{Rp:.2%}")
    colB.metric("Benchmark Return (Annualized)", f"{Rb:.2%}")
    colC.metric("M²", f"{M2:.2%}")

    # --------------------------------------
    # PLOT (REAL MONEY)
    # --------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(port_value, label="Portfolio (€)")
    ax.plot(bench_value, label="Benchmark (€)")
    ax.legend()
    ax.set_title("Portfolio Value Over Time (€)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (€)")
    st.pyplot(fig)


# ------------------------------------------------
# RESET BUTTON
# ------------------------------------------------
if st.button("Reset My Portfolio"):
    st.session_state.positions = []
    save_to_url()
    st.success("Your portfolio has been reset.")










