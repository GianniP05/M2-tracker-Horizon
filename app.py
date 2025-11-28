import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------
# QUANT UI STYLE
# ------------------------------------------------
st.set_page_config(page_title="M² Portfolio Tracker (Research Committee)", layout="wide")

# Inject custom CSS for quant-style look
st.markdown("""
    <style>

    /* Make page wider */
    .block-container {
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Reduce vertical space */
    .element-container {
        margin-bottom: 0.5rem !important;
    }

    /* Professional tables */
    table {
        border-collapse: collapse;
        width: 100%;
    }
    thead th {
        background-color: #0e1117 !important;
        color: white !important;
        border-bottom: 1px solid #444 !important;
    }
    tbody tr:nth-child(even) {
        background-color: #161a23 !important;
    }

    /* Clean metric display */
    .metric-container .metric-value {
        font-weight: 600 !important;
    }

    /* Section headers */
    h2, h3, h4 {
        margin-top: 0.2rem !important;
    }

    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
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
# LAYOUT ORGANIZATION
# ------------------------------------------------
col_left, col_right = st.columns([1.5, 1])

with col_left:
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

    st.subheader("📊 Current Portfolio")
    if len(st.session_state.positions) > 0:
        st.dataframe(pd.DataFrame(st.session_state.positions))
    else:
        st.info("Your current portfolio is empty.")

    remove_ticker = st.text_input("Remove ticker")
    if st.button("Remove"):
        st.session_state.positions = [
            p for p in st.session_state.positions if p["ticker"] != remove_ticker.upper()
        ]
        save_to_url()
        st.success(f"Removed {remove_ticker.upper()}")


with col_right:
    st.subheader("Benchmark")
    benchmark = st.text_input("Benchmark ticker (IWRD.L is used in the competition)", st.session_state.benchmark)
    st.session_state.benchmark = benchmark

    starting_value = st.number_input(
        "💰 Starting portfolio value (€)",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )

    start_date = st.date_input("Start date", value=pd.to_datetime("2020-01-01"))

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

    if weights.sum() == 0:
        st.error("Total weight cannot be zero.")
        st.stop()

    weights = weights / weights.sum()

    try:
        data = yf.download(tickers + [benchmark], start=start_date)["Close"]
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    if data is None or data.empty:
        st.error("No market data returned.")
        st.stop()

    rets = data.pct_change().dropna()

    if rets.empty or len(rets) < 2:
        st.error("Not enough data to compute returns.")
        st.stop()

    port_ret = rets[tickers] @ weights

    try:
        bench_ret = rets[benchmark]
    except KeyError:
        st.error("Benchmark ticker not found.")
        st.stop()

    rf = 0.05 / 252

    Rp = port_ret.mean() * 252
    Rb = bench_ret.mean() * 252
    sig_p = port_ret.std() * np.sqrt(252)
    sig_b = bench_ret.std() * np.sqrt(252)
    sharpe = (Rp - 0.05) / sig_p
    M2 = sharpe * sig_b + 0.05

    cum_p = (1 + port_ret).cumprod()
    cum_b = (1 + bench_ret).cumprod()

    port_value = cum_p * starting_value
    bench_value = cum_b * starting_value

    current_value = float(port_value.iloc[-1])
    profit = current_value - starting_value
    absolute_return = (current_value / starting_value) - 1

    # ------------------------------------------------
    # RESULTS DISPLAY
    # ------------------------------------------------
    st.subheader("💶 Portfolio Results (Real Money)")

    colA, colB, colC = st.columns(3)
    colA.metric("Starting Value", f"€{starting_value:,.2f}")
    colB.metric("Current Value", f"€{current_value:,.2f}", f"{absolute_return:.2%}")
    colC.metric("Profit / Loss", f"€{profit:,.2f}")

    st.subheader("📈 Risk-Adjusted Metrics")
    colX, colY, colZ = st.columns(3)
    colX.metric("Portfolio Return (Annualized)", f"{Rp:.2%}")
    colY.metric("Benchmark Return (Annualized)", f"{Rb:.2%}")
    colZ.metric("M²", f"{M2:.2%}")

    # ------------------------------------------------
    # CHART (unchanged look)
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(port_value, label="Portfolio (€)")
    ax.plot(bench_value, label="Benchmark (€)")
    ax.legend()
    ax.set_title("Portfolio Value Over Time (€)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (€)")
    st.pyplot(fig)

# Reset button
if st.button("Reset My Portfolio"):
    st.session_state.positions = []
    save_to_url()
    st.success("Your portfolio has been reset.")










