import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------
# STYLING — QUANT THEME
# ------------------------------------------------
st.set_page_config(page_title="M² Portfolio Tracker (Research Committee)", layout="wide")

custom_css = """
<style>
    /* Global background */
    .main {
        background-color: #0a0c10;
        color: #dcdcdc;
        font-family: 'Roboto', sans-serif;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #E0E0E0 !important;
        font-weight: 500;
    }

    /* Titles and labels */
    .stTextInput>div>div>input {
        background-color: #11141a;
        color: #E0E0E0;
        border: 1px solid #2c3038;
    }

    .stNumberInput>div>div>input {
        background-color: #11141a;
        color: #E0E0E0;
        border: 1px solid #2c3038;
    }

    /* DataFrame styling */
    .dataframe {
        background-color: #0f1116;
        color: #E0E0E0;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #4EA8DE;
        font-weight: 600;
        font-size: 32px;
    }
    [data-testid="stMetricLabel"] {
        color: #A9A9A9;
        font-size: 15px;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #1f6feb !important;
        color: white !important;
        border-radius: 6px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>M² Portfolio Tracker — Horizon Capital Research</h1>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

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
# SIDE INPUT PANEL
# ------------------------------------------------
with st.container():
    st.markdown("<h2>Portfolio Construction</h2>", unsafe_allow_html=True)

    colA, colB = st.columns([2,1])

    with colA:
        ticker = st.text_input("Add ticker")
    with colB:
        weight = st.number_input("Weight (0–1)", min_value=0.0, max_value=1.0, value=0.1)

    if st.button("Add Position"):
        if ticker.strip() != "":
            st.session_state.positions.append({"ticker": ticker.upper(), "weight": weight})
            save_to_url()
            st.success(f"Added {ticker.upper()}")
        else:
            st.error("Please enter a ticker.")

# ------------------------------------------------
# CURRENT PORTFOLIO TABLE
# ------------------------------------------------
st.markdown("<h3>Current Portfolio</h3>", unsafe_allow_html=True)
if len(st.session_state.positions) > 0:
    st.dataframe(pd.DataFrame(st.session_state.positions), use_container_width=True)
else:
    st.info("No holdings yet.")

remove_ticker = st.text_input("Remove ticker")
if st.button("Remove"):
    st.session_state.positions = [
        p for p in st.session_state.positions if p["ticker"] != remove_ticker.upper()
    ]
    save_to_url()
    st.success(f"Removed {remove_ticker.upper()}")

# ------------------------------------------------
# BENCHMARK & START VALUE
# ------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.session_state.benchmark = st.text_input("Benchmark", st.session_state.benchmark)
with col2:
    starting_value = st.number_input("Starting Value (€)", min_value=0.0, value=10000.0)
with col3:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))

run = st.button("Run Analysis")

# ------------------------------------------------
# PERFORMANCE CALCULATION
# ------------------------------------------------
if run:

    if len(st.session_state.positions) == 0:
        st.error("Portfolio is empty.")
        st.stop()

    tickers = [p["ticker"] for p in st.session_state.positions]
    weights = np.array([p["weight"] for p in st.session_state.positions])

    if weights.sum() == 0:
        st.error("Total weight cannot be zero.")
        st.stop()
    weights = weights / weights.sum()

    try:
        data = yf.download(tickers + [st.session_state.benchmark], start=start_date)["Close"]
    except:
        st.error("Could not download data.")
        st.stop()

    if data is None or data.empty:
        st.error("No data returned.")
        st.stop()

    rets = data.pct_change().dropna()

    port_ret = rets[tickers] @ weights
    bench_ret = rets[st.session_state.benchmark]

    # Risk-free
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

    # ------------------------------------------------
    # KPI SECTION (QUANT STYLE)
    # ------------------------------------------------
    st.markdown("<h2>Performance Overview</h2>", unsafe_allow_html=True)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Portfolio Return (Annualized)", f"{Rp:.2%}")
    kpi2.metric("Benchmark Return (Annualized)", f"{Rb:.2%}")
    kpi3.metric("M²", f"{M2:.2%}")

    kpi4, kpi5, kpi6 = st.columns(3)
    kpi4.metric("Starting Value", f"€{starting_value:,.0f}")
    kpi5.metric("Current Value", f"€{current_value:,.0f}")
    kpi6.metric("P/L", f"€{profit:,.0f}")

    # ------------------------------------------------
    # PLOT — Quant Theme
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(port_value, label="Portfolio (€)", linewidth=2)
    ax.plot(bench_value, label="Benchmark (€)", linewidth=2, alpha=0.8)

    ax.set_facecolor("#0f1116")
    fig.patch.set_facecolor("#0f1116")
    ax.tick_params(colors="#cccccc")
    ax.set_title("Portfolio Value Over Time (€)", color="#cccccc", fontsize=16)
    ax.set_xlabel("Date", color="#cccccc")
    ax.set_ylabel("Value (€)", color="#cccccc")
    ax.legend()

    st.pyplot(fig)

# ------------------------------------------------
# RESET
# ------------------------------------------------
if st.button("Reset Portfolio"):
    st.session_state.positions = []
    save_to_url()
    st.success("Portfolio reset.")












