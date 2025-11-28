import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------
# QUANT UI STYLE
# ------------------------------------------------
st.set_page_config(page_title="M² Portfolio Tracker (Research Committee)", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .element-container {
        margin-bottom: 0.4rem !important;
    }
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
    .metric-container .metric-value {
        font-weight: 600 !important;
    }
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
    entries = ",".join([p["entry"] for p in st.session_state.positions])
    st.experimental_set_query_params(tickers=tickers, weights=weights, entries=entries)


def load_from_url():
    params = st.experimental_get_query_params()

    tickers = params.get("tickers", [""])[0]
    weights = params.get("weights", [""])[0]
    entries = params.get("entries", [""])[0]

    positions = []
    if tickers.strip() != "":
        t_list = tickers.split(",")
        w_list = [float(w) for w in weights.split(",")]
        e_list = entries.split(",")

        for t, w, e in zip(t_list, w_list, e_list):
            positions.append({"ticker": t, "weight": w, "entry": e})

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
# LAYOUT (LEFT = PORTFOLIO / RIGHT = SETTINGS)
# ------------------------------------------------
col_left, col_right = st.columns([1.5, 1])


with col_left:
    st.subheader("Your Portfolio")

    ticker = st.text_input("Add ticker (e.g., AAPL)")
    weight = st.number_input("Weight (0–1)", min_value=0.0, max_value=1.0, value=0.1)
    entry_date = st.date_input("Entry date of this position")

    if st.button("Add position"):
        if ticker.strip() != "":
            st.session_state.positions.append({
                "ticker": ticker.upper(),
                "weight": weight,
                "entry": str(entry_date)
            })
            save_to_url()
            st.success(f"Added {ticker.upper()}")
        else:
            st.error("Please enter a ticker.")

    st.subheader("📊 Current Portfolio")
    if len(st.session_state.positions) > 0:
        df = pd.DataFrame(st.session_state.positions)
        df_display = df.rename(columns={
            "ticker": "Ticker",
            "weight": "Target Weight",
            "entry": "Entry Date"
        })
        st.dataframe(df_display)
    else:
        st.info("Your current portfolio is empty.")

    remove_ticker = st.text_input("Remove ticker")
    if st.button("Remove"):
        st.session_state.positions = [
            p for p in st.session_state.positions if p["ticker"] != remove_ticker.upper()
        ]
        save_to_url()
        st.success(f"Removed {remove_ticker.upper()}")

    # ------------------------------------------------
# SELL A POSITION (EXTREMELY CLEAR)
# ------------------------------------------------
st.subheader("📉 Sell a Position")

st.markdown("""
**How selling works (very clear):**
- You pick the **ticker** you want to sell.  
- You pick the **entry date** that matches the exact trade you want to close.  
- You pick the **sell date**, which is the day you exited the position.  
- The system removes ONLY that position (not all positions of that ticker).  
""")

if len(st.session_state.positions) > 0:

    # Dropdown of tickers in portfolio
    sell_ticker = st.selectbox(
        "Select ticker to sell",
        sorted(list({p["ticker"] for p in st.session_state.positions}))
    )

    # Filter entry dates for that ticker
    matching_entries = sorted(
        [p["entry"] for p in st.session_state.positions if p["ticker"] == sell_ticker]
    )

    sell_entry_date = st.selectbox(
        "Select the entry date of the position you want to sell",
        matching_entries
    )

    sell_date = st.date_input(
        "Select the date you sold this position"
    )

    if st.button("Sell Position"):
        before_count = len(st.session_state.positions)

        # Remove only the matching position
        st.session_state.positions = [
            p for p in st.session_state.positions
            if not (p["ticker"] == sell_ticker and p["entry"] == sell_entry_date)
        ]

        save_to_url()

        after_count = len(st.session_state.positions)

        if after_count < before_count:
            st.success(
                f"Sold {sell_ticker} (entered on {sell_entry_date}) on {sell_date}. "
                "The position has now been removed from your portfolio."
            )
        else:
            st.error(
                "No position was removed. Double-check the entry date."
            )

else:
    st.info("You must add positions before you can sell them.")


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
        st.error("Add at least one position.")
        st.stop()

    tickers = [p["ticker"] for p in st.session_state.positions]
    weights = np.array([p["weight"] for p in st.session_state.positions])

    if weights.sum() == 0:
        st.error("Total weight cannot be zero.")
        st.stop()

    weights = weights / weights.sum()

    # Use earliest entry date in portfolio
    min_entry_date = min(pd.to_datetime(p["entry"]) for p in st.session_state.positions)

    try:
        data = yf.download(tickers + [benchmark], start=min_entry_date)["Close"]
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    if data is None or data.empty:
        st.error("No price data returned.")
        st.stop()

    rets = data.pct_change().dropna()
    if rets.empty:
        st.error("Not enough price history.")
        st.stop()

    # ------------------------------------------------
    # POSITION-BASED RETURNS (STAGGERED ENTRIES)
    # ------------------------------------------------
    aligned_port_rets = []

    for p in st.session_state.positions:
        t = p["ticker"]
        w = p["weight"]
        e = pd.to_datetime(p["entry"])

        sliced = rets.loc[rets.index >= e, t]
        aligned_port_rets.append(sliced * w)

    aligned_df = pd.concat(aligned_port_rets, axis=1).fillna(0)
    port_ret = aligned_df.sum(axis=1)

    try:
        bench_ret = rets[benchmark]
    except KeyError:
        st.error("Benchmark not found.")
        st.stop()

    # ------------------------------------------------
    # ANNUALIZATION + M2
    # ------------------------------------------------
    rf = 0.05 / 252
    Rp = port_ret.mean() * 252
    Rb = bench_ret.mean() * 252
    sig_p = port_ret.std() * np.sqrt(252)
    sig_b = bench_ret.std() * np.sqrt(252)

    if sig_p == 0:
        st.error("Portfolio volatility is zero.")
        st.stop()

    sharpe = (Rp - 0.05) / sig_p
    M2 = sharpe * sig_b + 0.05

    # ------------------------------------------------
    # REAL MONEY PORTFOLIO VALUE
    # ------------------------------------------------
    cum_p = (1 + port_ret).cumprod()
    cum_b = (1 + bench_ret).cumprod()

    port_value = cum_p * starting_value
    bench_value = cum_b * starting_value

    current_value = float(port_value.iloc[-1])
    profit = current_value - starting_value
    absolute_return = (current_value / starting_value) - 1

    # ------------------------------------------------
    # MONEY WEIGHTS PER POSITION (€)
    # ------------------------------------------------
    money_weights = {}
    for p in st.session_state.positions:
        t = p["ticker"]
        w = p["weight"]
        e = pd.to_datetime(p["entry"])

        indiv_cum = (1 + rets.loc[rets.index >= e, t]).cumprod()
        indiv_val = float(indiv_cum.iloc[-1] * starting_value * w)

        money_weights[t] = indiv_val

    st.subheader("💰 Current Money Weights (€)")
    st.dataframe(pd.DataFrame.from_dict(money_weights, orient='index', columns=["Value (€)"]))

    # ------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------
    st.subheader("💶 Portfolio Results (Real Money)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Starting Value", f"€{starting_value:,.2f}")
    c2.metric("Current Value", f"€{current_value:,.2f}", f"{absolute_return:.2%}")
    c3.metric("Profit / Loss", f"€{profit:,.2f}")

    st.subheader("📈 Risk-Adjusted Metrics")
    r1, r2, r3 = st.columns(3)
    r1.metric("Portfolio Return (Annualized)", f"{Rp:.2%}")
    r2.metric("Benchmark Return (Annualized)", f"{Rb:.2%}")
    r3.metric("M²", f"{M2:.2%}")

    # ------------------------------------------------
    # CHART (UNCHANGED LOOK)
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
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







