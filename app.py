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
# URL STORAGE HELPERS (for OPEN positions only)
# ------------------------------------------------
def save_to_url():
    # Persist open positions only (ticker, amount, entry)
    tickers = ",".join([p["ticker"] for p in st.session_state.positions])
    amounts = ",".join([str(p["amount"]) for p in st.session_state.positions])
    entries = ",".join([p["entry"] for p in st.session_state.positions])
    st.experimental_set_query_params(tickers=tickers, amounts=amounts, entries=entries)


def load_from_url():
    params = st.experimental_get_query_params()

    tickers = params.get("tickers", [""])[0]
    amounts = params.get("amounts", [""])[0]
    entries = params.get("entries", [""])[0]

    positions = []
    if tickers.strip() != "":
        t_list = tickers.split(",")
        a_list = [float(a) for a in amounts.split(",")]
        e_list = entries.split(",")

        for t, a, e in zip(t_list, a_list, e_list):
            positions.append({"ticker": t, "amount": a, "entry": e})

    return positions


# ------------------------------------------------
# INITIALIZATION
# ------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.positions = load_from_url()   # open positions
    st.session_state.trades = []                  # closed trades (sell log)
    st.session_state.initialized = True

if "benchmark" not in st.session_state:
    st.session_state.benchmark = "IWRD.L"

if "force_run" not in st.session_state:
    st.session_state.force_run = False


# ------------------------------------------------
# LAYOUT (LEFT = PORTFOLIO / RIGHT = SETTINGS)
# ------------------------------------------------
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.subheader("Your Portfolio")

    ticker = st.text_input("Add ticker (e.g., AAPL)")
    amount = st.number_input(
        "Amount to invest in this position (€)",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )
    entry_date = st.date_input("Entry date of this position")

    if st.button("Add position"):
        if ticker.strip() != "":
            st.session_state.positions.append({
                "ticker": ticker.upper(),
                "amount": amount,
                "entry": str(entry_date)
            })
            save_to_url()
            st.success(f"Added {ticker.upper()} with €{amount:,.2f} invested from {entry_date}.")
        else:
            st.error("Please enter a ticker.")

    st.subheader("📊 Current Portfolio")
    if len(st.session_state.positions) > 0:
        df = pd.DataFrame(st.session_state.positions)
        df_display = df.rename(columns={
            "ticker": "Ticker",
            "amount": "Amount Invested (€)",
            "entry": "Entry Date"
        })
        st.dataframe(df_display)
    else:
        st.info("Your current portfolio is empty.")

    # ------------------------------------------------
    # SELL A POSITION (CLEAR UX)
    # ------------------------------------------------
    st.subheader("📉 Sell a Position")

    st.markdown("""
    **How selling works:**
    - Choose the **ticker** you want to sell.  
    - Choose the **entry date** of that specific position.  
    - Choose the **sell date** (when you exited).  
    - The system will:
        - Close that position,
        - Add the full value of that position (at sell date) into **cash**,
        - Log the trade with its **realized PnL**,
        - Recompute the metrics and graph automatically.
    """)

    if len(st.session_state.positions) > 0:
        sell_ticker = st.selectbox(
            "Select ticker to sell",
            sorted(list({p["ticker"] for p in st.session_state.positions}))
        )

        matching_entries = sorted(
            [p["entry"] for p in st.session_state.positions if p["ticker"] == sell_ticker]
        )

        sell_entry_date = st.selectbox(
            "Select the entry date of the position you want to sell",
            matching_entries
        )

        sell_date = st.date_input("Select the date you sold this position")

        if st.button("Sell Position"):
            # Find the position
            idx_to_remove = None
            for i, p in enumerate(st.session_state.positions):
                if p["ticker"] == sell_ticker and p["entry"] == sell_entry_date:
                    idx_to_remove = i
                    break

            if idx_to_remove is not None:
                pos = st.session_state.positions.pop(idx_to_remove)

                # Store as a closed trade (PnL will be computed in the performance block)
                st.session_state.trades.append({
                    "ticker": pos["ticker"],
                    "amount": pos["amount"],
                    "entry": pos["entry"],
                    "sell_date": str(sell_date)
                })

                save_to_url()
                st.session_state.force_run = True  # trigger auto recompute
                st.success(
                    f"Marked {sell_ticker} (entered {sell_entry_date}) as sold on {sell_date}. "
                    "Metrics and graph will be recomputed."
                )
                st.experimental_rerun()
            else:
                st.error("Could not find that exact position. Double-check the entry date.")
    else:
        st.info("You must add positions before you can sell them.")

    # OLD remove-ticker (hard delete without PnL) – optional to keep
    remove_ticker = st.text_input("Remove ticker (without sell/PnL, hard delete)")
    if st.button("Remove"):
        st.session_state.positions = [
            p for p in st.session_state.positions if p["ticker"] != remove_ticker.upper()
        ]
        save_to_url()
        st.success(f"Removed {remove_ticker.upper()} from open positions.")


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

    start_date = st.date_input("Start date for performance evaluation", value=pd.to_datetime("2020-01-01"))

    run_button = st.button("Compute Performance")

# Run either if user clicked button or a sale just happened
run = run_button or st.session_state.force_run


# ------------------------------------------------
# PERFORMANCE CALCULATION
# ------------------------------------------------
if run:

    if len(st.session_state.positions) == 0 and len(st.session_state.trades) == 0:
        st.error("Add at least one position or have at least one closed trade to compute performance.")
        st.session_state.force_run = False
        st.stop()

    # Collect all tickers involved (open + closed) plus benchmark
    all_tickers = set()
    for p in st.session_state.positions:
        all_tickers.add(p["ticker"])
    for tr in st.session_state.trades:
        all_tickers.add(tr["ticker"])
    all_tickers = list(all_tickers)

    # Determine earliest date needed (for any entry or sell, plus start_date)
    # ------------------------------------------------
    # DETERMINE EARLIEST DATE NEEDED (SAFE VERSION)
    # ------------------------------------------------
    dates_for_min = []
    
    # Collect entry/sell dates and convert safely
    for p in st.session_state.positions:
        dates_for_min.append(pd.to_datetime(p["entry"]))
    
    for tr in st.session_state.trades:
        dates_for_min.append(pd.to_datetime(tr["entry"]))
        dates_for_min.append(pd.to_datetime(tr["sell_date"]))
    
    # Determine earliest involved date
    if len(dates_for_min) > 0:
        min_entry_ts = min(dates_for_min)  # earliest trade date
    else:
        min_entry_ts = pd.to_datetime(start_date)
    
    start_ts = pd.to_datetime(start_date)
    
    # Final safe comparison
    download_start = min(min_entry_ts, start_ts)

    try:
        data = yf.download(all_tickers + [benchmark], start=download_start)["Close"]
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.session_state.force_run = False
        st.stop()

    if data is None or data.empty:
        st.error("No price data returned.")
        st.session_state.force_run = False
        st.stop()

    rets = data.pct_change().dropna()
    if rets.empty:
        st.error("Not enough price history.")
        st.session_state.force_run = False
        st.stop()

    dates = rets.index

    # ------------------------------------------------
    # PRECOMPUTE INFO FOR OPEN POSITIONS
    # ------------------------------------------------
    open_infos = []
    for p in st.session_state.positions:
        t = p["ticker"]
        amount = p["amount"]
        entry_raw = pd.to_datetime(p["entry"])

        # Align entry date to the first available date >= entry_raw
        valid_dates = dates[dates >= entry_raw]
        if len(valid_dates) == 0:
            continue
        entry_idx = valid_dates[0]

        indiv_rets = rets.loc[rets.index >= entry_idx, t]
        indiv_cum = (1 + indiv_rets).cumprod()

        open_infos.append({
            "ticker": t,
            "amount": amount,
            "entry_idx": entry_idx,
            "cum": indiv_cum
        })

    # ------------------------------------------------
    # PRECOMPUTE INFO FOR CLOSED TRADES (SELL LOG)
    # ------------------------------------------------
    trade_infos = []
    for tr in st.session_state.trades:
        t = tr["ticker"]
        amount = tr["amount"]
        entry_raw = pd.to_datetime(tr["entry"])
        sell_raw = pd.to_datetime(tr["sell_date"])

        # Align entry date
        valid_entry_dates = dates[dates >= entry_raw]
        if len(valid_entry_dates) == 0:
            continue
        entry_idx = valid_entry_dates[0]

        # Align sell date to last available date <= sell_raw
        valid_sell_dates = dates[dates <= sell_raw]
        if len(valid_sell_dates) == 0:
            continue
        sell_idx = valid_sell_dates[-1]

        indiv_rets = rets.loc[rets.index >= entry_idx, t]
        indiv_cum = (1 + indiv_rets).cumprod()

        # sale value at sell_idx
        if sell_idx in indiv_cum.index:
            sale_factor = indiv_cum.loc[sell_idx]
        else:
            # Fallback: use last available cum value
            sale_factor = indiv_cum.iloc[-1]
        sale_value = amount * sale_factor
        pnl = sale_value - amount

        trade_infos.append({
            "ticker": t,
            "amount": amount,
            "entry_idx": entry_idx,
            "sell_idx": sell_idx,
            "entry_str": tr["entry"],
            "sell_str": tr["sell_date"],
            "cum": indiv_cum,
            "sale_value": sale_value,
            "pnl": pnl
        })

    # ------------------------------------------------
    # BUILD CASH & INVESTED SERIES OVER TIME
    # ------------------------------------------------
    cash_series = pd.Series(index=dates, dtype=float)
    invested_series = pd.Series(index=dates, dtype=float)

    for d in dates:
        # CASH: start with initial capital
        cash = starting_value

        # Subtract all buy amounts whose entry <= d (open + closed)
        for info in open_infos:
            if d >= info["entry_idx"]:
                cash -= info["amount"]
        for info in trade_infos:
            if d >= info["entry_idx"]:
                cash -= info["amount"]

        # Add all sale values whose sell date <= d
        for info in trade_infos:
            if d >= info["sell_idx"]:
                cash += info["sale_value"]

        # INVESTED: value of active trades at date d
        invested = 0.0

        # Open positions: active from entry onwards
        for info in open_infos:
            if d >= info["entry_idx"]:
                if d in info["cum"].index:
                    factor = info["cum"].loc[d]
                else:
                    factor = info["cum"].iloc[-1]
                invested += info["amount"] * factor

        # Closed trades: active only between entry and sell date
        for info in trade_infos:
            if info["entry_idx"] <= d <= info["sell_idx"]:
                if d in info["cum"].index:
                    factor = info["cum"].loc[d]
                else:
                    factor = info["cum"].iloc[-1]
                invested += info["amount"] * factor

        cash_series.loc[d] = cash
        invested_series.loc[d] = invested

    portfolio_value_series = cash_series + invested_series

    # Restrict to user-chosen evaluation window
    mask = portfolio_value_series.index >= start_date
    port_value = portfolio_value_series[mask]

    if len(port_value) < 2:
        st.error("Not enough data after the selected start date to compute performance.")
        st.session_state.force_run = False
        st.stop()

    # Benchmark value series
    bench_ret = rets[benchmark].loc[mask]
    bench_cum = (1 + bench_ret).cumprod()
    bench_value = starting_value * bench_cum

    # Daily returns from value series
    port_ret = port_value.pct_change().dropna()
    bench_ret2 = bench_value.pct_change().dropna()

    # ------------------------------------------------
    # RISK METRICS / M²
    # ------------------------------------------------
    rf = 0.05 / 252  # 5% annual
    if len(port_ret) < 2 or len(bench_ret2) < 2:
        st.error("Not enough return points to compute risk metrics.")
        st.session_state.force_run = False
        st.stop()

    Rp = port_ret.mean() * 252
    Rb = bench_ret2.mean() * 252
    sig_p = port_ret.std() * np.sqrt(252)
    sig_b = bench_ret2.std() * np.sqrt(252)

    if sig_p == 0:
        st.error("Portfolio volatility is zero – cannot compute M².")
        st.session_state.force_run = False
        st.stop()

    sharpe = (Rp - 0.05) / sig_p
    M2 = sharpe * sig_b + 0.05

    # Final invested & cash
    last_date = port_value.index[-1]
    invested_final = invested_series.loc[last_date]
    cash_final = cash_series.loc[last_date]
    current_value = portfolio_value_series.loc[last_date]
    profit = current_value - starting_value
    absolute_return = (current_value / starting_value) - 1

    # ------------------------------------------------
    # MONEY WEIGHTS TABLE (OPEN POSITIONS + CASH)
    # ------------------------------------------------
    money_weights = {}

    # Value per open position at last date
    for info in open_infos:
        if last_date >= info["entry_idx"]:
            # If last_date in cum index, use it; otherwise last available
            if last_date in info["cum"].index:
                factor = info["cum"].loc[last_date]
            else:
                factor = info["cum"].iloc[-1]
            value_now = info["amount"] * factor
            money_weights[info["ticker"]] = value_now

    # Add cash
    money_weights["CASH"] = cash_final

    # ------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------
    st.subheader("💶 Portfolio Results (Real Money)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Starting Value", f"€{starting_value:,.2f}")
    c2.metric("Total Portfolio Value", f"€{current_value:,.2f}", f"{absolute_return:.2%}")
    c3.metric("Invested", f"€{invested_final:,.2f}")
    c4.metric("Cash", f"€{cash_final:,.2f}")

    st.subheader("📈 Risk-Adjusted Metrics")
    r1, r2, r3 = st.columns(3)
    r1.metric("Portfolio Return (Annualized)", f"{Rp:.2%}")
    r2.metric("Benchmark Return (Annualized)", f"{Rb:.2%}")
    r3.metric("M²", f"{M2:.2%}")

    st.subheader("💰 Current Money Weights (€)")
    mw_df = pd.DataFrame.from_dict(money_weights, orient="index", columns=["Value (€)"])
    st.dataframe(mw_df)

    # ------------------------------------------------
    # SOLD POSITIONS LOG (WITH PnL)
    # ------------------------------------------------
    if len(trade_infos) > 0:
        st.subheader("📜 Sold Positions Log (Realized PnL)")
        log_rows = []
        for info in trade_infos:
            log_rows.append({
                "Ticker": info["ticker"],
                "Amount Invested (€)": info["amount"],
                "Entry Date": info["entry_str"],
                "Sell Date": info["sell_str"],
                "Sale Value (€)": info["sale_value"],
                "PnL (€)": info["pnl"]
            })
        log_df = pd.DataFrame(log_rows)
        st.dataframe(log_df)
    else:
        st.info("No sold positions yet – sell a position to log realized PnL.")

    # ------------------------------------------------
    # CHART (UNCHANGED LOOK)
    # ------------------------------------------------
    st.subheader("📉 Portfolio vs Benchmark (Value in €)")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(port_value, label="Portfolio (€)")
    ax.plot(bench_value, label="Benchmark (€)")
    ax.legend()
    ax.set_title("Portfolio Value Over Time (€)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (€)")
    st.pyplot(fig)

    st.session_state.force_run = False  # reset trigger

# ------------------------------------------------
# RESET BUTTON
# ------------------------------------------------
if st.button("Reset My Portfolio"):
    st.session_state.positions = []
    st.session_state.trades = []
    save_to_url()
    st.success("Your portfolio (open positions + sold log) has been reset.")












