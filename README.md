# 📈 M² Portfolio Tracker (Streamlit)

A professional **real-money portfolio tracking dashboard** built with Streamlit, designed to track **open and closed positions**, cash balance, benchmark performance, and advanced **risk-adjusted metrics** including **Classic M²** and **B&R M²**.

Developed for the **Horizon Capital / Research Committee** investment workflow.

---

## What this app does

This app tracks a portfolio exactly as it would behave in reality:

- Supports **cash + invested capital**
- Handles **multiple buys and sells**
- Tracks **open positions and realized PnL**
- Compares performance to a benchmark
- Computes multiple risk-adjusted performance metrics

It is **not a backtester** and **not a prediction model** — it is a **portfolio accounting and evaluation tool**.

---

## Core features

### Portfolio management
- Add positions by:
  - **Ticker**
  - **Amount invested (€)**
  - **Entry date**
- Sell individual positions by selecting:
  - Ticker
  - Entry date
  - Sell date
- Automatically:
  - Moves sold value into **cash**
  - Logs **realized PnL**
  - Recomputes portfolio metrics and charts

### Cash-aware accounting
- Explicit cash tracking
- Cash decreases on buys
- Cash increases on sells
- Portfolio value = **cash + invested value**
- Partial investment (not fully invested portfolios supported)

---

## Performance & risk metrics

### Returns
- Real-money portfolio value over time (€)
- Benchmark value over time (€)
- Correct **geometric annualization** of returns
- Absolute and cumulative returns

### Risk metrics
- Annualized volatility
- Maximum drawdown
- Sharpe-based metrics

### Risk-adjusted performance
- **Classic M² (Modigliani–Modigliani)**  
- **B&R M²** (competition-style formulation using weekly returns)

Both metrics scale portfolio performance to the benchmark’s risk level.

---

## How the model works (high level)

1. Downloads historical closing prices using **Yahoo Finance**
2. Aligns buy and sell dates to the nearest available trading days
3. Tracks:
   - Cash balance
   - Value of each open position
   - Value of closed trades until exit
4. Builds a time series of total portfolio value
5. Computes returns and risk metrics
6. Displays results, logs, and charts interactively

---

## Getting started

### Installation
pip install streamlit yfinance numpy pandas matplotlib


