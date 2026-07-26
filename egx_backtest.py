#!/usr/bin/env python3
"""
EGX Backtester
==============
A long-only backtester for Egyptian Exchange (EGX) stocks, with realistic
transaction costs and walk-forward (out-of-sample) validation.

The walk-forward part matters more than the strategies. Any strategy can be
tuned to look good on past data. This tool tunes on one slice of history and
then reports performance on a slice it never saw. That out-of-sample number
is the only one worth believing.

USAGE
-----
    pip install pandas numpy yfinance

    # single stock, default strategy
    python egx_backtest.py --ticker EGS38191C010.CA

    # compare all strategies
    python egx_backtest.py --ticker EGS38191C010.CA --compare

    # walk-forward validation (the honest test)
    python egx_backtest.py --ticker EGS38191C010.CA --walkforward

    # use your own CSV instead of downloading
    python egx_backtest.py --csv mydata.csv

CSV format: columns Date,Open,High,Low,Close,Volume

EGX TICKERS ON YAHOO FINANCE
----------------------------
Yahoo uses ISIN-style codes with a .CA suffix. Verified examples:
    ^CASE30              EGX 30 index
    ^EGX30.CA            EGX 30
    EGS38191C010.CA      Abu Qir Fertilizers
    EGS69101C011.CA      EFG Hermes Holding
    EGS67031C012.CA      Saudi Egyptian Investment & Finance
    EGS38381C017.CA      Egyptian Financial & Industrial

To find others: search the company on finance.yahoo.com and copy the symbol
from the URL. Stick to EGX 30 constituents -- everything else is too thinly
traded for backtest results to mean anything.
"""

import argparse
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# EGX market frictions
# ---------------------------------------------------------------------------
# Round-trip cost estimate for an EGX retail account. Brokerage commission,
# EGX fee, MCDR clearing fee and stamp duty together land roughly here.
# CHECK YOUR OWN BROKER'S SCHEDULE AND CHANGE THIS -- it is the single number
# most likely to turn a "profitable" strategy into a losing one.
COMMISSION_PER_SIDE = 0.0015   # 0.15% each way
SLIPPAGE_PER_SIDE = 0.0010     # 0.10% each way; higher for illiquid names

TRADING_DAYS = 250             # EGX trades Sun-Thu


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_yahoo(ticker, start, end):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance not installed. Run: pip install yfinance")

    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        sys.exit(f"No data for '{ticker}'. Check the symbol on finance.yahoo.com.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    return _clean(df)


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    df = df.rename(columns=str.title)
    return _clean(df)


def _clean(df):
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < 250:
        print(f"WARNING: only {len(df)} bars. Results will be near-meaningless.",
              file=sys.stderr)
    return df


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, period):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
# Each returns a Series of desired position: 1 = long, 0 = flat.
# Signals are computed on bar t and ACTED ON at bar t+1's open (see backtest).
# That one-bar lag is what stops you from accidentally trading on information
# you would not have had. Removing it is the classic way to build a backtest
# that prints 400% a year and loses money live.

def strat_sma_cross(df, fast=20, slow=60):
    f = df["Close"].rolling(int(fast)).mean()
    s = df["Close"].rolling(int(slow)).mean()
    return (f > s).astype(int)


def strat_rsi_reversion(df, period=14, entry=30, exit_=55):
    r = rsi(df["Close"], int(period))
    pos, holding = [], 0
    for v in r:
        if holding == 0 and v < entry:
            holding = 1
        elif holding == 1 and v > exit_:
            holding = 0
        pos.append(holding)
    return pd.Series(pos, index=df.index)


def strat_donchian(df, lookback=55, exit_lookback=20):
    hi = df["Close"].rolling(int(lookback)).max()
    lo = df["Close"].rolling(int(exit_lookback)).min()
    pos, holding = [], 0
    for i in range(len(df)):
        c = df["Close"].iloc[i]
        if holding == 0 and not np.isnan(hi.iloc[i]) and c >= hi.iloc[i]:
            holding = 1
        elif holding == 1 and not np.isnan(lo.iloc[i]) and c <= lo.iloc[i]:
            holding = 0
        pos.append(holding)
    return pd.Series(pos, index=df.index)


def strat_trend_filter(df, ma=200):
    """Buy and hold, but only while price is above its long moving average."""
    return (df["Close"] > df["Close"].rolling(int(ma)).mean()).astype(int)


STRATEGIES = {
    "sma_cross":     (strat_sma_cross,     {"fast": [10, 20, 30, 50], "slow": [50, 60, 100, 150]}),
    "rsi_reversion": (strat_rsi_reversion, {"period": [7, 14, 21], "entry": [20, 25, 30], "exit_": [50, 55, 65]}),
    "donchian":      (strat_donchian,      {"lookback": [20, 40, 55, 80], "exit_lookback": [10, 20, 30]}),
    "trend_filter":  (strat_trend_filter,  {"ma": [100, 150, 200]}),
}


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def backtest(df, position, capital=100_000.0):
    """
    Long-only, all-in / all-out, next-open execution.
    No shorting (EGX retail generally cannot short).
    Costs charged on every position change.
    """
    pos = position.shift(1).fillna(0)          # act on the NEXT bar
    open_ = df["Open"].fillna(df["Close"])
    ret = open_.pct_change().fillna(0)

    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (COMMISSION_PER_SIDE + SLIPPAGE_PER_SIDE)

    strat_ret = pos * ret - cost
    equity = capital * (1 + strat_ret).cumprod()

    trades = _extract_trades(pos, open_)
    return equity, strat_ret, trades


def _extract_trades(pos, price):
    trades, entry_i = [], None
    p = pos.values
    for i in range(len(p)):
        if p[i] == 1 and (i == 0 or p[i - 1] == 0):
            entry_i = i
        elif entry_i is not None and p[i] == 0 and p[i - 1] == 1:
            gross = price.iloc[i] / price.iloc[entry_i] - 1
            net = gross - 2 * (COMMISSION_PER_SIDE + SLIPPAGE_PER_SIDE)
            trades.append({"entry": pos.index[entry_i], "exit": pos.index[i],
                           "days": i - entry_i, "net_return": net})
            entry_i = None
    if entry_i is not None:
        gross = price.iloc[-1] / price.iloc[entry_i] - 1
        trades.append({"entry": pos.index[entry_i], "exit": pos.index[-1],
                       "days": len(p) - entry_i,
                       "net_return": gross - 2 * (COMMISSION_PER_SIDE + SLIPPAGE_PER_SIDE),
                       "open": True})
    return pd.DataFrame(trades)


def metrics(equity, strat_ret, trades, benchmark=None):
    years = max(len(equity) / TRADING_DAYS, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    dd = equity / equity.cummax() - 1
    sd = strat_ret.std()
    sharpe = (strat_ret.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0

    m = {
        "total_return": total,
        "cagr": cagr,
        "max_drawdown": dd.min(),
        "sharpe": sharpe,
        "n_trades": len(trades),
        "win_rate": (trades["net_return"] > 0).mean() if len(trades) else np.nan,
        "avg_trade": trades["net_return"].mean() if len(trades) else np.nan,
        "time_in_market": (strat_ret != 0).mean(),
    }
    if benchmark is not None:
        bh = benchmark.iloc[-1] / benchmark.iloc[0] - 1
        m["buy_hold_return"] = bh
        m["excess_vs_buy_hold"] = total - bh
    return m


def show(name, m):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"  Total return       {m['total_return']:>10.1%}")
    print(f"  CAGR               {m['cagr']:>10.1%}")
    print(f"  Max drawdown       {m['max_drawdown']:>10.1%}")
    print(f"  Sharpe             {m['sharpe']:>10.2f}")
    print(f"  Trades             {m['n_trades']:>10d}")
    if m["n_trades"]:
        print(f"  Win rate           {m['win_rate']:>10.1%}")
        print(f"  Avg trade (net)    {m['avg_trade']:>10.2%}")
    print(f"  Time in market     {m['time_in_market']:>10.1%}")
    if "buy_hold_return" in m:
        print(f"  Buy & hold         {m['buy_hold_return']:>10.1%}")
        verdict = "BEAT" if m["excess_vs_buy_hold"] > 0 else "LOST TO"
        print(f"  --> {verdict} buy & hold by {abs(m['excess_vs_buy_hold']):.1%}")


# ---------------------------------------------------------------------------
# Parameter search + walk-forward
# ---------------------------------------------------------------------------
def param_grid(space):
    from itertools import product
    keys = list(space)
    for combo in product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))


def optimize(df, fn, space, min_trades=5):
    """Return the parameter set with the best Sharpe on THIS slice of data."""
    best, best_sharpe = None, -np.inf
    for params in param_grid(space):
        try:
            pos = fn(df, **params)
            eq, r, tr = backtest(df, pos)
            if len(tr) < min_trades:
                continue
            m = metrics(eq, r, tr)
            if m["sharpe"] > best_sharpe:
                best, best_sharpe = params, m["sharpe"]
        except Exception:
            continue
    return best, best_sharpe


def walk_forward(df, name, n_folds=4, train_frac=0.7):
    """
    Split history into folds. Tune on the training part of each fold, then
    trade the tuned parameters on the untouched test part. Stitch the test
    results together -- that stitched curve is your realistic expectation.
    """
    fn, space = STRATEGIES[name]
    fold_size = len(df) // n_folds
    all_ret, rows = [], []

    for k in range(n_folds):
        lo = k * fold_size
        hi = len(df) if k == n_folds - 1 else (k + 1) * fold_size
        fold = df.iloc[lo:hi]
        split = int(len(fold) * train_frac)
        train, test = fold.iloc[:split], fold.iloc[split:]
        if len(train) < 120 or len(test) < 40:
            continue

        params, is_sharpe = optimize(train, fn, space)
        if params is None:
            rows.append((k + 1, "no valid params", np.nan, np.nan))
            continue

        pos = fn(test, **params)
        eq, r, tr = backtest(test, pos)
        m = metrics(eq, r, tr, benchmark=test["Close"])
        all_ret.append(r)
        rows.append((k + 1, str(params), is_sharpe, m["total_return"]))

    print(f"\nWALK-FORWARD: {name}")
    print("=" * 72)
    print(f"{'Fold':<6}{'Tuned params':<38}{'In-samp SR':>12}{'Out-samp ret':>14}")
    for k, p, isr, osr in rows:
        isr_s = f"{isr:.2f}" if isr == isr else "-"
        osr_s = f"{osr:.1%}" if osr == osr else "-"
        print(f"{k:<6}{p[:36]:<38}{isr_s:>12}{osr_s:>14}")

    if all_ret:
        stitched = pd.concat(all_ret)
        eq = 100_000 * (1 + stitched).cumprod()
        m = metrics(eq, stitched, pd.DataFrame())
        print("\nCombined out-of-sample:")
        print(f"  Total return   {m['total_return']:>9.1%}")
        print(f"  CAGR           {m['cagr']:>9.1%}")
        print(f"  Max drawdown   {m['max_drawdown']:>9.1%}")
        print(f"  Sharpe         {m['sharpe']:>9.2f}")
        print("\nIf the parameters jump around between folds, or out-of-sample")
        print("returns are far below in-sample, the edge is not real.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="EGX backtester")
    ap.add_argument("--ticker", default="EGS38191C010.CA")
    ap.add_argument("--csv")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--strategy", default="sma_cross", choices=list(STRATEGIES))
    ap.add_argument("--compare", action="store_true", help="run every strategy")
    ap.add_argument("--walkforward", action="store_true", help="out-of-sample test")
    ap.add_argument("--capital", type=float, default=100_000)
    args = ap.parse_args()

    df = load_csv(args.csv) if args.csv else load_yahoo(args.ticker, args.start, args.end)
    label = args.csv or args.ticker
    print(f"\n{label}   {df.index[0].date()} to {df.index[-1].date()}   {len(df)} bars")
    print(f"Round-trip cost assumption: "
          f"{2 * (COMMISSION_PER_SIDE + SLIPPAGE_PER_SIDE):.2%}")

    if args.walkforward:
        names = list(STRATEGIES) if args.compare else [args.strategy]
        for n in names:
            walk_forward(df, n)
        return

    names = list(STRATEGIES) if args.compare else [args.strategy]
    for n in names:
        fn, _ = STRATEGIES[n]
        pos = fn(df)
        eq, r, tr = backtest(df, pos, args.capital)
        show(n, metrics(eq, r, tr, benchmark=df["Close"]))

    print("\nThese are IN-SAMPLE numbers on default parameters.")
    print("Run with --walkforward before believing any of them.")


if __name__ == "__main__":
    main()
