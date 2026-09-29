"""Step 2: run the base backtest, the 36-run parameter grid, and the Deflated Sharpe Ratio check.
Outputs go to results/."""
import itertools
import json
import time
from dataclasses import replace

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import data
from config import BASE, BT_START, GRID_LOOKBACK, GRID_SECTORS, GRID_STOPS, ORIGINAL, RESULTS_DIR
from strategy import MarketData, deflated_sharpe, oct_nov_returns, perf_stats, run_backtest

NAVY, ORANGE, GRAY = "#161C3C", "#C05621", "#9AA0AC"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ECEEF2", "figure.dpi": 150})


def load_market_data():
    kospi = data.load_kospi()
    wdates = data.weekly_dates(kospi.index)
    cap, val, close = data.load_snapshots(wdates, with_close=True)
    snaps = data.load_membership()
    sec_px = data.build_sector_prices(cap, snaps, kospi.index, close=close)
    tickers = data.candidate_tickers(cap, val, snaps)
    px = data.load_prices(tickers)
    print(f"[data] {len(kospi)} days, {sec_px.shape[1]} sectors, {px.shape[1]} stocks with prices")
    return MarketData(kospi, sec_px, px, cap, val, snaps), kospi


def fmt(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    md, kospi = load_market_data()

    # ------------------------------------------------------------ base case
    print("[base]", BASE.label())
    base = run_backtest(md, BASE, BT_START, keep_signals=True, verbose=True)
    bench = kospi.reindex(base.nav.index).ffill()
    bench = bench / bench.iloc[0]
    stats = perf_stats(base.nav, bench, base.turnover)
    stats["AvgHoldings"] = base.n_holdings.mean()
    stats["StopOuts"] = int((base.trades["action"] == "stop").sum())
    pd.Series(stats).to_csv(RESULTS_DIR / "base_metrics.csv", header=["value"])
    print(json.dumps(fmt(stats), indent=2))

    # original proposal (12% trailing stop, next-day execution) for comparison
    orig = run_backtest(md, ORIGINAL, BT_START)
    ostats = perf_stats(orig.nav, bench, orig.turnover)
    bstats = perf_stats(bench)
    comp = pd.DataFrame({"Final (weekly, Friday close, no stop)": stats, "Original proposal": ostats,
                         "KOSPI": {k: bstats.get(k) for k in stats}})
    comp.to_csv(RESULTS_DIR / "comparison.csv")
    print("\n", comp.loc[["CAGR", "Volatility", "Sharpe", "MaxDrawdown", "AvgWeeklyTurnover"]].round(4))

    pd.DataFrame({"strategy": base.nav, "original": orig.nav, "kospi": bench}).to_csv(RESULTS_DIR / "base_nav.csv")
    base.trades.to_csv(RESULTS_DIR / "base_trades.csv", index=False)
    hold = [{"date": s.date, "sector": sec, "ticker": t, "weight": s.weights.get(t, 0)}
            for s in base.signals for sec, ts in s.picks.items() for t in ts]
    pd.DataFrame(hold).to_csv(RESULTS_DIR / "base_weekly_picks.csv", index=False)

    on = oct_nov_returns(base.nav, bench)
    on.to_csv(RESULTS_DIR / "oct_nov_returns.csv")
    print("\nOct-Nov returns:\n", on.round(4))

    # ------------------------------------------------------------ charts
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax[0].plot(base.nav, color=ORANGE, lw=1.6, label="Final: weekly, Friday close, no stop")
    ax[0].plot(orig.nav, color=GRAY, lw=1.0, label="Original: 12% trailing stop, next-day execution")
    ax[0].plot(bench, color=NAVY, lw=1.2, label="KOSPI")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of 1 (log)"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_title("Sector Relative-Strength Momentum vs. KOSPI", loc="left", fontsize=11)
    dd = base.nav / base.nav.cummax() - 1
    ddb = bench / bench.cummax() - 1
    ax[1].fill_between(dd.index, dd, 0, color=ORANGE, alpha=0.35, lw=0, label="Final")
    ax[1].plot(ddb, color=NAVY, lw=0.8, label="KOSPI")
    ax[1].set_ylabel("Drawdown")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "equity_drawdown.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = range(len(on))
    ax.bar([i - 0.2 for i in x], on["Strategy"], 0.4, color=ORANGE, label="Strategy")
    ax.bar([i + 0.2 for i in x], on["KOSPI"], 0.4, color=NAVY, label="KOSPI")
    ax.set_xticks(list(x)); ax.set_xticklabels(on.index)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("October-November return by year", loc="left", fontsize=11); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "oct_nov.png"); plt.close(fig)

    # ------------------------------------------------------------ parameter grid
    rows, rets = [], {}
    grid = list(itertools.product(GRID_LOOKBACK, GRID_SECTORS, GRID_STOPS))
    for n, (lb, ns, stop) in enumerate(grid, 1):
        p = replace(BASE, lookback_w=lb, n_sectors=ns, stop=stop)
        r = run_backtest(md, p, BT_START)
        s = perf_stats(r.nav, bench, r.turnover)
        rows.append({"label": p.label(), "lookback": "/".join(f"{w:.2f}" for w in lb), "n_sectors": ns,
                     "stop": "none" if stop is None else stop, **s})
        rets[p.label()] = r.nav.pct_change()
        print(f"[grid {n:2d}/{len(grid)}] {p.label():32s} CAGR {s['CAGR']:7.2%}  Sharpe {s['Sharpe']:5.2f}  MDD {s['MaxDrawdown']:7.2%}")
    grid_df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    grid_df.to_csv(RESULTS_DIR / "grid_results.csv", index=False)

    rets = pd.DataFrame(rets).dropna(how="all")
    out = {"best_in_grid": deflated_sharpe(rets, grid_df.iloc[0]["label"])}
    if BASE.label() in rets.columns:
        out = {"base": deflated_sharpe(rets, BASE.label()), **out}
    pd.DataFrame(out).to_csv(RESULTS_DIR / "deflated_sharpe.csv")
    print("\nDeflated Sharpe:", {k: round(float(v["DSR"]), 4) for k, v in out.items()})

    fig, ax = plt.subplots(figsize=(8, 3.2))
    g = grid_df.sort_values("Sharpe")
    colors = [ORANGE if l == BASE.label() else GRAY for l in g["label"]]
    ax.bar(range(len(g)), g["Sharpe"], color=colors)
    ax.axhline(perf_stats(bench)["Sharpe"], color=NAVY, lw=1, ls="--", label="KOSPI Sharpe")
    ax.set_xticks([]); ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe ratio across 36 parameter sets (orange = final version)", loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "grid_sharpe.png"); plt.close(fig)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
