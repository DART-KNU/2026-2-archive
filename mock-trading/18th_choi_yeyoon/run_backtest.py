"""
3Q Earnings Surprise Drift Strategy — 백테스트 실행

  python run_backtest.py --download   # 1) 데이터 수집 (처음 한 번, 오래 걸림)
  python run_backtest.py              # 2) 백테스트 실행 → results/ 에 표와 차트 저장
  python run_backtest.py --demo       #    (합성 데이터로 코드 동작만 확인)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from backtest import metrics, simulate
from signals import build_events, prepare_panels


def fmt(m):
    out = {}
    for k, v in m.items():
        if isinstance(v, (int, np.integer)) or k in ("Avg Positions", "Profitable Names", "Weeks < 5%"):
            out[k] = f"{v:.1f}" if isinstance(v, float) else str(v)
        elif k in ("Sharpe", "Sortino", "Beta", "Avg HHI"):
            out[k] = f"{v:.2f}"
        else:
            out[k] = f"{v * 100:.2f}%" if pd.notna(v) else "n/a"
    return out


def plot_full(res, k200, out):
    nav = res["nav"] / res["nav"].iloc[0] * 100
    b = k200.reindex(nav.index).ffill()
    b = b / b.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(nav.index, nav, label="Strategy", color="#2F5E9E", lw=2)
    ax.plot(b.index, b, label="KOSPI200", color="#E08A2E", lw=1.5, ls="--")
    ax.axhline(100, color="#999", lw=0.8)
    ax.set_title("Cumulative Return (base = 100)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "full_cumulative.png"), dpi=160); plt.close(fig)

    dd = (res["nav"] / res["nav"].cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(dd.index, dd, 0, color="#C0504D", alpha=0.6)
    ax.set_title("Drawdown (%)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "full_drawdown.png"), dpi=160); plt.close(fig)


def plot_window(tbl, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(tbl))
    ax.bar(x - 0.2, tbl["Strategy"] * 100, 0.4, label="Strategy", color="#2F5E9E")
    ax.bar(x + 0.2, tbl["KOSPI200"] * 100, 0.4, label="KOSPI200", color="#E08A2E")
    ax.set_xticks(x); ax.set_xticklabels(tbl.index)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_title("Oct 1 – Nov 30 Return by Year (%)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "window_by_year.png"), dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="데이터 수집")
    ap.add_argument("--demo", action="store_true", help="합성 데이터로 실행")
    args = ap.parse_args()

    if args.download:
        from data_loader import download_all
        download_all()
        return

    if args.demo:
        from demo_data import make_demo
        panels, quarters, k200, sectors = make_demo()
        years, full_start, full_end = list(range(2016, 2025)), "2016-01-01", "2025-12-31"
        out = os.path.join(C.OUT_DIR, "demo")
    else:
        from data_loader import load_all
        panels, quarters, k200, sectors = load_all()
        years, full_start, full_end = C.CONTEST_YEARS, C.BACKTEST_START, C.END
        out = C.OUT_DIR
    os.makedirs(out, exist_ok=True)

    print("[1/4] 파생 데이터 계산")
    P = prepare_panels(panels, sectors)
    print("[2/4] 이벤트 신호 계산 (SUE · EAR · PIT 백분위)")
    ev = build_events(quarters, P)
    ev.to_csv(os.path.join(out, "events.csv"), index=False, encoding="utf-8-sig")
    print(f"      이벤트 {len(ev):,}건 / 진입조건 통과 {int(ev.passed.sum()):,}건")

    print("[3/4] 전체기간 백테스트 (PEAD 상시 운용)")
    full = simulate(P, ev, k200, full_start, full_end, contest=False, stage1=False)
    mf = metrics(full, k200)
    pd.Series(fmt(mf)).to_csv(os.path.join(out, "summary_full.csv"), header=["value"], encoding="utf-8-sig")
    full["nav"].to_csv(os.path.join(out, "full_nav.csv"), encoding="utf-8-sig")
    full["trades"].to_csv(os.path.join(out, "full_trades.csv"), index=False, encoding="utf-8-sig")
    plot_full(full, k200, out)

    print("[4/4] 연도별 10/1~11/30 대회 모드 백테스트 (Stage1 + Stage2 + 규정)")
    rows, details = {}, {}
    for y in years:
        res = simulate(P, ev, k200, f"{y}-10-01", f"{y}-11-30", contest=True, stage1=True)
        m = metrics(res, k200)
        rows[y] = {"Strategy": m["Total Return"], "KOSPI200": m["Benchmark (KOSPI200)"],
                   "Excess": m["Excess"], "Sharpe": m["Sharpe"], "MDD": m["MDD"],
                   "Avg Positions": m["Avg Positions"], "Top3 PnL Share": m["Top3 PnL Share"],
                   "Weeks < 5%": m["Weeks < 5%"]}
        details[y] = res
    tbl = pd.DataFrame(rows).T
    tbl.index.name = "Year"
    tbl.to_csv(os.path.join(out, "window_table.csv"), encoding="utf-8-sig")
    plot_window(tbl, out)

    print("\n══════ 전체기간 요약 ══════")
    for k, v in fmt(mf).items():
        print(f"  {k:<22}{v}")
    print("\n══════ 10~11월 구간 (연도별) ══════")
    show = tbl.copy()
    for c in ["Strategy", "KOSPI200", "Excess", "MDD", "Top3 PnL Share"]:
        show[c] = (show[c] * 100).map(lambda v: f"{v:6.2f}%" if pd.notna(v) else "   n/a")
    show["Sharpe"] = show["Sharpe"].map(lambda v: f"{v:5.2f}" if pd.notna(v) else "  n/a")
    show["Avg Positions"] = show["Avg Positions"].map(lambda v: f"{v:4.1f}")
    print(show.to_string())
    print(f"\n  초과수익 연도: {(tbl.Excess > 0).sum()} / {len(tbl)}")
    print(f"\n결과 저장 위치: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
