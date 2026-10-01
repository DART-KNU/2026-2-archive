"""
run_backtest.py
최종 전략 백테스트 실행: 최종 전략 vs 코어+현금 vs 코스피
- 기간: 2026년 7~9월, 2023년 7월 ~ 2026년 9월
- 결과: 콘솔 요약 + results/ 폴더에 차트(PNG)와 엑셀 저장
실행: python run_backtest.py   (data/ 폴더에 DataGuide 엑셀 3개 필요)
"""
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from backtest import PARAMS, Backtest, load_all, summarize

ME = "ME" if tuple(int(x) for x in pd.__version__.split(".")[:2]) >= (2, 2) else "M"
PERIODS = {
    "2026Q3": ("2026-07-01", "2026-09-28"),
    "full": ("2023-07-03", "2026-09-28"),
}
OUT = "results"


def main():
    os.makedirs(OUT, exist_ok=True)
    px, uni, kospi = load_all()
    print(f"유니버스 {len(uni)}종목 | 가격 {px.shape[1]}종목 x {px.shape[0]}일")

    # ---------------- 백테스트 ----------------
    results, runs = [], {}
    for per, (s, e) in PERIODS.items():
        for name, extra in {"Final strategy": {}, "Core + Cash": {"use_semis": False}}.items():
            b = Backtest(px, uni, s, e, **{**PARAMS, **extra}).run()
            runs[(per, name)] = b
            results.append(dict(period=per, **summarize(b, name)))
        base = kospi.loc[:pd.Timestamp(s) - pd.Timedelta(days=1)]
        if len(base):
            kp = kospi.loc[s:e]
            results.append(dict(period=per, strategy="KOSPI", ret=(kp.iloc[-1] / base.iloc[-1] - 1) * 100,
                                mdd=(kp / kp.cummax() - 1).min() * 100))
    summary = pd.DataFrame(results).set_index(["period", "strategy"])
    print(summary.round(2).to_string())

    # ---------------- 월별 수익률 (2026Q3) ----------------
    s, e = PERIODS["2026Q3"]
    M = {}
    for name in ["Final strategy", "Core + Cash"]:
        nav = runs[("2026Q3", name)].nav["nav"]
        m = pd.concat([pd.Series([1.0], index=[pd.Timestamp(s) - pd.Timedelta(days=1)]), nav]).resample(ME).last()
        M[name] = m.pct_change().dropna() * 100
    km = kospi.loc[:e].resample(ME).last()
    M["KOSPI"] = (km.pct_change() * 100).loc[s:]
    M = pd.DataFrame(M)
    M.index = M.index.strftime("%Y-%m")
    print(M.round(2).to_string())

    semi = pd.DataFrame(runs[("2026Q3", "Final strategy")].semi_log)
    print(semi[["ticker", "entry_date", "exit_date", "days", "ret", "reason"]].to_string())

    # ---------------- 차트 ----------------
    BLUE, GRAY, ORANGE = "#2E4A9E", "#8A919E", "#C05621"
    pct = FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E3E5EA", "legend.frameon": False})

    fig, ax = plt.subplots(figsize=(9, 4))
    start = pd.Timestamp(s) - pd.Timedelta(days=1)
    for name, c in [("Final strategy", BLUE), ("Core + Cash", GRAY)]:
        y = (pd.concat([pd.Series([1.0], index=[start]), runs[("2026Q3", name)].nav["nav"]]) - 1) * 100
        ax.plot(y.index, y, color=c, lw=2.2, label=name)
        ax.annotate(f"{y.iloc[-1]:+.1f}%", (y.index[-1], y.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontweight="bold")
    kq = kospi.loc[kospi.loc[:start].index[-1]:e]
    y = (kq / kq.iloc[0] - 1) * 100
    ax.plot(y.index, y, color=ORANGE, lw=2, ls="--", label="KOSPI")
    ax.annotate(f"{y.iloc[-1]:+.1f}%", (y.index[-1], y.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontweight="bold")
    ax.axhline(0, color="#B8BCC6", lw=1)
    ax.yaxis.set_major_formatter(pct)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.margins(x=0.08)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.set_title("Cumulative Return vs. KOSPI", loc="left", fontsize=13, fontweight="bold", pad=28)
    fig.savefig(f"{OUT}/cumulative_vs_kospi.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    cols = {"Core + Cash": GRAY, "Final strategy": BLUE, "KOSPI": ORANGE}
    w = 0.26
    for j, c in enumerate(cols):
        xs = [i + (j - 1) * (w + 0.02) for i in range(len(M))]
        bars = ax.bar(xs, M[c], width=w, color=cols[c], label=c, zorder=3)
        for b_, v in zip(bars, M[c]):
            ax.annotate(f"{v:+.1f}%", (b_.get_x() + b_.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -3),
                        textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#B8BCC6", lw=1)
    ax.set_xticks(range(len(M)))
    ax.set_xticklabels(pd.to_datetime(M.index).strftime("%B"))
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(M.min().min() - 5, M.max().max() + 5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.set_title("Monthly Return vs. KOSPI", loc="left", fontsize=13, fontweight="bold", pad=28)
    fig.savefig(f"{OUT}/monthly_vs_kospi.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---------------- 엑셀 저장 (로컬 확인용, 깃허브에는 올리지 않음) ----------------
    with pd.ExcelWriter(f"{OUT}/backtest_results.xlsx") as xw:
        summary.to_excel(xw, sheet_name="Summary")
        M.to_excel(xw, sheet_name="Monthly 2026Q3")
        for (per, name), b in runs.items():
            b.nav.to_excel(xw, sheet_name=f"NAV {per} {name}"[:31])
        pd.DataFrame(runs[("full", "Final strategy")].semi_log).to_excel(xw, sheet_name="Semi trades full", index=False)
        runs[("2026Q3", "Final strategy")].weekly.to_excel(xw, sheet_name="Weekly turnover Q3")
    print(f"saved -> {OUT}/")


if __name__ == "__main__":
    main()
