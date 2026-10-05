"""성과 지표, 40거래일 구간 분석, 그래프."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"Strategy": "#2a78d6", "Core only (a=0)": "#eb6834", "Satellite only (a=1)": "#1baf7a", "Market": "#8a8985"}
DISPLAY = {"Strategy": "Core 90 / Sat 10"}   # 그래프 범례 표기
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10})


def mdd(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def summary(navs: dict, turnovers: dict, min_to=5.0) -> pd.DataFrame:
    rows = []
    for k, nav in navs.items():
        r = nav.pct_change().dropna()
        yrs = len(r) / 252
        cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
        vol = r.std() * np.sqrt(252)
        to = turnovers.get(k)
        to_ = to.iloc[1:] if to is not None else None   # 첫 주(최초 매수) 제외
        rows.append({"portfolio": k, "total_return_%": (nav.iloc[-1] / nav.iloc[0] - 1) * 100,
                     "ann_return_%": cagr * 100, "ann_vol_%": vol * 100, "sharpe": r.mean() / r.std() * np.sqrt(252),
                     "mdd_%": mdd(nav) * 100,
                     "avg_weekly_turnover_%": to_.mean() if to is not None else np.nan,
                     "turnover_violation_weeks": int((to_ < min_to).sum()) if to is not None else np.nan})
    return pd.DataFrame(rows).set_index("portfolio").round(2)


def window_stats(navs: dict, bench: pd.Series, n=40) -> tuple[pd.DataFrame, dict]:
    """모든 n거래일 롤링 구간: 수익률 분포, 시장 대비 +5%/+10% 초과 비율, 구간 내 MDD."""
    rows, excess = [], {}
    b = bench.to_numpy()
    for k, nav in navs.items():
        v = nav.reindex(bench.index).to_numpy()
        m = len(v) - n
        R = v[n:] / v[:m] - 1
        B = b[n:] / b[:m] - 1
        dd = np.array([(v[i:i + n + 1] / np.maximum.accumulate(v[i:i + n + 1]) - 1).min() for i in range(m)])
        ex = R - B
        excess[k] = pd.Series(ex, index=bench.index[n:])
        rows.append({"portfolio": k, "windows": m, "mean_%": R.mean() * 100, "median_%": np.median(R) * 100,
                     "p5_%": np.percentile(R, 5) * 100, "p95_%": np.percentile(R, 95) * 100,
                     "beat_mkt_%": (ex > 0).mean() * 100,
                     "excess_gt5_%": (ex > 0.05).mean() * 100, "excess_gt10_%": (ex > 0.10).mean() * 100,
                     "window_mdd_mean_%": dd.mean() * 100, "window_mdd_p5_%": np.percentile(dd, 5) * 100,
                     "window_mdd_worst_%": dd.min() * 100})
    return pd.DataFrame(rows).set_index("portfolio").round(2), excess


def octnov_table(navs: dict) -> pd.DataFrame:
    """매년 10~11월 구간 수익률."""
    out = {}
    for k, nav in navs.items():
        r = {}
        for y in sorted(set(nav.index.year)):
            s = nav[(nav.index >= f"{y}-09-25") & (nav.index <= f"{y}-11-30")]
            pre = s[s.index < f"{y}-10-01"]
            s2 = s[s.index >= f"{y}-10-01"]
            if len(pre) and len(s2) > 30:
                r[y] = (s2.iloc[-1] / pre.iloc[-1] - 1) * 100
        out[k] = r
    return pd.DataFrame(out).round(2)


SLIDE = {"figsize": (4.52, 3.24), "rc": {"font.size": 11, "legend.fontsize": 10}}   # 슬라이드 칸(2.26"x1.62")의 2배


def plot_cum(navs: dict, path):
  with plt.rc_context(SLIDE["rc"]):
    fig, ax = plt.subplots(figsize=SLIDE["figsize"])
    ends = {}
    for k, nav in navs.items():
        v = nav / nav.iloc[0]
        ax.plot(nav.index, v, label=DISPLAY.get(k, k), color=COLORS.get(k), lw=2.4 if k == "Strategy" else 1.5,
                ls="--" if k == "Market" else "-")
        ends[k] = (v.index[-1], v.iloc[-1])
    # 끝값 라벨: 로그 축에서 최소 간격을 두고 위에서부터 배치
    order = sorted(ends, key=lambda k: -ends[k][1])
    ys, gap = [], 1.12
    for k in order:
        y = ends[k][1] if not ys else min(ends[k][1], ys[-1] / gap)
        ys.append(y)
        ax.annotate(f"{ends[k][1]:.2f}x", (ends[k][0], y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.75, 1, 1.5, 2, 3])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}x"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("Cumulative return, net of costs (log)", fontsize=12)
    ax.set_ylabel("Growth of 1")
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.grid(True)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5, borderaxespad=0.2)
    ax.set_xlim(pd.Timestamp("2017-10-01"), pd.Timestamp("2027-09-01"))
    ax.set_xticks(pd.to_datetime(["2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01", "2026-01-01"]))
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_regime(p_turb: pd.Series, a: pd.Series, index: pd.Series, path, lo=0.4, hi=0.6):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    ax1.plot(index.index, index, color=INK, lw=1.2, label="KOSPI")
    ax1.set_ylabel("KOSPI")
    ax1.set_title("Turbulent-regime probability (MS-GJR-GARCH, filtered, walk-forward) and aggressiveness a")
    turb = p_turb.reindex(index.index).ffill() >= hi
    ax1.fill_between(index.index, index.min(), index.max(), where=turb, color="#e34948", alpha=0.10, lw=0, label=f"Turbulent (p_turb ≥ {hi})")
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(True)
    ax2.plot(p_turb.index, p_turb, color="#e34948", lw=0.9, label="p_turb (daily, filtered)")
    ax2.step(a.index, a, where="post", color="#2a78d6", lw=1.6, label="Aggressiveness a (weekly)")
    ax2.axhline(lo, color="gray", ls=":", lw=0.8)
    ax2.axhline(hi, color="gray", ls=":", lw=0.8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax2.set_ylabel("Probability / a")
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_window_hist(excess: dict, path, keys=("Strategy", "Core only (a=0)", "Satellite only (a=1)")):
  with plt.rc_context(SLIDE["rc"]):
    fig, ax = plt.subplots(figsize=SLIDE["figsize"])
    bins = np.linspace(-0.3, 0.4, 36)
    for k in keys:
        if k in excess:
            ax.hist(excess[k].clip(-0.3, 0.4), bins=bins, histtype="step", lw=2.4 if k == "Strategy" else 1.5,
                    label=DISPLAY.get(k, k), color=COLORS.get(k))
    for x in (0.05, 0.10):
        ax.axvline(x, color=INK2, ls="--", lw=0.8)
        ax.text(x, ax.get_ylim()[1] * 0.97, f"+{int(x * 100)}% ", rotation=90, ha="right", va="top", fontsize=9, color=INK2)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("40-day excess return vs market (all windows)", fontsize=12)
    ax.set_xlabel("Excess return over market")
    ax.set_ylabel("Windows")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, loc="center right", handlelength=1.2, borderaxespad=0.2, fontsize=9)
    ax.grid(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300)
    plt.close(fig)
