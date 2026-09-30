[portfolio_screening.ipynb](https://github.com/user-attachments/files/32854641/portfolio_screening.ipynb)# [2026-2] 모의투자 포트폴리오 - [18기_고준성]

## 팩터 모멘텀 전략 
* **핵심 로직:** 모멘텀 기반 
* **카페 링크:** https://naver.me/FetRq2r5

* `portfolio_screening.ipynb` : [Upl[backtest_result.py](https://github.com/user-attachments/files/32854650/backtest_result.py)oading portfolio_screening.ipynb…]()


* `backtest_result.py` : 
"""유가·금리 매크로 노출 전략: 성과 분석 및 시각화.

portfolio_screening.ipynb 와 같은 스크리닝 규칙(strategy_core.py)을 과거 분기마다 적용해
코스피200과 성과를 비교한다.

비교 대상
    - 코스피200 (벤치마크)
    - 전략: 스크리닝 (전망 고정)        ← 본 전략
    - 대조군: 섹터 전종목 동일가중      ← 스크리닝(민감도 선별)의 효과를 보기 위한 비교군
    - 참고: 사후 정답 전망              ← 분기마다 실제 유가·금리 방향을 알았다면 (현실에선 불가능, 상한선)

사용법
    python backtest_result.py                       # 전망: 유가↑ 금리↑, 분기 리밸런싱
    python backtest_result.py --oil 1 --rate 0      # 유가 상승만 전망 (금융 몫은 코스피200)
    python backtest_result.py --rebal M --n 3       # 월간 리밸런싱, 섹터별 3종목
    python backtest_result.py --refresh             # 데이터 새로 받기

결과물 (results/ 폴더)
    cumulative.png   누적수익률 · 코스피200 대비 상대성과 · 낙폭
    annual.png       연도별 수익률 막대
    metrics.png      기간별 성과표 (CAGR · 샤프 · MDD · 베타 · 알파 · 정보비율)
    metrics.csv      위 수치 원본
    holdings.csv     리밸런싱별 편입 종목과 비중
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import strategy_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
matplotlib.rcParams["axes.unicode_minus"] = False
for font in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:   # 맥 / 윈도우 / 리눅스
    if font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rc("font", family=font)
        break

PERIODS = {
    "전체 (18.01~26.09)": ("2018-01-01", None),
    "트럼프 1기 (18.01~21.01)": ("2018-01-01", "2021-01-19"),
    "바이든 (21.01~25.01)": ("2021-01-20", "2025-01-19"),
    "트럼프 2기 (25.01~)": ("2025-01-20", None),
}
COLORS = {"코스피200 (BM)": "#222222", "전략": "#e1573b", "대조군": "#8e8e8e", "참고": "#2e86c1"}


def arrow(x):
    return {1: "↑", -1: "↓"}.get(int(x), "-")


def run_backtest(close, value, macro, wret, fac, view_fn, start, rebal, n, t_min, equal_all=False):
    """분기(월) 첫 거래일에 전날 종가로 스크리닝, 당일 종가 체결. 반환: (자산곡선, 편입 기록)"""
    assets = close.copy()
    assets[core.BENCH] = macro[core.BENCH]
    rets = assets.pct_change(fill_method=None)
    days = assets.loc[start:].index
    rebal_days = list(pd.Series(days, index=days).resample("QS" if rebal == "Q" else "MS").first())

    v, w, eq, log = 1.0, pd.Series(dtype=float), [], []
    for i, d in enumerate(days):
        if i > 0 and len(w):
            r = rets.loc[d, w.index].fillna(0)
            g = (w * (1 + r)).sum() + (1 - w.sum())
            w, v = w * (1 + r) / g, v * g
        if d in rebal_days:
            prev = assets.index[assets.index.get_loc(d) - 1]
            if equal_all:
                avail = [c for c in close.columns if pd.notna(close.loc[prev, c])]
                new = pd.Series(1 / len(avail), index=avail)
            else:
                vo, vr = view_fn(d, rebal_days)
                expo = core.exposures(wret, fac, prev)
                _, new = core.screen(expo, value, prev, vo, vr, n, t_min)
            idx = new.index.union(w.index)
            diff = new.reindex(idx, fill_value=0) - w.reindex(idx, fill_value=0)
            v *= 1 - diff.clip(lower=0).sum() * core.COST_BUY - (-diff.clip(upper=0)).sum() * core.COST_SELL
            w = new
            for c, wt in new.items():
                log.append({"리밸런싱일": d.date(), "종목코드": c,
                            "종목명": core.UNIVERSE.get(c, (c, ""))[0], "섹터": core.UNIVERSE.get(c, ("", "지수"))[1],
                            "비중": round(wt, 4)})
        eq.append(v)
    return pd.Series(eq, index=days), pd.DataFrame(log)


def oracle_view(macro):
    """그 분기에 실제로 일어난 유가·금리 방향 (사후 정답)."""
    def fn(d, rebal_days):
        i = rebal_days.index(d)
        nxt = rebal_days[i + 1] if i + 1 < len(rebal_days) else macro.index[-1]
        q = macro.loc[d:nxt]
        return (int(np.sign(q["WTI"].iloc[-1] - q["WTI"].iloc[0])),
                int(np.sign(q["US10Y"].iloc[-1] - q["US10Y"].iloc[0])))
    return fn


def fmt(df):
    out = df.copy().astype(object)
    for c in df.columns:
        if c in ("샤프비율", "베타", "정보비율"):
            out[c] = df[c].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        elif c in ("연 알파", "초과수익(CAGR차)"):
            out[c] = df[c].map(lambda x: "-" if pd.isna(x) else f"{x:+.1%}")
        else:
            out[c] = df[c].map(lambda x: "-" if pd.isna(x) else f"{x:.1%}")
    return out


def main():
    ap = argparse.ArgumentParser(description="유가·금리 노출 전략 백테스트")
    ap.add_argument("--oil", type=int, default=1, choices=[-1, 0, 1], help="유가 전망 (1 상승, -1 하락, 0 없음)")
    ap.add_argument("--rate", type=int, default=1, choices=[-1, 0, 1], help="금리 전망 (1 상승, -1 하락, 0 없음)")
    ap.add_argument("--n", type=int, default=core.N_PER_SECTOR, help="섹터별 편입 종목 수")
    ap.add_argument("--tmin", type=float, default=core.T_MIN, help="최소 t값")
    ap.add_argument("--rebal", choices=["Q", "M"], default="Q", help="리밸런싱 주기 (Q 분기, M 월)")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--refresh", action="store_true", help="데이터 새로 받기")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    close, value, macro = core.load_data(cache_dir=os.path.join(HERE, "data"), refresh=a.refresh)
    wret, fac = core.weekly_factors(close, macro)
    view = f"유가{arrow(a.oil)} 금리{arrow(a.rate)}"
    labels = {"전략": f"전략: 스크리닝 ({view})", "대조군": "대조군: 섹터 전종목 동일가중",
              "참고": "참고: 사후 정답 전망 (실현 불가)"}
    print(f"백테스트 {a.start} ~ {close.index[-1].date()} | 전망 {view} | 섹터별 {a.n}종목 | t≥{a.tmin:g} | "
          f"{'분기' if a.rebal == 'Q' else '월'} 리밸런싱")

    bm = macro[core.BENCH].loc[a.start:]
    curves = {"코스피200 (BM)": bm / bm.iloc[0]}
    curves[labels["전략"]], hold = run_backtest(close, value, macro, wret, fac, lambda d, r: (a.oil, a.rate),
                                               a.start, a.rebal, a.n, a.tmin)
    curves[labels["대조군"]], _ = run_backtest(close, value, macro, wret, fac, None, a.start, a.rebal, a.n, a.tmin,
                                              equal_all=True)
    curves[labels["참고"]], _ = run_backtest(close, value, macro, wret, fac, oracle_view(macro), a.start, a.rebal,
                                            a.n, a.tmin)
    color_of = {k: COLORS[next((key for key in COLORS if k.startswith(key)), "전략")] for k in curves}

    # ---------------- 기간별 지표
    tables = {}
    for p, (s, e) in PERIODS.items():
        s = max(pd.Timestamp(s), bm.index[0])
        rows = {k: core.metrics(c.loc[s:e], bm.loc[s:e]) for k, c in curves.items()}
        t = pd.DataFrame(rows).T
        t.loc["코스피200 (BM)", ["베타", "연 알파", "초과수익(CAGR차)", "정보비율"]] = np.nan
        tables[p] = t
        print(f"\n■ {p}")
        print(fmt(t)[["누적수익률", "CAGR", "샤프비율", "MDD", "베타", "초과수익(CAGR차)", "정보비율"]].to_string())
    pd.concat(tables).to_csv(os.path.join(OUT, "metrics.csv"), encoding="utf-8-sig")
    hold.to_csv(os.path.join(OUT, "holdings.csv"), index=False, encoding="utf-8-sig")

    # ---------------- 그림 1: 누적 · 상대 · 낙폭
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.3, 1.2]})
    for k, c in curves.items():
        main_line = k.startswith("코스피200") or k.startswith("전략")
        ls = "--" if k.startswith("참고") else "-"
        axes[0].plot(c, label=f"{k}  {c.iloc[-1] - 1:+.0%}", color=color_of[k], lw=2.3 if main_line else 1.3, ls=ls)
        axes[2].plot((c / c.cummax() - 1) * 100, color=color_of[k], lw=1.4 if main_line else 0.9, ls=ls)
        if not k.startswith("코스피200"):
            axes[1].plot(c / curves["코스피200 (BM)"], color=color_of[k], lw=1.8 if main_line else 1.2, ls=ls, label=k)
    axes[0].set_title(f"누적수익률 ({a.start[:7]} = 1.0, 배당 미반영)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].axhline(1, color="black", lw=0.8, ls="--")
    axes[1].set_title("코스피200 대비 상대성과 (선이 오르면 코스피200보다 우세)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[2].set_title("낙폭 (Drawdown, %)")
    for ax in axes:
        ax.grid(alpha=0.3)
        for p, (s, _) in list(PERIODS.items())[2:]:
            ax.axvline(pd.Timestamp(s), color="gray", lw=0.8, ls=":")
    for p, (s, e) in list(PERIODS.items())[1:]:
        s, e = pd.Timestamp(s), pd.Timestamp(e) if e else bm.index[-1]
        lo, hi = axes[0].get_ylim()
        axes[0].text(s + (e - s) / 2, lo + (hi - lo) * 0.02, p.split(" (")[0], ha="center", va="bottom",
                     fontsize=10, color="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cumulative.png"), dpi=130)
    plt.close()

    # ---------------- 그림 2: 연도별 수익률
    yearly = pd.DataFrame({k: c.resample("YE").last().pct_change().fillna(c.resample("YE").last().iloc[0] - 1)
                           for k, c in curves.items()})
    yearly.index = yearly.index.year
    fig, ax = plt.subplots(figsize=(12, 4.8))
    width = 0.8 / len(curves)
    for j, k in enumerate(curves):
        ax.bar(np.arange(len(yearly)) + (j - (len(curves) - 1) / 2) * width, yearly[k] * 100, width,
               label=k, color=color_of[k], alpha=0.55 if k.startswith("참고") else 1)
    ax.set_xticks(range(len(yearly)), [f"{y}{'*' if y == yearly.index[-1] else ''}" for y in yearly.index])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("수익률 (%)")
    ax.set_title("연도별 수익률 (* 올해는 연초 대비 현재까지)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "annual.png"), dpi=130)
    plt.close()
    print("\n연도별 수익률:")
    print(yearly.apply(lambda c: c.map("{:+.1%}".format)).to_string())

    # ---------------- 그림 3: 성과표
    cols = ["CAGR", "샤프비율", "MDD", "연변동성", "베타", "초과수익(CAGR차)", "정보비율"]
    fig, axs = plt.subplots(len(PERIODS), 1, figsize=(14, 2.2 * len(PERIODS)))
    for ax, p in zip(axs, PERIODS):
        ax.axis("off")
        f = fmt(tables[p])[cols]
        tb = ax.table(cellText=f.values, colLabels=cols, rowLabels=f.index, loc="center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 1.35)
        for (r, c), cell in tb.get_celld().items():
            if r == 0 or c == -1:
                cell.set_text_props(weight="bold")
            if r == 2:
                cell.set_facecolor("#fde8e2")
        ax.set_title(p, loc="left", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---------------- 편입 요약
    last = hold[hold["리밸런싱일"] == hold["리밸런싱일"].max()]
    print(f"\n마지막 리밸런싱 ({last['리밸런싱일'].iloc[0]}):",
          ", ".join(f"{r.종목명} {r.비중:.0%}" for r in last.itertuples()))
    share_bm = hold[hold["종목코드"] == core.BENCH].groupby("리밸런싱일")["비중"].sum()
    n_reb = hold["리밸런싱일"].nunique()
    print(f"코스피200으로 대체된 리밸런싱: {len(share_bm)}/{n_reb}회 (섹터에 조건 맞는 종목이 없을 때)")
    print(f"\n저장: {os.path.relpath(OUT)}/ cumulative.png, annual.png, metrics.png, metrics.csv, holdings.csv")


if __name__ == "__main__":
    main()
