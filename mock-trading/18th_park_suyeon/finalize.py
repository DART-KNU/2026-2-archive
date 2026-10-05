"""최종안 백테스트 + 발표용 차트/표 생성 (output/Final, output/figs)."""
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
from load_data import load_names
from run_backtest import load_all

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
NAVY, ORANGE, GOLD, BLUE, GRAY = "#1B2240", "#C05621", "#C9A227", "#4DA3FF", "#9AA0A6"
FINAL = bt.FINAL
FIG = bt.OUT / "figs"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data, kospi, sector = load_all()
    names = load_names()
    nav, tr, to, cl = bt.run_close(data, kospi, sector, FINAL)
    st, on = bt.save(FINAL, nav, tr, to, cl, kospi)
    orig = pd.read_csv(bt.OUT / "A_hard_filter" / "nav.csv", index_col=0, parse_dates=True)["nav"]
    k = kospi.reindex(nav.index).ffill()

    # equity + drawdown
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    a1.plot(k / k.iloc[0], color=GRAY, lw=1.3, label="KOSPI")
    a1.plot(orig / orig.iloc[0], color=BLUE, lw=1.1, label="초기 규칙")
    a1.plot(nav["nav"] / nav["nav"].iloc[0], color=ORANGE, lw=2.0, label="최종안")
    a1.set_ylabel("누적 (시작=1)"); a1.legend(loc="upper left", frameon=False); a1.grid(alpha=.25)
    a2.fill_between(nav.index, nav["nav"] / nav["nav"].cummax() - 1, 0, color=ORANGE, alpha=.4, label="최종안")
    a2.plot(k / k.cummax() - 1, color=GRAY, lw=1, label="KOSPI")
    a2.set_ylabel("낙폭"); a2.legend(loc="lower left", frameon=False); a2.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "equity_dd.png", dpi=220); plt.close(fig)

    # yearly
    ye = nav["nav"].resample("YE").last(); yr = ye.pct_change(); yr.iloc[0] = ye.iloc[0] - 1
    ky = kospi.resample("YE").last().pct_change().reindex(yr.index)
    ky.iloc[0] = kospi.loc[:"2017-12-31"].iloc[-1] / kospi.loc[:"2016-12-31"].iloc[-1] - 1
    fig, ax = plt.subplots(figsize=(9, 3.2)); x = np.arange(len(yr))
    ax.bar(x - .2, yr.values * 100, .4, color=ORANGE, label="최종안"); ax.bar(x + .2, ky.values * 100, .4, color=GRAY, label="KOSPI")
    ax.set_xticks(x); ax.set_xticklabels([f"{d.year}" + ("*" if d.year == 2026 else "") for d in yr.index]); ax.axhline(0, color="k", lw=.6)
    for i, v in enumerate(yr.values): ax.text(i - .2, v * 100 + (1.5 if v >= 0 else -5), f"{v*100:.0f}", ha="center", fontsize=8)
    ax.set_ylabel("연간 수익률 (%)"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "yearly.png", dpi=220); plt.close(fig)
    pd.DataFrame({"strategy": yr, "kospi": ky}).to_csv(bt.OUT / "Final" / "yearly.csv")

    # exit reasons
    lab = {"stop": "손절 -8%", "trail": "트레일링 -15%", "tp1": "1차 익절 +20%", "time": "시간청산 40일", "breakeven": "본전 이탈", "delisted/halt": "정지/폐지"}
    rs = tr.groupby("reason").agg(n=("ret", "size"), avg=("ret", "mean")).rename(index=lab).sort_values("avg")
    rs = rs[rs["n"] >= 5]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.barh(rs.index, rs["avg"] * 100, color=[ORANGE if v < 0 else NAVY for v in rs["avg"]])
    for i, (n, a) in enumerate(zip(rs["n"], rs["avg"])):
        ax.text(a * 100 + (0.8 if a >= 0 else -0.8), i, f"{a*100:.1f}% (n={n})", va="center", ha="left" if a >= 0 else "right", fontsize=8)
    ax.set_xlim(rs["avg"].min() * 100 - 22, rs["avg"].max() * 100 + 16)
    ax.axvline(0, color="k", lw=.6); ax.set_xlabel("평균 수익률 (%)"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "exit_reasons.png", dpi=220); plt.close(fig)

    # top contributors (수익 집중도)
    g = bt.position_returns(tr).sort_values(ascending=False).rename("pos_ret")
    top = g.head(8).reset_index(); top["name"] = top["code"].map(names)
    top.to_csv(bt.OUT / "Final" / "top_trades.csv", index=False, encoding="utf-8-sig")

    # screening funnel & watchlist as of last date
    c = data["close"]; kk = len(c) - 1
    C = c.to_numpy(float); HC = c.rolling(250, min_periods=125).max().to_numpy(float)
    MC = data["mcap"].to_numpy(float); TV = data["tv5"].to_numpy(float); codes = c.columns
    s0 = np.array([x.endswith("0") for x in codes]) & ~np.isnan(C[kk])
    s1 = s0 & (MC[kk] >= FINAL.min_mcap) & (TV[kk] > FINAL.min_tv5)
    s2 = s1 & (C[kk] >= HC[kk] * bt.WATCH_NEAR) & (C[kk] < HC[kk])
    rows = []
    for j in np.flatnonzero(s2):
        win = C[kk - 249:kk + 1, j]; pk = int(np.nanargmax(win)); bl = len(win) - 1 - pk
        rng = 1 - np.nanmin(win[pk:]) / HC[kk, j]
        base_ok = FINAL.base_min <= bl <= FINAL.base_max and rng <= FINAL.base_range
        ready = base_ok and C[kk, j] >= HC[kk, j] * FINAL.near_high
        rows.append(dict(code=codes[j], name=names.get(codes[j], ""), close=C[kk, j], breakout=HC[kk, j], gap=HC[kk, j] / C[kk, j] - 1,
                         base_days=bl, base_range=rng, mcap_조=MC[kk, j] / 1e12, tv5_억=TV[kk, j] / 1e8,
                         watch=base_ok, passed=ready))
    near = pd.DataFrame(rows)
    near["tier"] = np.where(near["passed"], 0, np.where(near["watch"], 1, 2))
    near = near.sort_values(["tier", "gap"])
    near.to_csv(bt.OUT / "Final" / "watchlist.csv", index=False, encoding="utf-8-sig")
    funnel = pd.Series({"상장 보통주": int(s0.sum()), "시총 1,000억↑ · 거래대금 30억↑": int(s1.sum()),
                        "52주 고가 90% 이내": int(s2.sum()), "관심: +10~120일 횡보·변동폭 40%": int(near["watch"].sum()),
                        "매수 대기: 고가 95% 이내": int(near["passed"].sum())})
    funnel.to_csv(bt.OUT / "Final" / "funnel.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 2.6))
    lbl, vals = funnel.index[::-1], funnel.values[::-1]
    ax.barh(lbl, vals, color=[ORANGE if i == 0 else NAVY for i in range(len(vals))])
    for i, v in enumerate(vals): ax.text(v + max(vals) * .01, i, f"{v:,}", va="center", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.15); ax.set_xticks([]); ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "funnel.png", dpi=220); plt.close(fig)

    # market filter chart
    kk2 = kospi.loc["2025-06-01":]; ma = kospi.rolling(60).mean().loc[kk2.index]; onf = bt.market_filter(kospi, FINAL.filter_mode).loc[kk2.index]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(kk2.index, kk2, color=NAVY, lw=1.4, label="KOSPI"); ax.plot(ma.index, ma, color=ORANGE, lw=1.2, ls="--", label="60일 이동평균")
    ax.fill_between(kk2.index, kk2.min() * .95, kk2.max() * 1.02, where=onf, color=BLUE, alpha=.15, label="필터 ON (신규 매수 가능)")
    ax.set_ylim(kk2.min() * .95, kk2.max() * 1.02); ax.legend(loc="upper left", frameon=False, fontsize=9); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "filter.png", dpi=220); plt.close(fig)

    # example trade chart (HD현대일렉트릭)
    close = c["A267260"].dropna(); s = close.loc["2023-01-01":"2024-09-30"]
    t = tr[(tr["code"] == "A267260")]; t = t[pd.to_datetime(t["entry_date"]).dt.year == 2024]
    assert len(t), "example trade missing"
    entry = pd.Timestamp(t["entry_date"].iloc[0]); exit_d = pd.to_datetime(t["exit_date"]).max()
    tp = t[t["reason"] == "tp1"]
    hc = close.rolling(250, min_periods=125).max()
    prev = close.index[close.index.get_loc(entry) - 1]
    pivot = hc.loc[prev]; peak_d = close.loc[:prev].iloc[-250:].idxmax()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(s.index, s.values, color=NAVY, lw=1.3)
    ax.axvspan(peak_d, entry, color=GOLD, alpha=.25, label=f"횡보 {close.index.get_loc(entry) - close.index.get_loc(peak_d)}거래일")
    ax.hlines(pivot, peak_d - pd.Timedelta(days=45), entry + pd.Timedelta(days=45), color=ORANGE, ls="--", lw=1.2, label="직전 52주 최고 종가 (돌파가)")
    ax.scatter([entry], [close.loc[entry]], color=ORANGE, s=60, zorder=5, label="매수 (종가 돌파)")
    if len(tp):
        d = pd.Timestamp(tp["exit_date"].iloc[0]); ax.scatter([d], [close.loc[d]], color=BLUE, s=55, zorder=5, label="1차 익절 +20% (절반)")
    ax.scatter([exit_d], [close.loc[exit_d]], color="black", marker="x", s=60, zorder=5, label="트레일링 -15% 청산")
    pos_ret = (t["ret"] * t["w"]).sum() / t["w"].sum()
    ax.set_title(f"사례: HD현대일렉트릭 — {entry:%Y.%m} 돌파 매수 → {exit_d:%Y.%m} 청산, 포지션 수익률 {pos_ret*100:+.0f}%", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=8); ax.grid(alpha=.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout(); fig.savefig(FIG / "example.png", dpi=220); plt.close(fig)

    print({k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()})
    print(on.round(3).to_string()); print(funnel); print(near.head(12).round(3).to_string(index=False)); print(top.round(3).to_string(index=False))
    print("filter now:", bool(bt.market_filter(kospi, FINAL.filter_mode).iloc[-1]))


if __name__ == "__main__":
    main()
