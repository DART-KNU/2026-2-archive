"""데이터 로드 → A안/B안/필터없음 비교 백테스트(종가 기준) → 결과표·차트 저장."""
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import backtest as bt
from load_data import load_item, CACHE

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_all():
    data = {k: load_item(k) for k in ("close", "mcap", "tv5")}
    close = data["close"]
    codes = close.columns[close.notna().any()]
    for k in data:
        data[k] = data[k].reindex(index=close.index, columns=codes)
    kospi = load_item("kospi").iloc[:, 0]
    sector = pd.read_pickle(CACHE / "sector.pkl") if (CACHE / "sector.pkl").exists() else pd.Series(dtype=str)
    return data, kospi, sector


def main():
    data, kospi, sector = load_all()
    print("universe:", data["close"].shape, data["close"].index[0].date(), "~", data["close"].index[-1].date())
    base = bt.Params()
    variants = [
        base,
        replace(base, name="B_soft_filter", weight_off=0.025),
        replace(base, name="C_no_filter", use_market_filter=False),
    ]
    if len(sys.argv) > 1:
        variants = [v for v in variants if v.name in sys.argv[1:]]
    summary, curves = {}, {}
    for p in variants:
        nav, tr, to, cl = bt.run_close(data, kospi, sector, p)
        st, on = bt.save(p, nav, tr, to, cl, kospi)
        summary[p.name] = st
        curves[p.name] = nav["nav"]
        print(f"\n== {p.name} ==")
        for k, v in st.items():
            print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
        print(on.round(4).to_string())
    pd.DataFrame(summary).to_csv(bt.OUT / "summary.csv")
    plot(curves, kospi)


def plot(curves: dict, kospi: pd.Series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    idx = next(iter(curves.values())).index
    k = kospi.reindex(idx).ffill()
    ax1.plot(k / k.iloc[0], color="#999999", lw=1.2, label="KOSPI")
    for name, s in curves.items():
        ax1.plot(s / s.iloc[0], lw=1.6, label=name)
        ax2.plot(s / s.cummax() - 1, lw=1.0, label=name)
    ax2.plot(k / k.cummax() - 1, color="#999999", lw=1.0)
    ax1.set_title("박스권 돌파형 52주 신고가 전략 — 누적수익 (시작=1)")
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)
    ax2.set_title("낙폭 (Drawdown)"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(bt.OUT / "equity.png", dpi=150)


if __name__ == "__main__":
    main()
