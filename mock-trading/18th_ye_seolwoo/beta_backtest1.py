"""
베타 그룹별 수익률 백테스트 (고베타 vs 저베타)
- 데이터: DataGuide 엑셀 (코드 x 아이템 행, 날짜 열) — 수정주가, 수정시가, 거래대금, 시가총액
- 시장수익률: 지수 데이터가 없으므로 '전 종목 시총가중 수익률'로 대체 (KOSPI+KOSDAQ 합산 시장)
- 매주 첫 거래일 리밸런싱, 신호는 전 거래일 종가까지 정보만 사용, 체결은 리밸런싱일 시가
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============================== 설정 ==============================
DATA_PATH   = "0929_data.xlsx"   # DataGuide 원본 엑셀
OUT_DIR     = Path("results")
BETA_WIN    = 60                 # 베타 추정 기간(거래일)
BETA_TYPE   = "all"              # "all": 일반 베타 / "down": 시장 하락일만 사용한 하방 베타
MIN_OBS     = 40                 # 베타 계산 최소 관측치
MIN_MCAP    = 1_000_000          # 시가총액 하한 (백만원) = 1조
MIN_VALUE   = 5_000_000_000      # 20일 평균 거래대금 하한 (원) = 50억
N_QUANT     = 5                  # 베타 분위 그룹 수
BUY_COST    = 0.0010             # 매수 수수료 10bp
SELL_COST   = 0.0030             # 매도 수수료 10bp + 거래세 20bp
RF          = 0.0                # 무위험수익률(연). CD91 넣으려면 여기 수정
# =================================================================


def load_dataguide(path):
    raw = pd.read_excel(path, header=8, engine="calamine")
    raw["코드"] = raw["코드"].astype(str)
    date_cols = [c for c in raw.columns[6:]]
    names = raw.drop_duplicates("코드").set_index("코드")["코드명"]
    items = {"수정주가(원)": "close", "수정시가(원)": "open",
             "거래대금(원)": "value", "시가총액(백만원)": "mcap"}
    out = {}
    for k, v in items.items():
        df = raw[raw["아이템명"] == k].set_index("코드")[date_cols].T
        df.index = pd.to_datetime(df.index)
        out[v] = df.apply(pd.to_numeric, errors="coerce")
    # 보통주만 (코드 끝자리 0), 기간 내 데이터 없는 종목 제거
    keep = [c for c in out["close"].columns
            if c.endswith("0") and out["close"][c].notna().any()]
    for v in out:
        out[v] = out[v][keep]
    return out, names


def rolling_beta(r, rm, win, min_obs, kind="all"):
    """r: (T,N) 종목수익률, rm: (T,) 시장수익률. t시점 베타는 t까지의 데이터로 계산."""
    if kind == "down":
        down = rm < 0
        rm_use = rm.where(down)
        r_use = r.where(np.broadcast_to(down.values[:, None], r.shape))
        win_eff = win * 2  # 하락일은 전체의 절반 정도라 창을 2배로
    else:
        rm_use, r_use, win_eff = rm, r, win
    valid = r_use.notna() & rm_use.notna().values[:, None]
    x = rm_use.values[:, None] * np.ones(r_use.shape)
    x = np.where(valid, x, np.nan)
    y = np.where(valid, r_use.values, np.nan)
    X, Y = pd.DataFrame(x, index=r.index, columns=r.columns), pd.DataFrame(y, index=r.index, columns=r.columns)
    n = X.rolling(win_eff, min_periods=1).count()
    mx, my = X.rolling(win_eff, min_periods=1).mean(), Y.rolling(win_eff, min_periods=1).mean()
    cov = (X * Y).rolling(win_eff, min_periods=1).mean() - mx * my
    var = (X * X).rolling(win_eff, min_periods=1).mean() - mx ** 2
    beta = cov / var
    return beta.where(n >= (min_obs if kind == "all" else min_obs // 2 + 10))


def simulate(C, O, dates, targets, reb_days):
    """targets: {rebalance_idx: np.array(weights)}. 시가 체결, 비용 반영, 일간 NAV 반환."""
    T, N = C.shape
    h = np.zeros(N); cash = 1.0
    nav = np.full(T, np.nan); turn = []
    start = min(reb_days)
    for t in range(start, T):
        c_prev = C[t - 1]
        if t in targets:
            r_open = np.where(np.isnan(O[t]) | np.isnan(c_prev), 1.0, O[t] / c_prev)
            h = h * r_open
            nav_open = h.sum() + cash
            tgt = targets[t] * nav_open
            buy = np.clip(tgt - h, 0, None).sum(); sell = np.clip(h - tgt, 0, None).sum()
            cost = buy * BUY_COST + sell * SELL_COST
            turn.append((buy + sell) / 2 / nav_open)
            scale = (nav_open - cost) / nav_open
            h = tgt * scale
            cash = (nav_open - cost) - h.sum()
            r_day = np.where(np.isnan(C[t]) | np.isnan(O[t]), 1.0, C[t] / O[t])
        else:
            r_day = np.where(np.isnan(C[t]) | np.isnan(c_prev), 1.0, C[t] / c_prev)
        h = h * r_day
        nav[t] = h.sum() + cash
    return pd.Series(nav, index=dates).dropna(), np.array(turn)


def stats(nav, rm_nav, label):
    r = nav.pct_change().dropna()
    rm = rm_nav.pct_change().reindex(r.index)
    yrs = len(r) / 252
    total = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + total) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - RF) / vol
    mdd = (nav / nav.cummax() - 1).min()
    beta = np.cov(r, rm)[0, 1] / rm.var()
    up, dn = rm > 0, rm < 0
    return {"그룹": label, "누적수익률": total, "연환산수익률": cagr, "연변동성": vol,
            "샤프": sharpe, "MDD": mdd, "실현베타": beta,
            "상승일 평균(bp)": r[up].mean() * 1e4, "하락일 평균(bp)": r[dn].mean() * 1e4}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    d, names = load_dataguide(DATA_PATH)
    close, opn, value, mcap = d["close"], d["open"], d["value"], d["mcap"]
    dates = close.index
    ret = close.pct_change(fill_method=None)

    # 시장수익률 = 전일 시총 가중 (전 종목)
    w_prev = mcap.shift(1).where(ret.notna())
    rm = (ret * w_prev).sum(axis=1) / w_prev.sum(axis=1)
    rm.iloc[0] = np.nan
    mkt_nav = (1 + rm.fillna(0)).cumprod()

    beta = rolling_beta(ret, rm, BETA_WIN, MIN_OBS, BETA_TYPE)
    val20 = value.rolling(20, min_periods=15).mean()
    elig = (mcap >= MIN_MCAP) & (val20 >= MIN_VALUE) & beta.notna() & close.notna()

    # 리밸런싱일: 매주 첫 거래일 (베타 계산 가능한 시점부터)
    wk = pd.Series(dates.isocalendar().year.astype(str) + "-" + dates.isocalendar().week.astype(str).str.zfill(2), index=dates)
    first_day = ~wk.duplicated()
    first_valid = elig.sum(axis=1).gt(50).idxmax()
    reb = [i for i, dt in enumerate(dates) if first_day.iloc[i] and dt > first_valid]

    groups = {"고베타(β>1)": {}, "저베타(β<1)": {}, "유니버스 동일가중": {}}
    groups.update({f"Q{q+1}" + (" (최저β)" if q == 0 else " (최고β)" if q == N_QUANT - 1 else ""): {} for q in range(N_QUANT)})
    qnames = [k for k in groups if k.startswith("Q")]
    log = []
    Ncols = close.shape[1]
    for t in reb:
        s = t - 1  # 신호일 = 전 거래일 종가
        ok = elig.iloc[s].values & opn.iloc[t].notna().values
        b = beta.iloc[s].values
        idx = np.where(ok)[0]
        def ew(sel):
            w = np.zeros(Ncols)
            if len(sel): w[sel] = 1 / len(sel)
            return w
        hi, lo = idx[b[idx] > 1], idx[b[idx] <= 1]
        groups["고베타(β>1)"][t] = ew(hi)
        groups["저베타(β<1)"][t] = ew(lo)
        groups["유니버스 동일가중"][t] = ew(idx)
        qs = pd.qcut(b[idx], N_QUANT, labels=False)
        for q in range(N_QUANT):
            groups[qnames[q]][t] = ew(idx[qs == q])
        log.append({"리밸런싱일": dates[t].date(), "유니버스": len(idx), "고베타 수": len(hi),
                    "저베타 수": len(lo), "베타 중앙값": np.nanmedian(b[idx])})

    C, O = close.values, opn.values
    navs, turns = {}, {}
    for g, tg in groups.items():
        navs[g], turns[g] = simulate(C, O, dates, tg, reb)
    start = dates[reb[0]]
    mkt = mkt_nav.loc[start:] / mkt_nav.loc[start:].iloc[0]
    navs["시장(시총가중)"] = mkt

    # 전체 기간 성과
    rows = [stats(n / n.iloc[0], mkt, g) | {"주간 회전율": turns.get(g, np.array([np.nan])).mean()}
            for g, n in navs.items()]
    res = pd.DataFrame(rows).set_index("그룹")

    # 최근 조정 국면: 시장 고점 → 현재
    peak = mkt.idxmax()
    rows2 = []
    for g, n in navs.items():
        seg = n.loc[peak:]
        rows2.append({"그룹": g, "고점 이후 수익률": seg.iloc[-1] / seg.iloc[0] - 1,
                      "고점 이후 MDD": (seg / seg.cummax() - 1).min()})
    res2 = pd.DataFrame(rows2).set_index("그룹")

    # 출력
    pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.3f}".format)
    print(f"백테스트 기간: {start.date()} ~ {dates[-1].date()}  |  베타: {BETA_TYPE}, {BETA_WIN}일")
    print(f"리밸런싱 {len(reb)}회, 평균 유니버스 {np.mean([l['유니버스'] for l in log]):.0f}종목\n")
    print(res.round(3)); print(f"\n시장 고점: {peak.date()}  → {dates[-1].date()}")
    print(res2.round(3))
    res.to_csv(OUT_DIR / f"summary_{BETA_TYPE}.csv", encoding="utf-8-sig")
    res2.to_csv(OUT_DIR / f"drawdown_{BETA_TYPE}.csv", encoding="utf-8-sig")
    pd.DataFrame(log).to_csv(OUT_DIR / f"rebalance_log_{BETA_TYPE}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(navs).to_csv(OUT_DIR / f"nav_{BETA_TYPE}.csv", encoding="utf-8-sig")

    # 차트
    import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    from matplotlib import font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    kfont = next((f for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR",
                              "Noto Sans CJK JP", "Noto Sans CJK SC"] if f in avail), "DejaVu Sans")
    plt.rcParams["font.family"] = kfont
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    style = {"고베타(β>1)": ("#d1495b", 2), "저베타(β<1)": ("#2e86ab", 2),
             "유니버스 동일가중": ("#888888", 1.2), "시장(시총가중)": ("#222222", 1.2)}
    for g, (col, lw) in style.items():
        n = navs[g]; ax[0].plot(n.index, n / n.iloc[0], color=col, lw=lw, label=g)
        ax[1].plot(n.index, n / n.cummax() - 1, color=col, lw=lw)
    ax[0].axvline(peak, color="k", ls=":", lw=1); ax[1].axvline(peak, color="k", ls=":", lw=1)
    ax[0].set_title(f"베타 그룹별 누적수익률 ({BETA_TYPE} beta, {BETA_WIN}일, 주간 리밸런싱)")
    ax[0].legend(frameon=False); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
    ax[1].set_ylabel("드로우다운")
    fig.tight_layout(); fig.savefig(OUT_DIR / f"beta_groups_{BETA_TYPE}.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    q = res.loc[qnames]
    ax.bar(range(len(q)), q["연환산수익률"], color=["#2e86ab", "#7fb3d5", "#bbbbbb", "#e59aa4", "#d1495b"])
    ax.set_xticks(range(len(q))); ax.set_xticklabels(qnames)
    ax.set_title("베타 5분위별 연환산수익률"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / f"beta_quintiles_{BETA_TYPE}.png", dpi=150)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1: DATA_PATH = sys.argv[1]
    if len(sys.argv) > 2: BETA_TYPE = sys.argv[2]
    main()
