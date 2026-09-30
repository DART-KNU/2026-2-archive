"""
베타 그룹별 수익률 백테스트 (고베타 vs 저베타)
- 데이터: DataGuide 엑셀 (long/wide 형식 자동 인식) — 수정주가 필수, 수정시가·거래대금·시가총액은 있으면 사용
- 실행: python beta_backtest.py <엑셀경로> <all|down> <결과폴더>
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
    """DataGuide 엑셀 두 형식 모두 지원
    - long: 행=코드x아이템, 열=날짜 (header 9행이 '코드, 코드명, ..., 날짜들')
    - wide: 행=날짜, 열=코드 (9~14행이 코드/코드명/유형/아이템코드/아이템명/집계주기)
    없는 아이템은 None 으로 반환"""
    items = {"수정주가(원)": "close", "수정시가(원)": "open",
             "거래대금(원)": "value", "시가총액(백만원)": "mcap"}
    head = pd.read_excel(path, header=None, nrows=10, engine="calamine")
    out = {v: None for v in items.values()}
    if str(head.iloc[8, 1]).startswith("A"):          # wide 형식
        raw = pd.read_excel(path, header=None, engine="calamine")
        codes, nm, itm = raw.iloc[8, 1:], raw.iloc[9, 1:], raw.iloc[12, 1:]
        body = raw.iloc[14:].set_index(0)
        body.index = pd.to_datetime(body.index)
        body.columns = range(1, raw.shape[1])
        names = pd.Series(nm.values, index=codes.astype(str).values).groupby(level=0).first()
        for k, v in items.items():
            cols = itm[itm == k].index
            if len(cols):
                df = body[cols].apply(pd.to_numeric, errors="coerce")
                df.columns = codes[cols].astype(str).values
                out[v] = df
    else:                                              # long 형식
        raw = pd.read_excel(path, header=8, engine="calamine")
        raw["코드"] = raw["코드"].astype(str)
        date_cols = list(raw.columns[6:])
        names = raw.drop_duplicates("코드").set_index("코드")["코드명"]
        for k, v in items.items():
            sub = raw[raw["아이템명"] == k]
            if len(sub):
                df = sub.set_index("코드")[date_cols].T
                df.index = pd.to_datetime(df.index)
                out[v] = df.apply(pd.to_numeric, errors="coerce")
    # 보통주만 (코드 끝자리 0), 기간 내 데이터 없는 종목 제거
    keep = [c for c in out["close"].columns
            if c.endswith("0") and out["close"][c].notna().any()]
    for v in out:
        if out[v] is not None:
            out[v] = out[v].reindex(columns=keep)
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
    """targets: {rebalance_idx: np.array(weights)}. 시가 체결, 비용 반영.
    반환: 일간 NAV, 리밸런싱별 회전율, 종목별 일간 손익(T x N, 비용 포함), 일간 비중(T x N)"""
    T, N = C.shape
    h = np.zeros(N); cash = 1.0
    nav = np.full(T, np.nan); turn = []
    pnl = np.zeros((T, N)); wts = np.zeros((T, N))
    start = min(reb_days)
    for t in range(start, T):
        c_prev = C[t - 1]
        if t in targets:
            r_open = np.where(np.isnan(O[t]) | np.isnan(c_prev), 1.0, O[t] / c_prev)
            pnl[t] += h * (r_open - 1)
            h = h * r_open
            nav_open = h.sum() + cash
            tgt = targets[t] * nav_open
            buy_i = np.clip(tgt - h, 0, None); sell_i = np.clip(h - tgt, 0, None)
            cost_i = buy_i * BUY_COST + sell_i * SELL_COST
            cost = cost_i.sum()
            turn.append((buy_i.sum() + sell_i.sum()) / 2 / nav_open)
            pnl[t] -= cost_i
            scale = (nav_open - cost) / nav_open
            h = tgt * scale
            cash = (nav_open - cost) - h.sum()
            r_day = np.where(np.isnan(C[t]) | np.isnan(O[t]), 1.0, C[t] / O[t])
        else:
            r_day = np.where(np.isnan(C[t]) | np.isnan(c_prev), 1.0, C[t] / c_prev)
        pnl[t] += h * (r_day - 1)
        h = h * r_day
        nav[t] = h.sum() + cash
        wts[t] = h / nav[t]
    return pd.Series(nav, index=dates).dropna(), np.array(turn), pnl, wts


def dispersion(pnl, wts, nav, dates, start, end=None):
    """대회 관리점수 근사: 포트 분산(비중 집중도) + 수익 분산(손익 집중도), 구간 [start, end]"""
    idx = np.where((dates >= pd.Timestamp(start)) & ((dates <= pd.Timestamp(end)) if end else True))[0]
    W = wts[idx]; held = W > 1e-9
    n_hold = held.sum(axis=1)
    top5 = np.sort(W, axis=1)[:, -5:].sum(axis=1)
    hhi = (W ** 2).sum(axis=1) / np.maximum(W.sum(axis=1), 1e-12) ** 2
    nav0 = nav.iloc[0] if start is None else nav.loc[:pd.Timestamp(start)].iloc[-1] if (nav.index <= pd.Timestamp(start)).any() else nav.iloc[0]
    P = pnl[idx].sum(axis=0) / nav0                   # 종목별 누적 손익 (구간 시작 NAV 대비)
    traded = (wts[idx] > 1e-9).any(axis=0)
    gains = P[P > 0]; losses = P[P < 0]
    g_sorted = np.sort(gains)[::-1]
    return {
        "평균 보유 종목 수": n_hold.mean(),
        "평균 Top5 편입비": top5.mean(),
        "비중 유효종목수(1/HHI)": (1 / hhi[hhi > 0]).mean(),
        "거래 종목 수": int(traded.sum()),
        "수익 종목 수": int((P > 0).sum()),
        "수익 종목 비율": (P > 0).sum() / max(traded.sum(), 1),
        "상위5 종목 이익 비중": g_sorted[:5].sum() / gains.sum() if len(gains) else np.nan,
        "최대 1종목 이익 비중": g_sorted[0] / gains.sum() if len(gains) else np.nan,
        "이익 유효종목수(1/HHI)": 1 / ((gains / gains.sum()) ** 2).sum() if len(gains) else np.nan,
        "총이익(NAV%)": gains.sum(), "총손실(NAV%)": losses.sum(),
    }


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

    has_mcap = mcap is not None
    if has_mcap:   # 시장수익률 = 전일 시총 가중 (전 종목)
        w_prev = mcap.shift(1).where(ret.notna())
        rm = (ret * w_prev).sum(axis=1) / w_prev.sum(axis=1)
        mkt_label = "시장(시총가중)"
    else:          # 시총 없으면 동일가중 시장 (극단값 ±30% 클립)
        rm = ret.clip(-0.3, 0.3).mean(axis=1)
        mkt_label = "시장(동일가중)"
    rm.iloc[0] = np.nan
    mkt_nav = (1 + rm.fillna(0)).cumprod()

    beta = rolling_beta(ret, rm, BETA_WIN, MIN_OBS, BETA_TYPE)
    elig = beta.notna() & close.notna()
    if has_mcap:
        elig &= mcap >= MIN_MCAP
    if value is not None:
        elig &= value.rolling(20, min_periods=15).mean() >= MIN_VALUE
    if not has_mcap and value is None:
        # 시총·거래대금이 없을 때 대체 필터: 최근 60일 중 가격 변동 없는 날 20% 이하 (비유동·거래정지 제외)
        zero = (ret == 0).astype(float).where(ret.notna())
        elig &= zero.rolling(60, min_periods=40).mean() <= 0.2
    if opn is None:   # 시가 없으면 리밸런싱일 종가 체결
        opn = close

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
    navs, turns, disp = {}, {}, {}
    peak_tmp = None
    for g, tg in groups.items():
        navs[g], turns[g], pnl_g, wts_g = simulate(C, O, dates, tg, reb)
        disp[g] = (pnl_g, wts_g)
    start = dates[reb[0]]
    mkt = mkt_nav.loc[start:] / mkt_nav.loc[start:].iloc[0]
    navs[mkt_label] = mkt

    # 전체 기간 성과
    rows = [stats(n / n.iloc[0], mkt, g) | {"주간 회전율": turns.get(g, np.array([np.nan])).mean()}
            for g, n in navs.items()]
    res = pd.DataFrame(rows).set_index("그룹")

    # 최근 조정 국면: 시장 고점 → 현재
    peak = mkt.idxmax()
    rows2 = []
    for g, n in navs.items():
        seg = n.loc[peak:]
        last = n.iloc[-43:]  # 마지막 2개월(42거래일) = 대회 기간 길이
        rows2.append({"그룹": g, "고점 이후 수익률": seg.iloc[-1] / seg.iloc[0] - 1,
                      "고점 이후 MDD": (seg / seg.cummax() - 1).min(),
                      "마지막 2개월 수익률": last.iloc[-1] / last.iloc[0] - 1,
                      "마지막 2개월 MDD": (last / last.cummax() - 1).min()})
    res2 = pd.DataFrame(rows2).set_index("그룹")

    # 대회 관리점수 근사: 포트 분산 + 수익 분산
    drows = []
    for g, (pnl_g, wts_g) in disp.items():
        for lab, a in [("전체 기간", start), ("시장 고점 이후", peak)]:
            drows.append({"그룹": g, "구간": lab} | dispersion(pnl_g, wts_g, navs[g], dates, a))
    res3 = pd.DataFrame(drows).set_index(["그룹", "구간"])

    # 출력
    pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.3f}".format)
    print(f"백테스트 기간: {start.date()} ~ {dates[-1].date()}  |  베타: {BETA_TYPE}, {BETA_WIN}일")
    print(f"리밸런싱 {len(reb)}회, 평균 유니버스 {np.mean([l['유니버스'] for l in log]):.0f}종목\n")
    print(res.round(3)); print(f"\n시장 고점: {peak.date()}  → {dates[-1].date()}")
    print(res2.round(3))
    print("\n[포트 분산 / 수익 분산]")
    print(res3.round(3).to_string())
    res3.to_csv(OUT_DIR / f"dispersion_{BETA_TYPE}.csv", encoding="utf-8-sig")
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
             "유니버스 동일가중": ("#888888", 1.2), mkt_label: ("#222222", 1.2)}
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
    if len(sys.argv) > 3: OUT_DIR = Path(sys.argv[3])
    main()
