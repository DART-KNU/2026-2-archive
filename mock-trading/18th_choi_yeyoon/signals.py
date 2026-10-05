"""
신호 계산
  SUE   = (이번 분기 영업이익 - 전년 동기) / 직전 8개 분기 (전년동기 대비 차이)의 표준편차
  EAR   = 이벤트일 t0 기준 [t-1, t+1] 3일간 (종목 수익률 - 섹터 평균 수익률) 합
  Score = 0.5 * pct(SUE) + 0.5 * pct(EAR)   ← pct는 '직전 1년간 같은 섹터 이벤트' 대비 백분위 (PIT)
  진입  = pct(Score) >= 0.8  &  SUE > 0  &  EAR > 0
"""
import numpy as np
import pandas as pd

import config as C


# ─── 파생 패널 ────────────────────────────────────────────
def prepare_panels(panels, sectors):
    P = {k: v for k, v in panels.items()}
    ret = P["ret"]
    P["dates"] = ret.index
    P["cols"] = list(ret.columns)
    P["tv20"] = P["tv"].rolling(20, min_periods=15).mean()
    P["listed"] = P["close"].notna().cumsum()
    P["vol60"] = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    P["ret21"] = (1 + ret.fillna(0)).rolling(21).apply(np.prod, raw=True) - 1
    sec = sectors.reindex(ret.columns).fillna("NA")
    P["sector"] = sec
    P["sec_mean"] = ret.T.groupby(sec.values).mean().T          # 날짜 × 섹터 평균 수익률
    return P


def eligible_mask(P, i):
    """i일 종가 기준 유니버스 조건 통과 여부 (종목 배열)"""
    return ((P["mcap"].values[i] >= C.MIN_MCAP) &
            (P["tv20"].values[i] >= C.MIN_TV20) &
            (P["listed"].values[i] >= C.MIN_LISTED_DAYS) &
            np.isfinite(P["vol60"].values[i]))


# ─── SUE ─────────────────────────────────────────────────
def _qrange(a, b):
    try:
        return pd.date_range(a, b, freq="QE-DEC")
    except ValueError:                                  # pandas < 2.2
        return pd.date_range(a, b, freq="Q-DEC")


def add_sue(q):
    out = []
    for t, g in q.groupby("ticker"):
        g = g.set_index("pend").sort_index()
        g = g[~g.index.duplicated()]
        g = g.reindex(_qrange(g.index.min(), g.index.max()))   # 빠진 분기는 NaN (연속성 보장)
        g["ticker"] = t
        g["d_op"] = g["op"] - g["op"].shift(4)
        sd = g["d_op"].shift(1).rolling(C.SUE_LOOKBACK_Q, min_periods=C.SUE_LOOKBACK_Q - 2).std()
        g["sue"] = g["d_op"] / sd.replace(0, np.nan)
        out.append(g.rename_axis("pend").reset_index())
    q = pd.concat(out, ignore_index=True)
    q["sue"] = q["sue"].replace([np.inf, -np.inf], np.nan)
    return q.dropna(subset=["event_date", "sue"]).reset_index(drop=True)


# ─── EAR ─────────────────────────────────────────────────
def add_ear(q, P):
    dates = P["dates"]
    ret = P["ret"].values
    smean = P["sec_mean"]
    col_ix = {t: j for j, t in enumerate(P["cols"])}
    sec = P["sector"]
    sm_ix = {s: k for k, s in enumerate(smean.columns)}
    sm = smean.values
    pos = dates.searchsorted(pd.to_datetime(q["event_date"]).values)   # 이벤트일 이후 첫 거래일
    ear, t0 = np.full(len(q), np.nan), np.full(len(q), -1)
    for k, (tk, p) in enumerate(zip(q["ticker"].values, pos)):
        j = col_ix.get(tk)
        if j is None or p < 1 or p + 2 > len(dates) - 1:
            continue
        r = ret[p - 1:p + 2, j]
        m = sm[p - 1:p + 2, sm_ix[sec.iloc[j]]]
        if np.isfinite(r).sum() >= 2:
            ear[k] = np.nansum(r - m)
            t0[k] = p
    q = q.copy()
    q["ear"] = ear
    q["t0_idx"] = t0
    q["t1_idx"] = np.where(t0 >= 0, t0 + 1, -1)            # t+1 종가에 신호 확정 → t+2 시가 매수
    q = q[q.t1_idx >= 0].reset_index(drop=True)
    q["known_date"] = dates[q.t1_idx.values]
    q["sector"] = q["ticker"].map(lambda t: sec.get(t, "NA"))
    return q


# ─── PIT 백분위 ──────────────────────────────────────────
def _pit_pct(df, col):
    """각 이벤트를 '신호 확정일 직전 1년간' 같은 섹터 이벤트들과 비교한 백분위 (미래 정보 없음)"""
    res = np.full(len(df), np.nan)
    kd = df["known_date"].values.astype("datetime64[D]").astype(np.int64)
    v = df[col].values
    lb = C.PCT_LOOKBACK_DAYS

    order_all = np.argsort(kd, kind="stable")
    kd_all, v_all = kd[order_all], v[order_all]

    for s, idx in df.groupby("sector").indices.items():
        idx = np.asarray(idx)
        o = idx[np.argsort(kd[idx], kind="stable")]
        kd_s, v_s = kd[o], v[o]
        for n, ii in enumerate(o):
            lo = np.searchsorted(kd_s, kd[ii] - lb, "left")
            hi = np.searchsorted(kd_s, kd[ii], "left")          # 같은 날 이전까지만
            if hi - lo >= C.PCT_MIN_PEERS:
                w = v_s[lo:hi]
            else:                                              # 섹터 표본 부족 → 전체 시장
                lo2 = np.searchsorted(kd_all, kd[ii] - lb, "left")
                hi2 = np.searchsorted(kd_all, kd[ii], "left")
                w = v_all[lo2:hi2]
            w = w[np.isfinite(w)]
            if len(w) >= C.PCT_MIN_PEERS and np.isfinite(v[ii]):
                res[ii] = (w <= v[ii]).mean()
    return res


def build_events(quarters, P):
    q = add_sue(quarters)
    q = add_ear(q, P)
    q = q.dropna(subset=["ear"]).reset_index(drop=True)
    q["pS"] = _pit_pct(q, "sue")
    q["pE"] = _pit_pct(q, "ear")
    q["score"] = 0.5 * q["pS"] + 0.5 * q["pE"]
    q["pScore"] = _pit_pct(q, "score")
    q["passed"] = (q["pScore"] >= 1 - C.TOP_PCT) & (q["sue"] > 0) & (q["ear"] > 0)
    return q


# ─── Stage 1 후보 (대회 시작일 i0 직전 종가 기준) ─────────
def stage1_candidates(P, ev, i0, year):
    i = i0 - 1
    q2_end = pd.Timestamp(year, 6, 30)
    q3_end = pd.Timestamp(year, 9, 30)
    known = ev[ev.t1_idx <= i]
    q2 = known[(known.pend == q2_end) & (known.pS >= 1 - C.S1_TOP_PCT) & (known.sue > 0)]
    already_q3 = set(known.loc[known.pend == q3_end, "ticker"])
    elig = eligible_mask(P, i)
    col_ix = {t: j for j, t in enumerate(P["cols"])}
    r21 = P["ret21"].values[i]
    sm = P["sec_mean"]
    sec_r21 = (1 + sm.iloc[max(0, i - 20):i + 1].fillna(0)).prod() - 1
    cands = []
    for _, e in q2.iterrows():
        j = col_ix.get(e.ticker)
        if j is None or e.ticker in already_q3 or not elig[j]:
            continue
        rs = r21[j] - sec_r21.get(P["sector"].iloc[j], 0.0)
        if np.isfinite(rs) and rs > 0:                      # 최근 1개월 섹터 대비 강세
            cands.append((e.ticker, float(e.pS) + 0.01 * rs))
    cands.sort(key=lambda x: -x[1])
    return cands
