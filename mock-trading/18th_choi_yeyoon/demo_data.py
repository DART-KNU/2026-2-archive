"""
가짜(합성) 데이터 생성기 — 코드가 끝까지 돌아가는지 확인하는 용도.
실제 데이터와 같은 형식(panels, quarters, k200, sectors)을 만들고,
실적 서프라이즈 이후 약한 드리프트를 일부러 심어 둡니다.
※ 여기서 나온 성과 숫자는 의미 없음. 발표에는 반드시 실제 데이터 결과를 쓰세요.
"""
import numpy as np
import pandas as pd


def make_demo(seed=7, n=350, start="2015-06-01", end="2025-12-31"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    tickers = [f"{100000 + (k + 1) * 10:06d}" for k in range(n)]
    tickers[0], tickers[1] = "005930", "000660"
    sec_names = ["반도체", "전기장비", "자동차", "화학", "은행", "소프트웨어", "제약", "화장품", "건설", "조선"]
    sectors = pd.Series(rng.choice(sec_names, n), index=tickers, name="sector")
    sectors["005930"] = sectors["000660"] = "반도체"

    mkt = rng.normal(0.0004, 0.011, T)
    sec_f = {s: rng.normal(0, 0.006, T) for s in sec_names}
    ret = np.empty((T, n))
    for j, t in enumerate(tickers):
        ret[:, j] = mkt * rng.uniform(0.7, 1.3) + sec_f[sectors[t]] + rng.normal(0, rng.uniform(0.012, 0.028), T)

    # 분기 실적 & 이벤트 (이벤트 후 드리프트 주입)
    qends = pd.date_range("2013-03-31", end, freq="QE-DEC")
    rows = []
    for j, t in enumerate(tickers):
        L = rng.lognormal(24, 1.0)
        season = rng.normal(0, 0.15, 4)
        ops = []
        for k, qe in enumerate(qends):
            s = rng.normal()
            op = L * (1 + season[qe.quarter - 1]) if k < 4 else ops[k - 4] + L * 0.1 * s
            ops.append(op)
            ev = qe + pd.Timedelta(days=int(rng.integers(20, 46)))
            rows.append((t, qe, op, ev, s if k >= 4 else 0.0))
            p = dates.searchsorted(ev)
            if k >= 4 and p + 22 < T:
                ret[p, j] += 0.015 * s                       # 발표일 반응
                ret[p + 1:p + 21, j] += 0.0006 * s           # 이후 드리프트 (PEAD)
    q = pd.DataFrame(rows, columns=["ticker", "pend", "op", "event_date", "_s"])
    q["avail"] = q["event_date"]
    q["prelim"] = pd.NaT
    q = q.drop(columns="_s")

    close = 10000 * np.cumprod(1 + ret, axis=0)
    open_ = np.vstack([close[:1], close[:-1]]) * (1 + rng.normal(0, 0.004, (T, n)))
    shares = rng.lognormal(17.5, 1.0, n)
    shares[0], shares[1] = 6e9, 7e8
    mcap = close * shares
    tv = mcap * rng.lognormal(np.log(0.03), 0.5, (T, n))

    df = lambda a: pd.DataFrame(a, index=dates, columns=tickers)
    panels = dict(open=df(open_), close=df(close), ret=df(ret), mcap=df(mcap), tv=df(tv))
    k200 = pd.Series(100 * np.cumprod(1 + mkt), index=dates, name="KOSPI200")
    return panels, q, k200, sectors
