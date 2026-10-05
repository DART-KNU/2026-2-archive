"""섹터 매핑·섹터 수익률, 시점별 매수 가능 종목과 파생 지표(모멘텀·변동성·ROE), 섹터 모멘텀."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ────────────────────────────── sectors ──────────────────────────────

GICS_ORDER = ["에너지", "소재", "산업", "임의소비재", "필수소비재", "헬스케어",
              "금융", "정보기술", "커뮤니케이션", "유틸리티", "부동산"]


def fg_to_gics_table(info_listed: pd.DataFrame, sector: pd.DataFrame):
    """상장 종목의 FnGuide 섹터 × 대회 GICS 교차표 → (교차표, 다수결 대응 dict)."""
    common = sector.index.intersection(info_listed.index)
    fg = info_listed.loc[common, "fg_code"]
    ok = fg.notna()
    ct = pd.crosstab(fg[ok], sector.loc[common[ok.to_numpy()], "sector"])
    mapping = ct.idxmax(axis=1).to_dict()
    return ct, mapping


def delisted_sector_monthly(sector_hist: pd.DataFrame, mapping: dict, sector: pd.DataFrame) -> pd.DataFrame:
    """상폐 종목 월별 GICS. 대회 분류표(sector)에 있는 종목은 그 분류가 우선."""
    out = sector_hist.apply(lambda s: s.map(mapping))
    override = [c for c in out.columns if c in sector.index]
    for c in override:
        out.loc[out[c].notna(), c] = sector.at[c, "sector"]
    return out


def sector_on(dates: pd.DatetimeIndex, static: pd.Series, monthly: pd.DataFrame) -> pd.DataFrame:
    """날짜별 종목 GICS (date × code). 상폐 종목은 해당 날짜 이전 가장 최근 월말 분류를 사용,
    첫 월말 이전 구간만 첫 관측값으로 채운다(분류 소급의 한계)."""
    m = monthly.reindex(monthly.index.union(dates)).ffill().bfill().reindex(dates)
    s = pd.DataFrame(np.repeat(static.to_numpy()[None, :], len(dates), axis=0), index=dates, columns=static.index)
    return pd.concat([s, m], axis=1)


def sector_returns(ret: pd.DataFrame, mcap: pd.DataFrame, sec: pd.DataFrame, elig: pd.DataFrame):
    """일간 섹터·시장 수익률 = 전일 시총 가중 평균. 유니버스 = 전일 매수 가능 종목.
    sec: 정수 섹터 코드(GICS_ORDER 인덱스, 없음=-1)."""
    r = ret.to_numpy()
    w = mcap.shift(1).where(elig.shift(1, fill_value=False)).to_numpy()
    s = sec.shift(1).fillna(-1).to_numpy()
    ok = np.isfinite(r) & np.isfinite(w)
    w = np.where(ok, w, 0.0)
    rw = np.where(ok, r * w, 0.0)
    out = {}
    for i, g in enumerate(GICS_ORDER):
        mk = s == i
        den = (w * mk).sum(1)
        out[g] = np.where(den > 0, (rw * mk).sum(1) / np.where(den > 0, den, 1), np.nan)
    den = w.sum(1)
    mkt = np.where(den > 0, rw.sum(1) / np.where(den > 0, den, 1), np.nan)
    return pd.DataFrame(out, index=ret.index), pd.Series(mkt, index=ret.index, name="market")


def market_sector_weights(mcap_row: pd.Series, sec_row: pd.Series) -> pd.Series:
    """시장 섹터 비중: 해당일 시총이 있는 전 종목(보통주, 스팩·ETF 제외) 기준."""
    ok = mcap_row.notna() & (sec_row >= 0)
    w = mcap_row[ok].groupby(sec_row[ok]).sum()
    w.index = [GICS_ORDER[i] for i in w.index]
    return (w / w.sum()).reindex(GICS_ORDER).fillna(0.0)


# ────────────────────────────── universe ──────────────────────────────


@dataclass
class Panels:
    price: pd.DataFrame
    mcap: pd.DataFrame
    value: pd.DataFrame
    fin: pd.DataFrame            # long: qend, code, ni, eq, avail
    stocks: pd.DataFrame
    sec_static: pd.Series        # 상장 종목 GICS
    sec_monthly: pd.DataFrame    # 상폐 종목 월별 GICS
    index: pd.DataFrame
    d: dict = field(default_factory=dict)   # 파생 패널


def load_panels(cfg) -> Panels:
    P = cfg["paths"]["processed"]
    stocks = pd.read_parquet(P / "stocks.parquet")
    price = pd.read_parquet(P / "price_adj.parquet")
    return Panels(
        price=price,
        mcap=pd.read_parquet(P / "mcap.parquet"),
        value=pd.read_parquet(P / "value.parquet"),
        fin=pd.read_parquet(P / "fin.parquet"),
        stocks=stocks,
        sec_static=stocks.loc[stocks.group == "listed", "gics"],
        sec_monthly=pd.read_parquet(P / "sector_monthly_delisted.parquet"),
        index=pd.read_parquet(P / "index.parquet"),
    )


def segment_age(price: pd.DataFrame) -> pd.DataFrame:
    """가격이 연속으로 존재한 거래일 수(상장일=1). 재상장 공백이 있으면 다시 1부터."""
    valid = price.notna().to_numpy()
    age = np.zeros(valid.shape, dtype=np.int32)
    run = np.zeros(valid.shape[1], dtype=np.int32)
    old = valid[0].copy()   # 데이터 시작일 이전부터 상장된 종목 → 첫 연속 구간은 충분히 오래된 것으로
    for i in range(valid.shape[0]):
        run = np.where(valid[i], run + 1, 0)
        old &= valid[i]
        age[i] = run + 1000 * old
    return pd.DataFrame(age, index=price.index, columns=price.columns)


def roe_panel(fin: pd.DataFrame, dates: pd.DatetimeIndex, codes, stale_days: int = 400) -> pd.DataFrame:
    """ROE = 최근 4개 분기 순이익 합(연속 4분기) ÷ 최신 분기 지배자본. 공시 가능일(avail) 이후부터 반영."""
    f = fin.sort_values(["code", "qend"]).copy()
    f["qn"] = f["qend"].dt.year * 4 + f["qend"].dt.month // 3
    g = f.groupby("code")
    ttm = g["ni"].rolling(4).sum().reset_index(level=0, drop=True)
    consec = (f["qn"] - g["qn"].shift(3)).eq(3)
    f["roe"] = np.where(consec & (f["eq"] > 0), ttm / f["eq"], np.nan)
    f = f.dropna(subset=["roe"])
    # 같은 공시 가능일에 여러 분기가 있으면 최신 분기
    f = f.sort_values(["code", "avail", "qend"]).drop_duplicates(["code", "avail"], keep="last")
    wide = f.pivot(index="avail", columns="code", values="roe")
    avail_d = f.pivot(index="avail", columns="code", values="avail")
    alld = wide.index.union(dates)
    roe = wide.reindex(alld).ffill().reindex(dates)
    last_avail = avail_d.reindex(alld).ffill().reindex(dates).to_numpy("datetime64[ns]")
    age_days = (dates.to_numpy()[:, None] - last_avail) / np.timedelta64(1, "D")
    roe = roe.where(age_days <= stale_days)
    return roe.reindex(columns=codes)


def prepare(pn: Panels, cfg) -> Panels:
    """파생 패널 계산 (모두 인과적)."""
    u, m = cfg["universe"], cfg["momentum"]
    price, mcap, value = pn.price, pn.mcap, pn.value
    dates, codes = price.index, price.columns
    d = pn.d
    d["ret"] = price.pct_change(fill_method=None)
    has_px = price.notna()
    v0 = value.where(has_px).fillna(0.0)
    d["halt"] = has_px & (v0 <= 0) if cfg["data"]["halt_if_zero_value"] else has_px & False
    d["value5"] = v0.where(has_px).rolling(5, min_periods=5).mean()
    d["age"] = segment_age(price)
    sec = sector_on(dates, pn.sec_static, pn.sec_monthly).reindex(columns=codes)
    sec = sec.where(has_px)
    code_of = {g: i for i, g in enumerate(GICS_ORDER)}
    d["sec"] = sec.apply(lambda s: s.map(code_of)).fillna(-1).astype(np.int8)
    # 매수 가능: 시총·거래대금·상장 경과·거래정지 아님·섹터 있음
    d["elig"] = (has_px & (mcap >= u["min_mcap"]) & (d["value5"] > u["min_value_5d"])
                 & (d["age"] > u["min_listed_days"]) & ~d["halt"] & (d["sec"] >= 0))
    lb, sk = m["stock_lookback"], m["stock_skip"]
    d["mom61"] = price.shift(sk) / price.shift(lb) - 1
    d["vol60"] = d["ret"].rolling(m["vol_window"], min_periods=int(m["vol_window"] * 0.67)).std()
    d["roe"] = roe_panel(pn.fin, dates, codes)
    return pn


# ────────────────────────────── momentum ──────────────────────────────


def _window(sr: pd.DataFrame, mkt: pd.Series, t, n: int):
    sr, mkt = sr.loc[:t], mkt.loc[:t]
    return sr.iloc[-n:], mkt.iloc[-n:]


def residual_momentum(sr: pd.DataFrame, mkt: pd.Series, t, beta_window=252, lookback=126, skip=21) -> pd.Series:
    """최근 beta_window일 회귀로 β → 잔차 → [t−lookback, t−skip] 잔차 합 ÷ 같은 구간 잔차 표준편차."""
    S, M = _window(sr, mkt, t, beta_window)
    out = {}
    for g in S.columns:
        y = S[g]
        ok = y.notna() & M.notna()
        if ok.sum() < beta_window * 0.8:
            out[g] = np.nan
            continue
        x, yy = M[ok].to_numpy(), y[ok].to_numpy()
        b = np.cov(x, yy, ddof=1)[0, 1] / np.var(x, ddof=1)
        a = yy.mean() - b * x.mean()
        e = (y - a - b * M)
        win = e.iloc[-lookback:len(e) - skip].dropna()
        if len(win) < (lookback - skip) * 0.8 or win.std() == 0:
            out[g] = np.nan
            continue
        out[g] = win.sum() / win.std()
    return pd.Series(out)


def relative_61(sr, mkt, t, lookback=126, skip=21, **_):
    S, M = _window(sr, mkt, t, lookback)
    S, M = S.iloc[: len(S) - skip], M.iloc[: len(M) - skip]
    return (1 + S).prod(min_count=int(len(S) * 0.8)) - 1 - ((1 + M).prod() - 1)


def slope_r2(sr, mkt, t, lookback=126, skip=21, **_):
    S, M = _window(sr, mkt, t, lookback)
    S, M = S.iloc[: len(S) - skip], M.iloc[: len(M) - skip]
    out = {}
    for g in S.columns:
        rel = np.log1p(S[g]) - np.log1p(M)
        rel = rel.dropna()
        if len(rel) < len(S) * 0.8:
            out[g] = np.nan
            continue
        y = rel.cumsum().to_numpy()
        x = np.arange(len(y))
        b, a = np.polyfit(x, y, 1)
        r2 = np.corrcoef(x, y)[0, 1] ** 2
        out[g] = b * 252 * r2
    return pd.Series(out)


METHODS = {"residual": residual_momentum, "rel61": relative_61, "slope_r2": slope_r2}


def sector_momentum(sr, mkt, t, cfg) -> pd.Series:
    m = cfg["momentum"]
    f = METHODS[m["method"]]
    return f(sr, mkt, t, beta_window=m["beta_window"], lookback=m["lookback"], skip=m["skip"])
