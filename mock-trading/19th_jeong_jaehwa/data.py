"""Data download (pykrx) with on-disk caching. Every step is resumable: re-running skips cached files.

Sources
- KOSPI index and weekly market-cap snapshots (all listed stocks): KRX (login required, see README)
- KRX industry classification per stock, semi-annual snapshots: KRX
- Adjusted daily closes per stock: pykrx get_market_ohlcv(adjusted=True)

Sector returns are built from the weekly snapshots as market-cap-weighted returns of member stocks,
so no sector-index constituent history is needed.
"""
import re
import time
from datetime import datetime

import pandas as pd

from config import DATA_DIR, DATA_START, DATA_END, KOSPI_CODE, MEMBERSHIP_MONTHS, BASE

SLEEP = 0.5   # be polite to the KRX server


def _stock():
    from pykrx import stock   # imported lazily so the engine can be tested without pykrx
    return stock


def _end():
    return DATA_END or datetime.today().strftime("%Y%m%d")


def _retry(fn, *args, tries=2, **kw):
    for k in range(tries):
        try:
            out = fn(*args, **kw)
            time.sleep(SLEEP)
            return out
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                print(f"   ! failed {fn.__name__}{args}: {e}")
                return None
            time.sleep(3)


# ---------------------------------------------------------------- calendar / KOSPI
def load_kospi(refresh=False):
    f = DATA_DIR / "kospi.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f, index_col=0, parse_dates=True)["close"]
    df = _retry(_stock().get_index_ohlcv, DATA_START, _end(), KOSPI_CODE)
    if df is None or df.empty or "종가" not in df.columns:
        if f.exists():
            print("   ! KOSPI refresh failed, using cached file")
            return pd.read_csv(f, index_col=0, parse_dates=True)["close"]
        raise RuntimeError("Could not download the KOSPI index. Check the KRX login (00_check_data.py).")
    s = df["종가"].astype(float).rename("close")
    s.index = pd.to_datetime(s.index)
    DATA_DIR.mkdir(exist_ok=True)
    s.to_frame().to_csv(f)
    return s


def weekly_dates(days):
    """Last trading day of each calendar week."""
    d = pd.Series(days, index=days)
    return pd.DatetimeIndex(d.groupby(d.index.to_period("W-SUN")).max().values)


# ---------------------------------------------------------------- weekly market-cap snapshots
def download_snapshots(wdates):
    out = DATA_DIR / "snap"; out.mkdir(parents=True, exist_ok=True)
    todo = [d for d in wdates if not (out / f"{d:%Y%m%d}.csv").exists()]
    print(f"[snapshots] {len(wdates)} weeks, {len(todo)} to download")
    for n, d in enumerate(todo, 1):
        df = _retry(_stock().get_market_cap, d.strftime("%Y%m%d"), market="ALL")
        if df is None or df.empty or "시가총액" not in df.columns:
            print(f"   ! snapshot {d:%Y-%m-%d} failed - will retry on the next run")
            continue
        df = df[["종가", "시가총액", "거래대금"]].rename(columns={"종가": "close", "시가총액": "cap", "거래대금": "value"})
        df.index.name = "ticker"
        df.to_csv(out / f"{d:%Y%m%d}.csv")
        if n % 25 == 0:
            print(f"   {n}/{len(todo)}  {d:%Y-%m-%d}")


def load_snapshots(wdates, with_close=False):
    """Returns (cap, value[, close]): DataFrames weekly date x ticker."""
    caps, vals, closes = {}, {}, {}
    for d in wdates:
        f = DATA_DIR / "snap" / f"{d:%Y%m%d}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, dtype={"ticker": str}).set_index("ticker")
        caps[d] = df["cap"]; vals[d] = df["value"]; closes[d] = df["close"]
    out = (pd.DataFrame(caps).T.sort_index(), pd.DataFrame(vals).T.sort_index())
    return out + (pd.DataFrame(closes).T.sort_index(),) if with_close else out


# ---------------------------------------------------------------- industry classification
# KRX renamed several KOSPI industries; map old names to current ones so a sector keeps one history.
ALIASES = {
    "음식료품": "음식료·담배", "섬유의복": "섬유·의류", "종이목재": "종이·목재", "의약품": "제약",
    "비금속광물": "비금속", "철강금속": "금속", "철강및금속": "금속", "기계": "기계·장비",
    "의료정밀": "의료·정밀기기", "운수장비": "운송장비·부품", "유통업": "유통", "전기가스업": "전기·가스",
    "건설업": "건설", "운수창고업": "운송·창고", "운수창고": "운송·창고", "통신업": "통신",
    "금융업": "금융", "은행": "금융", "서비스업": "일반서비스", "제조업": "제조",
}


def _norm(name):
    name = str(name).strip()
    return ALIASES.get(name, ALIASES.get(re.sub(r"[\s·]", "", name), name))


def membership_dates(days):
    d = pd.Series(days, index=days)
    first = d.groupby(d.index.to_period("M")).min()
    return pd.DatetimeIndex([x for x in first.values if pd.Timestamp(x).month in MEMBERSHIP_MONTHS])


def download_membership(mdates):
    out = DATA_DIR / "sector_class"; out.mkdir(parents=True, exist_ok=True)
    st = _stock()
    for d in mdates:
        f = out / f"{d:%Y%m%d}.csv"
        if f.exists():
            continue
        parts, ok = [], True
        for market in ("KOSPI", "KOSDAQ"):
            df = _retry(st.get_market_sector_classifications, d.strftime("%Y%m%d"), market)
            if df is None or df.empty or "업종명" not in df.columns:
                ok = False
                break
            parts.append(pd.DataFrame({"ticker": df.index.astype(str), "industry": df["업종명"].values,
                                       "market": market}))
        if not ok:
            print(f"   ! classification {d:%Y-%m-%d} failed - will retry on the next run")
            continue
        pd.concat(parts).to_csv(f, index=False)
        print(f"[classification] {d:%Y-%m-%d}: {sum(len(p) for p in parts)} stocks")


# ---- fallback classification (no KRX login): KSIC industries mapped to GICS-like industry groups
FDR_CACHE = "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/listing"

# (keyword in KSIC or KRX industry name, group). First match wins, so specific keywords come first.
GROUP_RULES = [
    ("반도체", "Semiconductors"), ("특수 목적용 기계", "Semiconductors"),
    ("전자부품", "Tech Hardware"), ("통신 및 방송 장비", "Tech Hardware"), ("영상 및 음향", "Tech Hardware"),
    ("컴퓨터 및 주변", "Tech Hardware"), ("측정, 시험", "Tech Hardware"), ("전기·전자", "Tech Hardware"),
    ("전지", "Electrical Equipment"), ("전동기", "Electrical Equipment"), ("절연선", "Electrical Equipment"),
    ("전기장비", "Electrical Equipment"),
    ("선박", "Capital Goods"), ("항공기", "Capital Goods"), ("무기", "Capital Goods"), ("일반 목적용 기계", "Capital Goods"),
    ("구조용 금속", "Capital Goods"), ("기타 운송장비", "Capital Goods"), ("기계장비 및 관련", "Capital Goods"),
    ("기계·장비", "Capital Goods"), ("운송장비·부품", "Capital Goods"), ("상품 종합 도매", "Capital Goods"),
    ("자동차 판매", "Retail"), ("자동차", "Autos"), ("고무제품", "Autos"),
    ("건설", "Construction"), ("건축기술", "Construction"), ("공사업", "Construction"),
    ("석유 정제", "Energy"),
    ("의료용 기기", "Health Care Equipment"), ("의료·정밀", "Health Care Equipment"),
    ("의약", "Pharma & Biotech"), ("제약", "Pharma & Biotech"), ("연구개발", "Pharma & Biotech"),
    ("화학", "Chemicals"), ("플라스틱", "Chemicals"), ("합성고무", "Chemicals"),
    ("철강", "Metals & Materials"), ("비철금속", "Metals & Materials"), ("금속", "Metals & Materials"),
    ("시멘트", "Metals & Materials"), ("유리", "Metals & Materials"), ("비금속", "Metals & Materials"),
    ("종이", "Metals & Materials"), ("골판지", "Metals & Materials"), ("나무제품", "Metals & Materials"),
    ("소프트웨어", "Software & Internet"), ("포털", "Software & Internet"), ("컴퓨터 프로그래밍", "Software & Internet"),
    ("정보 서비스", "Software & Internet"), ("IT 서비스", "Software & Internet"),
    ("영화", "Media & Entertainment"), ("오디오물", "Media & Entertainment"), ("방송", "Media & Entertainment"),
    ("광고", "Media & Entertainment"), ("출판", "Media & Entertainment"), ("오락", "Media & Entertainment"),
    ("숙박", "Consumer Services"), ("여행", "Consumer Services"), ("교습", "Consumer Services"),
    ("전기 통신", "Telecom"), ("통신서비스", "Telecom"), ("통신", "Telecom"),
    ("전기업", "Utilities"), ("가스", "Utilities"),
    ("은행", "Banks"), ("보험", "Insurance"),
    ("금융 지원", "Securities"), ("상품 중개", "Securities"), ("신탁업", "Securities"), ("증권", "Securities"),
    ("기타 금융", "Holdings & Other Financials"), ("기타금융", "Holdings & Other Financials"),
    ("회사 본부", "Holdings & Other Financials"), ("금융", "Holdings & Other Financials"),
    ("운송", "Transportation"), ("운수", "Transportation"),
    ("식품", "Food & Beverage"), ("음료", "Food & Beverage"), ("담배", "Food & Beverage"), ("음식료", "Food & Beverage"),
    ("유지", "Food & Beverage"), ("사료", "Food & Beverage"),
    ("의복", "Consumer Durables & Apparel"), ("신발", "Consumer Durables & Apparel"), ("가구", "Consumer Durables & Apparel"),
    ("가정용 기기", "Consumer Durables & Apparel"), ("섬유", "Consumer Durables & Apparel"),
    ("소매", "Retail"), ("도매", "Retail"), ("유통", "Retail"),
    ("부동산", "Real Estate"),
]


BANK_HOLDINGS = ("KB금융", "신한지주", "하나금융지주", "우리금융지주", "BNK금융지주", "JB금융지주", "DGB금융지주", "iM금융지주")


def industry_group(name, company=""):
    if company in BANK_HOLDINGS:
        return "Banks"
    name = str(name)
    for key, group in GROUP_RULES:
        if key in name:
            return group
    return "Other"


def download_static_classification():
    """Fallback without KRX login: KSIC industry of listed stocks and KRX industry of delisted stocks,
    from the FinanceDataReader KRX cache on GitHub, mapped to GICS-like industry groups."""
    import io
    import urllib.request
    f = DATA_DIR / "sector_static.csv"
    if f.exists():
        return True
    frames = {}
    for kind in ("desc", "delisting"):
        for back in range(0, 30):
            day = (pd.Timestamp.today() - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
            try:
                raw = urllib.request.urlopen(f"{FDR_CACHE}/{kind}/{day}.csv", timeout=30).read()
                frames[kind] = pd.read_csv(io.BytesIO(raw), dtype=str, index_col=0)
                print(f"[classification] {kind} list from {day}: {len(frames[kind])} rows")
                break
            except Exception:  # noqa: BLE001
                continue
    if "desc" not in frames:
        print("   ! could not download the classification cache from GitHub")
        return False
    d = frames["desc"]
    parts = [pd.DataFrame({"ticker": d["Code"].str.zfill(6), "name": d["Name"], "industry": d["Industry"]})]
    if "delisting" in frames:
        dl = frames["delisting"]
        parts.append(pd.DataFrame({"ticker": dl["Symbol"].str.zfill(6), "name": dl["Name"], "industry": dl["Industry"]}))
    df = pd.concat(parts).dropna(subset=["industry"]).drop_duplicates("ticker")
    df = df[~df["industry"].str.strip().isin(["", "-"])]
    df["sector"] = [industry_group(i, n) for i, n in zip(df["industry"], df["name"])]
    df.to_csv(f, index=False)
    print(f"[classification] {len(df)} stocks in {df['sector'].nunique()} industry groups")
    return True


def load_membership():
    """date -> DataFrame[ticker, sector].
    Preferred: KRX industry snapshots, sector = 'MARKET:industry' (renamed industries merged).
    Fallback: one static classification (sector_static.csv, GICS-like groups), applied to the whole period."""
    snaps = {}
    for f in sorted((DATA_DIR / "sector_class").glob("*.csv")):
        d = pd.Timestamp(datetime.strptime(f.stem, "%Y%m%d"))
        df = pd.read_csv(f, dtype={"ticker": str})
        df["sector"] = df["market"] + ":" + df["industry"].map(_norm)
        snaps[d] = df[["ticker", "sector"]]
    if snaps:
        return snaps
    f = DATA_DIR / "sector_static.csv"
    if f.exists():
        df = pd.read_csv(f, dtype={"ticker": str})
        df = df[df["sector"] != "Other"]
        return {pd.Timestamp("1990-01-01"): df[["ticker", "sector"]]}
    raise RuntimeError("No industry classification downloaded yet. Run 01_download_data.py.")


# ---------------------------------------------------------------- sector returns from snapshots
def build_sector_prices(cap, snaps, days, close=None, min_members=3, clip=(-0.6, 1.5)):
    """Market-cap-weighted weekly sector returns from the snapshots, as a daily (forward-filled) level series.
    A stock's weekly return is the price change, except when price and market-cap changes disagree by more
    than 5%p: then the smaller of the two is used (splits move the price, share issuance and mergers move
    the market cap, neither is a return)."""
    r = cap / cap.shift(1) - 1
    if close is not None:
        rp = (close / close.shift(1) - 1).reindex_like(r)
        r = rp.where((rp - r).abs() <= 0.05, rp.where(rp.abs() < r.abs(), r)).where(rp.notna(), r)
    r = r.clip(*clip)
    w = cap.shift(1)
    mdates = sorted(snaps)
    rows = {}
    for t in cap.index[1:]:
        prior = [d for d in mdates if d <= cap.index[cap.index.get_loc(t) - 1]]
        m = snaps[prior[-1] if prior else mdates[0]]
        g = pd.DataFrame({"r": r.loc[t], "w": w.loc[t]}).dropna()
        g = g.join(m.set_index("ticker")["sector"], how="inner")
        g = g[g["w"] > 0]
        cnt = g.groupby("sector")["r"].count()
        ret = (g["r"] * g["w"]).groupby(g["sector"]).sum() / g.groupby("sector")["w"].sum()
        rows[t] = ret[cnt >= min_members]
    ret = pd.DataFrame(rows).T.sort_index()
    started = ret.notna().cummax()                      # NaN before a sector first has enough members
    level = (1 + ret.fillna(0)).cumprod().where(started)
    return level.reindex(days).ffill()


# ---------------------------------------------------------------- adjusted stock prices
def candidate_tickers(cap, val, snaps=None, p=BASE):
    """Tickers that ever pass the cap / trading-value filter and have an industry classification."""
    avg_val = val.rolling(4, min_periods=1).mean()
    ok = (cap >= p.min_cap) & (avg_val >= p.min_value)
    ever = set(ok.columns[ok.any()])
    if snaps:
        ever &= set(pd.concat(snaps.values())["ticker"])
    return sorted(ever)


def download_prices(tickers, update=False):
    """Adjusted daily closes. update=True re-downloads names that are still trading."""
    out = DATA_DIR / "px"; out.mkdir(parents=True, exist_ok=True)
    todo = list(tickers) if update else [t for t in tickers if not (out / f"{t}.csv").exists()]
    print(f"[prices] {len(tickers)} tickers, {len(todo)} to download")
    for n, t in enumerate(todo, 1):
        f = out / f"{t}.csv"
        if update and f.exists():
            old = pd.read_csv(f, index_col=0, parse_dates=True)
            if len(old) == 0 or old.index[-1] < pd.Timestamp(_end()) - pd.Timedelta(days=30):
                continue   # delisted: nothing new to fetch
        df = _retry(_stock().get_market_ohlcv, DATA_START, _end(), t, adjusted=True)
        if df is None or df.empty or "종가" not in df.columns:
            if not f.exists():
                pd.DataFrame(columns=["close"]).to_csv(f)   # mark as tried
            continue
        df[["종가"]].rename(columns={"종가": "close"}).to_csv(f)
        if n % 50 == 0:
            print(f"   {n}/{len(todo)}")


def load_prices(tickers):
    series = {}
    for t in tickers:
        f = DATA_DIR / "px" / f"{t}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, index_col=0, parse_dates=True)["close"]
        s = s[s > 0]
        if len(s):
            series[t] = s
    return pd.DataFrame(series).sort_index()


def ticker_names(tickers):
    st = _stock()
    return {t: st.get_market_ticker_name(t) for t in tickers}
