"""
데이터 수집 & 가공
  - 가격/시총/거래대금 : pykrx (KRX)
  - 섹터               : pykrx KRX 업종 (data/sectors.csv 가 있으면 그걸 우선 사용 → WICS 등)
  - 실적/공시일        : OpenDART API

모든 결과는 data/ 폴더에 캐시되므로, 중간에 끊겨도 다시 실행하면 이어서 받습니다.
"""
import io
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

import config as C


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _p(*names):
    return os.path.join(C.DATA_DIR, *names)


# ════════════════════════════════════════════════════════════
# 1. 가격 (pykrx)
# ════════════════════════════════════════════════════════════
def trading_days(start, end):
    from pykrx import stock
    idx = stock.get_index_ohlcv_by_date(start.replace("-", ""), end.replace("-", ""), "1001")
    return [d.strftime("%Y%m%d") for d in idx.index]


def download_daily():
    """하루씩 전 종목 시가/종가/등락률/시총/거래대금을 받아 data/daily/YYYYMMDD.pkl 로 저장"""
    from pykrx import stock
    d_dir = _ensure(_p("daily"))
    days = trading_days(C.PRICE_START, C.END)
    todo = [d for d in days if not os.path.exists(os.path.join(d_dir, f"{d}.pkl"))]
    print(f"[price] 거래일 {len(days)}일 중 {len(todo)}일 다운로드 필요")
    if not todo:
        return 0
    for k, d in enumerate(todo):
        for attempt in range(3):
            try:
                o = stock.get_market_ohlcv(d, market="ALL")
                m = stock.get_market_cap(d, market="ALL")
                break
            except Exception as e:  # 네트워크 오류 등
                print(f"  retry {d}: {e}")
                time.sleep(3)
        else:
            print(f"  [skip] {d}")
            continue
        df = pd.DataFrame({
            "open": o["시가"],
            "close": o["종가"],
            "ret": o["등락률"] / 100.0,      # KRX 등락률 = 기준가 대비 → 액면분할 등 자동 보정
        }).join(m[["시가총액", "거래대금"]].rename(columns={"시가총액": "mcap", "거래대금": "tv"}), how="left")
        df.index.name = "ticker"
        df.to_pickle(os.path.join(d_dir, f"{d}.pkl"))
        time.sleep(C.PYKRX_SLEEP)
        if k % 50 == 0:
            print(f"  {d} ({k + 1}/{len(todo)})")
    return len(todo)


def build_panels():
    """일별 파일 → (날짜 × 종목) 패널. 보통주(코드 끝자리 0)만 남김"""
    d_dir = _p("daily")
    files = sorted(f for f in os.listdir(d_dir) if f.endswith(".pkl"))
    frames = []
    for f in files:
        df = pd.read_pickle(os.path.join(d_dir, f))
        df = df[df.index.astype(str).str.endswith("0")]
        df["date"] = pd.Timestamp(f[:8])
        frames.append(df.reset_index())
    long = pd.concat(frames, ignore_index=True)
    panels = {}
    for col in ["open", "close", "ret", "mcap", "tv"]:
        panels[col] = long.pivot(index="date", columns="ticker", values=col).sort_index().astype(float)
    pd.to_pickle(panels, _p("panels.pkl"))
    print(f"[price] 패널 저장: {panels['close'].shape}")
    return panels


def download_index():
    from pykrx import stock
    k = stock.get_index_ohlcv_by_date(C.PRICE_START.replace("-", ""), C.END.replace("-", ""), "1028")["종가"]
    k.index = pd.to_datetime(k.index)
    k.name = "KOSPI200"
    k.to_pickle(_p("k200.pkl"))
    return k


def download_sectors():
    """KRX 업종 분류. data/sectors.csv(ticker,sector)가 있으면 그걸 사용 (WICS 권장)"""
    csv = _p("sectors.csv")
    if os.path.exists(csv):
        s = pd.read_csv(csv, dtype={"ticker": str}).set_index("ticker")["sector"]
        print(f"[sector] sectors.csv 사용 ({len(s)}종목)")
    else:
        from pykrx import stock
        d = trading_days("2026-09-01", C.END)[-1]
        parts = []
        for mk in ["KOSPI", "KOSDAQ"]:
            try:
                df = stock.get_market_sector_classifications(d, mk)
                parts.append(df["업종명"])
            except Exception as e:
                print(f"[sector] {mk} 실패: {e}")
        s = pd.concat(parts) if parts else pd.Series(dtype=str)
        s.index = s.index.astype(str)
        print(f"[sector] KRX 업종 사용 ({len(s)}종목)")
    s.name = "sector"
    s.to_pickle(_p("sectors.pkl"))
    return s


# ════════════════════════════════════════════════════════════
# 2. OpenDART
# ════════════════════════════════════════════════════════════
DART = "https://opendart.fss.or.kr/api"


class DartHTMLError(RuntimeError):
    """DART가 JSON 대신 HTML 페이지를 돌려줌 (요청이 너무 길거나 일시 차단 등)"""


def _dart(path, params):
    if not C.DART_API_KEY:
        raise RuntimeError("DART_API_KEY 환경변수가 비어 있어요. OpenDART에서 인증키를 발급받아 넣어주세요.")
    params = dict(params, crtfc_key=C.DART_API_KEY)
    for attempt in range(5):
        try:
            r = requests.get(f"{DART}/{path}", params=params, timeout=30)
            time.sleep(C.DART_SLEEP)
            if not path.endswith(".json"):
                return r.content
            try:
                j = r.json()
            except ValueError:
                if attempt >= 1:
                    raise DartHTMLError(f"HTML 응답 (HTTP {r.status_code})")
                time.sleep(3)
                continue
            st = j.get("status")
            if st in ("000", "013"):          # 000 정상, 013 데이터 없음
                return j
            if st == "020":                    # 요청 한도 초과
                print("  [DART] 한도 초과 → 60초 대기")
                time.sleep(60)
                continue
            raise RuntimeError(f"DART {st}: {j.get('message')}")
        except requests.RequestException as e:
            print(f"  [DART] 네트워크 오류 재시도: {type(e).__name__}")
            time.sleep(3)
    raise RuntimeError(f"DART 요청 실패: {path}")


def corp_codes():
    fp = _p("corp_codes.pkl")
    if os.path.exists(fp):
        return pd.read_pickle(fp)
    content = _dart("corpCode.xml", {})
    z = zipfile.ZipFile(io.BytesIO(content))
    root = ET.fromstring(z.read(z.namelist()[0]))
    rows = [((e.findtext("corp_code") or "").strip(),
             (e.findtext("stock_code") or "").strip(),
             (e.findtext("corp_name") or "").strip()) for e in root.iter("list")]
    df = pd.DataFrame(rows, columns=["corp_code", "stock_code", "corp_name"])
    df = df[df.stock_code != ""].drop_duplicates("stock_code")
    df.to_pickle(fp)
    return df


def _list_all(corp_code, ty, bgn):
    items, page = [], 1
    while True:
        j = _dart("list.json", dict(corp_code=corp_code, bgn_de=bgn, end_de=C.END.replace("-", ""),
                                    pblntf_ty=ty, page_no=page, page_count=100))
        items += j.get("list", []) or []
        total = int(j.get("total_page", 1) or 1)
        if page >= total:
            return items
        page += 1


def download_disclosures(stock_codes):
    """종목별 정기공시(A)와 거래소공시(I, 잠정실적) 목록. data/disclosures.pkl 에 이어받기 저장"""
    fp = _p("disclosures.pkl")
    store = pd.read_pickle(fp) if os.path.exists(fp) else {}
    cc = corp_codes().set_index("stock_code")["corp_code"]
    todo = [s for s in stock_codes if s in cc.index and s not in store]
    print(f"[DART] 공시 목록: {len(todo)}개 종목 다운로드 필요")
    bgn = f"{C.FIN_START_YEAR}0101"
    for k, sc in enumerate(todo):
        a = _list_all(cc[sc], "A", bgn)
        i = _list_all(cc[sc], "I", bgn)
        store[sc] = {
            "A": [(x["report_nm"], x["rcept_dt"]) for x in a],
            "I": [(x["report_nm"], x["rcept_dt"]) for x in i if "영업(잠정)실적" in x["report_nm"]],
        }
        if k % 25 == 0:
            pd.to_pickle(store, fp)
            print(f"  {k + 1}/{len(todo)}")
    pd.to_pickle(store, fp)
    return store


def download_financials(stock_codes):
    """주요계정(fnlttMultiAcnt)에서 영업이익. 연도×보고서별 캐시"""
    f_dir = _ensure(_p("fin"))
    cc = corp_codes().set_index("stock_code")["corp_code"]
    codes = [cc[s] for s in stock_codes if s in cc.index]
    last_year = int(C.END[:4])
    for yr in range(C.FIN_START_YEAR, last_year + 1):
        for rc, q in [("11013", 1), ("11012", 2), ("11014", 3), ("11011", 4)]:
            fp = os.path.join(f_dir, f"{yr}_{rc}.pkl")
            if os.path.exists(fp) and yr < last_year:
                continue
            rows, failed = [], 0

            def fetch(chunk):
                """묶음 요청이 HTML로 거절되면 절반씩 쪼개서 다시 요청"""
                nonlocal failed
                try:
                    return _dart("fnlttMultiAcnt.json", dict(corp_code=",".join(chunk),
                                                            bsns_year=str(yr), reprt_code=rc)).get("list", []) or []
                except DartHTMLError:
                    if len(chunk) > 5:
                        mid = len(chunk) // 2
                        return fetch(chunk[:mid]) + fetch(chunk[mid:])
                    failed += 1
                    return []
                except RuntimeError as e:
                    failed += 1
                    print(f"  [fin] {yr} Q{q} 요청 실패 ({e})")
                    return []

            for s in range(0, len(codes), C.DART_FIN_CHUNK):
                for it in fetch(codes[s:s + C.DART_FIN_CHUNK]):
                    nm = (it.get("account_nm") or "").replace(" ", "")
                    if it.get("sj_div") == "IS" and nm.startswith("영업이익"):
                        amt = str(it.get("thstrm_amount", "")).replace(",", "").strip()
                        try:
                            val = float(amt)
                        except ValueError:
                            continue
                        rows.append((it.get("stock_code"), yr, q, it.get("fs_div"), val))
            df = pd.DataFrame(rows, columns=["stock_code", "year", "q", "fs_div", "amount"])
            if failed and not rows:
                raise RuntimeError("재무 API가 전부 HTML로 거절됨 → 요청 자체가 막힌 상태예요. "
                                   "잠시(30분~) 뒤 다시 실행하거나 진단 코드를 돌려주세요.")
            if failed:
                df.to_pickle(fp + ".partial")          # 실패 묶음이 있으면 정식 캐시로 저장하지 않음 → 재실행 시 다시 받음
                print(f"  [fin] {yr} Q{q}: {len(rows)}건 (실패 묶음 {failed}개, 다음 실행 때 재시도)")
            else:
                df.to_pickle(fp)
                print(f"  [fin] {yr} Q{q}: {len(rows)}건")


PERIODIC_RE = re.compile(r"(분기|반기|사업)보고서\s*\((\d{4})\.(\d{2})\)")


def build_quarters():
    """분기 영업이익 + 공시일 → quarters.pkl (ticker, pend, op, avail, prelim, event_date)"""
    # 영업이익: 1~3분기는 3개월 값, 4분기 = 연간 - (1+2+3분기)
    files = os.listdir(_p("fin"))
    full = {f for f in files if f.endswith(".pkl")}
    use = sorted(full) + sorted(f for f in files if f.endswith(".pkl.partial") and f[:-8] not in full)
    parts = [pd.read_pickle(os.path.join(_p("fin"), f)) for f in use]
    fin = pd.concat([x for x in parts if len(x)], ignore_index=True)
    fin["pri"] = (fin.fs_div != "CFS").astype(int)          # 연결(CFS) 우선
    fin = fin.sort_values("pri").drop_duplicates(["stock_code", "year", "q"])
    piv = fin.pivot_table(index=["stock_code", "year"], columns="q", values="amount", aggfunc="first")
    rows = []
    for (sc, yr), r in piv.iterrows():
        vals = {q: r.get(q) for q in (1, 2, 3)}
        if all(pd.notna([r.get(1), r.get(2), r.get(3), r.get(4)])):
            vals[4] = r.get(4) - r.get(1) - r.get(2) - r.get(3)
        for q, v in vals.items():
            if pd.notna(v):
                rows.append((sc, pd.Timestamp(yr, 3 * q, 1) + pd.offsets.MonthEnd(0), v))
    op = pd.DataFrame(rows, columns=["ticker", "pend", "op"])

    # 공시일
    store = pd.read_pickle(_p("disclosures.pkl"))
    per, pre, dec_fy = [], [], set()
    for sc, d in store.items():
        fy_months = []
        for nm, dt in d["A"]:
            if "정정" in nm:
                continue
            m = PERIODIC_RE.search(nm)
            if not m:
                continue
            kind, y, mo = m.group(1), int(m.group(2)), int(m.group(3))
            if kind == "사업":
                fy_months.append(mo)
            per.append((sc, pd.Timestamp(y, mo, 1) + pd.offsets.MonthEnd(0), pd.Timestamp(dt)))
        if fy_months and pd.Series(fy_months).mode().iloc[0] == 12:
            dec_fy.add(sc)                                   # 12월 결산법인만
        for nm, dt in d["I"]:
            t = pd.Timestamp(dt)
            qend = t - pd.offsets.QuarterEnd(startingMonth=12)   # 공시일 직전 분기말
            if (t - qend).days <= 75:
                pre.append((sc, qend, t))
    per = pd.DataFrame(per, columns=["ticker", "pend", "avail"]).groupby(["ticker", "pend"]).avail.min()
    pre = pd.DataFrame(pre, columns=["ticker", "pend", "prelim"]).groupby(["ticker", "pend"]).prelim.min()

    q = op.set_index(["ticker", "pend"]).join(per).join(pre).reset_index()
    q = q[q.ticker.isin(dec_fy) & q.avail.notna()]
    q["event_date"] = q[["avail", "prelim"]].min(axis=1) if C.USE_PRELIM_DATE else q["avail"]
    q = q.sort_values(["ticker", "pend"]).reset_index(drop=True)
    q.to_pickle(_p("quarters.pkl"))
    print(f"[quarters] {q.ticker.nunique()}개 기업, {len(q)}개 분기")
    return q


# ════════════════════════════════════════════════════════════
def download_all():
    _ensure(C.DATA_DIR)
    n_new = download_daily()
    if n_new == 0 and os.path.exists(_p("panels.pkl")):
        print("[price] 새로 받은 날짜 없음 → 기존 패널 사용")
        panels = pd.read_pickle(_p("panels.pkl"))
    else:
        panels = build_panels()
    if not os.path.exists(_p("k200.pkl")):
        download_index()
    if not os.path.exists(_p("sectors.pkl")):
        download_sectors()
    # 한 번이라도 시총 1,000억 이상이었던 보통주만 DART 조회 (호출 수 절약)
    ever = panels["mcap"].max()
    codes = sorted(ever[ever >= 1_000e8].index)
    print(f"[DART] 대상 종목 {len(codes)}개")
    download_disclosures(codes)
    download_financials(codes)
    build_quarters()


def load_all():
    panels = pd.read_pickle(_p("panels.pkl"))
    quarters = pd.read_pickle(_p("quarters.pkl"))
    k200 = pd.read_pickle(_p("k200.pkl"))
    sectors = pd.read_pickle(_p("sectors.pkl"))
    return panels, quarters, k200, sectors