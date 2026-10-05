"""DataGuide 6 엑셀/CSV 출력(달력기준, 일자×종목) → pandas 로더.

DataGuide 시트 구조
  1~8행 : 조회 조건 (Refresh, 기간 등)
  9행   : 코드 (A005930 ...)
  10행  : 코드명
  12행  : 아이템코드, 13행: 아이템명
  15행~ : A열 = 날짜(엑셀 serial 또는 yyyy-mm-dd), 나머지 = 값
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
CACHE = Path(__file__).parent / "data" / "cache"

HEADER_CODE_ROW = 8      # 0-based index of the "코드" row
HEADER_NAME_ROW = 9
DATA_START_ROW = 14
UNIT_SCALE = {"억원": 1e8, "백만원": 1e6, "천원": 1e3, "원": 1.0}   # 아이템명 괄호 속 단위 → 원


def read_dataguide_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return (wide DataFrame indexed by date with code columns, code→name Series)."""
    raw = pd.read_csv(path, header=None, dtype=str, encoding="utf-8-sig", low_memory=False)
    assert str(raw.iat[HEADER_CODE_ROW, 0]).strip() == "코드", f"unexpected layout in {path.name}"
    codes = raw.iloc[HEADER_CODE_ROW, 1:].str.strip()
    names = raw.iloc[HEADER_NAME_ROW, 1:].str.strip()
    body = raw.iloc[DATA_START_ROW:, :]
    dates = _parse_dates(body.iloc[:, 0])
    values = body.iloc[:, 1:].apply(lambda s: pd.to_numeric(s.str.replace(",", ""), errors="coerce"))
    item_name = str(raw.iat[HEADER_CODE_ROW + 4, 1])
    scale = next((v for k, v in UNIT_SCALE.items() if f"({k})" in item_name), 1.0)
    df = pd.DataFrame(values.to_numpy(dtype="float64") * scale, index=dates, columns=codes.to_numpy())
    df = df[~df.index.isna()].sort_index()
    df = df.loc[:, ~df.columns.duplicated()]
    return df, pd.Series(names.to_numpy(), index=codes.to_numpy()).drop_duplicates()


def _parse_dates(col: pd.Series) -> pd.DatetimeIndex:
    s = col.astype(str).str.strip()
    serial = pd.to_numeric(s, errors="coerce")
    out = pd.to_datetime(serial, unit="D", origin="1899-12-30", errors="coerce")
    text = pd.to_datetime(s.where(serial.isna()), errors="coerce")
    return pd.DatetimeIndex(out.fillna(text))


def load_item(name: str) -> pd.DataFrame:
    """Load one item (e.g. 'close') from cache parquet, building it from raw CSV if needed."""
    CACHE.mkdir(parents=True, exist_ok=True)
    pq = CACHE / f"{name}.pkl"
    if pq.exists():
        return pd.read_pickle(pq)
    df, names = read_dataguide_csv(RAW / f"{name}.csv")
    df.to_pickle(pq)
    names_path = CACHE / "names.pkl"
    if names_path.exists():
        names = pd.concat([pd.read_pickle(names_path), names]).groupby(level=0).first()
    names.to_pickle(names_path)
    return df


def load_names() -> pd.Series:
    return pd.read_pickle(CACHE / "names.pkl")
