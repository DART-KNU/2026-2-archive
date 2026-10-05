"""설정(config.yaml) 로드 + DataGuide(FnGuide) CSV 파서."""
from __future__ import annotations

from pathlib import Path
import yaml
import csv
import numpy as np
import pandas as pd


# ────────────────────────────── config ──────────────────────────────


ROOT = Path(__file__).resolve().parent


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for k in ("raw", "processed", "outputs"):
        cfg["paths"][k] = ROOT / cfg["paths"][k]
    return cfg


# ────────────────────────────── loader ──────────────────────────────


ENC = "utf-8-sig"


def _read_rows(path: Path) -> list[list[str]]:
    for enc in (ENC, "cp949"):
        try:
            with open(path, encoding=enc, newline="") as fh:
                return list(csv.reader(fh))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩 판별 실패: {path}")


def _to_num(block: list[list[str]]) -> np.ndarray:
    """쉼표 숫자 문자열 → float 배열. 빈칸은 NaN, 숫자가 아니면 에러."""
    flat = np.array(block, dtype=object).ravel()
    s = pd.Series(flat).str.replace(",", "", regex=False).replace("", None)
    out = pd.to_numeric(s, errors="raise").to_numpy(float)
    return out.reshape(len(block), -1)


def _find_row(rows, label, col):
    for i, r in enumerate(rows[:20]):
        if len(r) > col and r[col].strip() == label:
            return i
    raise ValueError(f"'{label}' 헤더 행을 찾지 못함")


def read_timeseries(path: Path, n_header: int = 14):
    """일간/월간 시계열 파일 → (날짜, 코드, 코드명, 문자열 본문)."""
    rows = _read_rows(Path(path))
    # 앞 빈 열 유무 판별: '코드' 라벨이 있는 열 위치
    off = 0 if rows[8][0].strip() == "코드" else 1
    if rows[8][off].strip() != "코드" or rows[9][off].strip() != "코드명":
        raise ValueError(f"{path}: 코드/코드명 헤더 위치가 예상과 다름")
    codes = [c.strip() for c in rows[8][off + 1:]]
    names = rows[9][off + 1:]
    data = rows[n_header:]
    if any(len(r) != len(rows[8]) for r in data):
        raise ValueError(f"{path}: 행 길이 불일치")
    dates = pd.to_datetime([r[off] for r in data], format="%Y-%m-%d")
    body = [r[off + 1:] for r in data]
    return dates, codes, names, body


def read_numeric_ts(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dates, codes, names, body = read_timeseries(path)
    df = pd.DataFrame(_to_num(body), index=dates, columns=codes)
    df.index.name = "date"
    return df, pd.Series(names, index=codes, name="name")


def read_string_ts(path: Path) -> pd.DataFrame:
    dates, codes, names, body = read_timeseries(path)
    df = pd.DataFrame(body, index=dates, columns=codes).replace("", np.nan)
    df.index.name = "date"
    return df


def read_quarterly(path: Path) -> pd.DataFrame:
    """분기 재무 → index=(분기말 날짜), columns=종목코드."""
    rows = _read_rows(Path(path))
    code_row = next(i for i, r in enumerate(rows[:13]) if len(r) > 2 and r[2].startswith("A") and r[0] == "")
    hdr_end = _find_row(rows, "회계연도", 0)
    codes = [c.strip() for c in rows[code_row][2:]]
    data = rows[hdr_end + 1:]
    yq = [(int(r[0]), int(r[1])) for r in data]
    idx = pd.to_datetime([f"{y}-{q:02d}-01" for y, q in yq]) + pd.offsets.MonthEnd(0)
    df = pd.DataFrame(_to_num([r[2:] for r in data]), index=idx, columns=codes)
    df.index.name = "qend"
    return df


def read_info(path: Path) -> pd.DataFrame:
    rows = _read_rows(Path(path))
    h = _find_row(rows, "코드", 0)
    cols = rows[h]
    df = pd.DataFrame(rows[h + 1:], columns=cols).replace("", np.nan)
    df = df.rename(columns={"코드": "code", "코드명": "name", "상장일": "list_date", "상장폐지일": "delist_date",
                            "상장된 시장": "market", "FnGuide Sector Code": "fg_code", "FnGuide Sector": "fg_sector"})
    for c in ("list_date", "delist_date"):
        df[c] = pd.to_datetime(df[c], format="%Y%m%d", errors="coerce")
    return df.set_index("code")


def read_sector_csv(path: Path) -> pd.DataFrame:
    """대회 GICS 분류표 (섹터코드, 섹터명, 종목코드, 종목명). 실제 파일은 data/raw/sector.txt (쉼표 구분)."""
    for enc in (ENC, "cp949"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    df.columns = ["sec_code", "sector", "code", "name"][: len(df.columns)]
    df["code"] = df["code"].str.strip()
    df.loc[~df["code"].str.startswith("A"), "code"] = "A" + df["code"].str.zfill(6)
    return df.set_index("code")
