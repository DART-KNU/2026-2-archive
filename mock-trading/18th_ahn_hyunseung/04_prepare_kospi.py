import pandas as pd
import duckdb

from settings import FACTOR_PANEL_FILE, KOSPI_FILE


def existing_covers(start_date, end_date):
    if not KOSPI_FILE.exists():
        return False
    try:
        df = pd.read_csv(KOSPI_FILE, encoding="utf-8-sig")
        date_col = "Date" if "Date" in df.columns else "date" if "date" in df.columns else None
        if date_col is None:
            return False
        d = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(d) == 0:
            return False
        return d.min().date() <= start_date and d.max().date() >= end_date
    except Exception:
        return False


def main():
    if not FACTOR_PANEL_FILE.exists():
        raise FileNotFoundError(FACTOR_PANEL_FILE)

    con = duckdb.connect()
    p = str(FACTOR_PANEL_FILE).replace("\\", "/")
    row = con.execute(
        f"SELECT MIN(date), MAX(date) FROM read_parquet('{p}')"
    ).fetchone()
    con.close()

    start_date = pd.Timestamp(row[0]).date()
    end_date = pd.Timestamp(row[1]).date()

    if existing_covers(start_date, end_date):
        print("KOSPI.csv already covers stock sample:", KOSPI_FILE)
        return

    try:
        import FinanceDataReader as fdr
    except ImportError as e:
        raise ImportError(
            "KOSPI 자동 다운로드용 패키지가 없습니다. "
            "pip install finance-datareader"
        ) from e

    print("Downloading KOSPI KS11:", start_date, "~", end_date)
    raw = fdr.DataReader("KS11", str(start_date), str(end_date)).reset_index()

    if "Date" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "Date"})

    out = raw[["Date", "Close"]].copy()
    out.to_csv(KOSPI_FILE, index=False, encoding="utf-8-sig")
    print("Saved:", KOSPI_FILE)


if __name__ == "__main__":
    main()
