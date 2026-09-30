from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

from settings import (
    RAW_STOCK_PARQUET_DIR,
    CLEAN_STOCK_PARQUET_DIR,
    FACTOR_PANEL_FILE,
)


REQUIRED = [
    "date",
    "code",
    "name",
    "open_adj",
    "close_adj",
    "trading_value_krw",
    "market_cap_krw",
]

CLEAN_STOCK_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR = CLEAN_STOCK_PARQUET_DIR.parent / "stock_qc"
QC_DIR.mkdir(exist_ok=True)


def normalize_code(s):
    s = s.astype("string").str.strip()
    no_a = ~s.str.startswith("A", na=False)
    digits = s.str.fullmatch(r"\d+", na=False)
    s.loc[digits & no_a] = "A" + s.loc[digits & no_a].str.zfill(6)
    has_a = s.str.fullmatch(r"A\d+", na=False)
    s.loc[has_a] = "A" + s.loc[has_a].str[1:].str.zfill(6)
    return s


def main():
    files = sorted(RAW_STOCK_PARQUET_DIR.glob("stock_daily_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"변환된 주가 parquet가 없습니다: {RAW_STOCK_PARQUET_DIR}"
        )

    summaries = []

    for file in files:
        print("QC:", file.name)
        df = pd.read_parquet(file)

        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{file.name} 필수 컬럼 누락: {missing}")

        rows_raw = len(df)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["code"] = normalize_code(df["code"])

        for c in [
            "open_adj",
            "close_adj",
            "trading_value_krw",
            "market_cap_krw",
        ]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        invalid_date = int(df["date"].isna().sum())
        duplicate_rows = int(
            df.duplicated(["date", "code"], keep=False).sum()
        )
        invalid_close = int(
            (df["close_adj"].notna() & (df["close_adj"] <= 0)).sum()
        )
        invalid_mcap = int(
            (df["market_cap_krw"].notna() & (df["market_cap_krw"] <= 0)).sum()
        )
        negative_value = int(
            (df["trading_value_krw"].notna() & (df["trading_value_krw"] < 0)).sum()
        )

        df = df.loc[
            df["date"].notna()
            & df["code"].notna()
        ].copy()

        df = (
            df.sort_values(["date", "code"])
            .drop_duplicates(["date", "code"], keep="last")
            .reset_index(drop=True)
        )

        year = int(df["date"].dt.year.mode().iloc[0])
        out = CLEAN_STOCK_PARQUET_DIR / f"stock_daily_{year}.parquet"
        df.to_parquet(out, index=False, compression="zstd")

        summaries.append({
            "file": file.name,
            "rows_raw": rows_raw,
            "rows_clean": len(df),
            "stocks": int(df["code"].nunique()),
            "dates": int(df["date"].nunique()),
            "date_min": df["date"].min(),
            "date_max": df["date"].max(),
            "invalid_date": invalid_date,
            "duplicate_rows": duplicate_rows,
            "invalid_close": invalid_close,
            "invalid_market_cap": invalid_mcap,
            "negative_trading_value": negative_value,
            "missing_open_rate": float(df["open_adj"].isna().mean()),
            "missing_close_rate": float(df["close_adj"].isna().mean()),
            "missing_mcap_rate": float(df["market_cap_krw"].isna().mean()),
        })

    qc = pd.DataFrame(summaries)
    qc.to_csv(QC_DIR / "stock_qc_summary.csv", index=False, encoding="utf-8-sig")

    input_pattern = str(
        CLEAN_STOCK_PARQUET_DIR / "stock_daily_*.parquet"
    ).replace("\\", "/")
    output_path = str(FACTOR_PANEL_FILE).replace("\\", "/")

    con = duckdb.connect()
    con.execute("SET threads = 6")

    con.execute(
        f"""
        COPY (
            SELECT
                CAST(date AS DATE) AS date,
                code,
                name,
                open_adj,
                close_adj,
                trading_value_krw,
                market_cap_krw
            FROM read_parquet('{input_pattern}', union_by_name=true)
            ORDER BY date, code
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    summary = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT code) AS stocks,
            COUNT(DISTINCT date) AS dates,
            MIN(date) AS date_min,
            MAX(date) AS date_max,
            MEDIAN(market_cap_krw) AS median_mcap_krw
        FROM read_parquet('{output_path}')
        """
    ).df()
    con.close()

    print("\nFACTOR PANEL COMPLETE")
    print(summary.to_string(index=False))
    print("Output:", FACTOR_PANEL_FILE)
    print("QC    :", QC_DIR / "stock_qc_summary.csv")


if __name__ == "__main__":
    main()
