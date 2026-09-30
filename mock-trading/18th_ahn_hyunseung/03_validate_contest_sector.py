import pandas as pd

from settings import CONTEST_FILE, BASE_DIR

VALID_SECTORS = {
    "En", "Ma", "In", "CD", "CS", "He", "Fi", "IT", "Co", "Ut", "Re"
}


def normalize_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.startswith("A") and s[1:].isdigit():
        return "A" + s[1:].zfill(6)
    if s.isdigit():
        return "A" + s.zfill(6)
    return s


def main():
    if not CONTEST_FILE.exists():
        raise FileNotFoundError(CONTEST_FILE)

    df = pd.read_csv(CONTEST_FILE, dtype=str, encoding="utf-8-sig")
    required = ["종목코드", "종목명", "섹터코드", "섹터명"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"대회 섹터 CSV 필수 컬럼 누락: {missing}")

    df = df[required].copy()
    df["종목코드"] = df["종목코드"].map(normalize_code)

    dup = df.duplicated("종목코드", keep=False)
    bad_sector = ~df["섹터코드"].isin(VALID_SECTORS)

    print("Rows           :", len(df))
    print("Unique codes   :", df["종목코드"].nunique())
    print("Duplicate rows :", int(dup.sum()))
    print("Unknown sector :", int(bad_sector.sum()))
    print("\nSector counts")
    print(df["섹터코드"].value_counts(dropna=False).sort_index())

    if dup.any():
        df.loc[dup].to_csv(
            BASE_DIR / "sector_duplicate_codes.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if bad_sector.any():
        df.loc[bad_sector].to_csv(
            BASE_DIR / "sector_unknown_codes.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise ValueError("정의되지 않은 섹터코드가 있습니다.")

    standardized = (
        df.drop_duplicates("종목코드")
        .sort_values("종목코드")
    )
    standardized.to_csv(
        BASE_DIR / "contest_sector_standardized.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSector validation complete.")


if __name__ == "__main__":
    main()
