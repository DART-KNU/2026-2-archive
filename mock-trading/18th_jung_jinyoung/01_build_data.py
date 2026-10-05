"""DataGuide 원본 → data/processed/*.parquet (상장·상폐 합친 패널)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data_io import load_config
from data_io import read_info, read_numeric_ts, read_quarterly, read_sector_csv, read_string_ts
from signals import delisted_sector_monthly, fg_to_gics_table


def main():
    cfg = load_config()
    R, P = cfg["paths"]["raw"], cfg["paths"]["processed"]
    P.mkdir(parents=True, exist_ok=True)

    sector = read_sector_csv(R / cfg["paths"]["sector_file"])
    info_l, info_d = read_info(R / "info_listed.csv"), read_info(R / "info_delisted.csv")
    sh_raw = read_string_ts(R / "sector_hist_delisted.csv")

    # ---- 종목 필터: 상장=sector 파일에 있는 코드, 상폐=sector_hist에 섹터 있음, 스팩 제외
    keep_l = [c for c in info_l.index if c in sector.index and "스팩" not in str(info_l.at[c, "name"])]
    keep_d = [c for c in info_d.index if sh_raw[c].notna().any() and "스팩" not in str(info_d.at[c, "name"])]
    print(f"종목 필터: 상장 {len(keep_l)}, 상폐 {len(keep_d)}")

    # ---- 섹터 대응표
    ct, mapping = fg_to_gics_table(info_l, sector)
    ct.to_csv(P / "fg_gics_crosstab.csv", encoding="utf-8-sig")
    pd.Series(mapping, name="gics").rename_axis("fg_code").to_csv(P / "fg_gics_map.csv", encoding="utf-8-sig")
    sh = delisted_sector_monthly(sh_raw[keep_d], mapping, sector)
    sh.to_parquet(P / "sector_monthly_delisted.parquet")

    # ---- 종목 정보
    stocks = pd.concat([
        info_l.loc[keep_l].assign(group="listed", gics=sector.loc[keep_l, "sector"]),
        info_d.loc[keep_d].assign(group="delisted", gics=None),
    ])[["name", "group", "market", "list_date", "delist_date", "fg_code", "gics"]]
    stocks.index.name = "code"
    stocks.to_parquet(P / "stocks.parquet")

    # ---- 일간 패널
    for item in ("price_adj", "mcap", "value"):
        l, _ = read_numeric_ts(R / f"{item}_listed.csv")
        d, _ = read_numeric_ts(R / f"{item}_delisted.csv")
        assert l.index.equals(d.index)
        panel = pd.concat([l[keep_l], d[keep_d]], axis=1)
        panel.to_parquet(P / f"{item}.parquet")
        print(f"{item}: {panel.shape}")

    idx, _ = read_numeric_ts(R / "index.csv")
    idx.rename(columns={"I.001": "KOSPI", "I.201": "KOSDAQ"}).to_parquet(P / "index.parquet")

    # ---- 재무 (long): 분기말 + 45일, 4분기 + 90일부터 사용 가능
    lag, lag4 = cfg["data"]["fin_lag_days"], cfg["data"]["fin_lag_days_q4"]
    fins = []
    for item, col in (("ni_ctrl", "ni"), ("eq_ctrl", "eq")):
        q = pd.concat([read_quarterly(R / f"{item}_listed.csv"), read_quarterly(R / f"{item}_delisted.csv")], axis=1)
        q = q[[c for c in q.columns if c in stocks.index]]
        fins.append(q.stack().rename(col))
    fin = pd.concat(fins, axis=1).reset_index().rename(columns={"level_1": "code"})
    fin.columns = ["qend", "code", "ni", "eq"]
    fin["avail"] = fin["qend"] + pd.to_timedelta(fin["qend"].dt.month.eq(12).map({True: lag4, False: lag}), unit="D")
    fin.to_parquet(P / "fin.parquet")
    print(f"fin: {len(fin)}행, 종목 {fin.code.nunique()}")


if __name__ == "__main__":
    main()
