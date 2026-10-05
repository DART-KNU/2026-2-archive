"""장 마감 후 실행: 시장 필터 상태 + 매수 대기 / 관심 종목 리스트.

사용법: python screen_today.py            (data/raw 의 DataGuide CSV를 최신으로 갱신한 뒤 실행)
출력  : output/screen_YYYYMMDD.csv
  tier 0 = 매수 대기 (52주 최고 종가 95% 이내 + 횡보 조건) → 다음 거래일 종가가 돌파가를 넘으면 매수
  tier 1 = 관심 종목 (90% 이내 + 횡보 조건) → 95% 안으로 들어오면 매수 대기로 승격
"""
import numpy as np
import pandas as pd

import backtest as bt
from load_data import load_names
from run_backtest import load_all


def screen(p: bt.Params = bt.FINAL) -> pd.DataFrame:
    data, kospi, _ = load_all()
    c = data["close"]
    k = len(c.index) - 1
    C = c.to_numpy(dtype="float64")
    HC = c.rolling(p.lookback, min_periods=p.lookback // 2).max().to_numpy(dtype="float64")
    MC, TV5 = data["mcap"].to_numpy(dtype="float64"), data["tv5"].to_numpy(dtype="float64")
    names = load_names()
    on = bt.market_filter(kospi.reindex(c.index).ffill(), p.filter_mode).iloc[-1]
    rows = []
    for j, code in enumerate(c.columns):
        pivot, px = HC[k, j], C[k, j]
        if not code.endswith("0") or np.isnan(pivot) or np.isnan(px):
            continue
        if not (pivot * bt.WATCH_NEAR <= px < pivot) or MC[k, j] < p.min_mcap or not TV5[k, j] > p.min_tv5:
            continue
        win = C[k - p.lookback + 1:k + 1, j]
        peak_pos = int(np.nanargmax(win))
        base_len = len(win) - 1 - peak_pos
        rng = 1 - np.nanmin(win[peak_pos:]) / pivot
        if not (p.base_min <= base_len <= p.base_max and rng <= p.base_range):
            continue
        tier = 0 if px >= pivot * p.near_high else 1
        rows.append(dict(tier=tier, code=code, name=names.get(code, ""), close=px, breakout_px=pivot,
                         to_breakout=pivot / px - 1, max_buy_px=pivot * (1 + p.chase_limit), base_days=base_len,
                         base_range=rng, mcap_억=MC[k, j] / 1e8, tv5_억=TV5[k, j] / 1e8))
    out = pd.DataFrame(rows).sort_values(["tier", "to_breakout"]) if rows else pd.DataFrame()
    day = c.index[k]
    ma = kospi.rolling(60).mean().iloc[-1]
    print(f"기준일 {day.date()} | 시장 필터: {'ON (신규매수 가능)' if on else 'OFF (신규매수 중단)'} | KOSPI {kospi.iloc[-1]:,.0f} / MA60 {ma:,.0f}")
    if len(out):
        for t, label in ((0, "매수 대기 (다음 거래일 종가가 돌파가 초과 & 최대매수가 이하이면 종가 매수)"), (1, "관심 종목")):
            part = out[out["tier"] == t]
            print(f"\n[{label}] {len(part)}종목")
            if len(part):
                print(part.drop(columns="tier").round(3).to_string(index=False))
        out.to_csv(bt.OUT / f"screen_{day:%Y%m%d}.csv", index=False, encoding="utf-8-sig")
    else:
        print("후보 없음")
    return out


if __name__ == "__main__":
    bt.OUT.mkdir(exist_ok=True)
    screen()
