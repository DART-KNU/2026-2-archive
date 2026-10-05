"""
진단: 신호 자체가 통하는가? (이벤트 스터디)
  - 새 전략을 시도하는 게 아니라, 사전 등록한 신호의 '사후 20거래일 섹터 대비 초과수익'을 그대로 측정
  - 신호가 통하는데 성과가 낮다면 → 구현(현금 비중) 문제
  - 신호 자체가 안 통한다면 → 한국 대형·유동 종목에서 PEAD가 약하다는 정직한 결론

  python diagnose.py          # 실제 데이터
  python diagnose.py --demo
"""
import argparse
import os

import numpy as np
import pandas as pd

import config as C
from signals import build_events, eligible_mask, prepare_panels


def car_after_entry(ev, P, hold=C.HOLD_DAYS):
    """t+2 ~ t+(hold+1) 동안의 (종목 - 섹터평균) 누적 초과수익"""
    ret = P["ret"].values
    sm = P["sec_mean"]
    sm_ix = {s: k for k, s in enumerate(sm.columns)}
    smv = sm.values
    col_ix = {t: j for j, t in enumerate(P["cols"])}
    T = len(P["dates"])
    cars, elig = np.full(len(ev), np.nan), np.zeros(len(ev), bool)
    masks = {}
    for k, (tk, t1) in enumerate(zip(ev["ticker"].values, ev["t1_idx"].values)):
        j = col_ix.get(tk)
        a, b = t1 + 1, t1 + 1 + hold
        if j is None or b > T:
            continue
        if t1 not in masks:
            masks[t1] = eligible_mask(P, t1)
        elig[k] = masks[t1][j]
        r = ret[a:b, j]
        m = smv[a:b, sm_ix[P["sector"].iloc[j]]]
        ok = np.isfinite(r) & np.isfinite(m)
        if ok.sum() >= hold * 0.7:
            cars[k] = np.sum(r[ok] - m[ok])
    ev = ev.copy()
    ev["car"] = cars
    ev["eligible"] = elig
    return ev


def summarize(x):
    x = x.dropna()
    n = len(x)
    if n < 2:
        return pd.Series({"N": n, "평균 CAR": np.nan, "t값": np.nan, "양수 비율": np.nan})
    return pd.Series({"N": n, "평균 CAR": x.mean(), "t값": x.mean() / (x.std(ddof=1) / np.sqrt(n)),
                      "양수 비율": (x > 0).mean()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        from demo_data import make_demo
        panels, quarters, k200, sectors = make_demo()
        out = os.path.join(C.OUT_DIR, "demo", "diagnose")
        y0 = 2017
    else:
        from data_loader import load_all
        panels, quarters, k200, sectors = load_all()
        out = os.path.join(C.OUT_DIR, "diagnose")
        y0 = int(C.BACKTEST_START[:4])
    os.makedirs(out, exist_ok=True)

    P = prepare_panels(panels, sectors)
    ev = build_events(quarters, P)
    ev = car_after_entry(ev, P)
    ev = ev[(ev.known_date.dt.year >= y0) & ev.car.notna()]
    el = ev[ev.eligible]
    el = el.assign(year=el.known_date.dt.year,
                   quintile=pd.cut(el.pScore, [0, .2, .4, .6, .8, 1.0001], labels=["Q1(하위)", "Q2", "Q3", "Q4", "Q5(상위)"]))

    t1 = pd.DataFrame({
        "진입조건 통과": summarize(el.loc[el.passed, "car"]),
        "미통과": summarize(el.loc[~el.passed, "car"]),
        "유니버스 전체": summarize(el["car"]),
    }).T
    t2 = el.groupby("quintile", observed=True)["car"].apply(summarize).unstack()
    t3 = el[el.passed].groupby("year")["car"].apply(summarize).unstack()
    per_month = el[el.passed].groupby(el[el.passed].known_date.dt.to_period("M")).size()

    fmt = lambda d: d.assign(**{c: d[c].map(lambda v: f"{v * 100:6.2f}%") for c in ["평균 CAR", "양수 비율"] if c in d},
                             **{"t값": d["t값"].map(lambda v: f"{v:5.2f}")}, N=d["N"].astype(int))
    print("\n══════ ① 진입 후 20거래일 섹터 대비 초과수익 (유니버스 통과 종목만) ══════")
    print(fmt(t1).to_string())
    print("\n══════ ② Score 5분위별 (위로 갈수록 커지면 신호가 작동) ══════")
    print(fmt(t2).to_string())
    print("\n══════ ③ 진입조건 통과 종목의 연도별 ══════")
    print(fmt(t3).to_string())
    print("\n══════ ④ 신호 공급량 ══════")
    print(f"  월평균 진입 가능 신호: {per_month.mean():.1f}개  (중앙값 {per_month.median():.0f}, 최대 {per_month.max()})")
    print(f"  → 20일 보유 기준 동시 보유 가능 종목 ≈ {per_month.mean() * C.HOLD_DAYS / 21:.1f}개 (목표 20~30개)")

    t1.to_csv(os.path.join(out, "car_by_group.csv"), encoding="utf-8-sig")
    t2.to_csv(os.path.join(out, "car_by_quintile.csv"), encoding="utf-8-sig")
    t3.to_csv(os.path.join(out, "car_by_year.csv"), encoding="utf-8-sig")
    print(f"\n결과 저장 위치: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
