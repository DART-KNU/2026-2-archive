"""
screen.py
실전 운용용 스크리닝: 데이터 마지막 날짜 기준으로
  1) 이번 주 코어 8종목과 리스크 패리티 목표 비중 (화요일 종가 기준으로 실행 → 수요일 주문)
  2) 섹터별 2위 후보 (금요일 회전율 보완 시 참고)
  3) 삼성전자·SK하이닉스 진입 신호 여부 (매일 장 마감 후 실행 → 다음 날 주문)
실행: python screen.py
      python screen.py --held 000660 005380 ...   (현재 보유 코어 종목을 넣으면 보유종목 우대 규칙 반영)
※ 대회 매수 불가 종목(거래대금 30억 이하, 시총 1,000억 미만, 투자유의·관리종목)은 주문 전 직접 확인
"""
import argparse

import numpy as np
import pandas as pd

from backtest import HYN, PARAMS, SAM, Backtest, load_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", nargs="*", default=[], help="현재 보유 중인 코어 종목코드 (6자리)")
    a = ap.parse_args()

    px, uni, _ = load_all(kospi_file=None)
    last = px.index[-1]
    bt = Backtest(px, uni, px.index[0], last, **PARAMS)
    names = uni.set_index("ticker")

    # ---- 1) 코어 목표 비중 ----
    tgt, meta = bt.core_targets(last, held=a.held)
    core = pd.DataFrame({"ticker": list(tgt), "weight_%": np.round(np.array(list(tgt.values())) * 100, 2)})
    core["name"] = core.ticker.map(names["name"])
    core["sector"] = core.ticker.map(names["sector"])
    core["runner_up"] = core.ticker.map(lambda t: names["name"].get(meta["runner"].get(t), "-"))
    print(f"\n[코어 목표 비중] 기준일 {last:%Y-%m-%d}, 합계 {core['weight_%'].sum():.1f}%")
    print(core.sort_values("weight_%", ascending=False)[["ticker", "name", "sector", "weight_%", "runner_up"]]
          .to_string(index=False))

    # ---- 2) 반도체 신호 ----
    i = len(px) - 1
    print("\n[반도체 신호]")
    for t, nm in [(SAM, "삼성전자"), (HYN, "SK하이닉스")]:
        c, ma = px[t], bt.ma20[t]
        trend = c.iloc[i - 2] > ma.iloc[i - 2]
        r2 = c.iloc[i] / c.iloc[i - 2] - 1
        sig = trend and r2 <= -0.05
        print(f"{nm}: 추세필터 {'O' if trend else 'X'} | 2일 수익률 {r2 * 100:+.2f}% | "
              f"{'>>> 내일 25% 매수' if sig else '신호 없음'}")


if __name__ == "__main__":
    main()
