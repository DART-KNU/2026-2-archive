"""
sensitivity.py
민감도 분석: 설정을 하나씩 바꿔 수익률·MDD·비용·회전율 미달 주 수를 비교
(익절 기준, 섹터 수, 보유종목 우대, 코스닥 포함, 회전율 보완 여부)
실행: python sensitivity.py
"""
import pandas as pd

from backtest import PARAMS, Backtest, load_all, summarize

PERIODS = {
    "2026Q3": ("2026-07-01", "2026-09-28"),
    "full": ("2023-07-03", "2026-09-28"),
}
TESTS = {
    "Base (final)": {},
    "TP 7%": {"tp": 0.07},
    "TP 10%": {"tp": 0.10},
    "No TP (expiry only)": {"tp": None},
    "10 sectors": {"n_sectors": 10},
    "No incumbent buffer": {"buffer": False},
    "Core incl. KOSDAQ": {"markets": ("코스피", "코스닥")},
    "No turnover top-up": {"topup": False},
}


def main():
    px, uni, _ = load_all(kospi_file=None)
    rows = []
    for per, (s, e) in PERIODS.items():
        for name, extra in TESTS.items():
            b = Backtest(px, uni, s, e, **{**PARAMS, **extra}).run()
            rows.append(dict(period=per, **summarize(b, name)))
            print(per, name, "done")
    cols = ["ret", "mdd", "cost", "semi_win", "weeks_below_5"]
    print(pd.DataFrame(rows).set_index(["period", "strategy"])[cols].round(2).to_string())


if __name__ == "__main__":
    main()
