"""
generate_mock_data.py
실제 시세 데이터가 없어도 부트캠프 실습이 가능하도록 가상의 일별 OHLCV 데이터를 생성합니다.
(랜덤워크 기반 - 실제 종목 데이터를 구하면 이 파일 대신 그 CSV를 써도 됩니다.
 컬럼 형식만 Date, Open, High, Low, Close, Volume 으로 맞추면 run_backtest.py 수정 없이 바로 동작합니다.)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_days = 500
dates = pd.date_range("2024-01-02", periods=n_days, freq="B")

# 일별 종가를 랜덤워크(기하 브라운 운동 근사)로 생성
daily_returns = np.random.normal(loc=0.0004, scale=0.013, size=n_days)
close = 50_000 * (1 + daily_returns).cumprod()

# 종가를 기준으로 시가/고가/저가를 그럴듯하게 파생
open_ = close * (1 + np.random.normal(0, 0.003, n_days))
high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.004, n_days)))
low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.004, n_days)))
volume = np.random.randint(100_000, 1_000_000, n_days)

df = pd.DataFrame({
    "Date": dates,
    "Open": open_.round(0),
    "High": high.round(0),
    "Low": low.round(0),
    "Close": close.round(0),
    "Volume": volume,
})

df.to_csv("data/mock_ohlcv.csv", index=False)
print(f"저장 완료: data/mock_ohlcv.csv ({n_days}행)")
