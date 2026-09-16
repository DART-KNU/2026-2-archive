"""
strategies/sma_cross.py

가장 단순한 교과서적 전략인 '이동평균 교차(SMA Crossover)'를 backtrader의
Strategy 클래스로 구현한 예시입니다.

설계 포인트 (부트캠프 설명용):
    - backtrader의 Strategy는 매 봉(bar)마다 자동으로 호출되는 next()에
      '이번 봉에서 무엇을 할지'만 정의하면 됩니다.
    - 실제 주문 체결/자금 관리/수수료 계산은 Broker가 알아서 처리하므로,
      전략 작성자는 오직 '매매 신호' 로직에만 집중하면 됩니다.
    - 이 분리(전략 로직 vs 체결/자금 관리)가 이벤트 기반 엔진 설계의 핵심입니다.
"""

import backtrader as bt


class SmaCross(bt.Strategy):
    params = dict(
        fast_period=10,   # 단기 이동평균 기간
        slow_period=30,   # 장기 이동평균 기간
    )

    def __init__(self):
        # 지표는 __init__에서 한 번만 선언 (매 봉마다 새로 계산하지 않음 - backtrader가 알아서 갱신)
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        # next()는 새 봉(하루치 데이터)이 들어올 때마다 자동 호출됩니다.
        # → 이 부분이 '이벤트 기반' 이라는 걸 가장 잘 보여주는 지점입니다.
        if not self.position:  # 보유 포지션이 없을 때
            if self.crossover > 0:  # 단기선이 장기선을 상향 돌파 -> 매수
                self.buy()
        else:
            if self.crossover < 0:  # 단기선이 장기선을 하향 돌파 -> 매도(청산)
                self.close()
