"""
run_backtest.py

백테스트 엔진(backtrader)의 5단계 구조를 그대로 보여주는 메인 실행 스크립트입니다.

    1) Cerebro   - 전체 백테스트를 지휘하는 실행 엔진
    2) Data Feed - 시세 데이터를 엔진에 공급
    3) Strategy  - 매매 로직 (strategies/sma_cross.py)
    4) Broker    - 주문 체결, 수수료, 초기 자금 관리
    5) Analyzer  - 성과 분석 (여기서는 자산곡선을 직접 기록하는 커스텀 Analyzer 사용)

실행 방법:
    python generate_mock_data.py   # (최초 1회, 이미 데이터가 있으면 생략 가능)
    python run_backtest.py
"""

import backtrader as bt
import pandas as pd
import numpy as np

from strategies.sma_cross import SmaCross
import quant_theme


# --------------------------------------------------------------
# 커스텀 Analyzer: 매 봉마다 (날짜, 자산 총액)을 기록
# backtrader 기본 Analyzer들은 통계 요약값 위주라, 자산곡선 자체를
# 뽑아내려면 이렇게 직접 만드는 게 가장 확실하고 이해하기 쉽습니다.
# --------------------------------------------------------------
class EquityCurve(bt.Analyzer):
    def start(self):
        self.dates = []
        self.values = []

    def next(self):
        self.dates.append(self.strategy.datetime.date(0))
        self.values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return {"dates": self.dates, "values": self.values}


def run():
    # 1) Cerebro 생성 - 이 객체가 데이터/전략/브로커를 전부 연결하고 실행합니다.
    cerebro = bt.Cerebro()

    # 2) Data Feed - CSV를 pandas로 읽어서 backtrader가 이해하는 형식으로 공급
    df = pd.read_csv("data/mock_ohlcv.csv", parse_dates=["Date"], index_col="Date")
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    # 3) Strategy 등록
    cerebro.addstrategy(SmaCross, fast_period=10, slow_period=30)

    # 4) Broker 설정 - 초기 자금, 수수료(슬리피지 개념 포함)
    cerebro.broker.setcash(10_000_000)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% 수수료 가정

    # 5) Analyzer 등록 - 자산곡선 + 기본 성과지표
    cerebro.addanalyzer(EquityCurve, _name="equity")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]

    # ------------------------------------------------------------
    # 결과 꺼내기 - Analyzer가 계산해둔 것을 읽기만 하면 됩니다.
    # ------------------------------------------------------------
    eq = strat.analyzers.equity.get_analysis()
    dates = pd.to_datetime(eq["dates"])
    equity = np.array(eq["values"])

    running_max = np.maximum.accumulate(equity)
    drawdown_pct = equity / running_max - 1

    monthly = pd.Series(equity, index=dates).resample("ME").last().pct_change().dropna()
    monthly_labels = [d.strftime("%Y-%m") for d in monthly.index]
    monthly_returns = monthly.values

    end_value = equity[-1]
    cumulative_return = end_value / start_value - 1
    n_years = (dates[-1] - dates[0]).days / 365.25
    annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1
    mdd = strat.analyzers.drawdown.get_analysis()["max"]["drawdown"] / 100
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio") or 0.0

    trade_info = strat.analyzers.trades.get_analysis()
    total_trades = trade_info.get("total", {}).get("total", 0)
    won_trades = trade_info.get("won", {}).get("total", 0)
    win_rate = (won_trades / total_trades) if total_trades else 0.0

    summary_df = pd.DataFrame({
        "Metric": ["Cumulative Return", "Annualized Return", "MDD", "Sharpe Ratio", "Win Rate", "Total Trades"],
        "Value": [
            f"{cumulative_return:+.2%}",
            f"{annualized_return:+.2%}",
            f"-{mdd:.1%}",
            f"{sharpe:.2f}",
            f"{win_rate:.1%}",
            f"{total_trades}",
        ],
    })

    # ------------------------------------------------------------
    # 산출물 통일 레이아웃 적용 (quant_theme.py)
    # ------------------------------------------------------------
    fig = quant_theme.build_backtest_report(
        dates=dates,
        equity=equity,
        drawdown_pct=drawdown_pct,
        monthly_labels=monthly_labels,
        monthly_returns=monthly_returns,
        summary_df=summary_df,
        suptitle="SMA Crossover Strategy - Backtest Report",
    )
    fig.savefig("sample_output/backtest_report.png", dpi=150, bbox_inches="tight")
    print("저장 완료: sample_output/backtest_report.png")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run()
