# [2026-2] 모의투자 포트폴리오 - [19기_정재화]

## 전략
- **핵심 로직**: 리더 섹터 상대강도 모멘텀 (Riding the Leaders). KOSPI 대비 상대강도 상위 3개 섹터를 고르고, 섹터별 모멘텀 상위 3종목(총 9종목)을 매주 금요일 종가에 리밸런싱
- **카페 링크**: [[19기 전략보고서] 리더 섹터 상대강도 모멘텀 전략](https://naver.me/FEgpmKIp)
- **발표 자료**: [strategy_presentation.pdf](strategy_presentation.pdf)

## 파일 설명
- `00_check_data.py` : 데이터 연결 테스트
- `01_download_data.py` : 가격·시가총액·업종 데이터 다운로드 및 캐시
- `02_run_backtest.py` : 백테스트 실행 (기본안, 36개 파라미터 조합, Deflated Sharpe)
- `03_live_signal.py` : 대회용 금요일 목표 포트폴리오 산출
- `config.py` / `data.py` / `strategy.py` : 파라미터 / 데이터 로딩 / 전략·성과 계산 모듈
- `results/` : 백테스트 결과 (성과 지표, 그래프, 매매 내역)

---

## Riding the Leaders: Sector Relative-Strength Momentum (details)

Backtest code for the DART 2026-2 virtual trading strategy (Timefolio Road to Fund Manager #13, Oct–Nov 2026).

## Strategy (final version)

1. **Universe.** KOSPI and KOSDAQ common stocks with market cap ≥ KRW 1tn and 4-week average daily trading value ≥ KRW 5bn.
2. **Sector score.** `0.2·RS(1M) + 0.4·RS(3M) + 0.4·RS(6M)`, where RS is the sector return minus the KOSPI return. Only sectors trading above their 60-day moving average are eligible. The top 3 are held; a held sector is replaced only when it falls out of the top 4.
3. **Stock score.** Within each selected sector: `rank(R3M) + rank(R6M) + rank(Price / 52-week high)`, with returns skipping the most recent week. Top 3 per sector; a held stock stays while it ranks in the top 5 of its sector.
4. **Weights.** Sectors 40 / 35 / 25%, equal-weighted within each sector. Samsung Electronics is set to 25% if selected. Single-stock cap 15%. A kept holding is resized only when it drifts more than 3%p from target. Unused weight (fewer than 3 eligible sectors) stays in cash.
5. **Execution.** Signals and trades on the last trading day of each week, at the close (in the contest: computed shortly before the close and executed in the closing auction). No stop-loss.
6. **Costs.** 10 bp commission each way, 20 bp sell tax, 10 bp slippage each way.

### Changes from the version presented on 2026-09-29

The presented version used a 12% trailing stop and traded at the next day's close. Its backtest (`comparison.csv`, grey line in `equity_drawdown.png`) lost money: frequent stop-outs of volatile momentum stocks that later recovered, and a one-day execution delay that gave up most of the short-term continuation after the signal. The final version removes the stop and trades at the signal-day close. These changes were made after seeing the results, so the 36-run grid and the Deflated Sharpe Ratio are reported to show how much of the result depends on parameter choice.

## How to run

```bash
pip install -r requirements.txt
python 00_check_data.py        # connectivity test (~10 s)
python 01_download_data.py     # download and cache data (first run 30-60 min, resumable)
python 02_run_backtest.py      # base case, 36-run grid, Deflated Sharpe Ratio
python 03_live_signal.py       # current target portfolio (run 01 with --update first)
```

## Outputs (`results/`)

| File | Content |
|---|---|
| `base_metrics.csv` | CAGR, volatility, Sharpe, max drawdown, turnover, hit rate |
| `comparison.csv` | Final version vs. original proposal vs. KOSPI |
| `equity_drawdown.png` | Growth of 1 and drawdown: final version, original proposal, KOSPI |
| `oct_nov_returns.csv`, `oct_nov.png` | Return over the contest months (Oct–Nov) by year |
| `grid_results.csv`, `grid_sharpe.png` | All 36 parameter sets (3 lookback weightings × 2/3/4 sectors × 8/12/15%/no stop) |
| `deflated_sharpe.csv` | Deflated Sharpe Ratio (Bailey & López de Prado, 2014) for the base case and the best grid run |
| `base_trades.csv`, `base_weekly_picks.csv` | Trade log and weekly selections |

## Data and limitations

- Prices, KOSPI and weekly market-cap snapshots: KRX via [pykrx](https://github.com/sharebook-kr/pykrx) (KRX data now requires a login: set `KRX_ID` / `KRX_PW`).
- Industry classification: KRX industry snapshots when available; otherwise (`01_download_data.py --static`) the KSIC industry of listed stocks and the KRX industry of delisted stocks from the [FinanceDataReader KRX cache](https://github.com/FinanceData/fdr_krx_data_cache), mapped to 25 GICS-like industry groups (`GROUP_RULES` in `data.py`). This static classification uses today's industry for the whole period.
- Sector returns are market-cap-weighted averages of member stocks, built from the weekly snapshots (all listed stocks, including later-delisted ones).
- Stocks without an industry (about 60% of delisted names) are excluded, which leaves some survivorship bias.
- Not modelled: the contest's sector cap (2× market weight), intraday execution, and the weekly 5% turnover rule (see `WeeksTurnover>=5%` in the metrics; weeks below 5% need small top-up trades in the contest).
- Sharpe ratios use a zero risk-free rate.
