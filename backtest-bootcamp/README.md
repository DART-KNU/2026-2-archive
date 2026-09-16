# 백테스트 부트캠프 실습 레포

금융데이터과학회 코딩 부트캠프용 실습 코드입니다. 실제 백테스트 엔진(**backtrader**)을 그대로 사용하고,
동아리 공통 산출물 스타일(`quant_theme.py`)을 적용해 리포트를 만드는 전체 흐름을 담았습니다.

## 폴더 구조

```
backtest-bootcamp/
├── data/
│   └── mock_ohlcv.csv       # 가상 일별 시세 데이터 (실제 데이터로 교체 가능)
├── strategies/
│   └── sma_cross.py         # 매매 전략 (이동평균 교차)
├── sample_output/
│   └── backtest_report.png  # run_backtest.py 실행 결과 예시
├── generate_mock_data.py    # 가상 데이터 생성 스크립트
├── run_backtest.py          # 메인 실행 스크립트 (엔진 구조 전체가 여기 담김)
├── quant_theme.py           # 동아리 공통 차트/표 스타일 모듈
└── requirements.txt
```

## 실행 방법

```bash
pip install -r requirements.txt
python generate_mock_data.py   # 이미 data/mock_ohlcv.csv가 있다면 생략 가능
python run_backtest.py
```

실행하면 `sample_output/backtest_report.png`에 자산곡선·낙폭·월별수익률·성과요약표가
한 장의 리포트로 저장됩니다.

## 왜 backtrader인가

이벤트 기반(event-driven) 구조라서, 실제 트레이딩 시스템(주문 체결 → 포지션 관리 →
자금 관리)과 개념이 가장 가깝습니다. 벡터화 엔진(예: vectorbt)보다 느리지만,
"내부에서 무슨 일이 일어나는지"가 코드로 명확히 드러나서 설계를 배우기에 적합합니다.

## 엔진 구조 (run_backtest.py 기준)

| 구성요소 | 역할 | 코드 위치 |
|---|---|---|
| **Cerebro** | 데이터/전략/브로커를 전부 연결하고 실행하는 총괄 엔진 | `run_backtest.py` |
| **Data Feed** | 시세 데이터를 엔진이 읽을 수 있는 형태로 공급 | `run_backtest.py` (`bt.feeds.PandasData`) |
| **Strategy** | 매매 로직 — "언제 사고 언제 팔지"만 정의 | `strategies/sma_cross.py` |
| **Broker** | 주문 체결, 수수료, 초기 자금 관리 | `run_backtest.py` (`cerebro.broker`) |
| **Analyzer** | 성과 분석 — 자산곡선 기록, MDD, 샤프비율, 승률 계산 | `run_backtest.py` (`EquityCurve`, `bt.analyzers.*`) |

각 구성요소가 독립적으로 분리되어 있다는 게 핵심입니다. **전략(Strategy)만 교체하면**
나머지(Cerebro/Data Feed/Broker/Analyzer)는 그대로 재사용할 수 있습니다.
부원들이 직접 전략을 만들 때도 `strategies/` 폴더에 새 파일만 추가하면 됩니다.

## 실제 데이터로 바꾸기

`data/mock_ohlcv.csv`를 컬럼명이 `Date, Open, High, Low, Close, Volume`인 다른 CSV로
교체하기만 하면 `run_backtest.py` 수정 없이 그대로 동작합니다.

## 산출물 스타일 통일

`quant_theme.py`의 `build_backtest_report()`가 자산곡선/낙폭/월별수익률/성과요약표를
동아리 공통 색상·폰트로 렌더링합니다. 다른 전략, 다른 데이터로 백테스트를 돌려도
이 함수 하나만 마지막에 호출하면 항상 같은 톤의 리포트가 나옵니다.
