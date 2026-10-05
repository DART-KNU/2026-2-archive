# [2026-2] 모의투자 포트폴리오 - [18기_정진영]

## 전략
* **핵심 로직:** 코어 90% / 위성 10% 섹터 로테이션. 코어는 시장 섹터 비중을 따르면서 섹터 안에서 저변동성 + ROE 상위 종목을 고르고(매주 1/13씩 순환 갱신), 위성은 잔차 모멘텀 상위 섹터의 6-1 모멘텀 + ROE 상위 종목을 매주 회전. 삼성전자·SK하이닉스는 항상 편입, 그 외 종목은 5% 상한. 원래 가설(섹터 모멘텀 + MS-GARCH 레짐 브레이크)을 미래참조 없는 워크포워드 백테스트로 검증한 뒤 수정한 전략이며, MS-GARCH는 위험 모니터링용으로만 사용.
* **백테스트 (2018-01 ~ 2026-09, 비용 차감):** 연 11.4% / 샤프 0.58 / MDD −42.4% (시장 11.5% / 0.55 / −45.0%), 대회 한도 위반 0주
* **카페 링크:** [[18기 전략 보고서] 섹터 중립과 모멘텀을 활용한 코어-위성 전략](https://cafe.naver.com/f-e/cafes/29307656/articles/1113?menuid=55&referrerAllArticles=false)
## 파일 설명
* `strategy_details.md` : 전략·대회 규칙·백테스트 결과·미래참조 방지 방법·실행 방법 상세 설명
* `config.yaml` : 모든 파라미터 (한도, 비용, 코어/위성 비중 등)
* `src/backtest.py` : 주간 리밸런싱 시뮬레이션 (백테스트·실전 공용 매매 로직)
* `src/allocate.py`, `src/rules.py` : 섹터·종목 배분과 대회 규칙(종목·섹터·소형주 한도, 회전율) 점검
* `src/universe.py`, `src/sectors.py`, `src/momentum.py` : 매수 가능 종목, 섹터 수익률, 모멘텀·변동성·ROE 지표
* `src/regime.py`, `src/regime_msgarch.r` : 2-레짐 MS-GJR-GARCH 워크포워드 추정 (R)
* `scripts/build_data.py`, `scripts/run_backtest.py` : 데이터 구축과 백테스트 실행
* `scripts/update_data.py`, `scripts/run_live.py`, `weekly.py` : 대회 기간 주간 데이터 갱신과 주문표 생성
* `tests/test_lookahead.py` : 미래참조 테스트
* `docs/images/` : 결과 그래프
* 데이터(DataGuide)는 저작권 문제로 포함하지 않음 — 받는 항목은 `dataguide_checklist.md` 참고
