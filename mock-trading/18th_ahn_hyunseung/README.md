# [2026-2] 모의투자 포트폴리오 - [18기 안현승]

## 전략
* **핵심 로직:** 중기적 이익전망 개선 및 단기적 컨센서스 상향 종목 선정 전
* **카페 링크:** [[19기 전략보고서] 중기적 이익전망 개선 및 최근 컨센서스 상향 기업 동일배분 전략](https://cafe.naver.com/knudart/1115)

## 파일 설명
* `settings.py` : 전체 프로젝트에서 공통으로 쓰는 경로와 주요 설정
* `00_convert_stock_dataguide.py` : 데이터가이드 원본 데이터 long format 변환
* '01_stock_qc_build_factor_panel.py' : 변환된 데이터 오류 점검 및 패널 데이터 제작
* '02_convert_consensus.py' : 컨센서스 지표들을 추출해 일별 컨센서스 데이터로 변환
* '03_validate_contest_sector.py' : 섹터 파일 읽어오기 및 종목코드 섹터 매핑 검사
* '04_prepare_kospi.py' : KOSPI 지수 준비 및 일별 수익률과 MA60 계산
* '05_final_strategy_vs_kospi_backtest.py' : 전략 백테스트
