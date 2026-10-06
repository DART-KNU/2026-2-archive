# [2026-2] 모의투자 포트폴리오 - [19기_윤상원]

## 펀더멘탈-모멘텀 전략 
* **핵심 로직:** 펀더멘탈이 좋은 기업을 선별 후 모멘텀을 비교하여 지속적인 리밸런싱
* **카페 링크:** [펀더멘탈-모멘텀 포트폴리오 전략](https://cafe.naver.com/knudart?iframe_url_utf8=%2Fca-fe%2Fcafes%2F29307656%2Farticles%2F1118%253FreferrerAllArticles%3Dtrue%2526oldPath%3D%252FArticleRead.nhn%253Fclubid%253D29307656%2526articleid%253D1118%2526referrerAllArticles%253Dtrue)

## 파일 설명 
* `Fundamental.py` : Fundamental Universe 확인
* `Momentum.py` : 당일 Momentum Score / Ranking 업데이트
* `Portfolio.py`: 리밸런싱일이면 새로운 Target 생성 / 비리밸런싱일이면 기존 Target 유지
* `backtest.py`: 최종 백테스트
* `daily_signal`: Current Weight 자동 계산, Target과 비교, BUY / SELL / HOLD 생성, Compliance 재검증
