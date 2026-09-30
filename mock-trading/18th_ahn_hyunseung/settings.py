from pathlib import Path

# ============================================================
# 0. PATHS
# ============================================================
# 이 파일만 수정하면 전체 파이프라인이 같은 경로를 사용합니다.

BASE_DIR = Path(
    r"C:\Users\sampo\Desktop\python\DART\2학기모의투자\converted_csv"
)

# DataGuide 주가 wide export 후보
STOCK_RAW_CANDIDATES = [
    BASE_DIR / "데이터1.txt",
    BASE_DIR / "붙여넣은 텍스트(1).txt",
    BASE_DIR / "stock_dataguide.txt",
    BASE_DIR / "데이터1.xlsx",
    BASE_DIR / "stock_dataguide.xlsx",
]

# DataGuide Fwd12M 컨센서스 export
CONSENSUS_XLSX = BASE_DIR / "데이터2.xlsx"

# 대회 섹터표
CONTEST_FILE = BASE_DIR / "grid_20260926.csv"

# 중간/결과 파일
RAW_STOCK_PARQUET_DIR = BASE_DIR / "raw_stock_parquet"
CLEAN_STOCK_PARQUET_DIR = BASE_DIR / "cleaned_stock_parquet"
FACTOR_PANEL_FILE = BASE_DIR / "factor_panel.parquet"
CONSENSUS_DIR = BASE_DIR / "consensus_daily"
KOSPI_FILE = BASE_DIR / "KOSPI.csv"
FINAL_OUTPUT_DIR = BASE_DIR / "final_strategy_vs_kospi_backtest"

# ============================================================
# 1. FINAL STRATEGY SPECIFICATION (FROZEN)
# ============================================================

BACKTEST_START = "2022-01-01"

# Universe
MCAP_CUTOFF = 1_000_000_000_000  # 1조원, PIT

# Portfolio
PORTFOLIO_N = 30
REBALANCE_DAYS = 5
OFFSETS = list(range(REBALANCE_DAYS))

# FreshRank hysteresis
ENTRY_CUTOFF = 0.70
HOLD_CUTOFF = 0.40
ENTRY_RELAXATION_LADDER = [0.70, 0.65, 0.60, 0.50, 0.00]

# Mom20 entry confirmation
MOMENTUM_FLOOR_LADDER = [0.30, 0.20, 0.10, 0.00]

# Trading cost
COMMISSION_BPS = 10.0
SELL_TAX_BPS = 20.0

# Sector hard constraints from contest
SECTOR_CAP_MULTIPLIER = 0.90
SECTOR_CAPS = {
    "En": 0.100,
    "Ma": 0.100,
    "In": 0.371,
    "CD": 0.110,
    "CS": 0.100,
    "He": 0.100,
    "Fi": 0.153,
    "IT": 1.085,
    "Co": 0.100,
    "Ut": 0.100,
    "Re": 0.100,
}

# Market regime
RUN_KOSPI_MA60_REGIME = True
REGIME_MA_WINDOW = 60
REGIME_RISK_ON_EXPOSURE = 1.00
REGIME_RISK_OFF_EXPOSURE = 0.80
