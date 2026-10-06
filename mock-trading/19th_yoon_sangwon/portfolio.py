# -*- coding: utf-8 -*-

import os
import re
import warnings
import importlib.util

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

from scipy.optimize import minimize


warnings.filterwarnings("ignore")


# =============================================================================
# 0. PATH
# =============================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKTEST_FILE = os.path.join(
    BASE_DIR,
    "backtest.py"
)

MOMENTUM_DIR = os.path.join(
    BASE_DIR,
    "momentum"
)

MOMENTUM_FILE = os.path.join(
    MOMENTUM_DIR,
    "momentum_scored.csv"
)

MOMENTUM_HISTORY_DIR = os.path.join(
    MOMENTUM_DIR,
    "history"
)

MARCAP_FILE = os.path.join(
    BASE_DIR,
    "marcap",
    "data",
    "marcap-2026.parquet"
)

PORTFOLIO_DIR = os.path.join(
    BASE_DIR,
    "portfolio"
)

PORTFOLIO_HISTORY_DIR = os.path.join(
    PORTFOLIO_DIR,
    "history"
)

PORTFOLIO_INPUT_DIR = os.path.join(
    PORTFOLIO_DIR,
    "input"
)

DIAGNOSTIC_DIR = os.path.join(
    PORTFOLIO_DIR,
    "diagnostics"
)


for folder in [
    PORTFOLIO_DIR,
    PORTFOLIO_HISTORY_DIR,
    PORTFOLIO_INPUT_DIR,
    DIAGNOSTIC_DIR
]:
    os.makedirs(
        folder,
        exist_ok=True
    )


LATEST_TARGET_FILE = os.path.join(
    PORTFOLIO_DIR,
    "portfolio_target_latest.csv"
)

RESTRICTED_FILE = os.path.join(
    PORTFOLIO_INPUT_DIR,
    "restricted_codes.csv"
)

LATEST_SECTOR_CHECK_FILE = os.path.join(
    DIAGNOSTIC_DIR,
    "sector_compliance_latest.csv"
)

LATEST_RULE_CHECK_FILE = os.path.join(
    DIAGNOSTIC_DIR,
    "rule_compliance_latest.csv"
)


# =============================================================================
# 1. STRATEGY
# =============================================================================

# Momentum hysteresis
ENTRY_RANK = 80
EXIT_RANK = 100

# 기본 목표 보유종목 수
TARGET_HOLDINGS = 25


# =============================================================================
# 2. 대회 공식 규칙
# =============================================================================

# -------------------------------------------------------------------------
# 종목별 한도
#
# 일반 종목 15%
# 삼성전자 40%
# -------------------------------------------------------------------------

OFFICIAL_GENERAL_STOCK_CAP = 0.15
OFFICIAL_SAMSUNG_CAP = 0.40


# -------------------------------------------------------------------------
# 섹터별 한도
#
# max(시장 섹터 비중 × 2, 10%)
# -------------------------------------------------------------------------

OFFICIAL_MIN_SECTOR_CAP = 0.10


# -------------------------------------------------------------------------
# 시총 1조 미만 종목 합계
#
# 공식:
# 시가총액 < 1조
# 포트폴리오 합계 <= 30%
# -------------------------------------------------------------------------

OFFICIAL_SMALL_CAP_THRESHOLD = 1_000_000_000_000
OFFICIAL_SMALL_CAP_LIMIT = 0.30


# =============================================================================
# 3. 내부 안전한도
# =============================================================================

# -------------------------------------------------------------------------
# 개별종목
#
# 공식 15%보다 훨씬 보수적으로 5%
# -------------------------------------------------------------------------

INTERNAL_STOCK_CAP = 0.05


# -------------------------------------------------------------------------
# 섹터
#
# 공식 섹터 한도의 90%까지만 사용
#
# 예)
# 공식 10% -> 내부 9%
# 공식 20% -> 내부 18%
# -------------------------------------------------------------------------

SECTOR_LIMIT_USAGE_RATIO = 0.90


# -------------------------------------------------------------------------
# 시총
#
# 공식:
# < 1.0조 합계 <= 30%
#
# 내부:
# < 1.1조 합계 <= 25%
#
# 1조 근처 종목이 가격하락으로
# 갑자기 공식 small-cap 그룹으로 진입하는 위험을 줄인다.
# -------------------------------------------------------------------------

INTERNAL_SMALL_CAP_THRESHOLD = 1_100_000_000_000
INTERNAL_SMALL_CAP_LIMIT = 0.25


# =============================================================================
# 4. 개별 종목 매수 가능 기준
# =============================================================================

# 시총 1,000억원 이상
MIN_MARKET_CAP = 100_000_000_000

# 최근 5일 평균 거래대금 30억원 초과
MIN_AVG_TRADING_VALUE_5D = 3_000_000_000

# Momentum 전략 자체의 최소 가격관측치
MIN_PRICE_OBS = 61

# 신규상장 최소 6영업일
MIN_LISTING_DAYS = 6


# =============================================================================
# 5. DATA SAFETY
# =============================================================================

MAX_MARCAP_LAG_DAYS = 10
MAX_INDEX_LAG_DAYS = 5

PRICE_LOOKBACK_DAYS = 220


# =============================================================================
# 6. HELPERS
# =============================================================================

def normalize_code(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    digits = re.sub(
        r"\D",
        "",
        text
    )

    if not digits:
        return None

    return digits.zfill(6)[-6:]


def first_existing_column(
    df,
    candidates
):

    for col in candidates:

        if col in df.columns:
            return col

    return None


# =============================================================================
# 7. BACKTEST MODULE
#
# 기존 백테스트에서
#
# - sector grid
# - 95 / 80 / 60 market regime
#
# 만 재사용한다.
#
# Portfolio optimizer는 여기서 직접 구현한다.
# =============================================================================

def load_backtest_module():

    if not os.path.exists(
        BACKTEST_FILE
    ):

        raise FileNotFoundError(
            f"backtest.py를 찾을 수 없습니다.\n"
            f"{BACKTEST_FILE}"
        )


    spec = importlib.util.spec_from_file_location(
        "bt",
        BACKTEST_FILE
    )


    if (
        spec is None
        or
        spec.loader is None
    ):

        raise RuntimeError(
            "backtest.py를 불러올 수 없습니다."
        )


    bt = importlib.util.module_from_spec(
        spec
    )


    spec.loader.exec_module(
        bt
    )


    for function_name in [
        "load_grid",
        "get_equity_exposure"
    ]:

        if not hasattr(
            bt,
            function_name
        ):

            raise RuntimeError(
                f"backtest.py에 "
                f"{function_name} 함수가 없습니다."
            )


    return bt


bt = load_backtest_module()


# =============================================================================
# 8. RESTRICTED CODES
# =============================================================================

def ensure_restricted_file():

    if os.path.exists(
        RESTRICTED_FILE
    ):

        return


    pd.DataFrame(
        columns=[
            "종목코드",
            "상태",
            "비고"
        ]
    ).to_csv(
        RESTRICTED_FILE,
        index=False,
        encoding="utf-8-sig"
    )


def load_manual_restricted():

    ensure_restricted_file()


    try:

        df = pd.read_csv(
            RESTRICTED_FILE,
            dtype=str
        )


    except Exception:

        return set()


    if (
        df.empty
        or
        "종목코드"
        not in df.columns
    ):

        return set()


    result = set()


    for value in df[
        "종목코드"
    ]:

        code = normalize_code(
            value
        )


        if code is not None:

            result.add(
                code
            )


    return result


# =============================================================================
# 9. MOMENTUM
# =============================================================================

def load_momentum():

    print()
    print("[1] Momentum 데이터")


    if not os.path.exists(
        MOMENTUM_FILE
    ):

        raise FileNotFoundError(
            f"{MOMENTUM_FILE}\n"
            f"먼저 momentum.py를 실행하세요."
        )


    df = pd.read_csv(
        MOMENTUM_FILE,
        dtype={
            "종목코드": str
        }
    )


    required = {
        "종목코드",
        "MomentumScore"
    }


    if not required.issubset(
        df.columns
    ):

        raise RuntimeError(
            "momentum_scored.csv에 "
            "종목코드 / MomentumScore가 없습니다."
        )


    df[
        "종목코드"
    ] = (
        df[
            "종목코드"
        ]
        .apply(
            normalize_code
        )
    )


    df[
        "MomentumScore"
    ] = pd.to_numeric(
        df[
            "MomentumScore"
        ],
        errors="coerce"
    )


    # =========================================================================
    # Absolute Momentum Pass
    # =========================================================================

    if "AbsoluteMomentumPass" in df.columns:

        values = (
            df[
                "AbsoluteMomentumPass"
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )


        mask = values.isin(
            [
                "true",
                "1",
                "yes",
                "y",
                "pass"
            ]
        )


        df = df[
            mask
        ].copy()


        print(
            "Absolute Momentum Pass 컬럼 사용: "
            "AbsoluteMomentumPass"
        )


    else:

        print(
            "[주의] AbsoluteMomentumPass 컬럼 없음"
        )


        df = df[
            df[
                "MomentumScore"
            ].notna()
        ].copy()


    df = (
        df
        .dropna(
            subset=[
                "종목코드",
                "MomentumScore"
            ]
        )
        .drop_duplicates(
            "종목코드"
        )
        .sort_values(
            "MomentumScore",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    df[
        "MomentumRank"
    ] = np.arange(
        1,
        len(
            df
        ) + 1
    )


    print(
        f"Absolute Momentum universe: "
        f"{len(df)}개"
    )


    if len(
        df
    ) < ENTRY_RANK:

        raise RuntimeError(
            "Absolute Momentum 후보가 "
            "80개보다 적습니다."
        )


    return df


# =============================================================================
# 10. SIGNAL DATE
# =============================================================================

def determine_signal_date(
    momentum
):

    print()
    print("[2] Signal Date")


    # =========================================================================
    # momentum_scored.csv 내부 날짜 우선
    # =========================================================================

    date_columns = [
        "기준일",
        "기준일자",
        "SignalDate",
        "Date",
        "PriceDate"
    ]


    for col in date_columns:

        if col not in momentum.columns:
            continue


        dates = pd.to_datetime(
            momentum[
                col
            ],
            errors="coerce"
        ).dropna()


        if dates.empty:
            continue


        signal_date = pd.Timestamp(
            dates.max()
        ).normalize()


        print(
            f"Signal Date : "
            f"{signal_date.date()}"
        )


        print(
            "Source      : momentum_scored.csv"
        )


        return signal_date


    # =========================================================================
    # history filename fallback
    # =========================================================================

    dates = []


    pattern = re.compile(
        r"momentum_scored_(\d{4}-\d{2}-\d{2})\.csv$"
    )


    if os.path.isdir(
        MOMENTUM_HISTORY_DIR
    ):

        for filename in os.listdir(
            MOMENTUM_HISTORY_DIR
        ):

            match = pattern.match(
                filename
            )


            if match:

                dates.append(
                    pd.Timestamp(
                        match.group(
                            1
                        )
                    )
                )


    if not dates:

        raise RuntimeError(
            "Momentum 기준일을 찾을 수 없습니다."
        )


    signal_date = max(
        dates
    ).normalize()


    print(
        f"Signal Date : "
        f"{signal_date.date()}"
    )


    print(
        "Source      : momentum history"
    )


    return signal_date


# =============================================================================
# 11. MARCAP
# =============================================================================

def load_marcap(
    signal_date
):

    print()
    print("[3] Marcap / 상장주식수")


    if not os.path.exists(
        MARCAP_FILE
    ):

        raise FileNotFoundError(
            f"marcap 파일 없음:\n"
            f"{MARCAP_FILE}"
        )


    df = pd.read_parquet(
        MARCAP_FILE
    )


    required = {
        "Date",
        "Code",
        "Name",
        "Market",
        "Marcap",
        "Stocks"
    }


    if not required.issubset(
        df.columns
    ):

        raise RuntimeError(
            "marcap parquet 필수 컬럼 누락"
        )


    df[
        "Date"
    ] = pd.to_datetime(
        df[
            "Date"
        ],
        errors="coerce"
    )


    usable_dates = (
        df.loc[
            df[
                "Date"
            ]
            <=
            signal_date,
            "Date"
        ]
        .dropna()
        .unique()
    )


    if len(
        usable_dates
    ) == 0:

        raise RuntimeError(
            "Signal Date 이전 marcap 데이터가 없습니다."
        )


    marcap_date = pd.Timestamp(
        max(
            usable_dates
        )
    ).normalize()


    lag = (
        signal_date
        -
        marcap_date
    ).days


    print(
        f"Signal Date : "
        f"{signal_date.date()}"
    )


    print(
        f"Marcap Date : "
        f"{marcap_date.date()}"
    )


    print(
        f"Marcap Lag  : "
        f"{lag} calendar day(s)"
    )


    if lag > MAX_MARCAP_LAG_DAYS:

        raise RuntimeError(
            "marcap 데이터가 너무 오래됐습니다."
        )


    latest = df[
        df[
            "Date"
        ]
        ==
        marcap_date
    ].copy()


    latest = latest[
        latest[
            "Market"
        ]
        .isin(
            [
                "KOSPI",
                "KOSDAQ"
            ]
        )
    ].copy()


    latest[
        "Code"
    ] = (
        latest[
            "Code"
        ]
        .apply(
            normalize_code
        )
    )


    latest[
        "Marcap"
    ] = pd.to_numeric(
        latest[
            "Marcap"
        ],
        errors="coerce"
    )


    latest[
        "Stocks"
    ] = pd.to_numeric(
        latest[
            "Stocks"
        ],
        errors="coerce"
    )


    latest = (
        latest
        .dropna(
            subset=[
                "Code",
                "Marcap",
                "Stocks"
            ]
        )
        .drop_duplicates(
            "Code"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        f"KOSPI/KOSDAQ 종목: "
        f"{len(latest)}개"
    )


    print(
        f"Marcap 결측치: "
        f"{latest['Marcap'].isna().sum()}"
    )


    print(
        f"Stocks 결측치: "
        f"{latest['Stocks'].isna().sum()}"
    )


    return (
        latest,
        marcap_date
    )


# =============================================================================
# 12. SECTOR LIMITS
# =============================================================================

def load_sector_info(
    listing
):

    print()
    print("[4] Sector Grid / Limits")


    grid = bt.load_grid().copy()


    required = {
        "종목코드",
        "섹터명"
    }


    if not required.issubset(
        grid.columns
    ):

        raise RuntimeError(
            "Sector Grid 컬럼 오류"
        )


    grid[
        "종목코드"
    ] = (
        grid[
            "종목코드"
        ]
        .apply(
            normalize_code
        )
    )


    grid = (
        grid
        .dropna(
            subset=[
                "종목코드",
                "섹터명"
            ]
        )
        .drop_duplicates(
            "종목코드"
        )
    )


    sector_map = dict(
        zip(
            grid[
                "종목코드"
            ],
            grid[
                "섹터명"
            ]
        )
    )


    merged = listing.merge(
        grid[
            [
                "종목코드",
                "섹터명"
            ]
        ],
        left_on="Code",
        right_on="종목코드",
        how="left"
    )


    total_market_cap = float(
        merged[
            "Marcap"
        ].sum()
    )


    if (
        not np.isfinite(
            total_market_cap
        )
        or
        total_market_cap
        <=
        0
    ):

        raise RuntimeError(
            "시장 전체 시가총액 계산 실패"
        )


    sector_market_cap = (
        merged
        .dropna(
            subset=[
                "섹터명"
            ]
        )
        .groupby(
            "섹터명"
        )[
            "Marcap"
        ]
        .sum()
    )


    sector_market_weight = (
        sector_market_cap
        /
        total_market_cap
    )


    rows = []


    for (
        sector,
        market_weight
    ) in sector_market_weight.items():


        # ---------------------------------------------------------------------
        # 공식
        # max(시장섹터비중 × 2, 10%)
        # ---------------------------------------------------------------------

        official_limit = max(
            float(
                market_weight
            )
            *
            2.0,

            OFFICIAL_MIN_SECTOR_CAP
        )


        # 포트폴리오는 100% 초과 불가능
        official_limit = min(
            official_limit,
            1.0
        )


        # ---------------------------------------------------------------------
        # 내부 버퍼
        # 공식한도의 90%
        # ---------------------------------------------------------------------

        internal_limit = (
            official_limit
            *
            SECTOR_LIMIT_USAGE_RATIO
        )


        rows.append(
            {
                "섹터명":
                    str(
                        sector
                    ),

                "시장비중":
                    float(
                        market_weight
                    ),

                "공식한도":
                    float(
                        official_limit
                    ),

                "내부한도":
                    float(
                        internal_limit
                    )
            }
        )


    sector_table = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "시장비중",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    internal_limits = dict(
        zip(
            sector_table[
                "섹터명"
            ],
            sector_table[
                "내부한도"
            ]
        )
    )


    display = sector_table.copy()


    for col in [
        "시장비중",
        "공식한도",
        "내부한도"
    ]:

        display[
            col
        ] *= 100


    print()
    print(
        display
        .rename(
            columns={
                "시장비중":
                    "시장비중(%)",

                "공식한도":
                    "공식한도(%)",

                "내부한도":
                    "내부한도(%)"
            }
        )
        .round(
            2
        )
        .to_string(
            index=False
        )
    )


    return (
        sector_map,
        sector_table,
        internal_limits
    )


# =============================================================================
# 13. ADMIN
# =============================================================================

def load_admin_codes():

    print()
    print("[5] KRX-ADMIN")


    try:

        df = fdr.StockListing(
            "KRX-ADMIN"
        )


    except Exception as e:

        print(
            "[주의] KRX-ADMIN 조회 실패"
        )


        print(
            e
        )


        return set()


    code_col = first_existing_column(
        df,
        [
            "Code",
            "Symbol",
            "종목코드"
        ]
    )


    if code_col is None:

        return set()


    result = set()


    for value in df[
        code_col
    ]:

        code = normalize_code(
            value
        )


        if code is not None:

            result.add(
                code
            )


    print(
        f"현재 KRX-ADMIN: "
        f"{len(result)}개"
    )


    return result


# =============================================================================
# 14. PREVIOUS TARGET
# =============================================================================

def load_previous_target():

    if not os.path.exists(
        LATEST_TARGET_FILE
    ):

        return (
            {},
            None
        )


    df = pd.read_csv(
        LATEST_TARGET_FILE,
        dtype={
            "종목코드": str
        }
    )


    if (
        df.empty
        or
        "종목코드"
        not in df.columns
        or
        "TargetWeight"
        not in df.columns
    ):

        return (
            {},
            None
        )


    df[
        "종목코드"
    ] = (
        df[
            "종목코드"
        ]
        .apply(
            normalize_code
        )
    )


    df[
        "TargetWeight"
    ] = pd.to_numeric(
        df[
            "TargetWeight"
        ],
        errors="coerce"
    ).fillna(
        0
    )


    target = {}


    for (
        code,
        weight
    ) in zip(
        df[
            "종목코드"
        ],
        df[
            "TargetWeight"
        ]
    ):

        if (
            code is not None
            and
            weight > 0
        ):

            target[
                code
            ] = float(
                weight
            )


    target_date = None


    if "TargetDate" in df.columns:

        dates = pd.to_datetime(
            df[
                "TargetDate"
            ],
            errors="coerce"
        ).dropna()


        if not dates.empty:

            target_date = pd.Timestamp(
                dates.iloc[
                    0
                ]
            ).normalize()


    return (
        target,
        target_date
    )


# =============================================================================
# 15. REBALANCE SCHEDULE
# =============================================================================

def load_trading_dates(
    signal_date
):

    start = (
        signal_date
        -
        pd.Timedelta(
            days=100
        )
    )


    df = fdr.DataReader(
        "005930",
        start.strftime(
            "%Y-%m-%d"
        ),
        (
            signal_date
            +
            pd.Timedelta(
                days=1
            )
        ).strftime(
            "%Y-%m-%d"
        )
    )


    if df is None or df.empty:

        raise RuntimeError(
            "거래일 데이터 없음"
        )


    dates = pd.DatetimeIndex(
        pd.to_datetime(
            df.index
        )
    ).normalize()


    return dates[
        dates
        <=
        signal_date
    ]


def build_rebalance_dates(
    trading_dates
):

    temp = pd.DataFrame(
        {
            "Date":
                trading_dates
        }
    )


    iso = (
        temp[
            "Date"
        ]
        .dt
        .isocalendar()
    )


    temp[
        "Year"
    ] = iso[
        "year"
    ]


    temp[
        "Week"
    ] = iso[
        "week"
    ]


    temp[
        "Weekday"
    ] = (
        temp[
            "Date"
        ]
        .dt
        .weekday
    )


    result = set()


    for _, group in temp.groupby(
        [
            "Year",
            "Week"
        ]
    ):

        group = group.sort_values(
            "Date"
        )


        # 월요일 또는 해당 주 첫 거래일
        first = pd.Timestamp(
            group.iloc[
                0
            ][
                "Date"
            ]
        ).normalize()


        result.add(
            first
        )


        # 목요일 이후 첫 거래일
        second_group = group[
            group[
                "Weekday"
            ]
            >=
            3
        ]


        if not second_group.empty:

            second = pd.Timestamp(
                second_group.iloc[
                    0
                ][
                    "Date"
                ]
            ).normalize()


            result.add(
                second
            )


    return result


# =============================================================================
# 16. MARKET INDEX
# =============================================================================

def load_index(
    label,
    tickers,
    signal_date
):

    start = (
        signal_date
        -
        pd.Timedelta(
            days=220
        )
    )


    best = None
    best_ticker = None
    best_date = None


    for ticker in tickers:

        try:

            df = fdr.DataReader(
                ticker,
                start.strftime(
                    "%Y-%m-%d"
                ),
                (
                    signal_date
                    +
                    pd.Timedelta(
                        days=1
                    )
                ).strftime(
                    "%Y-%m-%d"
                )
            )


        except Exception:

            continue


        if (
            df is None
            or
            df.empty
            or
            "Close"
            not in df.columns
        ):

            continue


        df = df.copy()


        df.index = pd.to_datetime(
            df.index
        )


        df = df[
            df.index
            <=
            signal_date
        ]


        if df.empty:

            continue


        last_date = pd.Timestamp(
            df.index[
                -1
            ]
        ).normalize()


        if (
            best_date is None
            or
            last_date > best_date
        ):

            best = df
            best_ticker = ticker
            best_date = last_date


    if best is None:

        raise RuntimeError(
            f"{label} 지수를 가져올 수 없습니다."
        )


    print(
        f"{label} Source    : "
        f"{best_ticker}"
    )


    print(
        f"{label} Last Date : "
        f"{best_date.date()}"
    )


    lag = (
        signal_date
        -
        best_date
    ).days


    if lag > MAX_INDEX_LAG_DAYS:

        raise RuntimeError(
            f"{label} 지수 데이터가 너무 오래됐습니다."
        )


    return best


def load_market_indices(
    signal_date
):

    print()
    print("[6] Market Regime Index")


    kospi = load_index(
        "KOSPI",
        [
            "YAHOO:^KS11",
            "^KS11",
            "KS11"
        ],
        signal_date
    )


    kosdaq = load_index(
        "KOSDAQ",
        [
            "YAHOO:^KQ11",
            "^KQ11",
            "KQ11"
        ],
        signal_date
    )


    return (
        kospi,
        kosdaq
    )


# =============================================================================
# 17. HYSTERESIS
# =============================================================================

def build_candidates(
    momentum,
    previous_target
):

    entry = momentum[
        momentum[
            "MomentumRank"
        ]
        <=
        ENTRY_RANK
    ].copy()


    retention = momentum[
        momentum[
            "MomentumRank"
        ]
        <=
        EXIT_RANK
    ].copy()


    retained_previous = (
        set(
            previous_target.keys()
        )
        &
        set(
            retention[
                "종목코드"
            ]
        )
    )


    candidate_codes = (
        set(
            entry[
                "종목코드"
            ]
        )
        |
        retained_previous
    )


    candidates = (
        momentum[
            momentum[
                "종목코드"
            ]
            .isin(
                candidate_codes
            )
        ]
        .copy()
        .sort_values(
            "MomentumScore",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    return (
        candidates,
        len(
            entry
        ),
        len(
            retained_previous
        )
    )


# =============================================================================
# 18. PRICE DOWNLOAD
# =============================================================================

def download_prices(
    codes,
    signal_date
):

    print()
    print("[7] 후보종목 가격 / 거래량")


    start = (
        signal_date
        -
        pd.Timedelta(
            days=PRICE_LOOKBACK_DAYS
        )
    )


    end = (
        signal_date
        +
        pd.Timedelta(
            days=1
        )
    )


    codes = list(
        dict.fromkeys(
            codes
        )
    )


    result = {}


    total = len(
        codes
    )


    for i, code in enumerate(
        codes,
        start=1
    ):

        try:

            df = fdr.DataReader(
                code,
                start.strftime(
                    "%Y-%m-%d"
                ),
                end.strftime(
                    "%Y-%m-%d"
                )
            )


        except Exception:

            continue


        if (
            df is None
            or
            df.empty
            or
            "Close"
            not in df.columns
            or
            "Volume"
            not in df.columns
        ):

            continue


        df = df.copy()


        df.index = pd.to_datetime(
            df.index
        )


        df = (
            df[
                df.index
                <=
                signal_date
            ]
            .sort_index()
        )


        if not df.empty:

            result[
                code
            ] = df


        if (
            i % 20 == 0
            or
            i == total
        ):

            print(
                f"{i}/{total}"
            )


    print(
        f"가격 확보: "
        f"{len(result)}/{total}개"
    )


    return result


# =============================================================================
# 19. LIVE METRICS
# =============================================================================

def get_close(
    code,
    signal_date,
    price_data
):

    if code not in price_data:

        return np.nan


    df = price_data[
        code
    ]


    df = df[
        df.index
        <=
        signal_date
    ]


    if df.empty:

        return np.nan


    return pd.to_numeric(
        df[
            "Close"
        ].iloc[
            -1
        ],
        errors="coerce"
    )


def get_obs(
    code,
    signal_date,
    price_data
):

    if code not in price_data:

        return 0


    return int(
        len(
            price_data[
                code
            ][
                price_data[
                    code
                ].index
                <=
                signal_date
            ]
        )
    )


def get_avg_value_5d(
    code,
    signal_date,
    price_data
):

    if code not in price_data:

        return np.nan


    df = price_data[
        code
    ]


    df = df[
        df.index
        <=
        signal_date
    ]


    if len(
        df
    ) < 5:

        return np.nan


    close = pd.to_numeric(
        df[
            "Close"
        ],
        errors="coerce"
    )


    volume = pd.to_numeric(
        df[
            "Volume"
        ],
        errors="coerce"
    )


    amount = (
        close
        *
        volume
    ).dropna()


    if len(
        amount
    ) < 5:

        return np.nan


    return float(
        amount.iloc[
            -5:
        ].mean()
    )


def get_market_cap(
    code,
    signal_date,
    price_data,
    listing_map
):

    if code not in listing_map.index:

        return np.nan


    info = listing_map.loc[
        code
    ]


    if isinstance(
        info,
        pd.DataFrame
    ):

        info = info.iloc[
            0
        ]


    stocks = pd.to_numeric(
        info[
            "Stocks"
        ],
        errors="coerce"
    )


    close = get_close(
        code,
        signal_date,
        price_data
    )


    if (
        pd.isna(
            stocks
        )
        or
        pd.isna(
            close
        )
    ):

        return np.nan


    return float(
        stocks
        *
        close
    )


# =============================================================================
# 20. TRADABILITY
# =============================================================================

def build_tradable(
    candidates,
    signal_date,
    price_data,
    listing_map,
    sector_map,
    admin_codes,
    manual_restricted
):

    passed = []
    rejected = []


    for _, row in candidates.iterrows():

        code = row[
            "종목코드"
        ]


        reasons = []


        # =========================================================================
        # Listing
        # =========================================================================

        if code not in listing_map.index:

            reasons.append(
                "NOT_IN_MARCAP"
            )


            name = ""
            market = ""


        else:

            info = listing_map.loc[
                code
            ]


            if isinstance(
                info,
                pd.DataFrame
            ):

                info = info.iloc[
                    0
                ]


            name = info[
                "Name"
            ]


            market = info[
                "Market"
            ]


        # =========================================================================
        # Price observation / new listing
        # =========================================================================

        obs = get_obs(
            code,
            signal_date,
            price_data
        )


        if obs < MIN_LISTING_DAYS:

            reasons.append(
                "NEW_LISTING_LT6"
            )


        if obs < MIN_PRICE_OBS:

            reasons.append(
                "OBS_LT61"
            )


        # =========================================================================
        # Market cap
        # =========================================================================

        market_cap = get_market_cap(
            code,
            signal_date,
            price_data,
            listing_map
        )


        if (
            pd.isna(
                market_cap
            )
            or
            market_cap
            <
            MIN_MARKET_CAP
        ):

            reasons.append(
                "MARKET_CAP"
            )


        # =========================================================================
        # Liquidity
        # =========================================================================

        trading_value = get_avg_value_5d(
            code,
            signal_date,
            price_data
        )


        if (
            pd.isna(
                trading_value
            )
            or
            trading_value
            <=
            MIN_AVG_TRADING_VALUE_5D
        ):

            reasons.append(
                "LIQUIDITY"
            )


        # =========================================================================
        # Restrictions
        # =========================================================================

        if code in admin_codes:

            reasons.append(
                "KRX_ADMIN"
            )


        if code in manual_restricted:

            reasons.append(
                "MANUAL_RESTRICTED"
            )


        # =========================================================================
        # Sector
        # =========================================================================

        sector = sector_map.get(
            code
        )


        if (
            sector is None
            or
            pd.isna(
                sector
            )
        ):

            reasons.append(
                "NO_SECTOR"
            )


        # =========================================================================
        # Row
        # =========================================================================

        record = row.to_dict()


        record.update(
            {
                "종목명":
                    name,

                "시장":
                    market,

                "섹터명":
                    sector,

                "가격관측수":
                    obs,

                "SignalClose":
                    get_close(
                        code,
                        signal_date,
                        price_data
                    ),

                "시가총액":
                    market_cap,

                "5일평균거래대금":
                    trading_value
            }
        )


        if reasons:

            record[
                "RejectReason"
            ] = ";".join(
                sorted(
                    set(
                        reasons
                    )
                )
            )


            rejected.append(
                record
            )


        else:

            passed.append(
                record
            )


    tradable = pd.DataFrame(
        passed
    )


    rejected = pd.DataFrame(
        rejected
    )


    if not tradable.empty:

        tradable = (
            tradable
            .sort_values(
                "MomentumScore",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


    if not rejected.empty:

        rejected = (
            rejected
            .sort_values(
                "MomentumRank"
            )
            .reset_index(
                drop=True
            )
        )


    return (
        tradable,
        rejected
    )


# =============================================================================
# 21. PORTFOLIO OPTIMIZER
# =============================================================================

def solve_portfolio(
    selected,
    requested_exposure,
    internal_sector_limits
):

    selected = (
        selected
        .copy()
        .reset_index(
            drop=True
        )
    )


    n = len(
        selected
    )


    if n == 0:

        return (
            None,
            False
        )


    # =========================================================================
    # Momentum raw weight
    # =========================================================================

    scores = pd.to_numeric(
        selected[
            "MomentumScore"
        ],
        errors="coerce"
    ).clip(
        lower=0
    )


    if (
        scores.isna().any()
        or
        scores.sum()
        <=
        0
    ):

        return (
            None,
            False
        )


    raw = (
        scores.to_numpy(
            dtype=float
        )
        /
        float(
            scores.sum()
        )
        *
        requested_exposure
    )


    # =========================================================================
    # Initial point
    # =========================================================================

    x0 = np.minimum(
        raw,
        INTERNAL_STOCK_CAP
    )


    if x0.sum() <= 0:

        return (
            None,
            False
        )


    # 기본 equal-ish feasible starting point도 허용
    x0 = np.full(
        n,
        requested_exposure
        /
        n
    )


    x0 = np.minimum(
        x0,
        INTERNAL_STOCK_CAP
    )


    # =========================================================================
    # Bounds
    # =========================================================================

    bounds = [
        (
            0.0,
            INTERNAL_STOCK_CAP
        )

        for _ in range(
            n
        )
    ]


    constraints = []


    # =========================================================================
    # Total exposure
    # =========================================================================

    constraints.append(
        {
            "type":
                "eq",

            "fun":
                lambda w:
                np.sum(
                    w
                )
                -
                requested_exposure
        }
    )


    # =========================================================================
    # Sector constraint
    # =========================================================================

    sectors = (
        selected[
            "섹터명"
        ]
        .astype(
            str
        )
        .to_numpy()
    )


    for sector in sorted(
        set(
            sectors
        )
    ):

        if sector not in internal_sector_limits:

            return (
                None,
                False
            )


        mask = (
            sectors
            ==
            sector
        ).astype(
            float
        )


        limit = float(
            internal_sector_limits[
                sector
            ]
        )


        constraints.append(
            {
                "type":
                    "ineq",

                "fun":
                    lambda w,
                    m=mask,
                    lim=limit:
                    lim
                    -
                    np.dot(
                        w,
                        m
                    )
            }
        )


    # =========================================================================
    # INTERNAL small-cap constraint
    #
    # < 1.1조
    # 합계 <= 25%
    # =========================================================================

    market_caps = pd.to_numeric(
        selected[
            "시가총액"
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )


    if np.isnan(
        market_caps
    ).any():

        return (
            None,
            False
        )


    internal_small_mask = (
        market_caps
        <
        INTERNAL_SMALL_CAP_THRESHOLD
    ).astype(
        float
    )


    constraints.append(
        {
            "type":
                "ineq",

            "fun":
                lambda w,
                m=internal_small_mask:
                INTERNAL_SMALL_CAP_LIMIT
                -
                np.dot(
                    w,
                    m
                )
        }
    )


    # =========================================================================
    # 공식 small-cap constraint도 별도 적용
    #
    # < 1조
    # 합계 <= 30%
    # =========================================================================

    official_small_mask = (
        market_caps
        <
        OFFICIAL_SMALL_CAP_THRESHOLD
    ).astype(
        float
    )


    constraints.append(
        {
            "type":
                "ineq",

            "fun":
                lambda w,
                m=official_small_mask:
                OFFICIAL_SMALL_CAP_LIMIT
                -
                np.dot(
                    w,
                    m
                )
        }
    )


    # =========================================================================
    # Objective
    #
    # Momentum raw weights와 가장 가까운 feasible portfolio
    # =========================================================================

    def objective(
        weights
    ):

        diff = (
            weights
            -
            raw
        )


        return float(
            np.dot(
                diff,
                diff
            )
        )


    # =========================================================================
    # Optimize
    # =========================================================================

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter":
                4000,

            "ftol":
                1e-12,

            "disp":
                False
        }
    )


    if not result.success:

        return (
            None,
            False
        )


    weights = np.asarray(
        result.x,
        dtype=float
    )


    tolerance = 2e-6


    # =========================================================================
    # Exposure
    # =========================================================================

    if abs(
        float(
            weights.sum()
        )
        -
        requested_exposure
    ) > tolerance:

        return (
            None,
            False
        )


    # =========================================================================
    # Stock cap
    # =========================================================================

    if (
        float(
            weights.max()
        )
        >
        INTERNAL_STOCK_CAP
        +
        tolerance
    ):

        return (
            None,
            False
        )


    # =========================================================================
    # Sector check
    # =========================================================================

    for sector in set(
        sectors
    ):

        mask = (
            sectors
            ==
            sector
        )


        actual = float(
            weights[
                mask
            ].sum()
        )


        limit = float(
            internal_sector_limits[
                sector
            ]
        )


        if (
            actual
            >
            limit
            +
            tolerance
        ):

            return (
                None,
                False
            )


    # =========================================================================
    # Internal small cap
    # =========================================================================

    internal_small_weight = float(
        weights[
            internal_small_mask.astype(
                bool
            )
        ].sum()
    )


    if (
        internal_small_weight
        >
        INTERNAL_SMALL_CAP_LIMIT
        +
        tolerance
    ):

        return (
            None,
            False
        )


    # =========================================================================
    # Official small cap
    # =========================================================================

    official_small_weight = float(
        weights[
            official_small_mask.astype(
                bool
            )
        ].sum()
    )


    if (
        official_small_weight
        >
        OFFICIAL_SMALL_CAP_LIMIT
        +
        tolerance
    ):

        return (
            None,
            False
        )


    # =========================================================================
    # Output
    # =========================================================================

    selected[
        "RawMomentumWeight"
    ] = raw


    selected[
        "TargetWeight"
    ] = weights


    selected = selected[
        selected[
            "TargetWeight"
        ]
        >
        1e-8
    ].copy()


    selected = (
        selected
        .sort_values(
            "TargetWeight",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    return (
        selected,
        True
    )


# =============================================================================
# 22. TOP25 → EXPAND IF NEEDED
# =============================================================================

def build_portfolio(
    tradable,
    requested_exposure,
    internal_sector_limits
):

    ordered = (
        tradable
        .sort_values(
            "MomentumScore",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    if len(
        ordered
    ) < TARGET_HOLDINGS:

        return (
            None,
            False,
            None
        )


    # =========================================================================
    # Top25부터
    #
    # 규정상 feasible하지 않으면
    # Top26, Top27 ... 순서로 확장
    # =========================================================================

    for candidate_count in range(
        TARGET_HOLDINGS,
        len(
            ordered
        ) + 1
    ):

        selected = ordered.iloc[
            :candidate_count
        ].copy()


        portfolio, success = solve_portfolio(
            selected,
            requested_exposure,
            internal_sector_limits
        )


        if success:

            return (
                portfolio,
                True,
                candidate_count
            )


    return (
        None,
        False,
        None
    )


# =============================================================================
# 23. FINAL COMPLIANCE
# =============================================================================

def validate_compliance(
    portfolio,
    requested_exposure,
    sector_table
):

    print()
    print(
        "[FINAL COMPLIANCE]"
    )


    p = portfolio.copy()


    p[
        "TargetWeight"
    ] = pd.to_numeric(
        p[
            "TargetWeight"
        ],
        errors="coerce"
    )


    p[
        "시가총액"
    ] = pd.to_numeric(
        p[
            "시가총액"
        ],
        errors="coerce"
    )


    if (
        p[
            "TargetWeight"
        ].isna().any()
        or
        p[
            "시가총액"
        ].isna().any()
    ):

        raise RuntimeError(
            "최종 Compliance 데이터에 NaN 존재"
        )


    # =========================================================================
    # Exposure
    # =========================================================================

    target_exposure = float(
        p[
            "TargetWeight"
        ].sum()
    )


    if abs(
        target_exposure
        -
        requested_exposure
    ) > 0.0025:

        raise RuntimeError(
            "Target Exposure 불일치"
        )


    # =========================================================================
    # Stock cap
    # =========================================================================

    for _, row in p.iterrows():

        code = row[
            "종목코드"
        ]


        weight = float(
            row[
                "TargetWeight"
            ]
        )


        official_cap = (
            OFFICIAL_SAMSUNG_CAP

            if code
            ==
            "005930"

            else
            OFFICIAL_GENERAL_STOCK_CAP
        )


        if (
            weight
            >
            official_cap
            +
            1e-8
        ):

            raise RuntimeError(
                f"공식 종목비중 한도 위반: "
                f"{code}"
            )


        if (
            weight
            >
            INTERNAL_STOCK_CAP
            +
            1e-8
        ):

            raise RuntimeError(
                f"내부 종목비중 한도 위반: "
                f"{code}"
            )


    # =========================================================================
    # Sector
    # =========================================================================

    sector_lookup = (
        sector_table
        .set_index(
            "섹터명"
        )
    )


    actual_sector = (
        p
        .groupby(
            "섹터명"
        )[
            "TargetWeight"
        ]
        .sum()
    )


    sector_rows = []


    for (
        sector,
        actual_weight
    ) in actual_sector.items():

        if sector not in sector_lookup.index:

            raise RuntimeError(
                f"섹터 정보 없음: "
                f"{sector}"
            )


        info = sector_lookup.loc[
            sector
        ]


        market_weight = float(
            info[
                "시장비중"
            ]
        )


        official_limit = float(
            info[
                "공식한도"
            ]
        )


        internal_limit = float(
            info[
                "내부한도"
            ]
        )


        official_pass = (
            actual_weight
            <=
            official_limit
            +
            1e-8
        )


        internal_pass = (
            actual_weight
            <=
            internal_limit
            +
            1e-8
        )


        sector_rows.append(
            {
                "섹터명":
                    sector,

                "시장비중":
                    market_weight,

                "공식한도":
                    official_limit,

                "내부한도":
                    internal_limit,

                "Target비중":
                    float(
                        actual_weight
                    ),

                "내부PASS":
                    bool(
                        internal_pass
                    ),

                "공식PASS":
                    bool(
                        official_pass
                    )
            }
        )


        if not internal_pass:

            raise RuntimeError(
                f"내부 섹터 한도 위반: "
                f"{sector}"
            )


        if not official_pass:

            raise RuntimeError(
                f"공식 섹터 한도 위반: "
                f"{sector}"
            )


    sector_check = pd.DataFrame(
        sector_rows
    )


    # =========================================================================
    # Small-cap INTERNAL
    #
    # < 1.1조 <= 25%
    # =========================================================================

    internal_small_weight = float(
        p.loc[
            p[
                "시가총액"
            ]
            <
            INTERNAL_SMALL_CAP_THRESHOLD,
            "TargetWeight"
        ]
        .sum()
    )


    if (
        internal_small_weight
        >
        INTERNAL_SMALL_CAP_LIMIT
        +
        1e-8
    ):

        raise RuntimeError(
            "내부 시총 1.1조 미만 비중 한도 위반"
        )


    # =========================================================================
    # Small-cap OFFICIAL
    #
    # < 1조 <= 30%
    # =========================================================================

    official_small_weight = float(
        p.loc[
            p[
                "시가총액"
            ]
            <
            OFFICIAL_SMALL_CAP_THRESHOLD,
            "TargetWeight"
        ]
        .sum()
    )


    if (
        official_small_weight
        >
        OFFICIAL_SMALL_CAP_LIMIT
        +
        1e-8
    ):

        raise RuntimeError(
            "공식 시총 1조 미만 비중 한도 위반"
        )


    # =========================================================================
    # Display
    # =========================================================================

    display = sector_check.copy()


    for col in [
        "시장비중",
        "공식한도",
        "내부한도",
        "Target비중"
    ]:

        display[
            col
        ] *= 100


    print()
    print(
        display
        .rename(
            columns={
                "시장비중":
                    "시장비중(%)",

                "공식한도":
                    "공식한도(%)",

                "내부한도":
                    "내부한도(%)",

                "Target비중":
                    "Target비중(%)"
            }
        )
        .round(
            2
        )
        .to_string(
            index=False
        )
    )


    print()
    print(
        "시총 버퍼 검증"
    )


    print(
        f"내부 기준 (<1.1조) : "
        f"{internal_small_weight * 100:.2f}% "
        f"/ "
        f"{INTERNAL_SMALL_CAP_LIMIT * 100:.2f}%"
    )


    print(
        f"공식 기준 (<1조)   : "
        f"{official_small_weight * 100:.2f}% "
        f"/ "
        f"{OFFICIAL_SMALL_CAP_LIMIT * 100:.2f}%"
    )


    print()
    print(
        "FINAL COMPLIANCE: ALL PASS"
    )


    return (
        sector_check,
        internal_small_weight,
        official_small_weight
    )


# =============================================================================
# 24. SAVE COMPLIANCE
# =============================================================================

def save_compliance(
    sector_check,
    internal_small_weight,
    official_small_weight,
    signal_date
):

    # =========================================================================
    # Sector
    # =========================================================================

    sector_check.to_csv(
        LATEST_SECTOR_CHECK_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    sector_check.to_csv(
        os.path.join(
            DIAGNOSTIC_DIR,
            (
                f"sector_compliance_"
                f"{signal_date:%Y-%m-%d}.csv"
            )
        ),
        index=False,
        encoding="utf-8-sig"
    )


    # =========================================================================
    # Overall rules
    # =========================================================================

    rule_df = pd.DataFrame(
        [
            {
                "규칙":
                    "종목별 비중",

                "공식기준":
                    "일반 15%, 삼성전자 40%",

                "내부기준":
                    "5%",

                "현재상태":
                    "PASS"
            },

            {
                "규칙":
                    "섹터별 비중",

                "공식기준":
                    "max(시장섹터비중×2, 10%)",

                "내부기준":
                    (
                        f"공식한도의 "
                        f"{SECTOR_LIMIT_USAGE_RATIO:.0%}"
                    ),

                "현재상태":
                    "PASS"
            },

            {
                "규칙":
                    "시총 내부 버퍼",

                "공식기준":
                    "<1조 30%",

                "내부기준":
                    "<1.1조 25%",

                "현재상태":
                    (
                        f"{internal_small_weight * 100:.2f}%"
                    )
            },

            {
                "규칙":
                    "시총 공식한도",

                "공식기준":
                    "<1조 30%",

                "내부기준":
                    "-",

                "현재상태":
                    (
                        f"{official_small_weight * 100:.2f}%"
                    )
            }
        ]
    )


    rule_df.to_csv(
        LATEST_RULE_CHECK_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    rule_df.to_csv(
        os.path.join(
            DIAGNOSTIC_DIR,
            (
                f"rule_compliance_"
                f"{signal_date:%Y-%m-%d}.csv"
            )
        ),
        index=False,
        encoding="utf-8-sig"
    )


# =============================================================================
# 25. SAVE TARGET
# =============================================================================

def save_target(
    portfolio,
    signal_date,
    requested_exposure,
    reason
):

    output = portfolio.copy()


    output.insert(
        0,
        "TargetDate",
        signal_date.strftime(
            "%Y-%m-%d"
        )
    )


    output.insert(
        1,
        "Recalculated",
        True
    )


    output.insert(
        2,
        "RecalcReason",
        reason
    )


    output[
        "RequestedExposure"
    ] = requested_exposure


    output.to_csv(
        LATEST_TARGET_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    history_file = os.path.join(
        PORTFOLIO_HISTORY_DIR,
        (
            f"portfolio_target_"
            f"{signal_date:%Y-%m-%d}.csv"
        )
    )


    output.to_csv(
        history_file,
        index=False,
        encoding="utf-8-sig"
    )


    return (
        output,
        history_file
    )


# =============================================================================
# 26. NON-REBALANCE DAY
#
# 중요:
# 비리밸런싱일에는 기존 Target을 건드리지 않는다.
# =============================================================================

def save_previous_target(
    previous_target,
    signal_date,
    listing_map,
    sector_map
):

    rows = []


    for (
        code,
        weight
    ) in previous_target.items():

        if code in listing_map.index:

            info = listing_map.loc[
                code
            ]


            if isinstance(
                info,
                pd.DataFrame
            ):

                info = info.iloc[
                    0
                ]


            name = info[
                "Name"
            ]


            market = info[
                "Market"
            ]


        else:

            name = ""
            market = ""


        rows.append(
            {
                "종목코드":
                    code,

                "종목명":
                    name,

                "시장":
                    market,

                "섹터명":
                    sector_map.get(
                        code,
                        ""
                    ),

                "TargetWeight":
                    weight
            }
        )


    output = pd.DataFrame(
        rows
    )


    output.insert(
        0,
        "TargetDate",
        signal_date.strftime(
            "%Y-%m-%d"
        )
    )


    output.insert(
        1,
        "Recalculated",
        False
    )


    output.insert(
        2,
        "RecalcReason",
        "KEEP_PREVIOUS_TARGET"
    )


    output[
        "RequestedExposure"
    ] = float(
        sum(
            previous_target.values()
        )
    )


    output.to_csv(
        LATEST_TARGET_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    history_file = os.path.join(
        PORTFOLIO_HISTORY_DIR,
        (
            f"portfolio_target_"
            f"{signal_date:%Y-%m-%d}.csv"
        )
    )


    output.to_csv(
        history_file,
        index=False,
        encoding="utf-8-sig"
    )


    return output


# =============================================================================
# 27. PRINT TARGET
# =============================================================================

def print_portfolio(
    portfolio,
    signal_date,
    requested_exposure,
    selected_count
):

    print()
    print(
        "=" * 150
    )


    print(
        "LIVE TARGET PORTFOLIO"
    )


    print(
        "=" * 150
    )


    print(
        f"Signal Date        : "
        f"{signal_date.date()}"
    )


    print(
        f"Requested Exposure : "
        f"{requested_exposure * 100:.2f}%"
    )


    print(
        f"Target Exposure    : "
        f"{portfolio['TargetWeight'].sum() * 100:.2f}%"
    )


    print(
        f"Target Count       : "
        f"{len(portfolio)}"
    )


    print(
        f"Candidate Used     : "
        f"{selected_count}"
    )


    display = portfolio.copy()


    display[
        "시가총액_억원"
    ] = (
        display[
            "시가총액"
        ]
        /
        100_000_000
    )


    display[
        "5일평균거래대금_억원"
    ] = (
        display[
            "5일평균거래대금"
        ]
        /
        100_000_000
    )


    display[
        "RawMomentumWeight_%"
    ] = (
        display[
            "RawMomentumWeight"
        ]
        *
        100
    )


    display[
        "TargetWeight_%"
    ] = (
        display[
            "TargetWeight"
        ]
        *
        100
    )


    columns = [
        "종목코드",
        "종목명",
        "시장",
        "MomentumRank",
        "MomentumScore",
        "섹터명",
        "시가총액_억원",
        "5일평균거래대금_억원",
        "RawMomentumWeight_%",
        "TargetWeight_%"
    ]


    print()
    print(
        display[
            columns
        ]
        .round(
            2
        )
        .to_string(
            index=False
        )
    )


# =============================================================================
# 28. MAIN
# =============================================================================

def main():

    print(
        "=" * 150
    )


    print(
        "DART QUANT - LIVE PORTFOLIO"
    )


    print(
        "Momentum100 / Rule-Safe / Buffered"
    )


    print(
        "=" * 150
    )


    # =========================================================================
    # Momentum
    # =========================================================================

    momentum = load_momentum()


    # =========================================================================
    # Signal Date
    # =========================================================================

    signal_date = determine_signal_date(
        momentum
    )


    # =========================================================================
    # Marcap
    # =========================================================================

    (
        listing,
        marcap_date
    ) = load_marcap(
        signal_date
    )


    listing_map = (
        listing
        .set_index(
            "Code"
        )
    )


    # =========================================================================
    # Sector
    # =========================================================================

    (
        sector_map,
        sector_table,
        internal_sector_limits
    ) = load_sector_info(
        listing
    )


    # =========================================================================
    # Previous Target
    # =========================================================================

    (
        previous_target,
        previous_target_date
    ) = load_previous_target()


    print()
    print(
        "[이전 Target]"
    )


    if previous_target:

        print(
            f"종목수: "
            f"{len(previous_target)}"
        )


        if previous_target_date is not None:

            print(
                f"Target Date: "
                f"{previous_target_date.date()}"
            )


    else:

        print(
            "이전 Target 없음"
        )


    # =========================================================================
    # Restrictions
    # =========================================================================

    admin_codes = load_admin_codes()


    manual_restricted = (
        load_manual_restricted()
    )


    print(
        f"Manual Restricted: "
        f"{len(manual_restricted)}개"
    )


    # =========================================================================
    # Schedule
    # =========================================================================

    trading_dates = load_trading_dates(
        signal_date
    )


    rebalance_dates = build_rebalance_dates(
        trading_dates
    )


    if not previous_target:

        recalculate = True
        reason = "INITIAL_TARGET"


    elif signal_date in rebalance_dates:

        recalculate = True
        reason = "SCHEDULED"


    else:

        recalculate = False
        reason = "KEEP_PREVIOUS_TARGET"


    print()
    print(
        "[Target Schedule]"
    )


    print(
        f"Signal Date : "
        f"{signal_date.date()}"
    )


    print(
        f"Recalculate : "
        f"{recalculate}"
    )


    print(
        f"Reason      : "
        f"{reason}"
    )


    # =========================================================================
    # NON-REBALANCE
    # =========================================================================

    if not recalculate:

        output = save_previous_target(
            previous_target,
            signal_date,
            listing_map,
            sector_map
        )


        print()
        print(
            "비리밸런싱일 -> 기존 Target 그대로 유지"
        )


        print(
            f"Target Exposure: "
            f"{output['TargetWeight'].sum() * 100:.2f}%"
        )


        return


    # =========================================================================
    # Market Regime
    # =========================================================================

    (
        kospi,
        kosdaq
    ) = load_market_indices(
        signal_date
    )


    requested_exposure = float(
        bt.get_equity_exposure(
            signal_date,
            kospi,
            kosdaq
        )
    )


    print()
    print(
        f"Requested Equity Exposure: "
        f"{requested_exposure * 100:.1f}%"
    )


    # =========================================================================
    # Hysteresis
    # =========================================================================

    (
        candidates,
        entry_count,
        retained_count
    ) = build_candidates(
        momentum,
        previous_target
    )


    print()
    print(
        "[Hysteresis]"
    )


    print(
        f"Top80 Entry                  : "
        f"{entry_count}개"
    )


    print(
        f"Previous Target Top100 유지 : "
        f"{retained_count}개"
    )


    print(
        f"Candidate Total              : "
        f"{len(candidates)}개"
    )


    # =========================================================================
    # Price
    # =========================================================================

    price_data = download_prices(
        candidates[
            "종목코드"
        ].tolist(),
        signal_date
    )


    # =========================================================================
    # Tradability
    # =========================================================================

    (
        tradable,
        rejected
    ) = build_tradable(
        candidates,
        signal_date,
        price_data,
        listing_map,
        sector_map,
        admin_codes,
        manual_restricted
    )


    print()
    print(
        "[Tradability]"
    )


    print(
        f"Tradable: "
        f"{len(tradable)}개"
    )


    print(
        f"Rejected: "
        f"{len(rejected)}개"
    )


    if not rejected.empty:

        print()
        print(
            "Reject Reason counts"
        )


        exploded = (
            rejected[
                "RejectReason"
            ]
            .str
            .split(
                ";"
            )
            .explode()
        )


        print(
            exploded
            .value_counts()
            .to_string()
        )


    rejected_file = os.path.join(
        DIAGNOSTIC_DIR,
        (
            f"rejected_candidates_"
            f"{signal_date:%Y-%m-%d}.csv"
        )
    )


    rejected.to_csv(
        rejected_file,
        index=False,
        encoding="utf-8-sig"
    )


    if len(
        tradable
    ) < TARGET_HOLDINGS:

        raise RuntimeError(
            "Tradable 후보가 25개보다 적습니다."
        )


    # =========================================================================
    # Optimize
    # =========================================================================

    (
        portfolio,
        success,
        selected_count
    ) = build_portfolio(
        tradable,
        requested_exposure,
        internal_sector_limits
    )


    if not success:

        raise RuntimeError(
            "현재 후보군으로 대회 규정과 내부 버퍼를 "
            "동시에 만족하는 포트폴리오를 만들 수 없습니다."
        )


    print()
    print(
        f"Constraint-safe portfolio candidate count: "
        f"{selected_count}"
    )


    # =========================================================================
    # FINAL COMPLIANCE
    #
    # 저장 전에 반드시 통과
    # =========================================================================

    (
        sector_check,
        internal_small_weight,
        official_small_weight
    ) = validate_compliance(
        portfolio,
        requested_exposure,
        sector_table
    )


    # =========================================================================
    # Compliance 저장
    # =========================================================================

    save_compliance(
        sector_check,
        internal_small_weight,
        official_small_weight,
        signal_date
    )


    # =========================================================================
    # Target 저장
    #
    # FINAL COMPLIANCE PASS 이후에만 실행
    # =========================================================================

    (
        output,
        history_file
    ) = save_target(
        portfolio,
        signal_date,
        requested_exposure,
        reason
    )


    # =========================================================================
    # Print
    # =========================================================================

    print_portfolio(
        portfolio,
        signal_date,
        requested_exposure,
        selected_count
    )


    print()
    print(
        "=" * 150
    )


    print(
        "Portfolio 생성 완료"
    )


    print(
        "=" * 150
    )


    print()
    print(
        "Latest Target:"
    )


    print(
        LATEST_TARGET_FILE
    )


    print()
    print(
        "History:"
    )


    print(
        history_file
    )


    print()
    print(
        "Sector Compliance:"
    )


    print(
        LATEST_SECTOR_CHECK_FILE
    )


    print()
    print(
        "Rule Compliance:"
    )


    print(
        LATEST_RULE_CHECK_FILE
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()