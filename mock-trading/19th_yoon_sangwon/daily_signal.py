# -*- coding: utf-8 -*-

import os
import re
import glob
import warnings
import importlib.util

import numpy as np
import pandas as pd
import FinanceDataReader as fdr


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

MARCAP_FILE = os.path.join(
    BASE_DIR,
    "marcap",
    "data",
    "marcap-2026.parquet"
)


# =============================================================================
# PORTFOLIO
# =============================================================================

PORTFOLIO_DIR = os.path.join(
    BASE_DIR,
    "portfolio"
)

TARGET_FILE = os.path.join(
    PORTFOLIO_DIR,
    "portfolio_target_latest.csv"
)

RESTRICTED_FILE = os.path.join(
    PORTFOLIO_DIR,
    "input",
    "restricted_codes.csv"
)


# =============================================================================
# DAILY SIGNAL
# =============================================================================

DAILY_DIR = os.path.join(
    BASE_DIR,
    "daily_signal"
)

HISTORY_DIR = os.path.join(
    DAILY_DIR,
    "history"
)

STATE_DIR = os.path.join(
    DAILY_DIR,
    "state"
)

DIAGNOSTIC_DIR = os.path.join(
    DAILY_DIR,
    "diagnostics"
)


for folder in [
    DAILY_DIR,
    HISTORY_DIR,
    STATE_DIR,
    DIAGNOSTIC_DIR
]:

    os.makedirs(
        folder,
        exist_ok=True
    )


LATEST_SIGNAL_FILE = os.path.join(
    DAILY_DIR,
    "daily_signal_latest.csv"
)

LATEST_ORDER_FILE = os.path.join(
    DAILY_DIR,
    "order_list_latest.csv"
)

LATEST_STATE_FILE = os.path.join(
    STATE_DIR,
    "portfolio_state_latest.csv"
)

LATEST_COMPLIANCE_FILE = os.path.join(
    DIAGNOSTIC_DIR,
    "daily_compliance_latest.csv"
)

LATEST_DIAGNOSTIC_FILE = os.path.join(
    DIAGNOSTIC_DIR,
    "daily_signal_diagnostic_latest.csv"
)


# =============================================================================
# 1. STRATEGY PARAMETERS
# =============================================================================

# -----------------------------------------------------------------------------
# Rebalancing Band
#
# 목표와 현재 비중 차이가 0.75%p 미만이면
# 기본적으로 HOLD
# -----------------------------------------------------------------------------

REBALANCE_BAND = 0.0075


# =============================================================================
# 2. COMPETITION RULES
# =============================================================================

# -----------------------------------------------------------------------------
# 개별 종목 공식 한도
# -----------------------------------------------------------------------------

OFFICIAL_GENERAL_STOCK_CAP = 0.15
OFFICIAL_SAMSUNG_CAP = 0.40


# -----------------------------------------------------------------------------
# 내부 개별 종목 한도
# -----------------------------------------------------------------------------

INTERNAL_STOCK_CAP = 0.05


# -----------------------------------------------------------------------------
# 공식 섹터 한도
#
# max(시장 섹터 비중 × 2, 10%)
# -----------------------------------------------------------------------------

OFFICIAL_MIN_SECTOR_CAP = 0.10


# -----------------------------------------------------------------------------
# 내부 섹터 한도
#
# 공식 한도의 90%
# -----------------------------------------------------------------------------

SECTOR_LIMIT_USAGE_RATIO = 0.90


# -----------------------------------------------------------------------------
# 공식 시총 규칙
#
# 시총 < 1조원 종목 비중 합계 <= 30%
# -----------------------------------------------------------------------------

OFFICIAL_SMALL_CAP_THRESHOLD = 1_000_000_000_000
OFFICIAL_SMALL_CAP_LIMIT = 0.30


# -----------------------------------------------------------------------------
# 내부 시총 버퍼
#
# 시총 < 1.1조원 종목 비중 합계 <= 25%
# -----------------------------------------------------------------------------

INTERNAL_SMALL_CAP_THRESHOLD = 1_100_000_000_000
INTERNAL_SMALL_CAP_LIMIT = 0.25


# =============================================================================
# 3. BUY ELIGIBILITY
# =============================================================================

# 시총 1,000억원 이상
MIN_MARKET_CAP = 100_000_000_000

# 최근 5일 평균 거래대금 30억원 초과
MIN_AVG_TRADING_VALUE_5D = 3_000_000_000

# Momentum 전략상 최소 가격 관측
MIN_PRICE_OBS = 61


# =============================================================================
# 4. DATA PARAMETERS
# =============================================================================

PRICE_LOOKBACK_DAYS = 220

MAX_MARCAP_LAG_DAYS = 10


# =============================================================================
# 5. WEEKLY TURNOVER
#
# 공식 산식은 아직 확인하지 못했으므로
# 실제 추천 변경비중 절대값 합계를 proxy로 사용
# =============================================================================

WEEKLY_TURNOVER_TARGET = 0.05


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


# =============================================================================
# 7. BACKTEST MODULE
# =============================================================================

def load_backtest_module():

    if not os.path.exists(
        BACKTEST_FILE
    ):

        raise FileNotFoundError(
            f"backtest.py 없음:\n"
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


    if not hasattr(
        bt,
        "load_grid"
    ):

        raise RuntimeError(
            "backtest.py에 load_grid()가 없습니다."
        )


    return bt


bt = load_backtest_module()


# =============================================================================
# 8. RESTRICTED FILE
# =============================================================================

def ensure_restricted_file():

    folder = os.path.dirname(
        RESTRICTED_FILE
    )

    os.makedirs(
        folder,
        exist_ok=True
    )


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
# 9. TARGET
# =============================================================================

def load_target():

    print()
    print(
        "[1] Portfolio Target"
    )


    if not os.path.exists(
        TARGET_FILE
    ):

        raise FileNotFoundError(
            "portfolio_target_latest.csv가 없습니다.\n"
            "먼저 Portfolio.py를 실행하세요."
        )


    df = pd.read_csv(
        TARGET_FILE,
        dtype={
            "종목코드": str
        }
    )


    required = {
        "종목코드",
        "TargetWeight",
        "TargetDate"
    }


    if not required.issubset(
        df.columns
    ):

        raise RuntimeError(
            "Target 파일 필수 컬럼이 없습니다."
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
    )


    dates = pd.to_datetime(
        df[
            "TargetDate"
        ],
        errors="coerce"
    ).dropna()


    if dates.empty:

        raise RuntimeError(
            "TargetDate를 읽을 수 없습니다."
        )


    signal_date = pd.Timestamp(
        dates.iloc[0]
    ).normalize()


    df = (
        df
        .dropna(
            subset=[
                "종목코드",
                "TargetWeight"
            ]
        )
        .drop_duplicates(
            "종목코드"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        f"Target Date     : "
        f"{signal_date.date()}"
    )


    print(
        f"Target Count    : "
        f"{len(df)}"
    )


    print(
        f"Target Exposure : "
        f"{df['TargetWeight'].sum() * 100:.2f}%"
    )


    return (
        df,
        signal_date
    )


# =============================================================================
# 10. PREVIOUS PORTFOLIO STATE
# =============================================================================

def load_previous_state():

    print()
    print(
        "[2] Previous Portfolio State"
    )


    if not os.path.exists(
        LATEST_STATE_FILE
    ):

        print(
            "이전 State 없음"
        )

        print(
            "→ 최초 운용일로 처리"
        )


        return (
            pd.DataFrame(
                columns=[
                    "종목코드",
                    "StateDate",
                    "ReferenceClose",
                    "StateWeight"
                ]
            ),
            None
        )


    df = pd.read_csv(
        LATEST_STATE_FILE,
        dtype={
            "종목코드": str
        }
    )


    required = {
        "종목코드",
        "StateDate",
        "ReferenceClose",
        "StateWeight"
    }


    if not required.issubset(
        df.columns
    ):

        raise RuntimeError(
            "portfolio_state_latest.csv 컬럼 오류"
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
        "ReferenceClose"
    ] = pd.to_numeric(
        df[
            "ReferenceClose"
        ],
        errors="coerce"
    )


    df[
        "StateWeight"
    ] = pd.to_numeric(
        df[
            "StateWeight"
        ],
        errors="coerce"
    )


    state_dates = pd.to_datetime(
        df[
            "StateDate"
        ],
        errors="coerce"
    ).dropna()


    if state_dates.empty:

        raise RuntimeError(
            "StateDate를 읽을 수 없습니다."
        )


    state_date = pd.Timestamp(
        state_dates.iloc[0]
    ).normalize()


    df = (
        df
        .dropna(
            subset=[
                "종목코드",
                "ReferenceClose",
                "StateWeight"
            ]
        )
        .drop_duplicates(
            "종목코드"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        f"State Date     : "
        f"{state_date.date()}"
    )


    print(
        f"State Count    : "
        f"{len(df)}"
    )


    print(
        f"State Exposure : "
        f"{df['StateWeight'].sum() * 100:.2f}%"
    )


    return (
        df,
        state_date
    )


# =============================================================================
# 11. ADMIN
# =============================================================================

def load_admin_codes():

    print()
    print(
        "[3] Restrictions"
    )


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


    code_col = None


    for candidate in [
        "Code",
        "Symbol",
        "종목코드"
    ]:

        if candidate in df.columns:

            code_col = candidate
            break


    if code_col is None:

        return set()


    result = set()


    for value in df[
        code_col
    ]:

        code = normalize_code(
            value
        )


        if code:

            result.add(
                code
            )


    print(
        f"KRX-ADMIN        : "
        f"{len(result)}개"
    )


    return result


# =============================================================================
# 12. MARCAP
# =============================================================================

def load_marcap(
    signal_date
):

    print()
    print(
        "[4] Marcap"
    )


    if not os.path.exists(
        MARCAP_FILE
    ):

        raise FileNotFoundError(
            MARCAP_FILE
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


    if lag > MAX_MARCAP_LAG_DAYS:

        raise RuntimeError(
            "Marcap 데이터가 너무 오래됐습니다."
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
        f"Marcap Date : "
        f"{marcap_date.date()}"
    )


    print(
        f"Lag         : "
        f"{lag} day(s)"
    )


    return (
        latest,
        marcap_date
    )


# =============================================================================
# 13. SECTOR
# =============================================================================

def load_sector_info(
    listing
):

    print()
    print(
        "[5] Sector Limits"
    )


    grid = bt.load_grid().copy()


    required = {
        "종목코드",
        "섹터명"
    }


    if not required.issubset(
        grid.columns
    ):

        raise RuntimeError(
            "Sector grid 컬럼 오류"
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


    sector_marcap = (
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


    rows = []


    for (
        sector,
        marcap
    ) in sector_marcap.items():

        market_weight = (
            float(
                marcap
            )
            /
            total_market_cap
        )


        official_limit = max(
            market_weight
            *
            2.0,

            OFFICIAL_MIN_SECTOR_CAP
        )


        official_limit = min(
            official_limit,
            1.0
        )


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
                    market_weight,

                "공식한도":
                    official_limit,

                "내부한도":
                    internal_limit
            }
        )


    sector_table = pd.DataFrame(
        rows
    )


    return (
        sector_map,
        sector_table
    )


# =============================================================================
# 14. PRICE DATA
# =============================================================================

def download_prices(
    codes,
    signal_date
):

    print()
    print(
        "[6] Price / Liquidity"
    )


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
        f"{len(result)}/{total}"
    )


    return result


# =============================================================================
# 15. PRICE HELPERS
# =============================================================================

def get_close(
    code,
    signal_date,
    prices
):

    if code not in prices:

        return np.nan


    df = prices[
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
    prices
):

    if code not in prices:

        return 0


    return len(
        prices[
            code
        ][
            prices[
                code
            ].index
            <=
            signal_date
        ]
    )


def get_5d_value(
    code,
    signal_date,
    prices
):

    if code not in prices:

        return np.nan


    df = prices[
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
    prices,
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
        prices
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
# 16. AUTO CURRENT WEIGHT
# =============================================================================

def build_auto_current_weights(
    previous_state,
    previous_state_date,
    signal_date,
    prices
):

    print()
    print(
        "[7] Auto Current Weight"
    )


    # =========================================================================
    # 최초 운용
    # =========================================================================

    if previous_state.empty:

        print(
            "최초 운용일"
        )

        print(
            "Current Equity : 0.00%"
        )

        print(
            "Current Cash   : 100.00%"
        )


        return pd.DataFrame(
            columns=[
                "종목코드",
                "CurrentWeight"
            ]
        )


    # =========================================================================
    # 같은 날짜 재실행
    #
    # 이미 저장된 State를 현재비중으로 사용
    # =========================================================================

    if (
        previous_state_date
        ==
        signal_date
    ):

        current = previous_state[
            [
                "종목코드",
                "StateWeight"
            ]
        ].copy()


        current = current.rename(
            columns={
                "StateWeight":
                    "CurrentWeight"
            }
        )


        equity = float(
            current[
                "CurrentWeight"
            ].sum()
        )


        print(
            "동일 날짜 State 존재"
        )

        print(
            "→ 저장된 StateWeight 사용"
        )


        print(
            f"Current Equity : "
            f"{equity * 100:.2f}%"
        )


        print(
            f"Current Cash   : "
            f"{(1 - equity) * 100:.2f}%"
        )


        return current


    # =========================================================================
    # 전일 State → 오늘 가격 변화 반영
    # =========================================================================

    previous_equity = float(
        previous_state[
            "StateWeight"
        ].sum()
    )


    previous_cash = max(
        0.0,
        1.0
        -
        previous_equity
    )


    rows = []


    for _, row in previous_state.iterrows():

        code = row[
            "종목코드"
        ]


        previous_weight = float(
            row[
                "StateWeight"
            ]
        )


        previous_close = float(
            row[
                "ReferenceClose"
            ]
        )


        current_close = get_close(
            code,
            signal_date,
            prices
        )


        if pd.isna(
            current_close
        ):

            raise RuntimeError(
                f"{code}의 현재 가격을 가져올 수 없습니다."
            )


        if previous_close <= 0:

            raise RuntimeError(
                f"{code} 이전 기준가격 오류"
            )


        price_ratio = (
            float(
                current_close
            )
            /
            previous_close
        )


        value_contribution = (
            previous_weight
            *
            price_ratio
        )


        rows.append(
            {
                "종목코드":
                    code,

                "PreviousWeight":
                    previous_weight,

                "PreviousClose":
                    previous_close,

                "CurrentClose":
                    float(
                        current_close
                    ),

                "PriceRatio":
                    price_ratio,

                "ValueContribution":
                    value_contribution
            }
        )


    temp = pd.DataFrame(
        rows
    )


    portfolio_value_factor = (
        previous_cash
        +
        float(
            temp[
                "ValueContribution"
            ].sum()
        )
    )


    if (
        not np.isfinite(
            portfolio_value_factor
        )
        or
        portfolio_value_factor
        <=
        0
    ):

        raise RuntimeError(
            "Portfolio Value Factor 계산 실패"
        )


    temp[
        "CurrentWeight"
    ] = (
        temp[
            "ValueContribution"
        ]
        /
        portfolio_value_factor
    )


    current = temp[
        [
            "종목코드",
            "CurrentWeight"
        ]
    ].copy()


    current = current[
        current[
            "CurrentWeight"
        ]
        >
        1e-10
    ].copy()


    current_equity = float(
        current[
            "CurrentWeight"
        ].sum()
    )


    print(
        f"Previous Equity : "
        f"{previous_equity * 100:.2f}%"
    )


    print(
        f"Previous Cash   : "
        f"{previous_cash * 100:.2f}%"
    )


    print(
        f"Current Equity  : "
        f"{current_equity * 100:.2f}%"
    )


    print(
        f"Current Cash    : "
        f"{(1 - current_equity) * 100:.2f}%"
    )


    return current


# =============================================================================
# 17. BUY ELIGIBILITY
# =============================================================================

def check_buy_eligibility(
    code,
    signal_date,
    prices,
    listing_map,
    admin_codes,
    manual_restricted
):

    reasons = []


    if code in admin_codes:

        reasons.append(
            "KRX_ADMIN"
        )


    if code in manual_restricted:

        reasons.append(
            "MANUAL_RESTRICTED"
        )


    obs = get_obs(
        code,
        signal_date,
        prices
    )


    if obs < MIN_PRICE_OBS:

        reasons.append(
            "OBS_LT61"
        )


    market_cap = get_market_cap(
        code,
        signal_date,
        prices,
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


    avg_value = get_5d_value(
        code,
        signal_date,
        prices
    )


    if (
        pd.isna(
            avg_value
        )
        or
        avg_value
        <=
        MIN_AVG_TRADING_VALUE_5D
    ):

        reasons.append(
            "LIQUIDITY"
        )


    return (
        len(
            reasons
        )
        ==
        0,
        reasons,
        market_cap,
        avg_value
    )


# =============================================================================
# 18. SIGNAL BUILDER
# =============================================================================

def build_signal(
    target,
    current,
    signal_date,
    prices,
    listing_map,
    sector_map,
    admin_codes,
    manual_restricted,
    ignore_band=False
):

    target_columns = [
        "종목코드",
        "TargetWeight"
    ]


    if "종목명" in target.columns:

        target_columns.append(
            "종목명"
        )


    if "섹터명" in target.columns:

        target_columns.append(
            "섹터명"
        )


    target_small = target[
        target_columns
    ].copy()


    merged = pd.merge(
        target_small,
        current,
        on="종목코드",
        how="outer"
    )


    merged[
        "TargetWeight"
    ] = (
        merged[
            "TargetWeight"
        ]
        .fillna(
            0
        )
    )


    merged[
        "CurrentWeight"
    ] = (
        merged[
            "CurrentWeight"
        ]
        .fillna(
            0
        )
    )


    if "종목명" not in merged.columns:

        merged[
            "종목명"
        ] = ""


    if "섹터명" not in merged.columns:

        merged[
            "섹터명"
        ] = ""


    # =========================================================================
    # Target에서 빠진 기존 보유종목 이름 / 섹터 보충
    # =========================================================================

    for idx in merged.index:

        code = merged.loc[
            idx,
            "종목코드"
        ]


        if (
            pd.isna(
                merged.loc[
                    idx,
                    "종목명"
                ]
            )
            or
            str(
                merged.loc[
                    idx,
                    "종목명"
                ]
            ).strip()
            ==
            ""
        ):

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


                merged.loc[
                    idx,
                    "종목명"
                ] = info.get(
                    "Name",
                    ""
                )


        if (
            pd.isna(
                merged.loc[
                    idx,
                    "섹터명"
                ]
            )
            or
            str(
                merged.loc[
                    idx,
                    "섹터명"
                ]
            ).strip()
            ==
            ""
        ):

            merged.loc[
                idx,
                "섹터명"
            ] = sector_map.get(
                code,
                ""
            )


    # =========================================================================
    # Difference
    # =========================================================================

    merged[
        "WeightDiff"
    ] = (
        merged[
            "TargetWeight"
        ]
        -
        merged[
            "CurrentWeight"
        ]
    )


    actions = []
    block_reasons = []
    market_caps = []
    liquidities = []


    for _, row in merged.iterrows():

        code = row[
            "종목코드"
        ]


        current_weight = float(
            row[
                "CurrentWeight"
            ]
        )


        target_weight = float(
            row[
                "TargetWeight"
            ]
        )


        diff = float(
            row[
                "WeightDiff"
            ]
        )


        market_cap = get_market_cap(
            code,
            signal_date,
            prices,
            listing_map
        )


        liquidity = get_5d_value(
            code,
            signal_date,
            prices
        )


        reasons = []


        # =====================================================================
        # Target에서 완전히 제외
        # =====================================================================

        if (
            current_weight
            >
            0
            and
            target_weight
            <=
            1e-12
        ):

            action = "SELL"


        # =====================================================================
        # BAND
        # =====================================================================

        elif (
            not ignore_band
            and
            abs(
                diff
            )
            <
            REBALANCE_BAND
        ):

            action = "HOLD"


        # =====================================================================
        # BUY
        # =====================================================================

        elif diff > 1e-12:

            (
                eligible,
                reasons,
                market_cap,
                liquidity
            ) = check_buy_eligibility(
                code,
                signal_date,
                prices,
                listing_map,
                admin_codes,
                manual_restricted
            )


            if eligible:

                action = "BUY"

            else:

                action = "BUY_BLOCKED"


        # =====================================================================
        # SELL
        # =====================================================================

        elif diff < -1e-12:

            action = "SELL"


        else:

            action = "HOLD"


        actions.append(
            action
        )


        block_reasons.append(
            ";".join(
                reasons
            )
        )


        market_caps.append(
            market_cap
        )


        liquidities.append(
            liquidity
        )


    merged[
        "Action"
    ] = actions


    merged[
        "BlockReason"
    ] = block_reasons


    merged[
        "시가총액"
    ] = market_caps


    merged[
        "5일평균거래대금"
    ] = liquidities


    # =========================================================================
    # 실제 추천 변경폭
    # =========================================================================

    merged[
        "RecommendedChange"
    ] = np.where(
        merged[
            "Action"
        ].isin(
            [
                "BUY",
                "SELL"
            ]
        ),

        merged[
            "WeightDiff"
        ],

        0.0
    )


    # =========================================================================
    # 매매 후 예상 비중
    #
    # BUY / SELL → Target
    # HOLD / BUY_BLOCKED → Current 유지
    # =========================================================================

    merged[
        "ProjectedWeight"
    ] = np.where(
        merged[
            "Action"
        ].isin(
            [
                "BUY",
                "SELL"
            ]
        ),

        merged[
            "TargetWeight"
        ],

        merged[
            "CurrentWeight"
        ]
    )


    return merged


# =============================================================================
# 19. COMPLIANCE CHECK
# =============================================================================

def check_projected_compliance(
    signal,
    sector_table
):

    p = signal.copy()


    p[
        "ProjectedWeight"
    ] = pd.to_numeric(
        p[
            "ProjectedWeight"
        ],
        errors="coerce"
    ).fillna(
        0
    )


    p[
        "시가총액"
    ] = pd.to_numeric(
        p[
            "시가총액"
        ],
        errors="coerce"
    )


    positive = p[
        p[
            "ProjectedWeight"
        ]
        >
        1e-10
    ].copy()


    # =========================================================================
    # 시총 데이터 없는 실제 보유종목이 있으면 검증 불가
    # =========================================================================

    if positive[
        "시가총액"
    ].isna().any():

        bad_codes = positive.loc[
            positive[
                "시가총액"
            ].isna(),
            "종목코드"
        ].tolist()


        raise RuntimeError(
            "시가총액을 검증할 수 없는 보유종목이 있습니다:\n"
            +
            ", ".join(
                bad_codes
            )
        )


    rows = []


    # =========================================================================
    # STOCK
    # =========================================================================

    stock_internal_pass = True
    stock_official_pass = True


    max_weight = 0.0


    for _, row in positive.iterrows():

        code = row[
            "종목코드"
        ]


        weight = float(
            row[
                "ProjectedWeight"
            ]
        )


        max_weight = max(
            max_weight,
            weight
        )


        official_limit = (
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
            INTERNAL_STOCK_CAP
            +
            1e-8
        ):

            stock_internal_pass = False


        if (
            weight
            >
            official_limit
            +
            1e-8
        ):

            stock_official_pass = False


    rows.append(
        {
            "구분":
                "종목별 비중",

            "현재":
                max_weight,

            "내부한도":
                INTERNAL_STOCK_CAP,

            "공식한도":
                OFFICIAL_GENERAL_STOCK_CAP,

            "내부PASS":
                stock_internal_pass,

            "공식PASS":
                stock_official_pass
        }
    )


    # =========================================================================
    # SECTOR
    # =========================================================================

    sector_lookup = sector_table.set_index(
        "섹터명"
    )


    sector_weights = (
        positive
        .groupby(
            "섹터명"
        )[
            "ProjectedWeight"
        ]
        .sum()
    )


    for (
        sector,
        weight
    ) in sector_weights.items():

        if (
            sector is None
            or
            str(
                sector
            ).strip()
            ==
            ""
        ):

            rows.append(
                {
                    "구분":
                        "섹터:UNKNOWN",

                    "현재":
                        float(
                            weight
                        ),

                    "내부한도":
                        np.nan,

                    "공식한도":
                        np.nan,

                    "내부PASS":
                        False,

                    "공식PASS":
                        False
                }
            )

            continue


        if sector not in sector_lookup.index:

            rows.append(
                {
                    "구분":
                        f"섹터:{sector}",

                    "현재":
                        float(
                            weight
                        ),

                    "내부한도":
                        np.nan,

                    "공식한도":
                        np.nan,

                    "내부PASS":
                        False,

                    "공식PASS":
                        False
                }
            )

            continue


        info = sector_lookup.loc[
            sector
        ]


        internal_limit = float(
            info[
                "내부한도"
            ]
        )


        official_limit = float(
            info[
                "공식한도"
            ]
        )


        rows.append(
            {
                "구분":
                    f"섹터:{sector}",

                "현재":
                    float(
                        weight
                    ),

                "내부한도":
                    internal_limit,

                "공식한도":
                    official_limit,

                "내부PASS":
                    (
                        float(
                            weight
                        )
                        <=
                        internal_limit
                        +
                        1e-8
                    ),

                "공식PASS":
                    (
                        float(
                            weight
                        )
                        <=
                        official_limit
                        +
                        1e-8
                    )
            }
        )


    # =========================================================================
    # INTERNAL SMALL CAP
    # =========================================================================

    internal_small_weight = float(
        positive.loc[
            positive[
                "시가총액"
            ]
            <
            INTERNAL_SMALL_CAP_THRESHOLD,
            "ProjectedWeight"
        ].sum()
    )


    rows.append(
        {
            "구분":
                "시총<1.1조",

            "현재":
                internal_small_weight,

            "내부한도":
                INTERNAL_SMALL_CAP_LIMIT,

            "공식한도":
                np.nan,

            "내부PASS":
                (
                    internal_small_weight
                    <=
                    INTERNAL_SMALL_CAP_LIMIT
                    +
                    1e-8
                ),

            "공식PASS":
                True
        }
    )


    # =========================================================================
    # OFFICIAL SMALL CAP
    # =========================================================================

    official_small_weight = float(
        positive.loc[
            positive[
                "시가총액"
            ]
            <
            OFFICIAL_SMALL_CAP_THRESHOLD,
            "ProjectedWeight"
        ].sum()
    )


    rows.append(
        {
            "구분":
                "시총<1조",

            "현재":
                official_small_weight,

            "내부한도":
                np.nan,

            "공식한도":
                OFFICIAL_SMALL_CAP_LIMIT,

            "내부PASS":
                True,

            "공식PASS":
                (
                    official_small_weight
                    <=
                    OFFICIAL_SMALL_CAP_LIMIT
                    +
                    1e-8
                )
        }
    )


    compliance = pd.DataFrame(
        rows
    )


    internal_pass = bool(
        compliance[
            "내부PASS"
        ].all()
    )


    official_pass = bool(
        compliance[
            "공식PASS"
        ].all()
    )


    return (
        compliance,
        internal_pass,
        official_pass,
        internal_small_weight,
        official_small_weight
    )


# =============================================================================
# 20. SAFE SIGNAL
# =============================================================================

def build_safe_signal(
    target,
    current,
    signal_date,
    prices,
    listing_map,
    sector_map,
    sector_table,
    admin_codes,
    manual_restricted,
    initial_run
):

    print()
    print(
        "[8] Rebalance Signal"
    )


    # =========================================================================
    # 최초 포트폴리오 구축 시에는 band 적용 안 함
    #
    # 0 → Target을 정확히 구축
    # =========================================================================

    if initial_run:

        signal = build_signal(
            target,
            current,
            signal_date,
            prices,
            listing_map,
            sector_map,
            admin_codes,
            manual_restricted,
            ignore_band=True
        )


        (
            compliance,
            internal_pass,
            official_pass,
            internal_small_weight,
            official_small_weight
        ) = check_projected_compliance(
            signal,
            sector_table
        )


        if not official_pass:

            raise RuntimeError(
                "최초 포트폴리오가 공식 규정을 만족하지 못합니다."
            )


        if not internal_pass:

            raise RuntimeError(
                "최초 포트폴리오가 내부 버퍼를 만족하지 못합니다."
            )


        print(
            "INITIAL BUILD: COMPLIANCE PASS"
        )


        return (
            signal,
            compliance,
            False,
            internal_small_weight,
            official_small_weight
        )


    # =========================================================================
    # 1차: 0.75%p band 적용
    # =========================================================================

    signal = build_signal(
        target,
        current,
        signal_date,
        prices,
        listing_map,
        sector_map,
        admin_codes,
        manual_restricted,
        ignore_band=False
    )


    (
        compliance,
        internal_pass,
        official_pass,
        internal_small_weight,
        official_small_weight
    ) = check_projected_compliance(
        signal,
        sector_table
    )


    if (
        internal_pass
        and
        official_pass
    ):

        print(
            "Normal Band Rebalance: "
            "COMPLIANCE PASS"
        )


        return (
            signal,
            compliance,
            False,
            internal_small_weight,
            official_small_weight
        )


    # =========================================================================
    # 2차: Compliance Override
    #
    # band 때문에 규정이 깨지면 band를 해제
    # =========================================================================

    print()
    print(
        "[COMPLIANCE OVERRIDE]"
    )


    print(
        "Band 적용 후 규정/내부버퍼 위반 → "
        "band 해제 후 재계산"
    )


    signal = build_signal(
        target,
        current,
        signal_date,
        prices,
        listing_map,
        sector_map,
        admin_codes,
        manual_restricted,
        ignore_band=True
    )


    (
        compliance,
        internal_pass,
        official_pass,
        internal_small_weight,
        official_small_weight
    ) = check_projected_compliance(
        signal,
        sector_table
    )


    if not official_pass:

        raise RuntimeError(
            "Compliance Override 이후에도 "
            "공식 규정을 만족하지 못합니다.\n"
            "주문 리스트 생성을 중단합니다."
        )


    if not internal_pass:

        raise RuntimeError(
            "Compliance Override 이후에도 "
            "내부 안전 버퍼를 만족하지 못합니다.\n"
            "주문 리스트 생성을 중단합니다."
        )


    print(
        "Compliance Override: PASS"
    )


    return (
        signal,
        compliance,
        True,
        internal_small_weight,
        official_small_weight
    )


# =============================================================================
# 21. RECOMMENDATION TEXT
# =============================================================================

def add_recommendation_text(
    signal
):

    signal = signal.copy()


    def make_text(
        row
    ):

        action = row[
            "Action"
        ]


        change = float(
            row[
                "RecommendedChange"
            ]
        )


        if action == "BUY":

            return (
                f"+{change * 100:.2f}%p 매수"
            )


        if action == "SELL":

            return (
                f"{change * 100:.2f}%p 매도"
            )


        if action == "BUY_BLOCKED":

            return (
                "매수 금지"
            )


        return (
            "변경 없음"
        )


    signal[
        "Recommendation"
    ] = signal.apply(
        make_text,
        axis=1
    )


    priority = {
        "SELL": 0,
        "BUY": 1,
        "BUY_BLOCKED": 2,
        "HOLD": 3
    }


    signal[
        "_priority"
    ] = (
        signal[
            "Action"
        ]
        .map(
            priority
        )
        .fillna(
            9
        )
    )


    signal = (
        signal
        .sort_values(
            [
                "_priority",
                "RecommendedChange"
            ],
            ascending=[
                True,
                True
            ]
        )
        .drop(
            columns=[
                "_priority"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return signal


# =============================================================================
# 22. SAVE PORTFOLIO STATE
#
# 중요:
# 이 State는 "오늘 제안한 매매를 실제로 모두 실행했다"고 가정한 상태.
# =============================================================================

def save_portfolio_state(
    signal,
    signal_date,
    prices
):

    rows = []


    for _, row in signal.iterrows():

        weight = float(
            row[
                "ProjectedWeight"
            ]
        )


        if weight <= 1e-10:

            continue


        code = row[
            "종목코드"
        ]


        close = get_close(
            code,
            signal_date,
            prices
        )


        if pd.isna(
            close
        ):

            raise RuntimeError(
                f"{code} State 저장용 가격 없음"
            )


        rows.append(
            {
                "종목코드":
                    code,

                "StateDate":
                    signal_date.strftime(
                        "%Y-%m-%d"
                    ),

                "ReferenceClose":
                    float(
                        close
                    ),

                "StateWeight":
                    weight
            }
        )


    state = pd.DataFrame(
        rows
    )


    state.to_csv(
        LATEST_STATE_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    history_file = os.path.join(
        STATE_DIR,
        (
            f"portfolio_state_"
            f"{signal_date:%Y-%m-%d}.csv"
        )
    )


    state.to_csv(
        history_file,
        index=False,
        encoding="utf-8-sig"
    )


    print()
    print(
        "[STATE SAVED]"
    )


    print(
        f"State Date     : "
        f"{signal_date.date()}"
    )


    print(
        f"State Count    : "
        f"{len(state)}"
    )


    print(
        f"State Exposure : "
        f"{state['StateWeight'].sum() * 100:.2f}%"
    )


# =============================================================================
# 23. WEEKLY TURNOVER PROXY
# =============================================================================

def calculate_weekly_turnover_proxy(
    signal_date,
    current_signal=None
):

    """
    공식 회전율 산식이 아직 확인되지 않았으므로
    운영용 proxy:

        주간 |RecommendedChange| 합계

    사용.

    날짜별 daily_signal history는 하루에 한 파일이므로
    같은 날 재실행해도 중복 합산되지 않는다.
    """

    current_iso = signal_date.isocalendar()


    total = 0.0


    pattern = os.path.join(
        HISTORY_DIR,
        "daily_signal_*.csv"
    )


    files = glob.glob(
        pattern
    )


    for file in files:

        filename = os.path.basename(
            file
        )


        match = re.match(
            r"daily_signal_(\d{4}-\d{2}-\d{2})\.csv$",
            filename
        )


        if not match:

            continue


        file_date = pd.Timestamp(
            match.group(
                1
            )
        )


        iso = file_date.isocalendar()


        if (
            iso.year
            !=
            current_iso.year
            or
            iso.week
            !=
            current_iso.week
        ):

            continue


        # 오늘 파일은 아직 저장 전이면
        # current_signal로 계산
        if file_date.normalize() == signal_date.normalize():

            continue


        try:

            df = pd.read_csv(
                file
            )

        except Exception:

            continue


        if "RecommendedChange" not in df.columns:

            continue


        change = pd.to_numeric(
            df[
                "RecommendedChange"
            ],
            errors="coerce"
        ).fillna(
            0
        )


        total += float(
            change.abs().sum()
        )


    if current_signal is not None:

        change = pd.to_numeric(
            current_signal[
                "RecommendedChange"
            ],
            errors="coerce"
        ).fillna(
            0
        )


        total += float(
            change.abs().sum()
        )


    return total


# =============================================================================
# 24. SAVE OUTPUTS
# =============================================================================

def save_outputs(
    signal,
    compliance,
    signal_date,
    compliance_override,
    weekly_turnover,
    internal_small_weight,
    official_small_weight
):

    # =========================================================================
    # Full Signal
    # =========================================================================

    signal.to_csv(
        LATEST_SIGNAL_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    signal.to_csv(
        os.path.join(
            HISTORY_DIR,
            (
                f"daily_signal_"
                f"{signal_date:%Y-%m-%d}.csv"
            )
        ),
        index=False,
        encoding="utf-8-sig"
    )


    # =========================================================================
    # Action List
    # =========================================================================

    orders = signal[
        signal[
            "Action"
        ]
        .isin(
            [
                "BUY",
                "SELL",
                "BUY_BLOCKED"
            ]
        )
    ].copy()


    orders.to_csv(
        LATEST_ORDER_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    orders.to_csv(
        os.path.join(
            HISTORY_DIR,
            (
                f"order_list_"
                f"{signal_date:%Y-%m-%d}.csv"
            )
        ),
        index=False,
        encoding="utf-8-sig"
    )


    # =========================================================================
    # Compliance
    # =========================================================================

    compliance.to_csv(
        LATEST_COMPLIANCE_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    compliance.to_csv(
        os.path.join(
            DIAGNOSTIC_DIR,
            (
                f"daily_compliance_"
                f"{signal_date:%Y-%m-%d}.csv"
            )
        ),
        index=False,
        encoding="utf-8-sig"
    )


    # =========================================================================
    # Diagnostic
    # =========================================================================

    diagnostic = pd.DataFrame(
        [
            {
                "Date":
                    signal_date.strftime(
                        "%Y-%m-%d"
                    ),

                "CurrentExposure":
                    float(
                        signal[
                            "CurrentWeight"
                        ].sum()
                    ),

                "TargetExposure":
                    float(
                        signal[
                            "TargetWeight"
                        ].sum()
                    ),

                "ProjectedExposure":
                    float(
                        signal[
                            "ProjectedWeight"
                        ].sum()
                    ),

                "BuyCount":
                    int(
                        (
                            signal[
                                "Action"
                            ]
                            ==
                            "BUY"
                        ).sum()
                    ),

                "SellCount":
                    int(
                        (
                            signal[
                                "Action"
                            ]
                            ==
                            "SELL"
                        ).sum()
                    ),

                "HoldCount":
                    int(
                        (
                            signal[
                                "Action"
                            ]
                            ==
                            "HOLD"
                        ).sum()
                    ),

                "BuyBlockedCount":
                    int(
                        (
                            signal[
                                "Action"
                            ]
                            ==
                            "BUY_BLOCKED"
                        ).sum()
                    ),

                "ComplianceOverride":
                    compliance_override,

                "InternalSmallCapWeight":
                    internal_small_weight,

                "OfficialSmallCapWeight":
                    official_small_weight,

                "WeeklyTurnoverProxy":
                    weekly_turnover,

                "WeeklyTurnoverTarget":
                    WEEKLY_TURNOVER_TARGET
            }
        ]
    )


    diagnostic.to_csv(
        LATEST_DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# =============================================================================
# 25. PRINT
# =============================================================================

def print_results(
    signal,
    compliance,
    compliance_override,
    weekly_turnover,
    internal_small_weight,
    official_small_weight
):

    print()
    print(
        "=" * 150
    )


    print(
        "DAILY REBALANCE LIST"
    )


    print(
        "=" * 150
    )


    display = signal.copy()


    for col in [
        "CurrentWeight",
        "TargetWeight",
        "WeightDiff",
        "RecommendedChange",
        "ProjectedWeight"
    ]:

        display[
            col
        ] *= 100


    columns = [
        "종목코드",
        "종목명",
        "섹터명",
        "CurrentWeight",
        "TargetWeight",
        "WeightDiff",
        "Action",
        "Recommendation",
        "ProjectedWeight",
        "BlockReason"
    ]


    print()
    print(
        display[
            columns
        ]
        .rename(
            columns={
                "CurrentWeight":
                    "현재비중(%)",

                "TargetWeight":
                    "목표비중(%)",

                "WeightDiff":
                    "차이(%p)",

                "ProjectedWeight":
                    "조정후비중(%)"
            }
        )
        .round(
            2
        )
        .to_string(
            index=False
        )
    )


    # =========================================================================
    # 변경 리스트
    # =========================================================================

    print()
    print(
        "=" * 80
    )


    print(
        "오늘 변경 리스트"
    )


    print(
        "=" * 80
    )


    actions = signal[
        signal[
            "Action"
        ]
        .isin(
            [
                "BUY",
                "SELL",
                "BUY_BLOCKED"
            ]
        )
    ]


    if actions.empty:

        print(
            "변경 필요 없음"
        )


    else:

        for _, row in actions.iterrows():

            code = row[
                "종목코드"
            ]


            name = row.get(
                "종목명",
                ""
            )


            if row[
                "Action"
            ] == "BUY_BLOCKED":

                print(
                    f"{code} {name}: "
                    f"BUY BLOCKED "
                    f"({row['BlockReason']})"
                )


            else:

                print(
                    f"{code} {name}: "
                    f"{row['Recommendation']}"
                )


    # =========================================================================
    # Compliance
    # =========================================================================

    print()
    print(
        "=" * 80
    )


    print(
        "조정 후 COMPLIANCE"
    )


    print(
        "=" * 80
    )


    compliance_display = compliance.copy()


    for col in [
        "현재",
        "내부한도",
        "공식한도"
    ]:

        compliance_display[
            col
        ] = (
            compliance_display[
                col
            ]
            *
            100
        )


    print(
        compliance_display
        .rename(
            columns={
                "현재":
                    "현재(%)",

                "내부한도":
                    "내부한도(%)",

                "공식한도":
                    "공식한도(%)"
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
        f"내부 small-cap (<1.1조): "
        f"{internal_small_weight * 100:.2f}% "
        f"/ "
        f"{INTERNAL_SMALL_CAP_LIMIT * 100:.2f}%"
    )


    print(
        f"공식 small-cap (<1조)  : "
        f"{official_small_weight * 100:.2f}% "
        f"/ "
        f"{OFFICIAL_SMALL_CAP_LIMIT * 100:.2f}%"
    )


    if compliance_override:

        print()
        print(
            "※ COMPLIANCE OVERRIDE 적용"
        )


        print(
            "0.75%p band보다 규정 준수를 우선했습니다."
        )


    print()
    print(
        "FINAL DAILY COMPLIANCE: ALL PASS"
    )


    # =========================================================================
    # Turnover
    # =========================================================================

    print()
    print(
        "-" * 80
    )


    print(
        f"주간 회전율 proxy : "
        f"{weekly_turnover * 100:.2f}%"
    )


    print(
        f"대회 기준         : "
        f"{WEEKLY_TURNOVER_TARGET * 100:.2f}% 이상"
    )


    if weekly_turnover >= WEEKLY_TURNOVER_TARGET:

        print(
            "상태              : 5% 이상"
        )

    else:

        print(
            "상태              : 5% 미만"
        )


    print()
    print(
        "[주의] 회전율은 공식 산식 확인 전 운영용 proxy입니다."
    )


# =============================================================================
# 26. MAIN
# =============================================================================

def main():

    print(
        "=" * 150
    )


    print(
        "DART QUANT - DAILY SIGNAL"
    )


    print(
        "Auto Current Weight / Double Compliance Check"
    )


    print(
        "=" * 150
    )


    # =========================================================================
    # Target
    # =========================================================================

    (
        target,
        signal_date
    ) = load_target()


    # =========================================================================
    # Previous State
    # =========================================================================

    (
        previous_state,
        previous_state_date
    ) = load_previous_state()


    initial_run = (
        previous_state.empty
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
        sector_table
    ) = load_sector_info(
        listing
    )


    # =========================================================================
    # Need price for Target + Previous Holdings
    # =========================================================================

    previous_codes = (
        previous_state[
            "종목코드"
        ].tolist()

        if not previous_state.empty

        else []
    )


    all_codes = sorted(
        set(
            target[
                "종목코드"
            ]
        )
        |
        set(
            previous_codes
        )
    )


    # =========================================================================
    # Prices
    # =========================================================================

    prices = download_prices(
        all_codes,
        signal_date
    )


    missing_codes = [
        code

        for code in all_codes

        if code not in prices
    ]


    if missing_codes:

        raise RuntimeError(
            "가격 데이터가 없는 종목:\n"
            +
            ", ".join(
                missing_codes
            )
        )


    # =========================================================================
    # Auto Current Weight
    # =========================================================================

    current = build_auto_current_weights(
        previous_state,
        previous_state_date,
        signal_date,
        prices
    )


    # =========================================================================
    # Safe Signal
    # =========================================================================

    (
        signal,
        compliance,
        compliance_override,
        internal_small_weight,
        official_small_weight
    ) = build_safe_signal(
        target,
        current,
        signal_date,
        prices,
        listing_map,
        sector_map,
        sector_table,
        admin_codes,
        manual_restricted,
        initial_run
    )


    signal = add_recommendation_text(
        signal
    )


    # =========================================================================
    # Weekly Turnover Proxy
    # =========================================================================

    weekly_turnover = (
        calculate_weekly_turnover_proxy(
            signal_date,
            current_signal=signal
        )
    )


    # =========================================================================
    # Save Output
    # =========================================================================

    save_outputs(
        signal,
        compliance,
        signal_date,
        compliance_override,
        weekly_turnover,
        internal_small_weight,
        official_small_weight
    )


    # =========================================================================
    # Save Assumed Executed State
    #
    # 오늘 추천 매매를 실제로 모두 실행했다고 가정
    # =========================================================================

    save_portfolio_state(
        signal,
        signal_date,
        prices
    )


    # =========================================================================
    # Print
    # =========================================================================

    print_results(
        signal,
        compliance,
        compliance_override,
        weekly_turnover,
        internal_small_weight,
        official_small_weight
    )


    print()
    print(
        "=" * 150
    )


    print(
        "Daily Signal 생성 완료"
    )


    print(
        "=" * 150
    )


    print()
    print(
        "전체 Signal:"
    )


    print(
        LATEST_SIGNAL_FILE
    )


    print()
    print(
        "변경 리스트:"
    )


    print(
        LATEST_ORDER_FILE
    )


    print()
    print(
        "Portfolio State:"
    )


    print(
        LATEST_STATE_FILE
    )


    print()
    print(
        "Compliance:"
    )


    print(
        LATEST_COMPLIANCE_FILE
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()