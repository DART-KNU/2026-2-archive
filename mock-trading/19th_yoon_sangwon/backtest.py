import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

from scipy.optimize import minimize, linprog


warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATH
# =============================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FUNDAMENTAL_FILE = os.path.join(
    BASE_DIR,
    "fundamental",
    "sector_fundamental_selected.csv"
)

GRID_FILE = os.path.join(
    BASE_DIR,
    "grid_20260929.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "backtest_independent_2m_rule_fixed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "independent_2m_summary.csv"
)

TARGET_FILE = os.path.join(
    OUTPUT_DIR,
    "independent_2m_target_history.csv"
)

DIAGNOSTIC_FILE = os.path.join(
    OUTPUT_DIR,
    "independent_2m_diagnostics.csv"
)

DAILY_FILE = os.path.join(
    OUTPUT_DIR,
    "independent_2m_daily.csv"
)

BLOCKED_BUY_FILE = os.path.join(
    OUTPUT_DIR,
    "blocked_buy_history.csv"
)

CONFIG_FILE = os.path.join(
    OUTPUT_DIR,
    "strategy_config.txt"
)


# =============================================================================
# 2. BACKTEST DESIGN
# =============================================================================

TOTAL_BACKTEST_MONTHS = 12

WINDOW_MONTHS = 2

ROLLING_STEP_MONTHS = 1


# =============================================================================
# 3. CAPITAL / COST
# =============================================================================

INITIAL_CAPITAL = 1_000_000_000

COMMISSION_RATE = 0.001

SELL_TAX_RATE = 0.002


# =============================================================================
# 4. MOMENTUM
# =============================================================================

SHORT_PERIOD = 20

MID_PERIOD = 60

SHORT_WEIGHT = 0.60

MID_WEIGHT = 0.40


MOMENTUM_ENTRY_RANK = 80

MOMENTUM_EXIT_RANK = 100


MIN_PRICE_OBSERVATIONS = 61


# =============================================================================
# 5. TRADABILITY
# =============================================================================

MIN_MARKET_CAP = 100_000_000_000

MIN_AVG_TRADING_VALUE_5D = 3_000_000_000


# =============================================================================
# 6. PORTFOLIO
# =============================================================================

TARGET_HOLDINGS = 25

MIN_EFFECTIVE_WEIGHT = 1e-8

INTERNAL_MAX_STOCK_WEIGHT = 0.05


# =============================================================================
# 7. OFFICIAL STOCK CAPS
# =============================================================================

SAMSUNG_CODE = "005930"

OFFICIAL_SAMSUNG_MAX = 0.40

OFFICIAL_OTHER_MAX = 0.15


# =============================================================================
# 8. MARKET REGIME
# =============================================================================

BULL_EXPOSURE = 0.95

MIXED_EXPOSURE = 0.80

BEAR_EXPOSURE = 0.60


# =============================================================================
# 9. REBALANCE
# =============================================================================

REBALANCE_BAND = 0.0075


# =============================================================================
# 10. SECTOR
# =============================================================================

OFFICIAL_MIN_SECTOR_LIMIT = 0.10

# 공식 섹터 한도의 90%만 내부 한도로 사용
SECTOR_LIMIT_USAGE_RATIO = 0.90

# 공식: 시가총액 < 1조원 합계 <= 30%
OFFICIAL_SMALL_CAP_THRESHOLD = 1_000_000_000_000
OFFICIAL_SMALL_CAP_LIMIT = 0.30

# 내부 버퍼: 시가총액 < 1.1조원 합계 <= 25%
INTERNAL_SMALL_CAP_THRESHOLD = 1_100_000_000_000
INTERNAL_SMALL_CAP_LIMIT = 0.25


# =============================================================================
# 11. PRICE LOOKBACK
# =============================================================================

WARMUP_CALENDAR_DAYS = 220


# =============================================================================
# 12. NORMALIZE CODE
# =============================================================================

def normalize_code(code):

    if pd.isna(code):
        return None

    code = str(code).strip()

    if code.endswith(".0"):
        code = code[:-2]

    if code.startswith("A"):
        code = code[1:]

    return code.zfill(6)


# =============================================================================
# 13. FUNDAMENTAL
# =============================================================================

def load_fundamental():

    print()
    print("[1] Fundamental Universe...")

    df = pd.read_csv(
        FUNDAMENTAL_FILE,
        dtype={
            "종목코드": str
        }
    )

    df["종목코드"] = (
        df["종목코드"]
        .apply(normalize_code)
    )

    df = (
        df
        .dropna(
            subset=[
                "종목코드"
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
        f"Fundamental 후보: {len(df):,}개"
    )

    return df


# =============================================================================
# 14. GRID
# =============================================================================

def load_grid():

    print()
    print("[2] Grid...")

    grid = pd.read_csv(
        GRID_FILE,
        dtype=str
    )

    grid["종목코드"] = (
        grid["종목코드"]
        .apply(normalize_code)
    )

    grid = (
        grid
        .drop_duplicates(
            "종목코드"
        )
        .reset_index(
            drop=True
        )
    )

    return grid


# =============================================================================
# 15. CURRENT LISTING
# =============================================================================

def load_current_listing():

    print()
    print("[3] 현재 KOSPI / KOSDAQ 정보...")

    frames = []

    for market in [
        "KOSPI",
        "KOSDAQ"
    ]:

        temp = (
            fdr.StockListing(
                market
            )
            .copy()
        )

        temp["시장"] = market

        frames.append(
            temp
        )

    listing = pd.concat(
        frames,
        ignore_index=True
    )

    listing["Code"] = (
        listing["Code"]
        .astype(str)
        .apply(normalize_code)
    )

    listing = (
        listing
        .drop_duplicates(
            "Code"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"현재 상장종목: {len(listing):,}개"
    )

    return listing


# =============================================================================
# 16. MARKET INDEX
# =============================================================================

def load_index_data():

    print()
    print("[4] 시장지수 데이터...")

    today = datetime.today()

    download_start = (
        today
        - timedelta(
            days=650
        )
    )

    kospi = (
        fdr.DataReader(
            "KS11",
            download_start,
            today
        )
        .sort_index()
    )

    kosdaq = (
        fdr.DataReader(
            "KQ11",
            download_start,
            today
        )
        .sort_index()
    )

    if kospi.empty:

        raise RuntimeError(
            "KOSPI 데이터를 불러오지 못했습니다."
        )

    end_date = (
        kospi.index[-1]
    )

    start_date = (
        end_date
        - pd.DateOffset(
            months=TOTAL_BACKTEST_MONTHS
        )
    )

    trading_dates = (
        kospi.index[
            (
                kospi.index
                >= start_date
            )
            &
            (
                kospi.index
                <= end_date
            )
        ]
    )

    print(
        f"전체 검증기간: "
        f"{trading_dates[0].date()} "
        f"~ "
        f"{trading_dates[-1].date()}"
    )

    return (
        kospi,
        kosdaq,
        trading_dates
    )


# =============================================================================
# 17. INDEPENDENT 2-MONTH WINDOWS
# =============================================================================

def build_independent_windows(
    trading_dates
):

    first_date = (
        trading_dates[0]
    )

    final_date = (
        trading_dates[-1]
    )

    windows = []

    start = pd.Timestamp(
        first_date
    )

    window_id = 1


    while True:

        raw_end = (
            start
            + pd.DateOffset(
                months=WINDOW_MONTHS
            )
        )

        if raw_end > final_date:
            break


        possible_start = (
            trading_dates[
                trading_dates
                >= start
            ]
        )

        possible_end = (
            trading_dates[
                trading_dates
                <= raw_end
            ]
        )


        if (
            len(possible_start) == 0
            or
            len(possible_end) == 0
        ):

            break


        actual_start = (
            possible_start[0]
        )

        actual_end = (
            possible_end[-1]
        )


        windows.append(
            {
                "Window":
                    window_id,

                "Start":
                    actual_start,

                "End":
                    actual_end
            }
        )


        window_id += 1

        start = (
            start
            + pd.DateOffset(
                months=ROLLING_STEP_MONTHS
            )
        )


    print()
    print(
        f"독립 2개월 Window: {len(windows)}개"
    )


    for window in windows:

        print(
            f"{window['Window']:02d}: "
            f"{window['Start'].date()} "
            f"~ "
            f"{window['End'].date()}"
        )


    return windows


# =============================================================================
# 18. TWICE-WEEKLY TARGET RECALCULATION
# =============================================================================

def build_twice_weekly_rebalance_dates(
    all_trading_dates
):

    temp = pd.DataFrame(
        {
            "Date":
                pd.DatetimeIndex(
                    all_trading_dates
                )
        }
    )


    iso = (
        temp["Date"]
        .dt
        .isocalendar()
    )


    temp["ISOYear"] = (
        iso["year"]
        .astype(int)
    )

    temp["ISOWeek"] = (
        iso["week"]
        .astype(int)
    )

    temp["Weekday"] = (
        temp["Date"]
        .dt
        .weekday
    )


    schedule = set()


    for _, group in (
        temp.groupby(
            [
                "ISOYear",
                "ISOWeek"
            ]
        )
    ):

        group = (
            group
            .sort_values(
                "Date"
            )
        )


        schedule.add(
            group[
                "Date"
            ]
            .iloc[0]
        )


        late_week = (
            group[
                group[
                    "Weekday"
                ]
                >= 3
            ]
        )


        if not late_week.empty:

            schedule.add(
                late_week[
                    "Date"
                ]
                .iloc[0]
            )


    return schedule


# =============================================================================
# 19. DOWNLOAD STOCK PRICES
# =============================================================================

def download_price_history(
    fundamental,
    trading_dates
):

    print()
    print("[5] 종목 가격데이터 다운로드...")

    start = (
        trading_dates[0]
        - timedelta(
            days=WARMUP_CALENDAR_DAYS
        )
    )

    end = (
        trading_dates[-1]
        + timedelta(
            days=3
        )
    )


    price_data = {}


    total = (
        len(
            fundamental
        )
    )


    for i, (_, row) in enumerate(
        fundamental.iterrows(),
        start=1
    ):

        code = (
            row[
                "종목코드"
            ]
        )


        try:

            price = (
                fdr.DataReader(
                    code,
                    start.strftime(
                        "%Y-%m-%d"
                    ),
                    end.strftime(
                        "%Y-%m-%d"
                    )
                )
                .sort_index()
            )

        except Exception:

            continue


        if (
            price is None
            or
            price.empty
            or
            "Close"
            not in price.columns
        ):

            continue


        price_data[
            code
        ] = price


        if (
            i % 25 == 0
            or
            i == total
        ):

            print(
                f"{i}/{total}"
            )


    print(
        f"가격 확보 종목: "
        f"{len(price_data):,}개"
    )


    return price_data


# =============================================================================
# 20. MARKET CAP REFERENCE
# =============================================================================

def build_market_cap_reference(
    listing,
    price_data
):

    cap_map = {}


    listing_map = (
        listing
        .set_index(
            "Code"
        )
    )


    for code, price in (
        price_data.items()
    ):

        if code not in listing_map.index:
            continue


        row = (
            listing_map.loc[
                code
            ]
        )


        if isinstance(
            row,
            pd.DataFrame
        ):

            row = (
                row.iloc[0]
            )


        if "Marcap" not in row.index:
            continue


        current_cap = (
            pd.to_numeric(
                row[
                    "Marcap"
                ],
                errors="coerce"
            )
        )


        if pd.isna(
            current_cap
        ):

            continue


        close = (
            pd.to_numeric(
                price[
                    "Close"
                ],
                errors="coerce"
            )
            .dropna()
        )


        if close.empty:
            continue


        current_price = float(
            close.iloc[-1]
        )


        if current_price <= 0:
            continue


        cap_map[
            code
        ] = {
            "current_cap":
                float(
                    current_cap
                ),

            "current_price":
                current_price
        }


    return cap_map


# =============================================================================
# 21. ESTIMATED HISTORICAL MARKET CAP
# =============================================================================

def estimate_market_cap(
    code,
    date,
    price_data,
    cap_reference
):

    if code not in cap_reference:
        return np.nan

    if code not in price_data:
        return np.nan


    history = (
        price_data[
            code
        ][
            price_data[
                code
            ].index
            <= date
        ]
    )


    if history.empty:
        return np.nan


    close = (
        pd.to_numeric(
            history[
                "Close"
            ],
            errors="coerce"
        )
        .dropna()
    )


    if close.empty:
        return np.nan


    historical_price = float(
        close.iloc[-1]
    )


    current_price = (
        cap_reference[
            code
        ][
            "current_price"
        ]
    )

    current_cap = (
        cap_reference[
            code
        ][
            "current_cap"
        ]
    )


    if current_price <= 0:
        return np.nan


    return (
        current_cap
        *
        historical_price
        /
        current_price
    )


# =============================================================================
# 22. SECTOR LIMITS
# =============================================================================

def build_sector_limits(
    listing,
    grid
):

    market = (
        listing[
            [
                "Code",
                "Marcap"
            ]
        ]
        .copy()
    )


    market = (
        market.rename(
            columns={
                "Code":
                    "종목코드",

                "Marcap":
                    "시가총액"
            }
        )
    )


    market["종목코드"] = (
        market[
            "종목코드"
        ]
        .apply(
            normalize_code
        )
    )


    market["시가총액"] = (
        pd.to_numeric(
            market[
                "시가총액"
            ],
            errors="coerce"
        )
    )


    merged = (
        grid[
            [
                "종목코드",
                "섹터명"
            ]
        ]
        .merge(
            market,
            on="종목코드",
            how="inner"
        )
        .dropna(
            subset=[
                "시가총액"
            ]
        )
    )


    total_cap = (
        merged[
            "시가총액"
        ]
        .sum()
    )


    sector = (
        merged
        .groupby(
            "섹터명"
        )[
            "시가총액"
        ]
        .sum()
        .reset_index()
    )


    sector[
        "시장비중"
    ] = (
        sector[
            "시가총액"
        ]
        /
        total_cap
    )


    sector[
        "공식한도"
    ] = (
        np.minimum(
            np.maximum(
                sector[
                    "시장비중"
                ]
                * 2,
                OFFICIAL_MIN_SECTOR_LIMIT
            ),
            1.0
        )
    )


    sector[
        "내부한도"
    ] = (
        sector[
            "공식한도"
        ]
        * SECTOR_LIMIT_USAGE_RATIO
    )


    return dict(
        zip(
            sector[
                "섹터명"
            ],
            sector[
                "내부한도"
            ]
        )
    )


# =============================================================================
# 23. DAILY MOMENTUM
# =============================================================================

def calculate_full_momentum(
    date,
    fundamental,
    price_data
):

    rows = []


    for _, stock in (
        fundamental.iterrows()
    ):

        code = (
            stock[
                "종목코드"
            ]
        )


        if code not in price_data:
            continue


        history = (
            price_data[
                code
            ][
                price_data[
                    code
                ].index
                <= date
            ]
        )


        close = (
            pd.to_numeric(
                history[
                    "Close"
                ],
                errors="coerce"
            )
            .dropna()
        )


        if (
            len(close)
            <
            MIN_PRICE_OBSERVATIONS
        ):

            continue


        current = float(
            close.iloc[-1]
        )

        close20 = float(
            close.iloc[
                -(SHORT_PERIOD + 1)
            ]
        )

        close60 = float(
            close.iloc[
                -(MID_PERIOD + 1)
            ]
        )


        ret20 = (
            current
            /
            close20
            - 1
        )

        ret60 = (
            current
            /
            close60
            - 1
        )


        ma20 = float(
            close
            .iloc[
                -SHORT_PERIOD:
            ]
            .mean()
        )


        absolute_pass = (
            ret20 > 0
            or
            current > ma20
        )


        if not absolute_pass:
            continue


        rows.append(
            {
                "종목코드":
                    code,

                "수익률20D":
                    ret20,

                "수익률60D":
                    ret60,

                "가격관측수":
                    len(
                        close
                    )
            }
        )


    df = pd.DataFrame(
        rows
    )


    if df.empty:
        return df


    df[
        "20DScore"
    ] = (
        df[
            "수익률20D"
        ]
        .rank(
            pct=True
        )
        * 100
    )


    df[
        "60DScore"
    ] = (
        df[
            "수익률60D"
        ]
        .rank(
            pct=True
        )
        * 100
    )


    df[
        "MomentumScore"
    ] = (
        SHORT_WEIGHT
        *
        df[
            "20DScore"
        ]
        +
        MID_WEIGHT
        *
        df[
            "60DScore"
        ]
    )


    df = (
        df
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
    ] = (
        np.arange(
            1,
            len(df) + 1
        )
    )


    return df


# =============================================================================
# 24. MOMENTUM HYSTERESIS
# =============================================================================

def apply_momentum_hysteresis(
    ranking,
    previous_target
):

    if ranking.empty:

        return (
            ranking,
            set(),
            0,
            0
        )


    entry = (
        ranking[
            ranking[
                "MomentumRank"
            ]
            <=
            MOMENTUM_ENTRY_RANK
        ]
    )


    retention = (
        ranking[
            ranking[
                "MomentumRank"
            ]
            <=
            MOMENTUM_EXIT_RANK
        ]
    )


    previous_codes = set(
        previous_target.keys()
    )


    retained_previous = (
        previous_codes
        &
        set(
            retention[
                "종목코드"
            ]
        )
    )


    allowed_codes = (
        set(
            entry[
                "종목코드"
            ]
        )
        |
        retained_previous
    )


    candidates = (
        ranking[
            ranking[
                "종목코드"
            ]
            .isin(
                allowed_codes
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    return (
        candidates,
        allowed_codes,
        len(
            entry
        ),
        len(
            retained_previous
        )
    )


# =============================================================================
# 25. HISTORICAL 5D TRADING VALUE
# =============================================================================

def historical_trading_value_5d(
    code,
    date,
    price_data
):

    if code not in price_data:
        return np.nan


    history = (
        price_data[
            code
        ][
            price_data[
                code
            ].index
            <= date
        ]
    )


    if len(history) < 5:
        return np.nan


    close = (
        pd.to_numeric(
            history[
                "Close"
            ],
            errors="coerce"
        )
    )

    volume = (
        pd.to_numeric(
            history[
                "Volume"
            ],
            errors="coerce"
        )
    )


    trading_value = (
        close
        *
        volume
    )


    trading_value = (
        trading_value
        .dropna()
        .iloc[-5:]
    )


    if len(trading_value) < 5:
        return np.nan


    return float(
        trading_value.mean()
    )


# =============================================================================
# 26. TARGET-CONSTRUCTION TRADABLE FILTER
# =============================================================================

def historical_tradable_filter(
    date,
    momentum,
    price_data,
    cap_reference,
    sector_map
):

    rows = []


    for _, row in (
        momentum.iterrows()
    ):

        code = (
            row[
                "종목코드"
            ]
        )


        market_cap = (
            estimate_market_cap(
                code,
                date,
                price_data,
                cap_reference
            )
        )


        trading_value = (
            historical_trading_value_5d(
                code,
                date,
                price_data
            )
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

            continue


        if (
            pd.isna(
                trading_value
            )
            or
            trading_value
            <=
            MIN_AVG_TRADING_VALUE_5D
        ):

            continue


        sector = (
            sector_map.get(
                code
            )
        )


        if pd.isna(
            sector
        ):

            continue


        new_row = (
            row.to_dict()
        )


        new_row[
            "시가총액"
        ] = (
            market_cap
        )


        new_row[
            "5일평균거래대금"
        ] = (
            trading_value
        )


        new_row[
            "섹터명"
        ] = (
            sector
        )


        rows.append(
            new_row
        )


    return pd.DataFrame(
        rows
    )


# =============================================================================
# 27. EXECUTION-DAY BUY ELIGIBILITY
#
# IMPORTANT:
# Actual trade on date T uses signal from previous trading day T-1.
#
# Therefore eligibility is checked using signal_date only.
#
# This prevents:
# - using today's close / volume to decide today's buy
# - unintended look-ahead in execution filter
#
# Only BUY is blocked.
# Existing holdings are not force-sold here.
# =============================================================================

def buy_eligible(
    code,
    signal_date,
    price_data,
    cap_reference
):

    market_cap = (
        estimate_market_cap(
            code,
            signal_date,
            price_data,
            cap_reference
        )
    )


    trading_value = (
        historical_trading_value_5d(
            code,
            signal_date,
            price_data
        )
    )


    market_cap_pass = (
        pd.notna(
            market_cap
        )
        and
        market_cap
        >=
        MIN_MARKET_CAP
    )


    liquidity_pass = (
        pd.notna(
            trading_value
        )
        and
        trading_value
        >
        MIN_AVG_TRADING_VALUE_5D
    )


    eligible = (
        market_cap_pass
        and
        liquidity_pass
    )


    return {
        "Eligible":
            eligible,

        "MarketCap":
            market_cap,

        "TradingValue5D":
            trading_value,

        "MarketCapPass":
            market_cap_pass,

        "LiquidityPass":
            liquidity_pass
    }


# =============================================================================
# 28. MARKET REGIME
# =============================================================================

def index_above_ma60(
    index_df,
    date
):

    history = (
        index_df[
            index_df.index
            <= date
        ]
    )


    close = (
        pd.to_numeric(
            history[
                "Close"
            ],
            errors="coerce"
        )
        .dropna()
    )


    if len(close) < 60:
        return False


    current = float(
        close.iloc[-1]
    )


    ma60 = float(
        close.iloc[-60:]
        .mean()
    )


    return (
        current > ma60
    )


def get_equity_exposure(
    date,
    kospi,
    kosdaq
):

    kospi_above = (
        index_above_ma60(
            kospi,
            date
        )
    )


    kosdaq_above = (
        index_above_ma60(
            kosdaq,
            date
        )
    )


    count = (
        int(
            kospi_above
        )
        +
        int(
            kosdaq_above
        )
    )


    if count == 2:
        return BULL_EXPOSURE


    if count == 1:
        return MIXED_EXPOSURE


    return BEAR_EXPOSURE


# =============================================================================
# 29. STOCK CAP
# =============================================================================

def stock_max_weight(
    code
):

    if code == SAMSUNG_CODE:

        official = (
            OFFICIAL_SAMSUNG_MAX
        )

    else:

        official = (
            OFFICIAL_OTHER_MAX
        )


    return min(
        official,
        INTERNAL_MAX_STOCK_WEIGHT
    )


# =============================================================================
# 30. FEASIBLE WEIGHTS
# =============================================================================

def feasible_weights(
    df,
    exposure,
    sector_limits
):

    n = (
        len(df)
    )


    if n == 0:
        return None


    bounds = [
        (
            0,
            stock_max_weight(
                code
            )
        )

        for code in
        df[
            "종목코드"
        ]
    ]


    A_ub = []

    b_ub = []


    for sector, limit in (
        sector_limits.items()
    ):

        mask = (
            df[
                "섹터명"
            ]
            .eq(
                sector
            )
            .astype(float)
            .values
        )


        if mask.sum() > 0:

            A_ub.append(
                mask
            )

            b_ub.append(
                limit
            )


    market_caps = pd.to_numeric(
        df[
            "시가총액"
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )

    if np.isnan(
        market_caps
    ).any():

        return None


    internal_small_mask = (
        market_caps
        <
        INTERNAL_SMALL_CAP_THRESHOLD
    ).astype(float)

    official_small_mask = (
        market_caps
        <
        OFFICIAL_SMALL_CAP_THRESHOLD
    ).astype(float)


    A_ub.append(
        internal_small_mask
    )

    b_ub.append(
        INTERNAL_SMALL_CAP_LIMIT
    )


    A_ub.append(
        official_small_mask
    )

    b_ub.append(
        OFFICIAL_SMALL_CAP_LIMIT
    )


    result = (
        linprog(
            np.zeros(
                n
            ),

            A_ub=(
                np.array(
                    A_ub
                )
                if A_ub
                else None
            ),

            b_ub=(
                np.array(
                    b_ub
                )
                if b_ub
                else None
            ),

            A_eq=
                np.ones(
                    (
                        1,
                        n
                    )
                ),

            b_eq=
                np.array(
                    [
                        exposure
                    ]
                ),

            bounds=
                bounds,

            method=
                "highs"
        )
    )


    if not result.success:
        return None


    return result.x


# =============================================================================
# 31. MOMENTUM WEIGHT OPTIMIZATION
# =============================================================================

def optimize_momentum_weight(
    df,
    exposure,
    sector_limits
):

    df = (
        df
        .copy()
        .sort_values(
            "MomentumScore",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    n = (
        len(df)
    )


    if n == 0:

        return (
            None,
            False
        )


    scores = (
        pd.to_numeric(
            df[
                "MomentumScore"
            ],
            errors="coerce"
        )
        .fillna(0)
        .clip(
            lower=0
        )
        .values
    )


    if scores.sum() <= 0:

        raw_fraction = (
            np.ones(
                n
            )
            /
            n
        )

    else:

        raw_fraction = (
            scores
            /
            scores.sum()
        )


    raw_weights = (
        raw_fraction
        *
        exposure
    )


    initial = (
        feasible_weights(
            df,
            exposure,
            sector_limits
        )
    )


    if initial is None:

        return (
            None,
            False
        )


    bounds = [
        (
            0,
            stock_max_weight(
                code
            )
        )

        for code in
        df[
            "종목코드"
        ]
    ]


    constraints = [
        {
            "type":
                "eq",

            "fun":
                lambda w:
                np.sum(w)
                -
                exposure
        }
    ]


    for sector, limit in (
        sector_limits.items()
    ):

        idx = np.where(
            df[
                "섹터명"
            ]
            .values
            ==
            sector
        )[0]


        if len(idx) == 0:
            continue


        constraints.append(
            {
                "type":
                    "ineq",

                "fun":
                    lambda w,
                    idx=idx,
                    limit=limit:
                    limit
                    -
                    np.sum(
                        w[
                            idx
                        ]
                    )
            }
        )


    market_caps = pd.to_numeric(
        df[
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


    internal_small_idx = np.where(
        market_caps
        <
        INTERNAL_SMALL_CAP_THRESHOLD
    )[0]

    official_small_idx = np.where(
        market_caps
        <
        OFFICIAL_SMALL_CAP_THRESHOLD
    )[0]


    constraints.append(
        {
            "type":
                "ineq",

            "fun":
                lambda w,
                idx=internal_small_idx:
                INTERNAL_SMALL_CAP_LIMIT
                -
                np.sum(
                    w[
                        idx
                    ]
                )
        }
    )


    constraints.append(
        {
            "type":
                "ineq",

            "fun":
                lambda w,
                idx=official_small_idx:
                OFFICIAL_SMALL_CAP_LIMIT
                -
                np.sum(
                    w[
                        idx
                    ]
                )
        }
    )


    def objective(
        weights
    ):

        return np.sum(
            (
                weights
                -
                raw_weights
            )
            ** 2
        )


    result = (
        minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter":
                    1000,

                "ftol":
                    1e-10,

                "disp":
                    False
            }
        )
    )


    if not result.success:

        return (
            None,
            False
        )


    output = (
        df.copy()
    )


    output[
        "RawMomentumWeight"
    ] = (
        raw_weights
    )


    output[
        "TargetWeight"
    ] = (
        result.x
    )


    return (
        output,
        True
    )


# =============================================================================
# 32. PORTFOLIO CONSTRUCTION
# =============================================================================

def momentum_weight_portfolio(
    df,
    exposure,
    sector_limits
):

    df = (
        df
        .copy()
        .sort_values(
            "MomentumScore",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    if df.empty:

        return (
            None,
            False
        )


    start_n = min(
        TARGET_HOLDINGS,
        len(df)
    )


    sizes = list(
        range(
            start_n,
            len(df) + 1
        )
    )


    if len(df) not in sizes:

        sizes.append(
            len(df)
        )


    for n in sizes:

        subset = (
            df
            .head(
                n
            )
            .copy()
            .reset_index(
                drop=True
            )
        )


        (
            portfolio,
            success
        ) = (
            optimize_momentum_weight(
                subset,
                exposure,
                sector_limits
            )
        )


        if (
            portfolio is None
            or
            not success
        ):

            continue


        portfolio = (
            portfolio[
                portfolio[
                    "TargetWeight"
                ]
                >=
                MIN_EFFECTIVE_WEIGHT
            ]
            .copy()
        )


        if portfolio.empty:
            continue


        return (
            portfolio,
            True
        )


    return (
        None,
        False
    )


# =============================================================================
# 33. BUILD TARGETS FOR ONE WINDOW
#
# FIXED LOGIC:
#
# - Momentum ranking is calculated every day
# - Actual target is recalculated only on scheduled days
# - Non-rebalance day target = previous_target.copy()
#
# This prevents unintended exposure decay.
# =============================================================================

def build_window_targets(
    window,
    all_trading_dates,
    fundamental,
    price_data,
    cap_reference,
    sector_map,
    sector_limits,
    kospi,
    kosdaq
):

    window_id = (
        window[
            "Window"
        ]
    )

    start = (
        window[
            "Start"
        ]
    )

    end = (
        window[
            "End"
        ]
    )


    window_dates = (
        all_trading_dates[
            (
                all_trading_dates
                >= start
            )
            &
            (
                all_trading_dates
                <= end
            )
        ]
    )


    normal_schedule = (
        build_twice_weekly_rebalance_dates(
            all_trading_dates
        )
    )


    rebalance_dates = {
        date

        for date in
        normal_schedule

        if (
            date >= start
            and
            date <= end
        )
    }


    rebalance_dates.add(
        start
    )


    previous_target = {}


    targets = {}

    allowed_sets = {}


    diagnostic_rows = []

    target_rows = []


    for date in window_dates:

        ranking = (
            calculate_full_momentum(
                date,
                fundamental,
                price_data
            )
        )


        absolute_count = (
            len(
                ranking
            )
        )


        # =====================================================================
        # RANKING EMPTY
        # =====================================================================

        if ranking.empty:

            target = (
                previous_target.copy()
            )


            targets[
                date
            ] = (
                target.copy()
            )


            allowed_sets[
                date
            ] = set(
                target.keys()
            )


            diagnostic_rows.append(
                {
                    "Window":
                        window_id,

                    "Date":
                        date,

                    "RebalanceDay":
                        date
                        in
                        rebalance_dates,

                    "AbsoluteMomentumCount":
                        0,

                    "EntryTop80Count":
                        0,

                    "RetainedPreviousCount":
                        0,

                    "HysteresisCandidateCount":
                        0,

                    "TradableCount":
                        np.nan,

                    "RequestedExposure":
                        np.nan,

                    "TargetCount":
                        len(
                            target
                        ),

                    "TargetExposure":
                        sum(
                            target.values()
                        ),

                    "PortfolioRun":
                        False,

                    "PortfolioSuccess":
                        False,

                    "FallbackPreviousTarget":
                        True
                }
            )


            previous_target = (
                target.copy()
            )


            continue


        # =====================================================================
        # DAILY HYSTERESIS CANDIDATES
        # =====================================================================

        (
            momentum_candidates,
            allowed_codes,
            entry_count,
            retained_previous_count
        ) = (
            apply_momentum_hysteresis(
                ranking,
                previous_target
            )
        )


        # =====================================================================
        # NON-REBALANCE DAY
        # =====================================================================

        if (
            date
            not in
            rebalance_dates
        ):

            target = (
                previous_target.copy()
            )


            targets[
                date
            ] = (
                target.copy()
            )


            allowed_sets[
                date
            ] = set(
                target.keys()
            )


            diagnostic_rows.append(
                {
                    "Window":
                        window_id,

                    "Date":
                        date,

                    "RebalanceDay":
                        False,

                    "AbsoluteMomentumCount":
                        absolute_count,

                    "EntryTop80Count":
                        entry_count,

                    "RetainedPreviousCount":
                        retained_previous_count,

                    "HysteresisCandidateCount":
                        len(
                            momentum_candidates
                        ),

                    "TradableCount":
                        np.nan,

                    "RequestedExposure":
                        np.nan,

                    "TargetCount":
                        len(
                            target
                        ),

                    "TargetExposure":
                        sum(
                            target.values()
                        ),

                    "PortfolioRun":
                        False,

                    "PortfolioSuccess":
                        np.nan,

                    "FallbackPreviousTarget":
                        False
                }
            )


            previous_target = (
                target.copy()
            )


            continue


        # =====================================================================
        # PORTFOLIO RECALCULATION DAY
        # =====================================================================

        tradable = (
            historical_tradable_filter(
                date,
                momentum_candidates,
                price_data,
                cap_reference,
                sector_map
            )
        )


        tradable_count = (
            len(
                tradable
            )
        )


        requested_exposure = (
            get_equity_exposure(
                date,
                kospi,
                kosdaq
            )
        )


        if (
            tradable.empty
            or
            tradable_count < 10
        ):

            target = (
                previous_target.copy()
            )

            success = False

            fallback = True


        else:

            (
                portfolio,
                success
            ) = (
                momentum_weight_portfolio(
                    tradable,
                    requested_exposure,
                    sector_limits
                )
            )


            if (
                portfolio is None
                or
                portfolio.empty
            ):

                target = (
                    previous_target.copy()
                )

                success = False

                fallback = True


            else:

                target = dict(
                    zip(
                        portfolio[
                            "종목코드"
                        ],
                        portfolio[
                            "TargetWeight"
                        ]
                    )
                )

                fallback = False


        targets[
            date
        ] = (
            target.copy()
        )


        allowed_sets[
            date
        ] = set(
            target.keys()
        )


        diagnostic_rows.append(
            {
                "Window":
                    window_id,

                "Date":
                    date,

                "RebalanceDay":
                    True,

                "AbsoluteMomentumCount":
                    absolute_count,

                "EntryTop80Count":
                    entry_count,

                "RetainedPreviousCount":
                    retained_previous_count,

                "HysteresisCandidateCount":
                    len(
                        momentum_candidates
                    ),

                "TradableCount":
                    tradable_count,

                "RequestedExposure":
                    requested_exposure,

                "TargetCount":
                    len(
                        target
                    ),

                "TargetExposure":
                    sum(
                        target.values()
                    ),

                "PortfolioRun":
                    True,

                "PortfolioSuccess":
                    success,

                "FallbackPreviousTarget":
                    fallback
            }
        )


        previous_target = (
            target.copy()
        )


    # =========================================================================
    # TARGET HISTORY
    # =========================================================================

    for date, target in (
        targets.items()
    ):

        for code, weight in (
            target.items()
        ):

            target_rows.append(
                {
                    "Window":
                        window_id,

                    "Date":
                        date,

                    "종목코드":
                        code,

                    "TargetWeight":
                        weight
                }
            )


    return (
        window_dates,
        targets,
        allowed_sets,
        pd.DataFrame(
            diagnostic_rows
        ),
        pd.DataFrame(
            target_rows
        )
    )


# =============================================================================
# 34. GET CLOSE
# =============================================================================

def get_close(
    code,
    date,
    price_data
):

    if code not in price_data:
        return np.nan


    history = (
        price_data[
            code
        ][
            price_data[
                code
            ].index
            <= date
        ]
    )


    if history.empty:
        return np.nan


    close = (
        pd.to_numeric(
            history[
                "Close"
            ],
            errors="coerce"
        )
        .dropna()
    )


    if close.empty:
        return np.nan


    return float(
        close.iloc[-1]
    )


# =============================================================================
# 35. BACKTEST ONE INDEPENDENT WINDOW
#
# IMPORTANT:
# Execution date T uses target generated on previous trading day T-1.
#
# BUY eligibility is also checked using T-1 information.
# =============================================================================

def backtest_independent_window(
    window,
    window_dates,
    targets,
    allowed_sets,
    price_data,
    cap_reference,
    kospi
):

    window_id = (
        window[
            "Window"
        ]
    )

    start = (
        window[
            "Start"
        ]
    )

    end = (
        window[
            "End"
        ]
    )


    # =========================================================================
    # WINDOW RESET
    # =========================================================================

    cash = float(
        INITIAL_CAPITAL
    )

    positions = {}

    previous_prices = {}


    daily_rows = []

    blocked_buy_rows = []


    total_cost = 0.0

    total_blocked_buy_value = 0.0

    blocked_buy_count = 0

    blocked_market_cap_count = 0

    blocked_liquidity_count = 0


    trade_days = 0


    previous_date = None


    for date in window_dates:

        # =====================================================================
        # MARK POSITIONS TO MARKET
        # =====================================================================

        for code in list(
            positions.keys()
        ):

            current_price = (
                get_close(
                    code,
                    date,
                    price_data
                )
            )


            previous_price = (
                previous_prices.get(
                    code
                )
            )


            if (
                pd.notna(
                    current_price
                )
                and
                previous_price is not None
                and
                previous_price > 0
            ):

                positions[
                    code
                ] *= (
                    current_price
                    /
                    previous_price
                )


            if pd.notna(
                current_price
            ):

                previous_prices[
                    code
                ] = (
                    current_price
                )


        portfolio_value_before = (
            cash
            +
            sum(
                positions.values()
            )
        )


        if portfolio_value_before <= 0:
            continue


        # =====================================================================
        # PREVIOUS-DAY SIGNAL
        # =====================================================================

        if previous_date is None:

            target = {}

            allowed_codes = set()


        else:

            target = (
                targets.get(
                    previous_date,
                    {}
                )
            )


            allowed_codes = (
                allowed_sets.get(
                    previous_date,
                    set(
                        target.keys()
                    )
                )
            )


        # =====================================================================
        # CURRENT WEIGHTS
        # =====================================================================

        current_weights = {}


        for code, value in (
            positions.items()
        ):

            current_weights[
                code
            ] = (
                value
                /
                portfolio_value_before
            )


        all_codes = (
            set(
                current_weights.keys()
            )
            |
            set(
                target.keys()
            )
        )


        trade_values = {}


        # =====================================================================
        # BAND
        # =====================================================================

        for code in all_codes:

            current_weight = (
                current_weights.get(
                    code,
                    0.0
                )
            )


            target_weight = (
                target.get(
                    code,
                    0.0
                )
            )


            diff = (
                target_weight
                -
                current_weight
            )


            force_exit = (
                current_weight > 0
                and
                code not in allowed_codes
                and
                code not in target
            )


            if force_exit:

                trade_values[
                    code
                ] = (
                    -positions.get(
                        code,
                        0.0
                    )
                )


            elif (
                abs(
                    diff
                )
                >=
                REBALANCE_BAND
            ):

                trade_values[
                    code
                ] = (
                    diff
                    *
                    portfolio_value_before
                )


        day_sell = 0.0

        day_buy = 0.0

        day_cost = 0.0

        day_blocked_buy = 0.0

        day_blocked_count = 0


        # =====================================================================
        # SELL FIRST
        #
        # Sell is never blocked by buy-side liquidity/cap rule.
        # =====================================================================

        for code, trade_value in (
            trade_values.items()
        ):

            if trade_value >= 0:
                continue


            current_value = (
                positions.get(
                    code,
                    0.0
                )
            )


            sell_value = min(
                abs(
                    trade_value
                ),
                current_value
            )


            if sell_value <= 0:
                continue


            cost = (
                sell_value
                *
                (
                    COMMISSION_RATE
                    +
                    SELL_TAX_RATE
                )
            )


            cash += (
                sell_value
                -
                cost
            )


            positions[
                code
            ] = (
                current_value
                -
                sell_value
            )


            if (
                positions[
                    code
                ]
                <= 1
            ):

                positions.pop(
                    code,
                    None
                )


            day_sell += (
                sell_value
            )


            day_cost += (
                cost
            )


        # =====================================================================
        # BUILD BUY LIST WITH EXECUTION ELIGIBILITY
        #
        # IMPORTANT:
        # Eligibility uses previous_date.
        # =====================================================================

        positive_trades = {}


        for code, value in (
            trade_values.items()
        ):

            if value <= 0:
                continue


            if previous_date is None:
                continue


            check = (
                buy_eligible(
                    code,
                    previous_date,
                    price_data,
                    cap_reference
                )
            )


            if not check[
                "Eligible"
            ]:

                blocked_buy_count += 1

                day_blocked_count += 1

                total_blocked_buy_value += (
                    value
                )

                day_blocked_buy += (
                    value
                )


                if not check[
                    "MarketCapPass"
                ]:

                    blocked_market_cap_count += 1


                if not check[
                    "LiquidityPass"
                ]:

                    blocked_liquidity_count += 1


                blocked_buy_rows.append(
                    {
                        "Window":
                            window_id,

                        "ExecutionDate":
                            date,

                        "SignalDate":
                            previous_date,

                        "종목코드":
                            code,

                        "RequestedBuyValue":
                            value,

                        "EstimatedMarketCap":
                            check[
                                "MarketCap"
                            ],

                        "TradingValue5D":
                            check[
                                "TradingValue5D"
                            ],

                        "MarketCapPass":
                            check[
                                "MarketCapPass"
                            ],

                        "LiquidityPass":
                            check[
                                "LiquidityPass"
                            ]
                    }
                )


                continue


            positive_trades[
                code
            ] = (
                value
            )


        requested_buy = (
            sum(
                positive_trades.values()
            )
        )


        if requested_buy > 0:

            max_buy = (
                cash
                /
                (
                    1
                    +
                    COMMISSION_RATE
                )
            )


            scale = min(
                1.0,
                max_buy
                /
                requested_buy
            )


        else:

            scale = 0.0


        # =====================================================================
        # BUY
        # =====================================================================

        for code, requested in (
            positive_trades.items()
        ):

            buy_value = (
                requested
                *
                scale
            )


            if buy_value <= 0:
                continue


            cost = (
                buy_value
                *
                COMMISSION_RATE
            )


            cash -= (
                buy_value
                +
                cost
            )


            positions[
                code
            ] = (
                positions.get(
                    code,
                    0.0
                )
                +
                buy_value
            )


            day_buy += (
                buy_value
            )


            day_cost += (
                cost
            )


            current_price = (
                get_close(
                    code,
                    date,
                    price_data
                )
            )


            if pd.notna(
                current_price
            ):

                previous_prices[
                    code
                ] = (
                    current_price
                )


        total_cost += (
            day_cost
        )


        day_turnover = (
            (
                day_buy
                +
                day_sell
            )
            /
            portfolio_value_before
        )


        if day_turnover > 0:

            trade_days += 1


        end_value = (
            cash
            +
            sum(
                positions.values()
            )
        )


        daily_rows.append(
            {
                "Window":
                    window_id,

                "Date":
                    date,

                "PortfolioValue":
                    end_value,

                "DailyTurnover":
                    day_turnover,

                "DailyCost":
                    day_cost,

                "Holdings":
                    len(
                        positions
                    ),

                "Cash":
                    cash,

                "BlockedBuyCount":
                    day_blocked_count,

                "BlockedBuyValue":
                    day_blocked_buy
            }
        )


        previous_date = (
            date
        )


    daily = (
        pd.DataFrame(
            daily_rows
        )
    )


    blocked_buy_df = (
        pd.DataFrame(
            blocked_buy_rows
        )
    )


    if daily.empty:

        return (
            None,
            daily,
            blocked_buy_df
        )


    # =========================================================================
    # RETURN
    # =========================================================================

    daily[
        "Return"
    ] = (
        daily[
            "PortfolioValue"
        ]
        .pct_change()
        .fillna(0)
    )


    final_value = float(
        daily[
            "PortfolioValue"
        ]
        .iloc[-1]
    )


    cumulative_return = (
        final_value
        /
        INITIAL_CAPITAL
        - 1
    )


    # =========================================================================
    # SHARPE
    # =========================================================================

    daily_mean = (
        daily[
            "Return"
        ]
        .mean()
    )


    daily_std = (
        daily[
            "Return"
        ]
        .std()
    )


    if (
        pd.notna(
            daily_std
        )
        and
        daily_std > 0
    ):

        sharpe = (
            daily_mean
            /
            daily_std
            *
            np.sqrt(
                252
            )
        )


    else:

        sharpe = np.nan


    # =========================================================================
    # MDD
    # =========================================================================

    cumulative = (
        (
            1
            +
            daily[
                "Return"
            ]
        )
        .cumprod()
    )


    running_max = (
        cumulative
        .cummax()
    )


    drawdown = (
        cumulative
        /
        running_max
        - 1
    )


    mdd = float(
        drawdown.min()
    )


    # =========================================================================
    # KOSPI
    # =========================================================================

    benchmark = (
        kospi[
            (
                kospi.index
                >= start
            )
            &
            (
                kospi.index
                <= end
            )
        ][
            "Close"
        ]
        .dropna()
    )


    if len(benchmark) >= 2:

        benchmark_return = (
            benchmark.iloc[-1]
            /
            benchmark.iloc[0]
            - 1
        )


    else:

        benchmark_return = (
            np.nan
        )


    excess_return = (
        cumulative_return
        -
        benchmark_return
    )


    summary = {
        "Window":
            window_id,

        "Start":
            start.strftime(
                "%Y-%m-%d"
            ),

        "End":
            end.strftime(
                "%Y-%m-%d"
            ),

        "StrategyReturn":
            cumulative_return,

        "KOSPIReturn":
            benchmark_return,

        "ExcessReturn":
            excess_return,

        "Sharpe":
            sharpe,

        "MDD":
            mdd,

        "AvgDailyTurnover":
            daily[
                "DailyTurnover"
            ]
            .mean(),

        "TotalCost":
            total_cost,

        "CostPctInitial":
            total_cost
            /
            INITIAL_CAPITAL,

        "AvgHoldings":
            daily[
                "Holdings"
            ]
            .mean(),

        "TradeDays":
            trade_days,

        "FinalValue":
            final_value,

        "BlockedBuyCount":
            blocked_buy_count,

        "BlockedMarketCapCount":
            blocked_market_cap_count,

        "BlockedLiquidityCount":
            blocked_liquidity_count,

        "BlockedBuyValuePctInitial":
            total_blocked_buy_value
            /
            INITIAL_CAPITAL
    }


    return (
        summary,
        daily,
        blocked_buy_df
    )


# =============================================================================
# 36. RUN ALL WINDOWS
# =============================================================================

def run_independent_backtests(
    windows,
    trading_dates,
    fundamental,
    price_data,
    cap_reference,
    sector_map,
    sector_limits,
    kospi,
    kosdaq
):

    print()
    print(
        "[6] Independent 2-Month Backtests..."
    )


    summary_rows = []

    daily_frames = []

    diagnostic_frames = []

    target_frames = []

    blocked_buy_frames = []


    for window in windows:

        print()
        print("=" * 100)


        print(
            f"Window "
            f"{window['Window']:02d} | "
            f"{window['Start'].date()} "
            f"~ "
            f"{window['End'].date()}"
        )


        print(
            "RESET → Cash 10억원 / Holdings 0 / "
            "Previous Target 0 / Hysteresis 0"
        )


        (
            window_dates,
            targets,
            allowed_sets,
            diagnostics,
            target_history
        ) = (
            build_window_targets(
                window,
                trading_dates,
                fundamental,
                price_data,
                cap_reference,
                sector_map,
                sector_limits,
                kospi,
                kosdaq
            )
        )


        (
            summary,
            daily,
            blocked_buy_df
        ) = (
            backtest_independent_window(
                window,
                window_dates,
                targets,
                allowed_sets,
                price_data,
                cap_reference,
                kospi
            )
        )


        if summary is None:
            continue


        summary_rows.append(
            summary
        )


        daily_frames.append(
            daily
        )


        diagnostic_frames.append(
            diagnostics
        )


        target_frames.append(
            target_history
        )


        if not blocked_buy_df.empty:

            blocked_buy_frames.append(
                blocked_buy_df
            )


        print(
            f"전략 "
            f"{summary['StrategyReturn'] * 100:+.2f}% | "
            f"KOSPI "
            f"{summary['KOSPIReturn'] * 100:+.2f}% | "
            f"Sharpe "
            f"{summary['Sharpe']:.2f} | "
            f"MDD "
            f"{summary['MDD'] * 100:.2f}% | "
            f"Turnover "
            f"{summary['AvgDailyTurnover'] * 100:.2f}% | "
            f"Cost "
            f"{summary['CostPctInitial'] * 100:.2f}% | "
            f"Blocked BUY "
            f"{summary['BlockedBuyCount']}"
        )


    summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )


    daily_df = (
        pd.concat(
            daily_frames,
            ignore_index=True
        )
        if daily_frames
        else
        pd.DataFrame()
    )


    diagnostics_df = (
        pd.concat(
            diagnostic_frames,
            ignore_index=True
        )
        if diagnostic_frames
        else
        pd.DataFrame()
    )


    target_df = (
        pd.concat(
            target_frames,
            ignore_index=True
        )
        if target_frames
        else
        pd.DataFrame()
    )


    blocked_buy_df = (
        pd.concat(
            blocked_buy_frames,
            ignore_index=True
        )
        if blocked_buy_frames
        else
        pd.DataFrame()
    )


    return (
        summary_df,
        daily_df,
        diagnostics_df,
        target_df,
        blocked_buy_df
    )


# =============================================================================
# 37. PRINT SUMMARY
# =============================================================================

def print_summary(
    summary,
    diagnostics,
    blocked_buy_df
):

    print()
    print("=" * 140)

    print(
        "INDEPENDENT 2-MONTH BACKTEST — RULE-FIXED VERSION"
    )

    print("=" * 140)


    display = (
        summary.copy()
    )


    for col in [
        "StrategyReturn",
        "KOSPIReturn",
        "ExcessReturn",
        "MDD",
        "AvgDailyTurnover",
        "CostPctInitial",
        "BlockedBuyValuePctInitial"
    ]:

        display[
            col
        ] *= 100


    print(
        display[
            [
                "Window",
                "Start",
                "End",
                "StrategyReturn",
                "KOSPIReturn",
                "ExcessReturn",
                "Sharpe",
                "MDD",
                "AvgDailyTurnover",
                "CostPctInitial",
                "AvgHoldings",
                "BlockedBuyCount",
                "BlockedMarketCapCount",
                "BlockedLiquidityCount",
                "BlockedBuyValuePctInitial"
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


    positive = (
        summary[
            "StrategyReturn"
        ]
        > 0
    )


    beat_market = (
        summary[
            "ExcessReturn"
        ]
        > 0
    )


    print()
    print("-" * 140)


    print(
        f"Window 수: "
        f"{len(summary)}"
    )


    print(
        f"양수수익 Window: "
        f"{positive.mean() * 100:.1f}%"
    )


    print(
        f"KOSPI 초과수익 Window: "
        f"{beat_market.mean() * 100:.1f}%"
    )


    print(
        f"평균 2개월 수익률: "
        f"{summary['StrategyReturn'].mean() * 100:+.2f}%"
    )


    print(
        f"중앙값 2개월 수익률: "
        f"{summary['StrategyReturn'].median() * 100:+.2f}%"
    )


    print(
        f"평균 KOSPI 수익률: "
        f"{summary['KOSPIReturn'].mean() * 100:+.2f}%"
    )


    print(
        f"평균 초과수익: "
        f"{summary['ExcessReturn'].mean() * 100:+.2f}%p"
    )


    print(
        f"평균 Realized Sharpe: "
        f"{summary['Sharpe'].mean():.2f}"
    )


    print(
        f"평균 MDD: "
        f"{summary['MDD'].mean() * 100:.2f}%"
    )


    print(
        f"평균 일일 Turnover: "
        f"{summary['AvgDailyTurnover'].mean() * 100:.2f}%"
    )


    print(
        f"평균 거래비용: "
        f"{summary['CostPctInitial'].mean() * 100:.2f}%"
    )


    print(
        f"평균 보유종목: "
        f"{summary['AvgHoldings'].mean():.1f}개"
    )


    print()
    print("=" * 140)

    print(
        "EXECUTION BUY RULE DIAGNOSTICS"
    )

    print("=" * 140)


    print(
        f"총 차단된 BUY 건수: "
        f"{int(summary['BlockedBuyCount'].sum())}"
    )


    print(
        f"시총 기준으로 차단된 건수: "
        f"{int(summary['BlockedMarketCapCount'].sum())}"
    )


    print(
        f"거래대금 기준으로 차단된 건수: "
        f"{int(summary['BlockedLiquidityCount'].sum())}"
    )


    print(
        f"Window 평균 차단 BUY 요청금액 / 초기자본: "
        f"{summary['BlockedBuyValuePctInitial'].mean() * 100:.2f}%"
    )


    if not blocked_buy_df.empty:

        both_fail = (
            (
                blocked_buy_df[
                    "MarketCapPass"
                ]
                ==
                False
            )
            &
            (
                blocked_buy_df[
                    "LiquidityPass"
                ]
                ==
                False
            )
        )


        print(
            f"시총+거래대금 동시 실패: "
            f"{int(both_fail.sum())}건"
        )


    print()
    print("=" * 140)

    print(
        "TARGET / EXPOSURE DIAGNOSTICS"
    )

    print("=" * 140)


    print(
        f"평균 Absolute Momentum 통과: "
        f"{diagnostics['AbsoluteMomentumCount'].mean():.1f}개"
    )


    print(
        f"평균 Hysteresis 후보: "
        f"{diagnostics['HysteresisCandidateCount'].mean():.1f}개"
    )


    print(
        f"평균 Target 종목: "
        f"{diagnostics['TargetCount'].mean():.1f}개"
    )


    print(
        f"전체 날짜 평균 Target Exposure: "
        f"{diagnostics['TargetExposure'].mean() * 100:.1f}%"
    )


    rebalance_diag = (
        diagnostics[
            diagnostics[
                "RebalanceDay"
            ]
            ==
            True
        ]
        .copy()
    )


    successful_rebalance = (
        rebalance_diag[
            rebalance_diag[
                "FallbackPreviousTarget"
            ]
            ==
            False
        ]
        .copy()
    )


    if not successful_rebalance.empty:

        print(
            f"정상 Portfolio 계산일 평균 Requested Exposure: "
            f"{successful_rebalance['RequestedExposure'].mean() * 100:.1f}%"
        )


        print(
            f"정상 Portfolio 계산일 평균 Target Exposure: "
            f"{successful_rebalance['TargetExposure'].mean() * 100:.1f}%"
        )


        exposure_error = (
            successful_rebalance[
                "TargetExposure"
            ]
            -
            successful_rebalance[
                "RequestedExposure"
            ]
        )


        print(
            f"정상 계산일 평균 Exposure 오차: "
            f"{exposure_error.mean() * 100:+.4f}%p"
        )


        print(
            f"정상 계산일 최대 절대 Exposure 오차: "
            f"{exposure_error.abs().max() * 100:.4f}%p"
        )


    portfolio_runs = (
        diagnostics[
            "PortfolioRun"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )


    fallback_days = (
        diagnostics[
            "FallbackPreviousTarget"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )


    print(
        f"Window 합산 Portfolio 계산일: "
        f"{int(portfolio_runs)}일"
    )


    print(
        f"Window 합산 Fallback: "
        f"{int(fallback_days)}일"
    )


# =============================================================================
# 38. SAVE CONFIG
# =============================================================================

def save_strategy_config():

    text = f"""
MAIN STRATEGY — RULE-FIXED IMPLEMENTATION
============================================================

Backtest
- Independent 2-Month Windows
- Each window resets Cash / Holdings / Previous Target / Hysteresis
- Previous-day signal is used for next trading day execution

Fundamental
- Fixed Fundamental Universe

Momentum
- 20D Weight: {SHORT_WEIGHT:.0%}
- 60D Weight: {MID_WEIGHT:.0%}
- Absolute Momentum: 20D > 0 OR Price > MA20
- Entry Rank: Top {MOMENTUM_ENTRY_RANK}
- Retention Rank: Top {MOMENTUM_EXIT_RANK}

Portfolio
- Target Holdings: {TARGET_HOLDINGS}
- Weighting: MomentumScore 100%
- Internal Stock Cap: {INTERNAL_MAX_STOCK_WEIGHT:.1%}
- Sector constraints applied

Market Regime
- Bull: {BULL_EXPOSURE:.0%}
- Mixed: {MIXED_EXPOSURE:.0%}
- Bear: {BEAR_EXPOSURE:.0%}

Target Recalculation
- Twice Weekly
- First trading day of each independent window also calculates initial target
- Non-rebalance days keep previous target unchanged

Execution BUY Eligibility
- Checked again using previous trading day's information
- Estimated Market Cap >= {MIN_MARKET_CAP:,.0f}
- 5D Average Trading Value > {MIN_AVG_TRADING_VALUE_5D:,.0f}
- Ineligible BUY is blocked
- Existing holdings are NOT force-sold only because execution-day BUY eligibility fails

Trading
- Rebalance Band: {REBALANCE_BAND:.2%}
- Commission: {COMMISSION_RATE:.2%}
- Sell Tax: {SELL_TAX_RATE:.2%}

Known Backtest Limitations
- Fundamental universe is not fully point-in-time
- Historical market cap is approximated from current market cap and price ratio
- Historical management / warning / caution / attention status is not fully point-in-time
- Current KRX-ADMIN snapshot must not be retroactively applied to historical windows
- Historical relisting status is not explicitly reconstructed
- 61-observation momentum requirement automatically exceeds the ordinary 6-business-day new-listing restriction
""".strip()


    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            text
        )


# =============================================================================
# 39. MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 140)

    print(
        "INDEPENDENT 2-MONTH MAIN BACKTEST"
    )

    print(
        "Fundamental → Momentum → Top25 → Momentum Weight"
    )

    print(
        "FIXES: Stable non-rebalance target + previous-day BUY eligibility"
    )

    print("=" * 140)


    fundamental = (
        load_fundamental()
    )


    grid = (
        load_grid()
    )


    listing = (
        load_current_listing()
    )


    (
        kospi,
        kosdaq,
        trading_dates
    ) = (
        load_index_data()
    )


    windows = (
        build_independent_windows(
            trading_dates
        )
    )


    price_data = (
        download_price_history(
            fundamental,
            trading_dates
        )
    )


    cap_reference = (
        build_market_cap_reference(
            listing,
            price_data
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


    sector_limits = (
        build_sector_limits(
            listing,
            grid
        )
    )


    (
        summary,
        daily,
        diagnostics,
        target_history,
        blocked_buy_history
    ) = (
        run_independent_backtests(
            windows,
            trading_dates,
            fundamental,
            price_data,
            cap_reference,
            sector_map,
            sector_limits,
            kospi,
            kosdaq
        )
    )


    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    daily.to_csv(
        DAILY_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    diagnostics.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    target_history.to_csv(
        TARGET_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    if not blocked_buy_history.empty:

        blocked_buy_history.to_csv(
            BLOCKED_BUY_FILE,
            index=False,
            encoding="utf-8-sig"
        )


    else:

        pd.DataFrame(
            columns=[
                "Window",
                "ExecutionDate",
                "SignalDate",
                "종목코드",
                "RequestedBuyValue",
                "EstimatedMarketCap",
                "TradingValue5D",
                "MarketCapPass",
                "LiquidityPass"
            ]
        ).to_csv(
            BLOCKED_BUY_FILE,
            index=False,
            encoding="utf-8-sig"
        )


    save_strategy_config()


    print_summary(
        summary,
        diagnostics,
        blocked_buy_history
    )


    print()
    print("=" * 140)

    print(
        "저장 완료"
    )

    print("=" * 140)


    print()

    print(
        "Summary:"
    )

    print(
        SUMMARY_FILE
    )


    print()

    print(
        "Target History:"
    )

    print(
        TARGET_FILE
    )


    print()

    print(
        "Diagnostics:"
    )

    print(
        DIAGNOSTIC_FILE
    )


    print()

    print(
        "Daily:"
    )

    print(
        DAILY_FILE
    )


    print()

    print(
        "Blocked BUY History:"
    )

    print(
        BLOCKED_BUY_FILE
    )


    print()

    print(
        "Config:"
    )

    print(
        CONFIG_FILE
    )


    print()
    print("=" * 140)

    print(
        "Backtest 완료"
    )

    print("=" * 140)