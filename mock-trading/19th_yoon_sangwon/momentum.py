import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import FinanceDataReader as fdr


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

MOMENTUM_DIR = os.path.join(
    BASE_DIR,
    "momentum"
)

HISTORY_DIR = os.path.join(
    MOMENTUM_DIR,
    "history"
)


# -----------------------------------------------------------------------------
# 항상 최신 상태로 덮어쓰는 파일
# -----------------------------------------------------------------------------

SCORED_FILE = os.path.join(
    MOMENTUM_DIR,
    "momentum_scored.csv"
)

SELECTED_FILE = os.path.join(
    MOMENTUM_DIR,
    "momentum_selected.csv"
)


# =============================================================================
# 2. MOMENTUM SETTINGS
# =============================================================================

SHORT_PERIOD = 20
MID_PERIOD = 60

SHORT_WEIGHT = 0.60
MID_WEIGHT = 0.40

TOP_N = 80

LOOKBACK_CALENDAR_DAYS = 180

MIN_PRICE_OBSERVATIONS = 61


# =============================================================================
# 3. CREATE DIRECTORIES
# =============================================================================

os.makedirs(
    MOMENTUM_DIR,
    exist_ok=True
)

os.makedirs(
    HISTORY_DIR,
    exist_ok=True
)


# =============================================================================
# 4. STOCK CODE
# =============================================================================

def clean_stock_code(code):

    if pd.isna(code):
        return None

    code = str(code).strip()

    if code.endswith(".0"):
        code = code[:-2]

    if code.startswith("A"):
        code = code[1:]

    return code.zfill(6)


# =============================================================================
# 5. PRICE DATA
# =============================================================================

def get_price_data(code):

    end_date = datetime.today()

    start_date = (
        end_date
        - timedelta(
            days=LOOKBACK_CALENDAR_DAYS
        )
    )

    try:

        price = fdr.DataReader(
            code,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

    except Exception as e:

        print(
            f"[가격 오류] "
            f"{code}: {e}"
        )

        return None

    if price is None or price.empty:
        return None

    if "Close" not in price.columns:
        return None

    price = (
        price
        .sort_index()
        .copy()
    )

    return price


# =============================================================================
# 6. CALCULATE ONE STOCK
# =============================================================================

def calculate_momentum(
    code,
    name
):

    price = get_price_data(
        code
    )

    if price is None:
        return None

    close = (
        pd.to_numeric(
            price["Close"],
            errors="coerce"
        )
        .dropna()
    )

    price_count = len(
        close
    )

    if (
        price_count
        < MIN_PRICE_OBSERVATIONS
    ):

        return None

    # -------------------------------------------------------------------------
    # Prices
    # -------------------------------------------------------------------------

    current_close = float(
        close.iloc[-1]
    )

    close_20 = float(
        close.iloc[
            -(SHORT_PERIOD + 1)
        ]
    )

    close_60 = float(
        close.iloc[
            -(MID_PERIOD + 1)
        ]
    )

    # -------------------------------------------------------------------------
    # 20D / 60D cumulative return
    # -------------------------------------------------------------------------

    return_20 = (
        current_close
        / close_20
        - 1
    )

    return_60 = (
        current_close
        / close_60
        - 1
    )

    # -------------------------------------------------------------------------
    # MA20
    # -------------------------------------------------------------------------

    ma20 = float(
        close
        .iloc[
            -SHORT_PERIOD:
        ]
        .mean()
    )

    above_ma20 = (
        current_close
        > ma20
    )

    # -------------------------------------------------------------------------
    # 20D volatility
    # -------------------------------------------------------------------------

    daily_return = (
        close
        .pct_change()
        .dropna()
    )

    volatility_20 = float(
        daily_return
        .iloc[
            -SHORT_PERIOD:
        ]
        .std()
    )

    annualized_volatility_20 = (
        volatility_20
        * np.sqrt(252)
    )

    # -------------------------------------------------------------------------
    # Absolute Momentum Filter
    #
    # 20D Return > 0
    # OR
    # Current Price > MA20
    # -------------------------------------------------------------------------

    absolute_pass = (
        (return_20 > 0)
        or
        above_ma20
    )

    # -------------------------------------------------------------------------
    # Actual price-data date
    # -------------------------------------------------------------------------

    base_date = (
        close.index[-1]
        .strftime(
            "%Y-%m-%d"
        )
    )

    return {

        "종목코드":
        code,

        "종목명":
        name,

        "기준일":
        base_date,

        "가격관측수":
        price_count,

        "종가":
        current_close,

        "수익률20D":
        return_20,

        "수익률60D":
        return_60,

        "MA20":
        ma20,

        "MA20상회여부":
        above_ma20,

        "변동성20D":
        volatility_20,

        "연율화변동성20D":
        annualized_volatility_20,

        "AbsoluteMomentumPass":
        absolute_pass
    }


# =============================================================================
# 7. LOAD FUNDAMENTAL UNIVERSE
# =============================================================================

def load_fundamental():

    if not os.path.exists(
        FUNDAMENTAL_FILE
    ):

        raise FileNotFoundError(
            f"Fundamental 파일 없음:\n"
            f"{FUNDAMENTAL_FILE}"
        )

    candidates = pd.read_csv(
        FUNDAMENTAL_FILE,
        dtype={
            "종목코드": str
        }
    )

    if "종목코드" not in candidates.columns:

        raise RuntimeError(
            "Fundamental 파일에 "
            "'종목코드' 컬럼이 없습니다."
        )

    candidates[
        "종목코드"
    ] = (
        candidates[
            "종목코드"
        ]
        .apply(
            clean_stock_code
        )
    )

    candidates = (
        candidates
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

    return candidates


# =============================================================================
# 8. RUN MOMENTUM
# =============================================================================

def run_momentum():

    print("=" * 80)
    print("Momentum v2")
    print(
        "20D / 60D + Absolute Momentum "
        "+ Relative Ranking"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Fundamental Universe
    # -------------------------------------------------------------------------

    candidates = (
        load_fundamental()
    )

    print()
    print(
        f"Fundamental 후보군: "
        f"{len(candidates):,}개"
    )

    results = []

    failures = []

    total = len(
        candidates
    )

    # -------------------------------------------------------------------------
    # Calculate each stock
    # -------------------------------------------------------------------------

    for i, (_, row) in enumerate(
        candidates.iterrows(),
        start=1
    ):

        code = (
            row[
                "종목코드"
            ]
        )

        if "종목명" in candidates.columns:

            name = (
                row[
                    "종목명"
                ]
            )

        elif "corp_name" in candidates.columns:

            name = (
                row[
                    "corp_name"
                ]
            )

        else:

            name = code

        try:

            result = (
                calculate_momentum(
                    code,
                    name
                )
            )

            if result is not None:

                results.append(
                    result
                )

            else:

                failures.append(
                    {
                        "종목코드":
                        code,

                        "종목명":
                        name,

                        "사유":
                        "가격데이터 부족"
                    }
                )

        except Exception as e:

            failures.append(
                {
                    "종목코드":
                    code,

                    "종목명":
                    name,

                    "사유":
                    str(e)
                }
            )

        if (
            i % 25 == 0
            or
            i == total
        ):

            print(
                f"{i:,}/{total:,} 진행"
            )

    momentum_df = pd.DataFrame(
        results
    )

    if momentum_df.empty:

        raise RuntimeError(
            "Momentum 계산 결과가 없습니다."
        )

    print()
    print(
        f"Momentum 계산 성공: "
        f"{len(momentum_df):,}개"
    )

    print(
        f"Momentum 계산 실패: "
        f"{len(failures):,}개"
    )

    # =========================================================================
    # 9. ABSOLUTE MOMENTUM
    # =========================================================================

    passed = (
        momentum_df[
            momentum_df[
                "AbsoluteMomentumPass"
            ]
        ]
        .copy()
    )

    failed = (
        momentum_df[
            ~momentum_df[
                "AbsoluteMomentumPass"
            ]
        ]
        .copy()
    )

    print()
    print(
        f"Absolute Momentum 통과: "
        f"{len(passed):,}개"
    )

    print(
        f"Absolute Momentum 탈락: "
        f"{len(failed):,}개"
    )

    if passed.empty:

        raise RuntimeError(
            "Absolute Momentum 통과 종목이 없습니다."
        )

    # =========================================================================
    # 10. RELATIVE MOMENTUM SCORE
    #
    # IMPORTANT:
    # Absolute Momentum을 통과한 종목들끼리만 percentile 계산
    # =========================================================================

    passed[
        "20DScore"
    ] = (
        passed[
            "수익률20D"
        ]
        .rank(
            pct=True,
            ascending=True
        )
        * 100
    )

    passed[
        "60DScore"
    ] = (
        passed[
            "수익률60D"
        ]
        .rank(
            pct=True,
            ascending=True
        )
        * 100
    )

    passed[
        "MomentumScore"
    ] = (
        SHORT_WEIGHT
        * passed[
            "20DScore"
        ]
        +
        MID_WEIGHT
        * passed[
            "60DScore"
        ]
    )

    passed = (
        passed
        .sort_values(
            "MomentumScore",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    passed[
        "MomentumRank"
    ] = (
        np.arange(
            1,
            len(passed) + 1
        )
    )

    # =========================================================================
    # 11. ADD FAILED STOCKS BACK TO SCORED FILE
    # =========================================================================

    failed[
        "20DScore"
    ] = np.nan

    failed[
        "60DScore"
    ] = np.nan

    failed[
        "MomentumScore"
    ] = np.nan

    failed[
        "MomentumRank"
    ] = np.nan

    scored = pd.concat(
        [
            passed,
            failed
        ],
        ignore_index=True
    )

    # =========================================================================
    # 12. FUNDAMENTAL INFORMATION MERGE
    # =========================================================================

    merge_columns = [
        "종목코드"
    ]

    optional_columns = [

        "업종",
        "섹터명",

        "FundamentalModel",
        "FundamentalScore",
        "SectorFundamentalScore",

        "평가기준"
    ]

    for column in optional_columns:

        if (
            column
            in candidates.columns
        ):

            merge_columns.append(
                column
            )

    if (
        len(
            merge_columns
        )
        > 1
    ):

        fundamental_info = (
            candidates[
                merge_columns
            ]
            .drop_duplicates(
                "종목코드"
            )
        )

        scored = (
            scored.merge(
                fundamental_info,
                on="종목코드",
                how="left"
            )
        )

        passed = (
            passed.merge(
                fundamental_info,
                on="종목코드",
                how="left"
            )
        )

    # =========================================================================
    # 13. TOP 80
    # =========================================================================

    selected = (
        passed
        .head(
            min(
                TOP_N,
                len(passed)
            )
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    # =========================================================================
    # 14. HUMAN-READABLE COLUMNS
    # =========================================================================

    for df in [
        scored,
        selected
    ]:

        df[
            "수익률20D_pct"
        ] = (
            df[
                "수익률20D"
            ]
            * 100
        )

        df[
            "수익률60D_pct"
        ] = (
            df[
                "수익률60D"
            ]
            * 100
        )

        df[
            "연율화변동성20D_pct"
        ] = (
            df[
                "연율화변동성20D"
            ]
            * 100
        )

    # =========================================================================
    # 15. BASE DATE
    #
    # 컴퓨터 날짜가 아니라 실제 가격 데이터의 최신 기준일
    # =========================================================================

    base_date = (
        momentum_df[
            "기준일"
        ]
        .max()
    )

    # =========================================================================
    # 16. LATEST FILES
    # =========================================================================

    scored.to_csv(
        SCORED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    selected.to_csv(
        SELECTED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # =========================================================================
    # 17. HISTORY FILES
    # =========================================================================

    scored_history_file = os.path.join(
        HISTORY_DIR,
        f"momentum_scored_{base_date}.csv"
    )

    selected_history_file = os.path.join(
        HISTORY_DIR,
        f"momentum_selected_{base_date}.csv"
    )

    scored.to_csv(
        scored_history_file,
        index=False,
        encoding="utf-8-sig"
    )

    selected.to_csv(
        selected_history_file,
        index=False,
        encoding="utf-8-sig"
    )

    # =========================================================================
    # 18. OUTPUT
    # =========================================================================

    print()
    print("=" * 80)
    print("Momentum v2 결과")
    print("=" * 80)

    print(
        f"기준일: "
        f"{base_date}"
    )

    print(
        f"전체 계산 성공: "
        f"{len(momentum_df):,}개"
    )

    print(
        f"Absolute 통과: "
        f"{len(passed):,}개"
    )

    print(
        f"Absolute 탈락: "
        f"{len(failed):,}개"
    )

    print(
        f"Portfolio 전달 Top N: "
        f"{len(selected):,}개"
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("Top N 요약")

    print(
        f"평균 20D 수익률: "
        f"{selected['수익률20D'].mean() * 100:.2f}%"
    )

    print(
        f"평균 60D 수익률: "
        f"{selected['수익률60D'].mean() * 100:.2f}%"
    )

    print(
        f"MA20 상회 비율: "
        f"{selected['MA20상회여부'].mean() * 100:.1f}%"
    )

    print(
        f"평균 연율화 20D 변동성: "
        f"{selected['연율화변동성20D'].mean() * 100:.2f}%"
    )

    # -------------------------------------------------------------------------
    # Top 20
    # -------------------------------------------------------------------------

    print()
    print("Momentum Top 20")

    display_columns = [
        "MomentumRank",
        "종목코드",
        "종목명",
        "수익률20D_pct",
        "수익률60D_pct",
        "20DScore",
        "60DScore",
        "MomentumScore",
        "MA20상회여부",
        "연율화변동성20D_pct",
        "가격관측수"
    ]

    print(
        selected[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("저장 완료")
    print("=" * 80)

    print()
    print(
        "최신 전체 Momentum:"
    )

    print(
        SCORED_FILE
    )

    print()
    print(
        "최신 Portfolio 전달용 Top 80:"
    )

    print(
        SELECTED_FILE
    )

    print()
    print(
        "전체 Momentum History:"
    )

    print(
        scored_history_file
    )

    print()
    print(
        "Top 80 History:"
    )

    print(
        selected_history_file
    )


# =============================================================================
# 19. RUN
# =============================================================================

if __name__ == "__main__":

    run_momentum()