import io
import os
import re
import time
import zipfile
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import FinanceDataReader as fdr


# ============================================================
# 1. 기본 설정
# ============================================================

# ★ 본인 OpenDART API KEY 입력
API_KEY = os.getenv("DART_API_KEY")

# 2026년 대회 시작 기준 최근 확정 연간 사업보고서
BSNS_YEAR = "2025"
REPRT_CODE = "11011"

# None = 전체 실행
# 테스트 시 100 등으로 변경
MAX_COMPANIES = None


# ------------------------------------------------------------
# 일반기업 상대평가
# ------------------------------------------------------------

SELECT_RATIO = 0.25
MIN_SECTOR_SIZE = 6


# ------------------------------------------------------------
# 금융모델 상대평가
# ------------------------------------------------------------

FINANCIAL_SELECT_RATIO = 0.25
FINANCIAL_MIN_GROUP_SIZE = 3


# ------------------------------------------------------------
# 경로
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "fundamental"
)

GRID_FILE = os.path.join(
    BASE_DIR,
    "grid_20260929.csv"
)

# 선택파일
#
# 없어도 정상 실행.
#
# 있으면 다음 컬럼 사용:
# 종목코드,CET1,BIS,NPL,NCR,KICS,CSM성장률
REGULATORY_FILE = os.path.join(
    BASE_DIR,
    "financial_regulatory_metrics.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CACHE_FILE = os.path.join(
    OUTPUT_DIR,
    "fundamental_cache.csv"
)

RAW_FILE = os.path.join(
    OUTPUT_DIR,
    "fundamental_raw.csv"
)

SCORED_FILE = os.path.join(
    OUTPUT_DIR,
    "fundamental_scored.csv"
)

SECTOR_SCORED_FILE = os.path.join(
    OUTPUT_DIR,
    "sector_fundamental_scored.csv"
)

SECTOR_SELECTED_FILE = os.path.join(
    OUTPUT_DIR,
    "sector_fundamental_selected.csv"
)


# ============================================================
# 2. 숫자 변환
# ============================================================

def to_number(value):

    if value is None:
        return None

    text = str(value).strip()

    if text in [
        "",
        "-",
        "None",
        "nan",
        "NaN"
    ]:
        return None

    text = (
        text
        .replace(",", "")
        .replace(" ", "")
    )

    try:
        return float(text)

    except ValueError:
        return None


# ============================================================
# 3. 보통주 판별
# ============================================================

def is_common_stock(name):

    name = str(name).strip()

    preferred_patterns = [
        r"우$",
        r"\d우$",
        r"\d우B$",
        r"\d우C$",
        r"우B$",
        r"우C$"
    ]

    for pattern in preferred_patterns:

        if re.search(
            pattern,
            name
        ):
            return False

    return True


# ============================================================
# 4. 현재 KOSPI / KOSDAQ 보통주
# ============================================================

def get_current_listed_stocks():

    print()
    print(
        "[1] KOSPI / KOSDAQ 상장종목 불러오는 중..."
    )

    kospi = fdr.StockListing(
        "KOSPI"
    ).copy()

    kosdaq = fdr.StockListing(
        "KOSDAQ"
    ).copy()

    kospi["시장"] = "KOSPI"
    kosdaq["시장"] = "KOSDAQ"

    listed = pd.concat(
        [
            kospi,
            kosdaq
        ],
        ignore_index=True
    )

    listed["Code"] = (
        listed["Code"]
        .astype(str)
        .str.zfill(6)
    )

    listed = listed.drop_duplicates(
        subset="Code",
        keep="first"
    )

    before = len(
        listed
    )

    listed = listed[
        listed["Name"]
        .apply(
            is_common_stock
        )
    ].copy()

    print(
        f"전체 상장 종목: "
        f"{before:,}개"
    )

    print(
        f"보통주 필터 후: "
        f"{len(listed):,}개"
    )

    return listed


# ============================================================
# 5. DART 기업 목록
# ============================================================

def get_dart_companies():

    print()
    print(
        "[2] DART 기업 목록 불러오는 중..."
    )

    if (
        not API_KEY
        or API_KEY == "YOUR_DART_API_KEY"
    ):

        raise RuntimeError(
            "코드 상단 API_KEY에 "
            "본인 OpenDART API KEY를 입력해주세요."
        )

    url = (
        "https://opendart.fss.or.kr/api/"
        "corpCode.xml"
    )

    response = requests.get(
        url,
        params={
            "crtfc_key": API_KEY
        },
        timeout=30
    )

    if not response.content.startswith(
        b"PK"
    ):

        print(
            response.text[:500]
        )

        raise RuntimeError(
            "DART corpCode 응답 오류. "
            "API KEY를 확인해주세요."
        )

    with zipfile.ZipFile(
        io.BytesIO(
            response.content
        )
    ) as zf:

        xml_data = zf.read(
            "CORPCODE.xml"
        )

    root = ET.fromstring(
        xml_data
    )

    rows = []

    for item in root.findall(
        "list"
    ):

        corp_code = item.findtext(
            "corp_code"
        )

        corp_name = item.findtext(
            "corp_name"
        )

        stock_code = item.findtext(
            "stock_code"
        )

        if stock_code:

            stock_code = (
                stock_code.strip()
            )

        if stock_code:

            rows.append(
                {
                    "corp_code": corp_code,
                    "corp_name": corp_name,
                    "stock_code": (
                        stock_code.zfill(6)
                    )
                }
            )

    dart_df = pd.DataFrame(
        rows
    )

    dart_df = dart_df.drop_duplicates(
        subset="stock_code",
        keep="first"
    )

    print(
        f"DART 종목코드 보유 기업: "
        f"{len(dart_df):,}개"
    )

    return dart_df


# ============================================================
# 6. 현재 보통주 ∩ DART
# ============================================================

def build_current_universe():

    listed = (
        get_current_listed_stocks()
    )

    dart_df = (
        get_dart_companies()
    )

    universe = listed.merge(
        dart_df,
        left_on="Code",
        right_on="stock_code",
        how="inner"
    )

    universe = universe[
        [
            "Code",
            "Name",
            "시장",
            "corp_code",
            "corp_name"
        ]
    ].copy()

    universe = universe.rename(
        columns={
            "Code": "종목코드",
            "Name": "종목명"
        }
    )

    if MAX_COMPANIES is not None:

        universe = universe.head(
            MAX_COMPANIES
        )

    print()
    print(
        f"현재 보통주 ∩ DART: "
        f"{len(universe):,}개"
    )

    return universe


# ============================================================
# 7. DART 재무제표
# ============================================================

def get_financial_statement(
    corp_code
):

    url = (
        "https://opendart.fss.or.kr/api/"
        "fnlttSinglAcntAll.json"
    )

    base_params = {

        "crtfc_key": API_KEY,
        "corp_code": corp_code,
        "bsns_year": BSNS_YEAR,
        "reprt_code": REPRT_CODE

    }

    # 연결재무제표 우선
    # 없으면 별도재무제표
    for fs_div in [
        "CFS",
        "OFS"
    ]:

        params = (
            base_params.copy()
        )

        params[
            "fs_div"
        ] = fs_div

        try:

            response = requests.get(
                url,
                params=params,
                timeout=20
            )

            data = (
                response.json()
            )

        except Exception:

            continue

        if (
            data.get("status")
            == "000"
        ):

            rows = data.get(
                "list",
                []
            )

            if rows:

                return (
                    pd.DataFrame(
                        rows
                    ),
                    fs_div
                )

    return None, None


# ============================================================
# 8. 계정명 정규화
# ============================================================

def normalize_account_name(
    text
):

    return (
        str(text)
        .replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
    )


# ============================================================
# 9. 3개년 계정 검색
# ============================================================

def find_account_3yr(
    fs_df,
    account_names,
    statement_types=None
):

    if (
        fs_df is None
        or fs_df.empty
    ):

        return (
            None,
            None,
            None
        )

    temp = (
        fs_df.copy()
    )

    if statement_types is not None:

        temp = temp[
            temp["sj_div"]
            .isin(
                statement_types
            )
        ]

    if temp.empty:

        return (
            None,
            None,
            None
        )

    temp[
        "계정명정규화"
    ] = (
        temp[
            "account_nm"
        ]
        .astype(str)
        .apply(
            normalize_account_name
        )
    )

    # 정확 일치 → 포함 검색
    for exact_only in [
        True,
        False
    ]:

        for account_name in (
            account_names
        ):

            target = (
                normalize_account_name(
                    account_name
                )
            )

            if exact_only:

                matched = temp[
                    temp[
                        "계정명정규화"
                    ]
                    == target
                ]

            else:

                matched = temp[
                    temp[
                        "계정명정규화"
                    ]
                    .str.contains(
                        target,
                        regex=False,
                        na=False
                    )
                ]

            if not matched.empty:

                row = (
                    matched.iloc[0]
                )

                current = to_number(
                    row.get(
                        "thstrm_amount"
                    )
                )

                previous = to_number(
                    row.get(
                        "frmtrm_amount"
                    )
                )

                previous2 = to_number(
                    row.get(
                        "bfefrmtrm_amount"
                    )
                )

                return (
                    current,
                    previous,
                    previous2
                )

    return (
        None,
        None,
        None
    )


# ============================================================
# 10. 2개년 계정 검색
# ============================================================

def find_account(
    fs_df,
    account_names,
    statement_types=None
):

    (
        current,
        previous,
        _
    ) = find_account_3yr(
        fs_df,
        account_names,
        statement_types
    )

    return (
        current,
        previous
    )


# ============================================================
# 11. 성장률 계산
# ============================================================

def normal_growth(
    current,
    previous
):

    if (
        current is None
        or previous is None
    ):
        return None

    if previous <= 0:
        return None

    growth = (
        (
            current
            - previous
        )
        / abs(previous)
        * 100
    )

    return max(
        -100,
        min(
            100,
            growth
        )
    )


def signed_profit_growth(
    current,
    previous
):

    if (
        current is None
        or previous is None
    ):
        return None

    # 흑자 → 흑자
    if (
        previous > 0
        and current > 0
    ):

        growth = (
            (
                current
                - previous
            )
            / abs(previous)
            * 100
        )

    # 적자 → 흑자
    elif (
        previous <= 0
        and current > 0
    ):

        growth = 100

    # 흑자 → 적자
    elif (
        previous > 0
        and current <= 0
    ):

        growth = -100

    # 적자 → 적자
    else:

        growth = -100

    return max(
        -100,
        min(
            100,
            growth
        )
    )


# ============================================================
# 12. 3개년 이익 안정성
# ============================================================

def earnings_stability_raw(
    current,
    previous,
    previous2
):

    values = [
        current,
        previous,
        previous2
    ]

    if any(
        v is None
        for v in values
    ):

        return None

    mean_abs = (
        sum(
            abs(v)
            for v in values
        )
        / len(values)
    )

    if mean_abs == 0:

        return None

    mean_value = (
        sum(values)
        / len(values)
    )

    variance = (
        sum(
            (
                v
                - mean_value
            ) ** 2
            for v in values
        )
        / len(values)
    )

    std = (
        variance ** 0.5
    )

    cv = (
        std
        / mean_abs
    )

    # 0~1
    # 높을수록 안정
    return (
        1
        / (
            1
            + cv
        )
    )


# ============================================================
# 13. 기업 하나 분석
# ============================================================

def analyze_company(
    row
):

    corp_code = (
        row["corp_code"]
    )

    fs_df, fs_div = (
        get_financial_statement(
            corp_code
        )
    )

    if fs_df is None:

        return None

    # --------------------------------------------------------
    # 재무상태표
    # --------------------------------------------------------

    assets, prev_assets = (
        find_account(
            fs_df,
            [
                "자산총계"
            ],
            [
                "BS"
            ]
        )
    )

    (
        liabilities,
        prev_liabilities
    ) = find_account(
        fs_df,
        [
            "부채총계"
        ],
        [
            "BS"
        ]
    )

    equity, prev_equity = (
        find_account(
            fs_df,
            [
                "자본총계"
            ],
            [
                "BS"
            ]
        )
    )

    # --------------------------------------------------------
    # 매출 / 영업수익
    # --------------------------------------------------------

    (
        revenue,
        prev_revenue,
        prev2_revenue
    ) = find_account_3yr(
        fs_df,
        [
            "매출액",
            "수익(매출액)",
            "영업수익",
            "영업수익합계"
        ],
        [
            "IS",
            "CIS"
        ]
    )

    # --------------------------------------------------------
    # 영업이익
    # --------------------------------------------------------

    (
        operating_profit,
        prev_operating_profit,
        prev2_operating_profit
    ) = find_account_3yr(
        fs_df,
        [
            "영업이익",
            "영업이익(손실)"
        ],
        [
            "IS",
            "CIS"
        ]
    )

    # --------------------------------------------------------
    # 순이익
    # --------------------------------------------------------

    (
        net_income,
        prev_net_income,
        prev2_net_income
    ) = find_account_3yr(
        fs_df,
        [
            "당기순이익",
            "당기순이익(손실)",
            "연결당기순이익"
        ],
        [
            "IS",
            "CIS"
        ]
    )

    # --------------------------------------------------------
    # 영업현금흐름
    # --------------------------------------------------------

    ocf, prev_ocf = (
        find_account(
            fs_df,
            [
                "영업활동으로 인한 현금흐름",
                "영업활동현금흐름",
                "영업활동 현금흐름"
            ],
            [
                "CF"
            ]
        )
    )

    # 금융기업을 위해
    # 매출 / 영업이익 / OCF는
    # 여기서 필수로 강제하지 않음
    if (
        equity is None
        or net_income is None
    ):

        return None

    # --------------------------------------------------------
    # ROE
    # --------------------------------------------------------

    if (
        prev_equity is not None
        and equity is not None
        and (
            prev_equity
            + equity
        ) != 0
    ):

        average_equity = (
            (
                prev_equity
                + equity
            )
            / 2
        )

    else:

        average_equity = equity

    if (
        average_equity is not None
        and average_equity != 0
    ):

        roe = (
            net_income
            / average_equity
            * 100
        )

    else:

        roe = None

    # --------------------------------------------------------
    # 영업이익률
    # --------------------------------------------------------

    if (
        revenue is not None
        and revenue != 0
        and operating_profit is not None
    ):

        operating_margin = (
            operating_profit
            / revenue
            * 100
        )

    else:

        operating_margin = None

    # --------------------------------------------------------
    # 부채비율
    # --------------------------------------------------------

    if (
        equity is not None
        and equity != 0
        and liabilities is not None
    ):

        debt_ratio = (
            liabilities
            / equity
            * 100
        )

    else:

        debt_ratio = None

    revenue_growth = (
        normal_growth(
            revenue,
            prev_revenue
        )
    )

    op_profit_growth = (
        signed_profit_growth(
            operating_profit,
            prev_operating_profit
        )
        if operating_profit is not None
        else None
    )

    net_income_growth = (
        signed_profit_growth(
            net_income,
            prev_net_income
        )
    )

    earnings_stability = (
        earnings_stability_raw(
            net_income,
            prev_net_income,
            prev2_net_income
        )
    )

    return {

        "종목코드": row[
            "종목코드"
        ],

        "종목명": row[
            "종목명"
        ],

        "시장": row[
            "시장"
        ],

        "corp_code": corp_code,

        "재무제표": fs_div,

        "자산": assets,
        "부채": liabilities,
        "자본": equity,

        "매출액": revenue,
        "전기매출액": prev_revenue,
        "전전기매출액": prev2_revenue,

        "영업이익": operating_profit,
        "전기영업이익": (
            prev_operating_profit
        ),
        "전전기영업이익": (
            prev2_operating_profit
        ),

        "당기순이익": net_income,
        "전기순이익": prev_net_income,
        "전전기순이익": (
            prev2_net_income
        ),

        "영업현금흐름": ocf,

        "ROE": roe,

        "영업이익률": (
            operating_margin
        ),

        "매출성장률": (
            revenue_growth
        ),

        "영업이익성장률": (
            op_profit_growth
        ),

        "순이익성장률": (
            net_income_growth
        ),

        "이익안정성": (
            earnings_stability
        ),

        "부채비율": (
            debt_ratio
        )
    }


# ============================================================
# 14. 전체 재무분석 + 캐시
# ============================================================

def run_fundamental_analysis(
    universe
):

    print()
    print(
        "[3] Fundamental 재무정보 수집..."
    )

    if os.path.exists(
        CACHE_FILE
    ):

        cache_df = pd.read_csv(
            CACHE_FILE,
            dtype={
                "종목코드": str
            }
        )

        cache_df[
            "종목코드"
        ] = (
            cache_df[
                "종목코드"
            ]
            .astype(str)
            .str.zfill(6)
        )

        print(
            f"기존 캐시: "
            f"{len(cache_df):,}개"
        )

    else:

        cache_df = (
            pd.DataFrame()
        )

    cached_codes = set()

    if not cache_df.empty:

        cached_codes = set(
            cache_df[
                "종목코드"
            ]
        )

    new_results = []

    total = len(
        universe
    )

    for i, (
        _,
        row
    ) in enumerate(
        universe.iterrows(),
        start=1
    ):

        code = (
            row[
                "종목코드"
            ]
        )

        if code in cached_codes:

            continue

        try:

            result = (
                analyze_company(
                    row
                )
            )

            if result is not None:

                new_results.append(
                    result
                )

        except Exception as e:

            print(
                f"\n오류 "
                f"{code} "
                f"{row['종목명']}: "
                f"{e}"
            )

        if (
            i % 50 == 0
            or i == total
        ):

            print(
                f"{i:,}/{total:,} 진행"
            )

        time.sleep(
            0.03
        )

        # 100개마다 중간저장
        if (
            len(new_results) > 0
            and len(new_results) % 100 == 0
        ):

            temp_df = pd.concat(
                [
                    cache_df,
                    pd.DataFrame(
                        new_results
                    )
                ],
                ignore_index=True
            )

            temp_df = (
                temp_df
                .drop_duplicates(
                    "종목코드",
                    keep="last"
                )
            )

            temp_df.to_csv(
                CACHE_FILE,
                index=False,
                encoding="utf-8-sig"
            )

    new_df = pd.DataFrame(
        new_results
    )

    if cache_df.empty:

        result_df = (
            new_df
        )

    elif new_df.empty:

        result_df = (
            cache_df
        )

    else:

        result_df = pd.concat(
            [
                cache_df,
                new_df
            ],
            ignore_index=True
        )

    result_df = (
        result_df
        .drop_duplicates(
            "종목코드",
            keep="last"
        )
    )

    # 현재 보통주 유니버스에 존재하는 종목만 유지
    current_codes = set(
        universe[
            "종목코드"
        ]
    )

    result_df = result_df[
        result_df[
            "종목코드"
        ].isin(
            current_codes
        )
    ].copy()

    result_df.to_csv(
        CACHE_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    result_df.to_csv(
        RAW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"재무분석 성공: "
        f"{len(result_df):,}개"
    )

    return result_df


# ============================================================
# 15. Percentile
# ============================================================

def percentile_score(
    series,
    higher_is_better=True
):

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    return (
        numeric.rank(
            pct=True,
            ascending=higher_is_better
        )
        * 100
    )


def neutralized_percentile(
    series,
    higher_is_better=True
):

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = (
        numeric.dropna()
    )

    # 결측치는 중립 50점
    result = pd.Series(
        50.0,
        index=series.index,
        dtype=float
    )

    if len(valid) >= 2:

        result.loc[
            valid.index
        ] = percentile_score(
            valid,
            higher_is_better
        )

    elif len(valid) == 1:

        result.loc[
            valid.index
        ] = 50.0

    return result


# ============================================================
# 16. KIND 업종
# ============================================================

def get_kind_sector_data():

    print()
    print(
        "[4] KIND 업종 데이터 불러오는 중..."
    )

    url = (
        "https://kind.krx.co.kr/"
        "corpgeneral/corpList.do"
    )

    params = {
        "method": "download",
        "searchType": "13"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    tables = pd.read_html(
        io.BytesIO(
            response.content
        )
    )

    kind_df = (
        tables[0].copy()
    )

    kind_df[
        "종목코드"
    ] = (
        kind_df[
            "종목코드"
        ]
        .astype(str)
        .str.zfill(6)
    )

    kind_df = (
        kind_df
        .drop_duplicates(
            subset="종목코드",
            keep="first"
        )
    )

    print(
        f"KIND 업종: "
        f"{len(kind_df):,}개"
    )

    return kind_df


# ============================================================
# 17. 대회 Sector Grid
# ============================================================

def get_competition_sector_grid():

    print()
    print(
        "[5] 대회 Sector Grid 불러오는 중..."
    )

    if not os.path.exists(
        GRID_FILE
    ):

        raise FileNotFoundError(
            f"Grid 파일 없음: "
            f"{GRID_FILE}"
        )

    grid_df = pd.read_csv(
        GRID_FILE,
        dtype=str
    )

    required = {
        "섹터코드",
        "섹터명",
        "종목코드",
        "종목명"
    }

    missing = (
        required
        - set(
            grid_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Grid 필수컬럼 누락: "
            f"{sorted(missing)}"
        )

    grid_df[
        "종목코드"
    ] = (
        grid_df[
            "종목코드"
        ]
        .astype(str)
        .str.replace(
            "^A",
            "",
            regex=True
        )
        .str.zfill(6)
    )

    grid_df = (
        grid_df
        .drop_duplicates(
            "종목코드",
            keep="first"
        )
    )

    print(
        f"Grid 종목: "
        f"{len(grid_df):,}개"
    )

    return grid_df


# ============================================================
# 18. 현재 시장 데이터
# ============================================================

def get_current_valuation_data():

    print()
    print(
        "[6] PBR / PER / 배당 / 시총 불러오는 중..."
    )

    frames = []

    for market_name in [
        "KOSPI",
        "KOSDAQ"
    ]:

        frames.append(
            fdr.StockListing(
                market_name
            ).copy()
        )

    valuation = pd.concat(
        frames,
        ignore_index=True
    )

    valuation[
        "Code"
    ] = (
        valuation[
            "Code"
        ]
        .astype(str)
        .str.zfill(6)
    )

    keep = [
        col
        for col in [
            "Code",
            "PBR",
            "PER",
            "DIV",
            "Marcap"
        ]
        if col in valuation.columns
    ]

    valuation = (
        valuation[
            keep
        ]
        .drop_duplicates(
            "Code",
            keep="first"
        )
        .rename(
            columns={
                "Code": "종목코드",
                "PBR": "시장PBR",
                "PER": "시장PER",
                "DIV": "배당수익률",
                "Marcap": "시가총액"
            }
        )
    )

    for col in [
        "시장PBR",
        "시장PER",
        "배당수익률",
        "시가총액"
    ]:

        if col in valuation.columns:

            valuation[
                col
            ] = pd.to_numeric(
                valuation[
                    col
                ],
                errors="coerce"
            )

    return valuation


# ============================================================
# 19. 금융 규제지표
# ============================================================

def load_regulatory_metrics():

    columns = [

        "종목코드",
        "CET1",
        "BIS",
        "NPL",
        "NCR",
        "KICS",
        "CSM성장률"

    ]

    if not os.path.exists(
        REGULATORY_FILE
    ):

        print()
        print(
            "[규제지표] 별도 파일 없음 "
            "→ 해당 규제지표 점수는 50점 중립 처리"
        )

        return pd.DataFrame(
            columns=columns
        )

    reg = pd.read_csv(
        REGULATORY_FILE,
        dtype={
            "종목코드": str
        }
    )

    reg[
        "종목코드"
    ] = (
        reg[
            "종목코드"
        ]
        .astype(str)
        .str.replace(
            "^A",
            "",
            regex=True
        )
        .str.zfill(6)
    )

    for col in columns[1:]:

        if col not in reg.columns:

            reg[
                col
            ] = pd.NA

        reg[
            col
        ] = pd.to_numeric(
            reg[
                col
            ],
            errors="coerce"
        )

    return (
        reg[
            columns
        ]
        .drop_duplicates(
            "종목코드",
            keep="last"
        )
    )


# ============================================================
# 20. Grid 금융 여부
# ============================================================

def is_grid_financial(
    sector
):

    text = (
        str(
            sector
        )
        .strip()
        .lower()
    )

    return (
        text == "금융"
        or text == "financials"
    )


# ============================================================
# 21. Fundamental 평가모델 분류
# ============================================================

def classify_fundamental_model(
    row
):

    """
    중요

    Grid 섹터와 Fundamental 모델을 분리한다.

    Grid 금융 여부:
        포트폴리오 Sector 제약에 사용

    FundamentalModel:
        실제 사업형태에 맞춰 재무평가 방식 선택
    """

    grid_financial = bool(
        row.get(
            "Grid금융여부",
            False
        )
    )

    # Grid상 금융이 아니면 일반기업
    if not grid_financial:

        return "일반기업"

    name = str(
        row.get(
            "종목명",
            ""
        )
    ).strip()

    industry = str(
        row.get(
            "업종",
            ""
        )
    ).strip()

    name_lower = (
        name.lower()
    )

    # ========================================================
    # 1. 은행 / 금융지주
    # ========================================================

    bank_holding_names = {

        "KB금융",
        "신한지주",
        "하나금융지주",
        "우리금융지주",

        "BNK금융지주",
        "JB금융지주",
        "iM금융지주",

        "메리츠금융지주",
        "한국금융지주",

        "기업은행",
        "카카오뱅크",
        "케이뱅크",
        "제주은행",
        "푸른저축은행"

    }

    if (
        name in bank_holding_names

        or "은행" in industry

        or "저축기관" in industry

        or "금융지주" in name
    ):

        return (
            "은행/금융지주"
        )

    # ========================================================
    # 2. 증권
    # ========================================================

    if (
        "증권" in name
        or "선물" in name
        or "증권" in industry
    ):

        return "증권"

    # ========================================================
    # 3. 보험
    #
    # 실제 보험회사.
    # GA/보험대리점은 금융서비스로 분류.
    # ========================================================

    insurer_names = {

        "DB손해보험",
        "롯데손해보험",

        "미래에셋생명",
        "삼성생명",
        "삼성화재",

        "서울보증보험",

        "코리안리",

        "한화생명",
        "한화손해보험",

        "현대해상",
        "흥국화재"

    }

    if (
        name in insurer_names

        or industry == "보험업"

        or "재 보험업" in industry
    ):

        return "보험"

    # ========================================================
    # 4. 투자 / 캐피탈
    # ========================================================

    explicit_investment_names = {

        "컴퍼니케이",
        "스톤브릿지벤처스"

    }

    investment_keywords = [

        "인베스트",
        "벤처",
        "창투",
        "캐피탈",
        "IB투자",
        "파트너스"

    ]

    if (
        name in explicit_investment_names

        or "신탁업 및 집합투자업"
        in industry

        or any(
            keyword.lower()
            in name_lower
            for keyword
            in investment_keywords
        )
    ):

        return (
            "투자/캐피탈"
        )

    # ========================================================
    # 5. 금융서비스
    #
    # PG / VAN / 카드 / 핀테크 /
    # 신용평가 / 금융데이터 / 보험GA 등
    # ========================================================

    explicit_financial_service_names = {

        "아이지넷",
        "더즌"

    }

    financial_service_keywords = [

        "카드",
        "파이낸셜",

        "KCP",
        "이니시스",

        "정보통신",

        "평가정보",
        "기업평가",

        "에프앤가이드",

        "쿠콘",
        "한패스",

        "머니트리",

        "NICE",
        "나이스"

    ]

    if (
        name in explicit_financial_service_names

        or "금융 지원 서비스업"
        in industry

        or "보험 및 연금관련 서비스업"
        in industry

        or any(
            keyword.lower()
            in name_lower
            for keyword
            in financial_service_keywords
        )
    ):

        return (
            "금융서비스"
        )

    # ========================================================
    # 6. 그 외
    #
    # Grid상 금융이라도 실제 영업기업이면
    # Fundamental은 일반기업 모델 적용.
    #
    # 예:
    # 다우기술 / 다우데이타 등
    #
    # 단, Grid 섹터 자체는 금융으로 유지됨.
    # ========================================================

    return "일반기업"


# ============================================================
# 22. 금융 관련 추가지표
# ============================================================

def build_financial_metrics(
    scored
):

    scored = (
        scored.copy()
    )

    # 자기자본비율
    scored[
        "자기자본비율"
    ] = (
        scored[
            "자본"
        ]
        / scored[
            "자산"
        ].replace(
            0,
            pd.NA
        )
        * 100
    )

    # --------------------------------------------------------
    # PBR
    # --------------------------------------------------------

    if (
        "시장PBR"
        in scored.columns
    ):

        scored[
            "PBR"
        ] = pd.to_numeric(
            scored[
                "시장PBR"
            ],
            errors="coerce"
        )

    else:

        scored[
            "PBR"
        ] = pd.NA

    # PBR 없으면
    # 시가총액 / 자기자본
    if (
        "시가총액"
        in scored.columns
    ):

        calculated_pbr = (
            scored[
                "시가총액"
            ]
            / scored[
                "자본"
            ].replace(
                0,
                pd.NA
            )
        )

        scored[
            "PBR"
        ] = (
            scored[
                "PBR"
            ]
            .where(
                scored[
                    "PBR"
                ] > 0,
                calculated_pbr
            )
        )

    scored[
        "PBR"
    ] = scored[
        "PBR"
    ].where(
        scored[
            "PBR"
        ] > 0
    )

    # --------------------------------------------------------
    # 은행 자본적정성
    # CET1 우선 → BIS
    # --------------------------------------------------------

    scored[
        "자본적정성"
    ] = (
        scored[
            "CET1"
        ]
        .where(
            scored[
                "CET1"
            ].notna(),
            scored[
                "BIS"
            ]
        )
    )

    return scored


# ============================================================
# 23. 일반기업 점수
# ============================================================

def score_general_companies(
    group_df
):

    """
    일반기업 Fundamental

    ROE              25%
    영업이익률         25%
    매출성장률         20%
    영업이익성장률      20%
    부채비율           10%
    """

    g = (
        group_df.copy()
    )

    specs = [

        (
            "ROE",
            0.25,
            True
        ),

        (
            "영업이익률",
            0.25,
            True
        ),

        (
            "매출성장률",
            0.20,
            True
        ),

        (
            "영업이익성장률",
            0.20,
            True
        ),

        (
            "부채비율",
            0.10,
            False
        )

    ]

    total = pd.Series(
        0.0,
        index=g.index,
        dtype=float
    )

    for (
        metric,
        weight,
        higher
    ) in specs:

        score_col = (
            f"{metric}점수"
        )

        g[
            score_col
        ] = neutralized_percentile(
            g[
                metric
            ],
            higher
        )

        total = (
            total
            + g[
                score_col
            ]
            * weight
        )

    g[
        "FundamentalScore"
    ] = total

    return g


# ============================================================
# 24. 금융모델별 Fundamental Score
# ============================================================

def score_financial_group(
    group_df,
    model
):

    g = (
        group_df.copy()
    )

    metric_specs = {

        # ----------------------------------------------------
        # 은행 / 금융지주
        # ----------------------------------------------------

        "은행/금융지주": [

            (
                "ROE",
                0.25,
                True
            ),

            (
                "PBR",
                0.20,
                False
            ),

            (
                "순이익성장률",
                0.15,
                True
            ),

            (
                "자본적정성",
                0.15,
                True
            ),

            (
                "NPL",
                0.10,
                False
            ),

            (
                "배당수익률",
                0.10,
                True
            ),

            (
                "매출성장률",
                0.05,
                True
            )
        ],

        # ----------------------------------------------------
        # 증권
        # ----------------------------------------------------

        "증권": [

            (
                "ROE",
                0.25,
                True
            ),

            (
                "PBR",
                0.15,
                False
            ),

            (
                "순이익성장률",
                0.20,
                True
            ),

            (
                "매출성장률",
                0.10,
                True
            ),

            (
                "NCR",
                0.20,
                True
            ),

            (
                "이익안정성",
                0.10,
                True
            )
        ],

        # ----------------------------------------------------
        # 보험
        # ----------------------------------------------------

        "보험": [

            (
                "ROE",
                0.20,
                True
            ),

            (
                "PBR",
                0.15,
                False
            ),

            (
                "순이익성장률",
                0.15,
                True
            ),

            (
                "KICS",
                0.20,
                True
            ),

            (
                "CSM성장률",
                0.20,
                True
            ),

            (
                "이익안정성",
                0.10,
                True
            )
        ],

        # ----------------------------------------------------
        # 투자 / 캐피탈
        # ----------------------------------------------------

        "투자/캐피탈": [

            (
                "ROE",
                0.30,
                True
            ),

            (
                "PBR",
                0.20,
                False
            ),

            (
                "순이익성장률",
                0.20,
                True
            ),

            (
                "매출성장률",
                0.15,
                True
            ),

            (
                "이익안정성",
                0.15,
                True
            )
        ],

        # ----------------------------------------------------
        # 금융서비스
        # ----------------------------------------------------

        "금융서비스": [

            (
                "ROE",
                0.25,
                True
            ),

            (
                "영업이익률",
                0.20,
                True
            ),

            (
                "매출성장률",
                0.20,
                True
            ),

            (
                "영업이익성장률",
                0.15,
                True
            ),

            (
                "순이익성장률",
                0.10,
                True
            ),

            (
                "부채비율",
                0.10,
                False
            )
        ]
    }

    specs = (
        metric_specs[
            model
        ]
    )

    total = pd.Series(
        0.0,
        index=g.index,
        dtype=float
    )

    for (
        metric,
        weight,
        higher
    ) in specs:

        if metric not in g.columns:

            g[
                metric
            ] = pd.NA

        score_col = (
            f"{model}_{metric}_점수"
        )

        g[
            score_col
        ] = neutralized_percentile(
            g[
                metric
            ],
            higher
        )

        total = (
            total
            + g[
                score_col
            ]
            * weight
        )

    g[
        "FundamentalScore"
    ] = total

    return g


# ============================================================
# 25. Quality Gate + Fundamental 점수
# ============================================================

def build_quality_universe(
    raw_df,
    kind_df,
    grid_df
):

    print()
    print(
        "[7] Quality Gate + Fundamental 평가..."
    )

    scored = (
        raw_df.copy()
    )

    scored[
        "종목코드"
    ] = (
        scored[
            "종목코드"
        ]
        .astype(str)
        .str.zfill(6)
    )

    # --------------------------------------------------------
    # KIND
    # --------------------------------------------------------

    kind_temp = kind_df[
        [
            "종목코드",
            "업종"
        ]
    ].copy()

    kind_temp[
        "종목코드"
    ] = (
        kind_temp[
            "종목코드"
        ]
        .astype(str)
        .str.zfill(6)
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    grid_temp = grid_df[
        [
            "종목코드",
            "섹터코드",
            "섹터명"
        ]
    ].copy()

    valuation = (
        get_current_valuation_data()
    )

    regulatory = (
        load_regulatory_metrics()
    )

    scored = scored.merge(
        kind_temp,
        on="종목코드",
        how="left"
    )

    scored = scored.merge(
        grid_temp,
        on="종목코드",
        how="left"
    )

    scored = scored.merge(
        valuation,
        on="종목코드",
        how="left"
    )

    scored = scored.merge(
        regulatory,
        on="종목코드",
        how="left"
    )

    # ========================================================
    # 중요
    #
    # 여기서는
    # 시총 1,000억원
    # 5일 평균 거래대금
    # 관리종목 등
    #
    # 매수가능성 필터를 적용하지 않는다.
    #
    # → portfolio.py에서 당일 기준 적용
    # ========================================================

    scored[
        "Grid금융여부"
    ] = (
        scored[
            "섹터명"
        ]
        .apply(
            is_grid_financial
        )
    )

    scored[
        "FundamentalModel"
    ] = (
        scored.apply(
            classify_fundamental_model,
            axis=1
        )
    )

    scored = (
        build_financial_metrics(
            scored
        )
    )

    print()
    print(
        "Fundamental 평가모델 분포:"
    )

    print(
        scored[
            "FundamentalModel"
        ].value_counts()
    )

    # ========================================================
    # Quality Gate
    # ========================================================

    scored[
        "QualityGate"
    ] = False

    # --------------------------------------------------------
    # 일반기업
    # --------------------------------------------------------

    general_mask = (
        scored[
            "FundamentalModel"
        ]
        == "일반기업"
    )

    general_quality = (

        general_mask

        & (
            scored[
                "ROE"
            ] > 0
        )

        & (
            scored[
                "영업이익률"
            ] > 0
        )

        & (
            scored[
                "자본"
            ] > 0
        )

        & (
            scored[
                "영업현금흐름"
            ] > 0
        )
    )

    # --------------------------------------------------------
    # 금융서비스
    #
    # 일반 영업기업과 유사한 Gate
    # --------------------------------------------------------

    service_mask = (
        scored[
            "FundamentalModel"
        ]
        == "금융서비스"
    )

    service_quality = (

        service_mask

        & (
            scored[
                "ROE"
            ] > 0
        )

        & (
            scored[
                "영업이익률"
            ] > 0
        )

        & (
            scored[
                "자본"
            ] > 0
        )

        & (
            scored[
                "영업현금흐름"
            ] > 0
        )
    )

    # --------------------------------------------------------
    # 순수 금융기업
    # --------------------------------------------------------

    pure_financial_models = [

        "은행/금융지주",
        "증권",
        "보험",
        "투자/캐피탈"

    ]

    financial_mask = (
        scored[
            "FundamentalModel"
        ].isin(
            pure_financial_models
        )
    )

    financial_quality = (

        financial_mask

        & (
            scored[
                "ROE"
            ] > 0
        )

        & (
            scored[
                "당기순이익"
            ] > 0
        )

        & (
            scored[
                "자본"
            ] > 0
        )

        & (
            scored[
                "자산"
            ] > 0
        )

        & (
            scored[
                "PBR"
            ] > 0
        )
    )

    scored.loc[
        (
            general_quality
            | service_quality
            | financial_quality
        ),
        "QualityGate"
    ] = True

    scored = scored[
        scored[
            "QualityGate"
        ]
    ].copy()

    print()
    print(
        f"Quality Gate 통과: "
        f"{len(scored):,}개"
    )

    print()

    print(
        scored[
            "FundamentalModel"
        ].value_counts()
    )

    # ========================================================
    # Fundamental Score
    # ========================================================

    scored[
        "FundamentalScore"
    ] = pd.NA

    # --------------------------------------------------------
    # 일반기업
    # --------------------------------------------------------

    general_idx = scored.index[
        scored[
            "FundamentalModel"
        ]
        == "일반기업"
    ]

    if len(
        general_idx
    ) > 0:

        temp = (
            score_general_companies(
                scored.loc[
                    general_idx
                ]
            )
        )

        for col in temp.columns:

            if col not in scored.columns:

                scored[
                    col
                ] = pd.NA

            scored.loc[
                general_idx,
                col
            ] = temp[
                col
            ]

    # --------------------------------------------------------
    # 금융 모델
    # --------------------------------------------------------

    for model in [

        "은행/금융지주",
        "증권",
        "보험",
        "투자/캐피탈",
        "금융서비스"

    ]:

        idx = scored.index[
            scored[
                "FundamentalModel"
            ]
            == model
        ]

        if len(
            idx
        ) == 0:

            continue

        temp = (
            score_financial_group(
                scored.loc[
                    idx
                ],
                model
            )
        )

        for col in temp.columns:

            if col not in scored.columns:

                scored[
                    col
                ] = pd.NA

            scored.loc[
                idx,
                col
            ] = temp[
                col
            ]

    scored[
        "FundamentalScore"
    ] = pd.to_numeric(
        scored[
            "FundamentalScore"
        ],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 결측 점검
    # --------------------------------------------------------

    missing = scored[
        scored[
            "FundamentalScore"
        ].isna()
    ].copy()

    print()
    print(
        f"FundamentalScore 결측: "
        f"{len(missing):,}개"
    )

    if not missing.empty:

        print(
            missing[
                [
                    "종목코드",
                    "종목명",
                    "FundamentalModel",
                    "업종"
                ]
            ].to_string(
                index=False
            )
        )

    scored = scored[
        scored[
            "FundamentalScore"
        ].notna()
    ].copy()

    scored.to_csv(
        SCORED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    return scored


# ============================================================
# 26. 업종 / 금융모델 상대평가
# ============================================================

def build_sector_scores(
    fundamental_df
):

    print()
    print(
        "[8] 업종 / 금융모델 상대평가 시작..."
    )

    df = (
        fundamental_df.copy()
    )

    # ========================================================
    # SPAC 제거
    # ========================================================

    spac_mask = (
        df[
            "종목명"
        ]
        .astype(str)
        .str.contains(
            "스팩|SPAC|기업인수목적",
            case=False,
            na=False
        )
    )

    print(
        f"SPAC 제외: "
        f"{spac_mask.sum():,}개"
    )

    df = df[
        ~spac_mask
    ].copy()

    # ========================================================
    # A. 일반기업
    # ========================================================

    general_df = df[
        df[
            "FundamentalModel"
        ]
        == "일반기업"
    ].copy()

    general_df[
        "업종"
    ] = (
        general_df[
            "업종"
        ]
        .fillna(
            "업종 미매칭"
        )
    )

    general_df[
        "업종기업수"
    ] = (
        general_df
        .groupby(
            "업종"
        )[
            "종목코드"
        ]
        .transform(
            "count"
        )
    )

    general_metrics = [

        (
            "ROE",
            "업종_ROE점수",
            True
        ),

        (
            "영업이익률",
            "업종_영업이익률점수",
            True
        ),

        (
            "매출성장률",
            "업종_매출성장점수",
            True
        ),

        (
            "영업이익성장률",
            "업종_영업이익성장점수",
            True
        ),

        (
            "부채비율",
            "업종_부채비율점수",
            False
        )

    ]

    for (
        source_col,
        score_col,
        higher
    ) in general_metrics:

        general_df[
            score_col
        ] = (
            general_df
            .groupby(
                "업종",
                group_keys=False
            )[
                source_col
            ]
            .transform(
                lambda s:
                neutralized_percentile(
                    s,
                    higher
                )
            )
        )

    general_df[
        "SectorFundamentalScore"
    ] = (

        general_df[
            "업종_ROE점수"
        ] * 0.25

        + general_df[
            "업종_영업이익률점수"
        ] * 0.25

        + general_df[
            "업종_매출성장점수"
        ] * 0.20

        + general_df[
            "업종_영업이익성장점수"
        ] * 0.20

        + general_df[
            "업종_부채비율점수"
        ] * 0.10
    )

    general_df[
        "업종내순위비율"
    ] = (
        general_df
        .groupby(
            "업종"
        )[
            "SectorFundamentalScore"
        ]
        .rank(
            pct=True,
            ascending=False,
            method="first"
        )
    )

    enough_general = (
        general_df[
            "업종기업수"
        ]
        >= MIN_SECTOR_SIZE
    )

    general_global_cutoff = (
        general_df[
            "FundamentalScore"
        ]
        .quantile(
            1
            - SELECT_RATIO
        )
    )

    general_df[
        "선정여부"
    ] = (

        (
            enough_general

            & (
                general_df[
                    "업종내순위비율"
                ]
                <= SELECT_RATIO
            )
        )

        |

        (
            ~enough_general

            & (
                general_df[
                    "FundamentalScore"
                ]
                >= general_global_cutoff
            )
        )
    )

    general_df[
        "평가기준"
    ] = (
        "업종 상대평가"
    )

    general_df.loc[
        ~enough_general,
        "평가기준"
    ] = (
        "일반기업 전체평가"
    )

    # ========================================================
    # B. 금융모델
    # ========================================================

    financial_df = df[
        df[
            "FundamentalModel"
        ]
        != "일반기업"
    ].copy()

    if not financial_df.empty:

        financial_df[
            "모델기업수"
        ] = (
            financial_df
            .groupby(
                "FundamentalModel"
            )[
                "종목코드"
            ]
            .transform(
                "count"
            )
        )

        financial_df[
            "SectorFundamentalScore"
        ] = (
            financial_df[
                "FundamentalScore"
            ]
        )

        financial_df[
            "모델내순위비율"
        ] = (
            financial_df
            .groupby(
                "FundamentalModel"
            )[
                "FundamentalScore"
            ]
            .rank(
                pct=True,
                ascending=False,
                method="first"
            )
        )

        enough_financial = (
            financial_df[
                "모델기업수"
            ]
            >= FINANCIAL_MIN_GROUP_SIZE
        )

        finance_global_cutoff = (
            financial_df[
                "FundamentalScore"
            ]
            .quantile(
                1
                - FINANCIAL_SELECT_RATIO
            )
        )

        financial_df[
            "선정여부"
        ] = (

            (
                enough_financial

                & (
                    financial_df[
                        "모델내순위비율"
                    ]
                    <= FINANCIAL_SELECT_RATIO
                )
            )

            |

            (
                ~enough_financial

                & (
                    financial_df[
                        "FundamentalScore"
                    ]
                    >= finance_global_cutoff
                )
            )
        )

        financial_df[
            "평가기준"
        ] = (
            "금융모델 상대평가"
        )

        financial_df.loc[
            ~enough_financial,
            "평가기준"
        ] = (
            "금융 전체평가"
        )

    else:

        financial_df = pd.DataFrame(
            columns=df.columns
        )

    # ========================================================
    # C. 통합
    # ========================================================

    sector_df = pd.concat(
        [
            general_df,
            financial_df
        ],
        ignore_index=True,
        sort=False
    )

    sector_selected = (
        sector_df[
            sector_df[
                "선정여부"
            ]
        ]
        .copy()
        .sort_values(
            "FundamentalScore",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    sector_df.to_csv(
        SECTOR_SCORED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    sector_selected.to_csv(
        SECTOR_SELECTED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 결과
    # ========================================================

    print()
    print(
        "=" * 75
    )

    print(
        "Fundamental Universe 선정 결과"
    )

    print(
        "=" * 75
    )

    print(
        f"전체 평가 기업: "
        f"{len(sector_df):,}개"
    )

    print(
        f"최종 Fundamental 후보: "
        f"{len(sector_selected):,}개"
    )

    print()
    print(
        "최종 평가모델별 후보:"
    )

    print(
        sector_selected[
            "FundamentalModel"
        ].value_counts()
    )

    if not financial_df.empty:

        print()
        print(
            "금융 평가모델별 Gate 통과 / 최종 선발"
        )

        print(
            financial_df
            .groupby(
                "FundamentalModel"
            )
            .agg(

                Gate통과=(
                    "종목코드",
                    "count"
                ),

                최종선발=(
                    "선정여부",
                    "sum"
                )

            )
        )

    return (
        sector_df,
        sector_selected
    )


# ============================================================
# 27. 주요 종목 점검
# ============================================================

def print_major_check(
    sector_df
):

    print()
    print(
        "=" * 75
    )

    print(
        "주요 금융 / 경계기업 점검"
    )

    print(
        "=" * 75
    )

    names = [

        # 은행
        "KB금융",
        "신한지주",
        "하나금융지주",
        "우리금융지주",
        "한국금융지주",
        "기업은행",
        "카카오뱅크",

        # 증권
        "키움증권",
        "미래에셋증권",
        "삼성증권",
        "NH투자증권",

        # 보험
        "삼성생명",
        "삼성화재",
        "현대해상",

        # 금융서비스
        "삼성카드",
        "NHN KCP",
        "KG이니시스",
        "에프앤가이드",
        "아이지넷",
        "더즌",

        # 투자/캐피탈
        "에이티넘인베스트",
        "스톤브릿지벤처스",
        "컴퍼니케이",
        "DSC인베스트먼트",

        # Grid 금융이지만 일반기업 모델을 써야 할 후보
        "다우기술",
        "다우데이타"

    ]

    temp = sector_df[
        sector_df[
            "종목명"
        ].isin(
            names
        )
    ].copy()

    cols = [

        "종목코드",
        "종목명",

        "섹터명",
        "업종",

        "FundamentalModel",

        "ROE",
        "PBR",

        "FundamentalScore",

        "선정여부",
        "평가기준"

    ]

    cols = [
        col
        for col in cols
        if col
        in temp.columns
    ]

    print(
        temp[
            cols
        ].sort_values(
            [
                "FundamentalModel",
                "종목명"
            ]
        ).to_string(
            index=False
        )
    )


# ============================================================
# 28. Grid 금융섹터 전체 분류 점검
# ============================================================

def print_financial_grid_check(
    sector_df
):

    print()
    print(
        "=" * 75
    )

    print(
        "Grid 금융섹터 → FundamentalModel 점검"
    )

    print(
        "=" * 75
    )

    grid_fin = sector_df[
        sector_df[
            "Grid금융여부"
        ]
    ].copy()

    cols = [

        "종목코드",
        "종목명",
        "섹터명",
        "업종",

        "FundamentalModel",

        "FundamentalScore",
        "선정여부"

    ]

    cols = [
        col
        for col in cols
        if col
        in grid_fin.columns
    ]

    print(
        grid_fin[
            cols
        ]
        .sort_values(
            [
                "FundamentalModel",
                "종목명"
            ]
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 29. 실행
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 75
    )

    print(
        "DART Fundamental Screening FINAL"
    )

    print(
        "General / Bank / Securities / Insurance / "
        "Investment-Capital / Financial Services"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # 1. 현재 상장 보통주
    # --------------------------------------------------------

    universe = (
        build_current_universe()
    )

    # --------------------------------------------------------
    # 2. DART 재무정보
    # --------------------------------------------------------

    raw_df = (
        run_fundamental_analysis(
            universe
        )
    )

    # --------------------------------------------------------
    # 3. KIND 업종
    # --------------------------------------------------------

    kind_df = (
        get_kind_sector_data()
    )

    # --------------------------------------------------------
    # 4. 대회 Sector Grid
    # --------------------------------------------------------

    grid_df = (
        get_competition_sector_grid()
    )

    # --------------------------------------------------------
    # 5. Quality Gate + Fundamental Score
    # --------------------------------------------------------

    fundamental_df = (
        build_quality_universe(
            raw_df,
            kind_df,
            grid_df
        )
    )

    # --------------------------------------------------------
    # 6. 업종 / 금융모델 상대평가
    # --------------------------------------------------------

    (
        sector_scored,
        sector_selected
    ) = build_sector_scores(
        fundamental_df
    )

    # --------------------------------------------------------
    # 7. 주요 종목 점검
    # --------------------------------------------------------

    print_major_check(
        sector_scored
    )

    # --------------------------------------------------------
    # 8. Grid 금융 전체 점검
    # --------------------------------------------------------

    print_financial_grid_check(
        sector_scored
    )

    # --------------------------------------------------------
    # 완료
    # --------------------------------------------------------

    print()
    print(
        "=" * 75
    )

    print(
        "완료"
    )

    print(
        "=" * 75
    )

    print(
        f"최종 Fundamental 후보군: "
        f"{len(sector_selected):,}개"
    )

    print()

    print(
        "Momentum 입력파일:"
    )

    print(
        SECTOR_SELECTED_FILE
    )

    print()

    print(
        "※ Grid 섹터는 포트폴리오 산업비중 제한용입니다."
    )

    print(
        "※ FundamentalModel은 실제 사업모델에 따라 별도로 분류합니다."
    )

    print()

    print(
        "※ Fundamental 단계에서는 "
        "시가총액 1,000억원 조건을 적용하지 않습니다."
    )

    print(
        "※ Fundamental 단계에서는 "
        "5일 평균 거래대금 30억원 조건도 적용하지 않습니다."
    )

    print()

    print(
        "※ 시총 / 거래대금 / 투자주의 / 투자경고 / 투자위험 / "
        "투자주의환기 / 관리종목 / 신규·재상장 6영업일 조건은 "
        "portfolio.py에서 주문일 기준으로 검사합니다."
    )