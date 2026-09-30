# ============================================================
# 16_final_strategy_vs_kospi_backtest.py
#
# FINAL STRATEGY BACKTEST
# ------------------------------------------------------------
# Universe
#   - PIT market cap >= KRW 1 trillion
#   - BothUp20:
#       EPS Fwd12M_t > EPS Fwd12M_t-20
#       OP  Fwd12M_t > OP  Fwd12M_t-20
#
# Primary alpha
#   FreshRank =
#       0.5 * Rank(EPSRev5)
#     + 0.5 * Rank(OPRev5)
#
# Entry confirmation
#   - New names prefer Mom20Rank >= 0.30
#   - If N=30 / sector feasibility fails:
#       0.30 -> 0.20 -> 0.10 -> 0.00
#
# Hysteresis
#   - New entry: FreshRank >= 0.70
#   - Incumbent hold: FreshRank >= 0.40
#   - If still infeasible after momentum relaxation:
#       entry cutoff 0.70 -> 0.65 -> 0.60 -> 0.50 -> 0.00
#
# Portfolio
#   - N = 30
#   - Equal Weight
#   - Contest sector target = 90% of hard cap
#   - 5 trading-day rebalance
#   - signal close(t) -> execute open(t+1)
#
# Costs
#   - Buy  : 10bp
#   - Sell : 30bp
#
# Benchmark
#   - KOSPI
#
# Risk overlay
#   - BASE: 100% exposure
#   - KOSPI_MA60_2STATE:
#       KOSPI close >= MA60 -> 100%
#       KOSPI close <  MA60 -> 80%
#
# Outputs
#   performance_by_offset.csv
#   performance_summary.csv
#   kospi_summary.csv
#   selection_schedule.csv
#   relaxation_diagnostics_summary.csv
#   daily_nav_all_offsets.csv
#   daily_nav_average.csv
#   profit_contribution_by_stock.csv
#   trade_log.csv
#   final_strategy_vs_kospi.png
# ============================================================

from pathlib import Path
from settings import (
    BASE_DIR,
    FACTOR_PANEL_FILE,
    CONSENSUS_DIR,
    CONTEST_FILE,
    KOSPI_FILE,
    FINAL_OUTPUT_DIR,
    BACKTEST_START,
    MCAP_CUTOFF,
    PORTFOLIO_N,
    REBALANCE_DAYS,
    OFFSETS,
    ENTRY_CUTOFF,
    HOLD_CUTOFF,
    ENTRY_RELAXATION_LADDER,
    MOMENTUM_FLOOR_LADDER,
    COMMISSION_BPS,
    SELL_TAX_BPS,
    SECTOR_CAP_MULTIPLIER,
    SECTOR_CAPS,
    RUN_KOSPI_MA60_REGIME,
    REGIME_MA_WINDOW,
    REGIME_RISK_ON_EXPOSURE,
    REGIME_RISK_OFF_EXPOSURE,
)
import math
import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIG
# ============================================================

FACTOR_FILE = FACTOR_PANEL_FILE if FACTOR_PANEL_FILE.exists() else None
OUTPUT_DIR = FINAL_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KOSPI_FILE_CANDIDATES = [
    KOSPI_FILE,
    BASE_DIR / "kospi.csv",
    BASE_DIR / "KOSPI_index.csv",
    BASE_DIR / "kospi_index.csv",
]

SIM_START = pd.Timestamp(BACKTEST_START)
NAME_WEIGHT = 1.0 / PORTFOLIO_N

BUY_COST_RATE = COMMISSION_BPS / 10000.0
SELL_COST_RATE = (COMMISSION_BPS + SELL_TAX_BPS) / 10000.0

SAFE_SECTOR_CAPS = {
    sector: cap * SECTOR_CAP_MULTIPLIER
    for sector, cap in SECTOR_CAPS.items()
}
SECTOR_ORDER = list(SECTOR_CAPS.keys())


# ============================================================
# 1. INPUT CHECK
# ============================================================

if FACTOR_FILE is None:
    raise FileNotFoundError(
        "factor_panel.parquet가 없습니다."
    )

if not CONTEST_FILE.exists():
    raise FileNotFoundError(
        f"Contest sector file 없음: {CONTEST_FILE}"
    )

if not CONSENSUS_DIR.exists():
    raise FileNotFoundError(
        f"Consensus folder 없음: {CONSENSUS_DIR}"
    )

if not list(CONSENSUS_DIR.glob("consensus_*.parquet")):
    raise FileNotFoundError(
        f"consensus_*.parquet 없음: {CONSENSUS_DIR}"
    )

print("=" * 110)
print("FINAL STRATEGY VS KOSPI")
print("=" * 110)
print("Factor   :", FACTOR_FILE)
print("Consensus:", CONSENSUS_DIR)
print("Contest  :", CONTEST_FILE)
print("Output   :", OUTPUT_DIR)


# ============================================================
# 2. CONTEST SECTOR MAP
# ============================================================

contest = pd.read_csv(
    CONTEST_FILE,
    dtype=str,
    encoding="utf-8-sig",
)

required_cols = [
    "종목코드",
    "종목명",
    "섹터코드",
    "섹터명",
]

for c in required_cols:
    if c not in contest.columns:
        raise ValueError(f"Contest file에 {c} 없음")

contest = (
    contest[required_cols]
    .rename(
        columns={
            "종목코드": "code",
            "종목명": "contest_name",
            "섹터코드": "sector_code",
            "섹터명": "sector_name",
        }
    )
    .drop_duplicates("code")
    .copy()
)

contest["code"] = (
    contest["code"]
    .astype(str)
    .str.strip()
)

contest = contest.loc[
    contest["sector_code"].isin(SECTOR_ORDER)
].copy()

sector_map = (
    contest
    .set_index("code")["sector_code"]
    .to_dict()
)


# ============================================================
# 3. FACTOR SCHEMA
# ============================================================

factor_path = str(FACTOR_FILE).replace("\\", "/")
consensus_glob = str(
    CONSENSUS_DIR / "consensus_*.parquet"
).replace("\\", "/")

con = duckdb.connect()

schema = con.execute(
    f"""
    DESCRIBE
    SELECT *
    FROM read_parquet('{factor_path}')
    """
).df()

factor_cols = set(schema["column_name"])


def pick_col(candidates):
    for c in candidates:
        if c in factor_cols:
            return c
    raise ValueError(f"필수 컬럼 없음: {candidates}")


DATE_COL = pick_col(["date"])
CODE_COL = pick_col(["code"])
NAME_COL = pick_col(
    ["name", "stock_name", "code_name"]
)
OPEN_COL = pick_col(
    ["open_adj", "adj_open", "adjusted_open", "open"]
)
CLOSE_COL = pick_col(
    ["close_adj", "adj_close", "adjusted_close", "close"]
)
MCAP_COL = pick_col(
    ["market_cap_krw", "market_cap"]
)
VALUE_COL = pick_col(
    ["trading_value_krw", "trading_value"]
)


# ============================================================
# 4. BUILD DATA PANEL
#
# 중요:
#   - 시총 1조 조건은 "당일 투자자격"에만 적용한다.
#   - 과거 20일 가격/보유종목 평가 데이터는 시총 조건으로 잘라내지 않는다.
#     그래야 1조를 막 넘은 종목의 Mom20과, 보유 후 1조 아래로 내려간
#     종목의 가격 평가가 왜곡되지 않는다.
# ============================================================

con.register(
    "contest_map",
    contest[["code", "sector_code", "sector_name"]],
)

signal_query = f"""
WITH px_all AS (
    SELECT
        CAST(f.{DATE_COL} AS DATE) AS date,
        f.{CODE_COL} AS code,
        f.{NAME_COL} AS name,
        f.{OPEN_COL} AS open_price,
        f.{CLOSE_COL} AS close_price,
        f.{MCAP_COL} AS market_cap_krw,
        f.{VALUE_COL} AS trading_value_krw,
        c.sector_code,
        c.sector_name
    FROM read_parquet('{factor_path}') AS f
    INNER JOIN contest_map AS c
        ON f.{CODE_COL} = c.code
    WHERE f.{CLOSE_COL} IS NOT NULL
),
calendar AS (
    SELECT
        date,
        LAG(date, 5) OVER (ORDER BY date) AS lag5_date,
        LAG(date, 20) OVER (ORDER BY date) AS lag20_date
    FROM (
        SELECT DISTINCT date
        FROM px_all
    )
),
eligible_today AS (
    SELECT *
    FROM px_all
    WHERE
        market_cap_krw >= {MCAP_CUTOFF}
        AND trading_value_krw > 0
        AND open_price IS NOT NULL
        AND close_price IS NOT NULL
),
merged AS (
    SELECT
        p.*,
        cal.lag5_date,
        cal.lag20_date,
        c0.eps_fwd12m,
        c0.op_fwd12m,
        c5.eps_fwd12m AS eps_fwd12m_lag5,
        c5.op_fwd12m AS op_fwd12m_lag5,
        c20.eps_fwd12m AS eps_fwd12m_lag20,
        c20.op_fwd12m AS op_fwd12m_lag20,
        p20.close_price AS close_lag20
    FROM eligible_today AS p
    INNER JOIN calendar AS cal
        USING(date)
    LEFT JOIN read_parquet('{consensus_glob}') AS c0
        ON c0.code = p.code
        AND CAST(c0.date AS DATE) = p.date
    LEFT JOIN read_parquet('{consensus_glob}') AS c5
        ON c5.code = p.code
        AND CAST(c5.date AS DATE) = cal.lag5_date
    LEFT JOIN read_parquet('{consensus_glob}') AS c20
        ON c20.code = p.code
        AND CAST(c20.date AS DATE) = cal.lag20_date
    LEFT JOIN px_all AS p20
        ON p20.code = p.code
        AND p20.date = cal.lag20_date
)
SELECT *
FROM merged
ORDER BY date, code
"""

print("\nBuilding signal panel...")
sig = con.execute(signal_query).df()

# Execution / valuation panel: no market-cap screen.
price_query = f"""
SELECT
    CAST(f.{DATE_COL} AS DATE) AS date,
    f.{CODE_COL} AS code,
    f.{NAME_COL} AS name,
    f.{OPEN_COL} AS open_price,
    f.{CLOSE_COL} AS close_price,
    f.{MCAP_COL} AS market_cap_krw,
    c.sector_code,
    c.sector_name
FROM read_parquet('{factor_path}') AS f
INNER JOIN contest_map AS c
    ON f.{CODE_COL} = c.code
WHERE
    f.{OPEN_COL} IS NOT NULL
    OR f.{CLOSE_COL} IS NOT NULL
ORDER BY date, code
"""

px = con.execute(price_query).df()
con.close()

sig["date"] = pd.to_datetime(sig["date"])
px["date"] = pd.to_datetime(px["date"])

for c in [
    "market_cap_krw",
    "trading_value_krw",
    "eps_fwd12m",
    "op_fwd12m",
    "eps_fwd12m_lag5",
    "op_fwd12m_lag5",
    "eps_fwd12m_lag20",
    "op_fwd12m_lag20",
    "close_price",
    "close_lag20",
]:
    sig[c] = pd.to_numeric(sig[c], errors="coerce")


# ============================================================
# 5. SIGNAL CONSTRUCTION
# ============================================================

def symmetric_revision(new, old):
    new = pd.to_numeric(new, errors="coerce")
    old = pd.to_numeric(old, errors="coerce")

    denom = new.abs() + old.abs()

    out = 2.0 * (new - old) / denom
    out.loc[denom <= 1e-12] = np.nan

    return out.clip(-2, 2)


sig["eps_rev5"] = symmetric_revision(
    sig["eps_fwd12m"],
    sig["eps_fwd12m_lag5"],
)

sig["op_rev5"] = symmetric_revision(
    sig["op_fwd12m"],
    sig["op_fwd12m_lag5"],
)

sig["eps_rev20"] = symmetric_revision(
    sig["eps_fwd12m"],
    sig["eps_fwd12m_lag20"],
)

sig["op_rev20"] = symmetric_revision(
    sig["op_fwd12m"],
    sig["op_fwd12m_lag20"],
)

sig["both_up20"] = (
    (sig["eps_fwd12m"] > sig["eps_fwd12m_lag20"])
    &
    (sig["op_fwd12m"] > sig["op_fwd12m_lag20"])
)

sig["mom20"] = (
    sig["close_price"]
    / sig["close_lag20"]
    - 1.0
)

sig.loc[
    (
        (sig["close_price"] <= 0)
        |
        (sig["close_lag20"] <= 0)
    ),
    "mom20",
] = np.nan


# ============================================================
# 6. FINAL UNIVERSE / RANKS
# ============================================================

eligible = (
    sig["both_up20"].fillna(False)
    &
    sig["eps_rev5"].notna()
    &
    sig["op_rev5"].notna()
)

panel = sig.loc[eligible].copy()

panel["fresh_eps_rank"] = (
    panel.groupby("date")["eps_rev5"]
    .rank(
        pct=True,
        method="average",
    )
)

panel["fresh_op_rank"] = (
    panel.groupby("date")["op_rev5"]
    .rank(
        pct=True,
        method="average",
    )
)

panel["fresh_rank"] = (
    panel[
        [
            "fresh_eps_rank",
            "fresh_op_rank",
        ]
    ].mean(axis=1)
)

panel["r20_eps_rank"] = (
    panel.groupby("date")["eps_rev20"]
    .rank(
        pct=True,
        method="average",
    )
)

panel["r20_op_rank"] = (
    panel.groupby("date")["op_rev20"]
    .rank(
        pct=True,
        method="average",
    )
)

panel["revision20_rank"] = (
    panel[
        [
            "r20_eps_rank",
            "r20_op_rank",
        ]
    ].mean(axis=1)
)

panel["revision20_rank"] = (
    panel["revision20_rank"]
    .fillna(-np.inf)
)

panel["mom20_rank"] = (
    panel.groupby("date")["mom20"]
    .rank(
        pct=True,
        method="average",
    )
)

# Missing momentum = neutral
panel["mom20_rank"] = (
    panel["mom20_rank"]
    .fillna(0.5)
)

panel_by_date = {
    d: g.copy()
    for d, g in panel.groupby(
        "date",
        sort=False,
    )
}

candidate_count = (
    panel.groupby("date")
    .size()
)

print(
    "Avg BothUp20 candidates/day:",
    f"{candidate_count.mean():.1f}",
)


# ============================================================
# 7. PRICE MATRICES
# ============================================================

open_wide = (
    px.pivot(
        index="date",
        columns="code",
        values="open_price",
    )
    .sort_index()
)

close_wide = (
    px.pivot(
        index="date",
        columns="code",
        values="close_price",
    )
    .sort_index()
)

market_dates = (
    open_wide.index
    .intersection(close_wide.index)
    .sort_values()
)

market_dates = market_dates[
    market_dates >= SIM_START
]

print(
    "Backtest:",
    market_dates.min(),
    "~",
    market_dates.max(),
)


# ============================================================
# 8. KOSPI
# ============================================================

def load_local_kospi(path):
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    date_col = None
    close_col = None

    for c in df.columns:
        key = str(c).strip().lower()

        if (
            date_col is None
            and key in ["date", "날짜", "일자"]
        ):
            date_col = c

        if (
            close_col is None
            and key in ["close", "종가", "kospi", "코스피"]
        ):
            close_col = c

    if date_col is None or close_col is None:
        raise ValueError(
            "KOSPI CSV는 Date/Close 또는 날짜/종가 컬럼 필요"
        )

    out = df[
        [date_col, close_col]
    ].copy()

    out.columns = [
        "date",
        "kospi_close",
    ]

    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    )

    out["kospi_close"] = (
        out["kospi_close"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    out["kospi_close"] = pd.to_numeric(
        out["kospi_close"],
        errors="coerce",
    )

    return (
        out
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
    )


kospi_file = next(
    (
        p
        for p in KOSPI_FILE_CANDIDATES
        if p.exists()
    ),
    None,
)

if kospi_file is not None:
    print("KOSPI local:", kospi_file)
    kospi = load_local_kospi(kospi_file)

else:
    print("KOSPI CSV 없음 -> FinanceDataReader KS11")

    try:
        import FinanceDataReader as fdr
    except ImportError as e:
        raise ImportError(
            "KOSPI CSV가 없으면: pip install finance-datareader"
        ) from e

    raw = (
        fdr.DataReader(
            "KS11",
            str(
                (
                    market_dates.min()
                    - pd.Timedelta(days=150)
                ).date()
            ),
            str(
                (
                    market_dates.max()
                    + pd.Timedelta(days=5)
                ).date()
            ),
        )
        .reset_index()
    )

    if "Date" not in raw.columns:
        raw = raw.rename(
            columns={
                raw.columns[0]: "Date"
            }
        )

    kospi = (
        raw[
            ["Date", "Close"]
        ]
        .rename(
            columns={
                "Date": "date",
                "Close": "kospi_close",
            }
        )
        .copy()
    )

    kospi["date"] = pd.to_datetime(
        kospi["date"]
    )

kospi = (
    kospi
    .set_index("date")
    .sort_index()
)

kospi["kospi_ret"] = (
    kospi["kospi_close"]
    .pct_change()
)

kospi["kospi_ma60"] = (
    kospi["kospi_close"]
    .rolling(
        REGIME_MA_WINDOW,
        min_periods=REGIME_MA_WINDOW,
    )
    .mean()
)

kospi = (
    kospi
    .reindex(market_dates)
    .ffill()
)


# ============================================================
# 9. EXACT N=30 SELECTION
#
# DP objective:
#   1) retain incumbents as much as possible
#   2) maximize FreshRank
#   3) Revision20Rank exact tie-break
# ============================================================

def better_value(candidate, incumbent):
    if incumbent is None:
        return True

    c_stay, c_fresh, c_r20 = candidate
    i_stay, i_fresh, i_r20 = incumbent

    if c_stay != i_stay:
        return c_stay > i_stay

    if abs(c_fresh - i_fresh) > 1e-12:
        return c_fresh > i_fresh

    return c_r20 > i_r20 + 1e-12


def sorted_sector(g):
    return (
        g.sort_values(
            [
                "is_current",
                "fresh_rank",
                "revision20_rank",
                "code",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def prefix_stats(g):
    n = len(g)

    stay = np.zeros(
        n + 1,
        dtype=int,
    )

    fresh = np.zeros(
        n + 1,
        dtype=float,
    )

    r20 = np.zeros(
        n + 1,
        dtype=float,
    )

    for k in range(
        1,
        n + 1,
    ):
        row = g.iloc[
            k - 1
        ]

        stay[k] = (
            stay[k - 1]
            + int(
                row["is_current"]
            )
        )

        fresh[k] = (
            fresh[k - 1]
            + float(
                row["fresh_rank"]
            )
        )

        rr = row[
            "revision20_rank"
        ]

        r20[k] = (
            r20[k - 1]
            + (
                float(rr)
                if np.isfinite(rr)
                else 0.0
            )
        )

    return (
        stay,
        fresh,
        r20,
    )


def exact_select(candidates):
    if len(candidates) == 0:
        return None

    sector_quota = {
        sector:
            int(
                math.floor(
                    SAFE_SECTOR_CAPS[
                        sector
                    ]
                    / NAME_WEIGHT
                    + 1e-12
                )
            )
        for sector in SECTOR_ORDER
    }

    sector_data = {}

    for sector in SECTOR_ORDER:
        g = candidates.loc[
            candidates[
                "sector_code"
            ]
            == sector
        ].copy()

        sector_data[
            sector
        ] = sorted_sector(g)

    dp = {
        0: {
            "value":
                (0, 0.0, 0.0),
            "choices": [],
        }
    }

    for sector in SECTOR_ORDER:
        g = sector_data[
            sector
        ]

        (
            stay,
            fresh,
            r20,
        ) = prefix_stats(g)

        max_k = min(
            len(g),
            PORTFOLIO_N,
            sector_quota[
                sector
            ],
        )

        options = []

        for k in range(
            max_k + 1
        ):
            options.append(
                {
                    "k": k,
                    "value": (
                        int(stay[k]),
                        float(fresh[k]),
                        float(r20[k]),
                    ),
                }
            )

        new_dp = {}

        for used, state in dp.items():

            for opt in options:
                new_used = (
                    used
                    + opt["k"]
                )

                if new_used > PORTFOLIO_N:
                    continue

                old_v = state[
                    "value"
                ]

                add_v = opt[
                    "value"
                ]

                new_v = (
                    old_v[0] + add_v[0],
                    old_v[1] + add_v[1],
                    old_v[2] + add_v[2],
                )

                old_state = new_dp.get(
                    new_used
                )

                if (
                    old_state is None
                    or better_value(
                        new_v,
                        old_state[
                            "value"
                        ],
                    )
                ):
                    new_dp[
                        new_used
                    ] = {
                        "value":
                            new_v,
                        "choices":
                            (
                                state[
                                    "choices"
                                ]
                                + [
                                    (
                                        sector,
                                        opt["k"],
                                    )
                                ]
                            ),
                    }

        dp = new_dp

        if not dp:
            return None

    final = dp.get(
        PORTFOLIO_N
    )

    if final is None:
        return None

    parts = []

    for sector, k in final[
        "choices"
    ]:
        if k > 0:
            parts.append(
                sector_data[
                    sector
                ]
                .iloc[:k]
                .copy()
            )

    if not parts:
        return None

    selected = pd.concat(
        parts,
        ignore_index=True,
    )

    if len(selected) != PORTFOLIO_N:
        raise RuntimeError(
            "DP selection count mismatch"
        )

    return selected


# ============================================================
# 10. FINAL TARGET GENERATION
# ============================================================

def make_target(
    signal_date,
    current_codes,
):
    g = panel_by_date.get(
        signal_date
    )

    if g is None:
        return None, None

    g = g.copy()

    g["is_current"] = (
        g["code"]
        .isin(
            current_codes
        )
    )

    attempts = []

    # First relax momentum, then Fresh entry threshold
    for entry_cutoff in (
        ENTRY_RELAXATION_LADDER
    ):

        for mom_floor in (
            MOMENTUM_FLOOR_LADDER
        ):

            allowed_current = (
                g["is_current"]
                &
                (
                    g["fresh_rank"]
                    >= HOLD_CUTOFF
                )
            )

            allowed_new = (
                ~g["is_current"]
                &
                (
                    g["fresh_rank"]
                    >= entry_cutoff
                )
                &
                (
                    g["mom20_rank"]
                    >= mom_floor
                )
            )

            candidates = g.loc[
                allowed_current
                | allowed_new
            ].copy()

            attempts.append(
                {
                    "entry_cutoff":
                        entry_cutoff,
                    "mom_floor":
                        mom_floor,
                    "candidate_n":
                        len(candidates),
                }
            )

            selected = exact_select(
                candidates
            )

            if selected is None:
                continue

            selected[
                "target_weight"
            ] = NAME_WEIGHT

            diag = {
                "entry_cutoff_used":
                    entry_cutoff,

                "entry_relaxed":
                    (
                        entry_cutoff
                        < ENTRY_CUTOFF
                        - 1e-12
                    ),

                "mom_floor_initial":
                    MOMENTUM_FLOOR_LADDER[
                        0
                    ],

                "mom_floor_used":
                    mom_floor,

                "momentum_relaxed":
                    (
                        mom_floor
                        <
                        MOMENTUM_FLOOR_LADDER[
                            0
                        ]
                        - 1e-12
                    ),

                "retained_count":
                    int(
                        selected[
                            "is_current"
                        ].sum()
                    ),

                "new_count":
                    int(
                        (
                            ~selected[
                                "is_current"
                            ]
                        ).sum()
                    ),

                "mean_fresh_rank":
                    float(
                        selected[
                            "fresh_rank"
                        ].mean()
                    ),

                "mean_mom20_rank":
                    float(
                        selected[
                            "mom20_rank"
                        ].mean()
                    ),

                "n_attempts":
                    len(attempts),
            }

            return (
                selected,
                diag,
            )

    return None, {
        "failed": True,
        "n_attempts":
            len(attempts),
    }


# ============================================================
# 11. BUILD REBALANCE SCHEDULE
# ============================================================

def build_schedule(offset):
    signal_dates = set(
        market_dates[
            offset
            :
            -1
            :
            REBALANCE_DAYS
        ]
    )

    current_codes = set()

    schedule = {}
    diag_rows = []

    for i, date in enumerate(
        market_dates
    ):
        if date not in signal_dates:
            continue

        if (
            i + 1
            >= len(
                market_dates
            )
        ):
            continue

        selected, diag = (
            make_target(
                signal_date=date,
                current_codes=current_codes,
            )
        )

        exec_date = (
            market_dates[
                i + 1
            ]
        )

        if selected is None:
            diag_rows.append(
                {
                    "offset":
                        offset,
                    "signal_date":
                        date,
                    "exec_date":
                        exec_date,
                    "success":
                        False,
                    **(
                        diag
                        if diag is not None
                        else {}
                    ),
                }
            )
            continue

        schedule[
            exec_date
        ] = {
            "signal_date":
                date,
            "target":
                selected,
            "diag":
                diag,
        }

        current_codes = set(
            selected[
                "code"
            ]
        )

        diag_rows.append(
            {
                "offset":
                    offset,
                "signal_date":
                    date,
                "exec_date":
                    exec_date,
                "success":
                    True,
                **diag,
            }
        )

    return (
        schedule,
        pd.DataFrame(
            diag_rows
        ),
    )


# ============================================================
# 12. SIMULATION
# ============================================================

def get_price(
    row,
    code,
    fallback,
):
    p = row.get(
        code,
        np.nan,
    )

    if pd.notna(p) and p > 0:
        return float(p)

    return fallback.get(
        code,
        np.nan,
    )


def simulate(
    schedule,
    offset,
):
    shares = {}
    cash = 1.0
    last_close = {}

    daily_rows = []
    trade_rows = []

    contribution = {}

    prev_nav = 1.0
    started = False

    for date in market_dates:

        open_row = open_wide.loc[
            date
        ]

        close_row = close_wide.loc[
            date
        ]

        # ----------------------------------------------------
        # A. Overnight P&L
        # ----------------------------------------------------

        for code, qty in list(
            shares.items()
        ):
            prev_c = last_close.get(
                code,
                np.nan,
            )

            open_p = get_price(
                open_row,
                code,
                last_close,
            )

            if (
                pd.notna(prev_c)
                and pd.notna(open_p)
            ):
                pnl = (
                    qty
                    * (
                        open_p
                        - prev_c
                    )
                )

                contribution[
                    code
                ] = (
                    contribution.get(
                        code,
                        0.0,
                    )
                    + pnl
                )

        buy_value = 0.0
        sell_value = 0.0
        transaction_cost = 0.0

        # ----------------------------------------------------
        # B. Rebalance at open
        # ----------------------------------------------------

        if date in schedule:
            started = True

            target = schedule[
                date
            ]["target"]

            weight_map = (
                target
                .set_index("code")[
                    "target_weight"
                ]
                .to_dict()
            )

            target_codes = set(
                weight_map
            )

            equity_open = cash

            for code, qty in shares.items():

                p = get_price(
                    open_row,
                    code,
                    last_close,
                )

                if pd.notna(p):
                    equity_open += (
                        qty * p
                    )

            if equity_open <= 0:
                raise RuntimeError(
                    "equity_open <= 0"
                )

            # SELL FIRST
            sell_orders = []

            for code in list(
                shares.keys()
            ):
                p = get_price(
                    open_row,
                    code,
                    last_close,
                )

                if pd.isna(p) or p <= 0:
                    continue

                current_value = (
                    shares[
                        code
                    ]
                    * p
                )

                desired_value = (
                    weight_map.get(
                        code,
                        0.0,
                    )
                    * equity_open
                )

                amount = (
                    current_value
                    - desired_value
                )

                if amount > 1e-12:
                    sell_orders.append(
                        (
                            code,
                            p,
                            amount,
                        )
                    )

            for code, p, amount in sell_orders:

                qty = (
                    amount
                    / p
                )

                shares[
                    code
                ] -= qty

                fee = (
                    amount
                    * SELL_COST_RATE
                )

                cash += (
                    amount
                    - fee
                )

                sell_value += amount
                transaction_cost += fee

                if (
                    abs(
                        shares[
                            code
                        ]
                    )
                    < 1e-12
                ):
                    del shares[
                        code
                    ]

            # BUY
            buy_orders = []

            for code in target_codes:

                p = get_price(
                    open_row,
                    code,
                    last_close,
                )

                if pd.isna(p) or p <= 0:
                    continue

                current_value = (
                    shares.get(
                        code,
                        0.0,
                    )
                    * p
                )

                desired_value = (
                    weight_map[
                        code
                    ]
                    * equity_open
                )

                need = (
                    desired_value
                    - current_value
                )

                if need > 1e-12:
                    buy_orders.append(
                        (
                            code,
                            p,
                            need,
                        )
                    )

            total_need = sum(
                x[2]
                for x in buy_orders
            )

            required_cash = (
                total_need
                * (
                    1.0
                    + BUY_COST_RATE
                )
            )

            buy_scale = (
                min(
                    1.0,
                    cash
                    / required_cash,
                )
                if required_cash > 0
                else 0.0
            )

            for code, p, need0 in buy_orders:

                amount = (
                    need0
                    * buy_scale
                )

                if amount <= 0:
                    continue

                fee = (
                    amount
                    * BUY_COST_RATE
                )

                outflow = (
                    amount + fee
                )

                if (
                    outflow
                    > cash
                    + 1e-10
                ):
                    continue

                qty = (
                    amount / p
                )

                shares[
                    code
                ] = (
                    shares.get(
                        code,
                        0.0,
                    )
                    + qty
                )

                cash -= outflow
                buy_value += amount
                transaction_cost += fee

            trade_rows.append(
                {
                    "date":
                        date,
                    "offset":
                        offset,
                    "buy_value":
                        buy_value,
                    "sell_value":
                        sell_value,
                    "transaction_cost":
                        transaction_cost,
                    "gross_turnover":
                        (
                            buy_value
                            + sell_value
                        )
                        / equity_open,
                }
            )

        # ----------------------------------------------------
        # C. Intraday P&L
        # ----------------------------------------------------

        for code, qty in shares.items():

            open_p = get_price(
                open_row,
                code,
                last_close,
            )

            close_p = get_price(
                close_row,
                code,
                last_close,
            )

            if (
                pd.notna(open_p)
                and pd.notna(close_p)
            ):
                pnl = (
                    qty
                    * (
                        close_p
                        - open_p
                    )
                )

                contribution[
                    code
                ] = (
                    contribution.get(
                        code,
                        0.0,
                    )
                    + pnl
                )

                last_close[
                    code
                ] = (
                    close_p
                )

        # ----------------------------------------------------
        # D. Close NAV / management diagnostics
        # ----------------------------------------------------

        nav = cash
        values = {}

        for code, qty in shares.items():

            p = get_price(
                close_row,
                code,
                last_close,
            )

            if pd.notna(p):
                last_close[
                    code
                ] = p

                value = (
                    qty * p
                )

                values[
                    code
                ] = value

                nav += value

        if nav <= 0:
            raise RuntimeError(
                "NAV <= 0"
            )

        weights = {
            code:
                value / nav
            for code, value
            in values.items()
        }

        w_arr = np.array(
            list(
                weights.values()
            ),
            dtype=float,
        )

        if len(w_arr):
            weight_hhi = float(
                np.sum(
                    w_arr ** 2
                )
            )

            effective_n = (
                1.0
                / weight_hhi
            )

            top5_weight = float(
                np.sort(
                    w_arr
                )[-5:].sum()
            )

            max_name_weight = float(
                w_arr.max()
            )

        else:
            weight_hhi = np.nan
            effective_n = np.nan
            top5_weight = np.nan
            max_name_weight = np.nan

        sector_weights = {}

        for code, weight in weights.items():

            sector = (
                sector_map.get(
                    code
                )
            )

            if sector is None:
                continue

            sector_weights[
                sector
            ] = (
                sector_weights.get(
                    sector,
                    0.0,
                )
                + weight
            )

        sec_arr = np.array(
            list(
                sector_weights.values()
            ),
            dtype=float,
        )

        sector_hhi = (
            float(
                np.sum(
                    sec_arr ** 2
                )
            )
            if len(sec_arr)
            else np.nan
        )

        daily_return = (
            nav
            / prev_nav
            - 1.0
        )

        daily_rows.append(
            {
                "date":
                    date,
                "offset":
                    offset,
                "nav":
                    nav,
                "daily_return":
                    daily_return,
                "cash_weight":
                    cash / nav,
                "n_holdings":
                    len(weights),
                "effective_n":
                    effective_n,
                "weight_hhi":
                    weight_hhi,
                "top5_weight":
                    top5_weight,
                "max_name_weight":
                    max_name_weight,
                "sector_hhi":
                    sector_hhi,
                "transaction_cost":
                    transaction_cost,
                "started":
                    started,
            }
        )

        prev_nav = nav

    daily = pd.DataFrame(
        daily_rows
    )

    if daily["started"].any():

        start_idx = (
            daily.index[
                daily["started"]
            ][0]
        )

        daily = (
            daily.loc[
                start_idx:
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        first_nav = (
            daily.iloc[0][
                "nav"
            ]
        )

        daily[
            "nav_norm"
        ] = (
            daily["nav"]
            / first_nav
        )

        daily.loc[
            0,
            "daily_return"
        ] = 0.0

    else:
        daily[
            "nav_norm"
        ] = np.nan

    contribution = (
        pd.Series(
            contribution,
            dtype=float,
        )
        .sort_values(
            ascending=False
        )
    )

    return (
        daily,
        pd.DataFrame(
            trade_rows
        ),
        contribution,
    )


# ============================================================
# 13. PERFORMANCE METRICS
# ============================================================

def max_drawdown(nav):
    nav = (
        pd.Series(nav)
        .dropna()
    )

    if len(nav) == 0:
        return np.nan

    peak = nav.cummax()

    return float(
        (
            nav
            / peak
            - 1.0
        ).min()
    )


def performance_metrics(
    returns,
    benchmark_returns=None,
):
    r = (
        pd.Series(
            returns
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .astype(float)
    )

    if len(r) < 2:
        return {}

    nav = (
        1.0 + r
    ).cumprod()

    n = len(r)

    total_return = (
        nav.iloc[-1]
        - 1.0
    )

    cagr = (
        nav.iloc[-1]
        ** (
            252.0 / n
        )
        - 1.0
    )

    ann_vol = (
        r.std(ddof=1)
        * np.sqrt(252.0)
    )

    ann_mean = (
        r.mean()
        * 252.0
    )

    sharpe = (
        ann_mean
        / ann_vol
        if ann_vol > 0
        else np.nan
    )

    downside = (
        np.sqrt(
            np.mean(
                np.minimum(
                    r.values,
                    0.0,
                ) ** 2
            )
        )
        * np.sqrt(252.0)
    )

    sortino = (
        ann_mean
        / downside
        if downside > 0
        else np.nan
    )

    mdd = max_drawdown(
        nav
    )

    calmar = (
        cagr
        / abs(mdd)
        if (
            pd.notna(mdd)
            and mdd < 0
        )
        else np.nan
    )

    out = {
        "n_days":
            n,
        "total_return":
            total_return,
        "cagr":
            cagr,
        "ann_vol":
            ann_vol,
        "sharpe":
            sharpe,
        "sortino":
            sortino,
        "max_drawdown":
            mdd,
        "calmar":
            calmar,
        "positive_day_rate":
            float(
                (
                    r > 0
                ).mean()
            ),
    }

    if benchmark_returns is not None:

        b = pd.Series(
            benchmark_returns,
            index=r.index,
        )

        aligned = pd.concat(
            [
                r.rename(
                    "strategy"
                ),
                b.rename(
                    "benchmark"
                ),
            ],
            axis=1,
        ).dropna()

        if len(aligned) >= 30:

            s = aligned[
                "strategy"
            ]

            bm = aligned[
                "benchmark"
            ]

            var_b = bm.var(
                ddof=1
            )

            beta = (
                s.cov(bm)
                / var_b
                if var_b > 0
                else np.nan
            )

            corr = s.corr(
                bm
            )

            alpha_daily = (
                s.mean()
                - beta
                * bm.mean()
                if pd.notna(beta)
                else np.nan
            )

            excess = (
                s - bm
            )

            tracking_error = (
                excess.std(
                    ddof=1
                )
                * np.sqrt(252.0)
            )

            information_ratio = (
                excess.mean()
                * 252.0
                / tracking_error
                if tracking_error > 0
                else np.nan
            )

            out.update(
                {
                    "beta_vs_kospi":
                        beta,
                    "corr_vs_kospi":
                        corr,
                    "jensen_alpha_ann":
                        (
                            alpha_daily
                            * 252.0
                        ),
                    "tracking_error":
                        tracking_error,
                    "information_ratio":
                        information_ratio,
                }
            )

    return out


# ============================================================
# 14. MANAGEMENT SCORE PROXIES
# ============================================================

def management_metrics(
    daily,
    contribution,
):
    active = daily.loc[
        daily[
            "n_holdings"
        ] > 0
    ].copy()

    out = {
        "avg_effective_n":
            active[
                "effective_n"
            ].mean(),

        "min_effective_n":
            active[
                "effective_n"
            ].min(),

        "avg_weight_hhi":
            active[
                "weight_hhi"
            ].mean(),

        "avg_top5_weight":
            active[
                "top5_weight"
            ].mean(),

        "max_name_weight_seen":
            active[
                "max_name_weight"
            ].max(),

        "avg_sector_hhi":
            active[
                "sector_hhi"
            ].mean(),
    }

    contrib = (
        contribution
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if len(contrib) == 0:
        out.update(
            {
                "contributor_count":
                    0,
                "positive_contributor_count":
                    0,
                "positive_contributor_ratio":
                    np.nan,
                "profit_contribution_hhi":
                    np.nan,
                "top1_positive_profit_share":
                    np.nan,
                "top5_positive_profit_share":
                    np.nan,
            }
        )
        return out

    pos = contrib.loc[
        contrib > 0
    ]

    out[
        "contributor_count"
    ] = len(
        contrib
    )

    out[
        "positive_contributor_count"
    ] = len(
        pos
    )

    out[
        "positive_contributor_ratio"
    ] = (
        len(pos)
        / len(contrib)
    )

    if (
        len(pos)
        and pos.sum() > 0
    ):
        shares = (
            pos
            / pos.sum()
        ).sort_values(
            ascending=False
        )

        out[
            "profit_contribution_hhi"
        ] = float(
            np.sum(
                shares.values ** 2
            )
        )

        out[
            "top1_positive_profit_share"
        ] = float(
            shares.iloc[0]
        )

        out[
            "top5_positive_profit_share"
        ] = float(
            shares.iloc[
                :5
            ].sum()
        )

    else:
        out[
            "profit_contribution_hhi"
        ] = np.nan

        out[
            "top1_positive_profit_share"
        ] = np.nan

        out[
            "top5_positive_profit_share"
        ] = np.nan

    return out


# ============================================================
# 15. KOSPI MA60 OVERLAY
# ============================================================

def regime_overlay(
    daily,
    offset,
):
    d = (
        daily[
            [
                "date",
                "daily_return",
            ]
        ]
        .copy()
        .set_index("date")
        .sort_index()
    )

    dates = d.index

    signal_dates = set(
        dates[
            offset
            :
            -1
            :
            REBALANCE_DAYS
        ]
    )

    exposure = 1.0
    pending_exp = None
    pending_date = None

    rows = []
    nav = 1.0

    for i, date in enumerate(
        dates
    ):

        overlay_cost = 0.0

        if (
            pending_exp is not None
            and pending_date == date
        ):
            delta = (
                pending_exp
                - exposure
            )

            if delta > 0:
                overlay_cost = (
                    delta
                    * BUY_COST_RATE
                )

            elif delta < 0:
                overlay_cost = (
                    abs(delta)
                    * SELL_COST_RATE
                )

            exposure = (
                pending_exp
            )

            pending_exp = None
            pending_date = None

        base_return = float(
            d.loc[
                date,
                "daily_return"
            ]
        )

        strategy_return = (
            exposure
            * base_return
            - overlay_cost
        )

        nav *= (
            1.0
            + strategy_return
        )

        rows.append(
            {
                "date":
                    date,
                "daily_return":
                    strategy_return,
                "nav_norm":
                    nav,
                "exposure":
                    exposure,
                "overlay_cost":
                    overlay_cost,
            }
        )

        if (
            date in signal_dates
            and i + 1 < len(dates)
        ):
            close = kospi.loc[
                date,
                "kospi_close"
            ]

            ma60 = kospi.loc[
                date,
                "kospi_ma60"
            ]

            if (
                pd.notna(close)
                and pd.notna(ma60)
            ):
                pending_exp = (
                    REGIME_RISK_ON_EXPOSURE
                    if close >= ma60
                    else REGIME_RISK_OFF_EXPOSURE
                )

                pending_date = (
                    dates[
                        i + 1
                    ]
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 16. RUN ALL 5 OFFSETS
# ============================================================

performance_rows = []
daily_rows = []
trade_rows = []
contribution_rows = []
schedule_rows = []

for offset in OFFSETS:

    print(
        "\n",
        "=" * 110,
    )

    print(
        "RUN FINAL STRATEGY | offset=",
        offset,
    )

    schedule, sched_diag = (
        build_schedule(
            offset
        )
    )

    if len(sched_diag):
        schedule_rows.append(
            sched_diag
        )

    (
        daily,
        trades,
        contribution,
    ) = simulate(
        schedule,
        offset,
    )

    if len(daily) < 30:
        continue

    dates = pd.DatetimeIndex(
        daily[
            "date"
        ]
    )

    kret = (
        kospi
        .reindex(
            dates
        )[
            "kospi_ret"
        ]
        .fillna(0.0)
    )

    strategy_return = pd.Series(
        daily[
            "daily_return"
        ].values,
        index=dates,
    )

    perf = performance_metrics(
        strategy_return,
        kret,
    )

    manage = management_metrics(
        daily,
        contribution,
    )

    annual_turnover = (
        trades[
            "gross_turnover"
        ].sum()
        * 252.0
        / len(daily)
        if len(trades)
        else 0.0
    )

    total_cost = (
        trades[
            "transaction_cost"
        ].sum()
        if len(trades)
        else 0.0
    )

    failures = (
        int(
            (
                ~sched_diag[
                    "success"
                ]
            ).sum()
        )
        if (
            len(sched_diag)
            and "success"
            in sched_diag.columns
        )
        else 0
    )

    performance_rows.append(
        {
            "offset":
                offset,
            "exposure_mode":
                "BASE",
            **perf,
            **manage,
            "annual_turnover":
                annual_turnover,
            "total_transaction_cost":
                total_cost,
            "schedule_count":
                len(schedule),
            "schedule_failures":
                failures,
        }
    )

    dsave = daily.copy()
    dsave[
        "exposure_mode"
    ] = "BASE"

    daily_rows.append(
        dsave
    )

    if len(trades):
        trade_rows.append(
            trades
        )

    if len(contribution):

        csave = (
            contribution
            .rename(
                "cumulative_price_pnl"
            )
            .reset_index()
            .rename(
                columns={
                    "index":
                        "code"
                }
            )
        )

        csave[
            "offset"
        ] = offset

        contribution_rows.append(
            csave
        )

    # --------------------------------------------------------
    # KOSPI MA60 100/80
    # --------------------------------------------------------

    if RUN_KOSPI_MA60_REGIME:

        reg = regime_overlay(
            daily,
            offset,
        )

        reg_dates = pd.DatetimeIndex(
            reg[
                "date"
            ]
        )

        reg_return = pd.Series(
            reg[
                "daily_return"
            ].values,
            index=reg_dates,
        )

        reg_kret = (
            kospi
            .reindex(
                reg_dates
            )[
                "kospi_ret"
            ]
            .fillna(0.0)
        )

        reg_perf = performance_metrics(
            reg_return,
            reg_kret,
        )

        performance_rows.append(
            {
                "offset":
                    offset,
                "exposure_mode":
                    "KOSPI_MA60_2STATE",
                **reg_perf,
                **manage,
                "annual_turnover":
                    annual_turnover,
                "total_transaction_cost":
                    (
                        total_cost
                        +
                        reg[
                            "overlay_cost"
                        ].sum()
                    ),
                "avg_exposure":
                    reg[
                        "exposure"
                    ].mean(),
                "schedule_count":
                    len(schedule),
                "schedule_failures":
                    failures,
            }
        )


performance = pd.DataFrame(
    performance_rows
)

daily_all = (
    pd.concat(
        daily_rows,
        ignore_index=True,
    )
    if daily_rows
    else pd.DataFrame()
)

trades_all = (
    pd.concat(
        trade_rows,
        ignore_index=True,
    )
    if trade_rows
    else pd.DataFrame()
)

contrib_all = (
    pd.concat(
        contribution_rows,
        ignore_index=True,
    )
    if contribution_rows
    else pd.DataFrame()
)

schedule_all = (
    pd.concat(
        schedule_rows,
        ignore_index=True,
    )
    if schedule_rows
    else pd.DataFrame()
)


# ============================================================
# 17. SUMMARY ACROSS OFFSETS
# ============================================================

summary_metrics = [
    "cagr",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "beta_vs_kospi",
    "corr_vs_kospi",
    "jensen_alpha_ann",
    "tracking_error",
    "information_ratio",

    "avg_effective_n",
    "min_effective_n",
    "avg_weight_hhi",
    "avg_top5_weight",
    "max_name_weight_seen",
    "avg_sector_hhi",

    "positive_contributor_count",
    "positive_contributor_ratio",
    "profit_contribution_hhi",
    "top1_positive_profit_share",
    "top5_positive_profit_share",

    "annual_turnover",
    "total_transaction_cost",
    "avg_exposure",

    "schedule_count",
    "schedule_failures",
]

agg = {}

for col in summary_metrics:

    if col not in performance.columns:
        continue

    agg[
        f"{col}_mean"
    ] = (
        col,
        "mean",
    )

    agg[
        f"{col}_median"
    ] = (
        col,
        "median",
    )

    agg[
        f"{col}_min"
    ] = (
        col,
        "min",
    )

    agg[
        f"{col}_max"
    ] = (
        col,
        "max",
    )

summary = (
    performance
    .groupby(
        [
            "exposure_mode",
        ],
        as_index=False,
    )
    .agg(
        **agg
    )
)


# ============================================================
# 18. KOSPI SUMMARY
# ============================================================

kret_full = (
    kospi.loc[
        market_dates,
        "kospi_ret"
    ]
    .fillna(0.0)
)

kospi_perf = performance_metrics(
    kret_full
)

kospi_summary = pd.DataFrame(
    [
        {
            "benchmark":
                "KOSPI",
            **kospi_perf,
        }
    ]
)


# ============================================================
# 19. RELAXATION DIAGNOSTICS
# ============================================================

if len(schedule_all):

    success_sched = (
        schedule_all.loc[
            schedule_all[
                "success"
            ]
            == True
        ]
        .copy()
    )

    relaxation_summary = pd.DataFrame(
        [
            {
                "n_success":
                    len(success_sched),

                "entry_relax_rate":
                    success_sched[
                        "entry_relaxed"
                    ]
                    .fillna(False)
                    .mean(),

                "momentum_relax_rate":
                    success_sched[
                        "momentum_relaxed"
                    ]
                    .fillna(False)
                    .mean(),

                "avg_entry_cutoff_used":
                    success_sched[
                        "entry_cutoff_used"
                    ].mean(),

                "avg_mom_floor_used":
                    success_sched[
                        "mom_floor_used"
                    ].mean(),

                "avg_retained_count":
                    success_sched[
                        "retained_count"
                    ].mean(),

                "avg_new_count":
                    success_sched[
                        "new_count"
                    ].mean(),

                "avg_mean_fresh_rank":
                    success_sched[
                        "mean_fresh_rank"
                    ].mean(),

                "avg_mean_mom20_rank":
                    success_sched[
                        "mean_mom20_rank"
                    ].mean(),
            }
        ]
    )

else:
    relaxation_summary = pd.DataFrame()


# ============================================================
# 20. AVERAGE DAILY NAV
# ============================================================

daily_average_parts = []

if len(daily_all):

    base_pivot = (
        daily_all.pivot(
            index="date",
            columns="offset",
            values="nav_norm",
        )
        .sort_index()
    )

    avg_nav = (
        base_pivot.mean(
            axis=1,
            skipna=True,
        )
    )

    temp = pd.DataFrame(
        {
            "date":
                avg_nav.index,
            "strategy_base_nav":
                avg_nav.values,
        }
    )

    # KOSPI normalized to common start
    first_date = (
        temp[
            "date"
        ].min()
    )

    k = kospi.loc[
        kospi.index >= first_date,
        [
            "kospi_close"
        ]
    ].copy()

    if len(k):

        k[
            "kospi_nav"
        ] = (
            k[
                "kospi_close"
            ]
            / k[
                "kospi_close"
            ].iloc[0]
        )

        temp = temp.merge(
            k[
                [
                    "kospi_nav"
                ]
            ]
            .reset_index()
            .rename(
                columns={
                    "index":
                        "date"
                }
            ),
            on="date",
            how="left",
        )

    daily_average_parts.append(
        temp
    )

daily_average = (
    daily_average_parts[0]
    if daily_average_parts
    else pd.DataFrame()
)


# ============================================================
# 21. SAVE
# ============================================================

performance.to_csv(
    OUTPUT_DIR
    / "performance_by_offset.csv",
    index=False,
    encoding="utf-8-sig",
)

summary.to_csv(
    OUTPUT_DIR
    / "performance_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

kospi_summary.to_csv(
    OUTPUT_DIR
    / "kospi_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

if len(schedule_all):

    schedule_all.to_csv(
        OUTPUT_DIR
        / "selection_schedule.csv",
        index=False,
        encoding="utf-8-sig",
    )

if len(relaxation_summary):

    relaxation_summary.to_csv(
        OUTPUT_DIR
        / "relaxation_diagnostics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

if len(daily_all):

    daily_all.to_csv(
        OUTPUT_DIR
        / "daily_nav_all_offsets.csv",
        index=False,
        encoding="utf-8-sig",
    )

if len(daily_average):

    daily_average.to_csv(
        OUTPUT_DIR
        / "daily_nav_average.csv",
        index=False,
        encoding="utf-8-sig",
    )

if len(contrib_all):

    contrib_all.to_csv(
        OUTPUT_DIR
        / "profit_contribution_by_stock.csv",
        index=False,
        encoding="utf-8-sig",
    )

if len(trades_all):

    trades_all.to_csv(
        OUTPUT_DIR
        / "trade_log.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 22. PLOT: FINAL STRATEGY VS KOSPI
# ============================================================

plt.figure(
    figsize=(
        15,
        8,
    )
)

if len(daily_all):

    pivot = (
        daily_all.pivot(
            index="date",
            columns="offset",
            values="nav_norm",
        )
        .sort_index()
    )

    avg_nav = (
        pivot.mean(
            axis=1,
            skipna=True,
        )
    )

    plt.plot(
        avg_nav.index,
        avg_nav.values,
        label=
            "Final Strategy - BASE",
        linewidth=2.2,
    )

    # Average MA60 overlay NAV reconstructed offset-by-offset
    reg_nav_series = []

    for offset in OFFSETS:

        d = daily_all.loc[
            daily_all[
                "offset"
            ]
            == offset
        ].copy()

        if len(d) == 0:
            continue

        reg = regime_overlay(
            d,
            offset,
        )

        s = (
            reg.set_index(
                "date"
            )[
                "nav_norm"
            ]
        )

        reg_nav_series.append(
            s.rename(
                offset
            )
        )

    if reg_nav_series:

        reg_pivot = pd.concat(
            reg_nav_series,
            axis=1,
        )

        reg_avg = reg_pivot.mean(
            axis=1,
            skipna=True,
        )

        plt.plot(
            reg_avg.index,
            reg_avg.values,
            label=
                "Final Strategy - KOSPI MA60 100/80",
            linewidth=2.0,
        )

# KOSPI NAV
if len(market_dates):

    kret = (
        kospi.loc[
            market_dates,
            "kospi_ret"
        ]
        .fillna(0.0)
    )

    knav = (
        1.0 + kret
    ).cumprod()

    if len(knav):

        knav = (
            knav
            / knav.iloc[0]
        )

        plt.plot(
            knav.index,
            knav.values,
            label=
                "KOSPI",
            linewidth=2.5,
        )

plt.axhline(
    1.0,
    linewidth=0.8,
)

plt.title(
    "Final Strategy vs KOSPI"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Cumulative NAV"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "final_strategy_vs_kospi.png",
    dpi=180,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 23. CONSOLE SUMMARY
# ============================================================

pd.set_option(
    "display.max_columns",
    160,
)

pd.set_option(
    "display.width",
    320,
)

show_cols = [
    "exposure_mode",
    "cagr_mean",
    "ann_vol_mean",
    "sharpe_mean",
    "sortino_mean",
    "max_drawdown_mean",
    "calmar_mean",
    "beta_vs_kospi_mean",
    "corr_vs_kospi_mean",
    "jensen_alpha_ann_mean",
    "tracking_error_mean",
    "information_ratio_mean",
    "avg_effective_n_mean",
    "avg_top5_weight_mean",
    "positive_contributor_ratio_mean",
    "profit_contribution_hhi_mean",
    "top5_positive_profit_share_mean",
    "annual_turnover_mean",
    "schedule_failures_mean",
]

show_cols = [
    c
    for c in show_cols
    if c in summary.columns
]

print("\n")
print("=" * 180)
print("FINAL STRATEGY PERFORMANCE")
print("=" * 180)

print(
    summary[
        show_cols
    ].to_string(
        index=False
    )
)

print("\n")
print("=" * 180)
print("KOSPI BENCHMARK")
print("=" * 180)

print(
    kospi_summary.to_string(
        index=False
    )
)

if len(relaxation_summary):

    print("\n")
    print("=" * 180)
    print("RELAXATION DIAGNOSTICS")
    print("=" * 180)

    print(
        relaxation_summary.to_string(
            index=False
        )
    )

print("\nSaved:")
print(OUTPUT_DIR)
