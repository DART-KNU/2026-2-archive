"""Global settings for the Sector Relative-Strength Momentum backtest."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Data window: one extra year before the backtest start for 6M momentum / 52W high.
DATA_START = "20140101"
DATA_END = None            # None = today
BT_START = "2015-02-02"

KOSPI_CODE = "1001"
SAMSUNG = "005930"

# Industry classification snapshots are taken on the first trading day of these months.
MEMBERSHIP_MONTHS = (1, 7)



@dataclass
class Params:
    lookback_w: Tuple[float, float, float] = (0.2, 0.4, 0.4)   # weights on RS(1M), RS(3M), RS(6M)
    n_sectors: int = 3
    n_stocks: int = 3                                           # per sector
    buffer: int = 1                                             # held sector kept while rank <= n_sectors + buffer
    stock_buffer: int = 2                                       # held stock kept while rank <= n_stocks + stock_buffer
    band: float = 0.03                                          # resize a kept holding only if |w - target| > band
    rebalance_weeks: int = 1                                    # signal every N weeks (0 = every day)
    exec_lag: int = 0                                           # 0 = trade at the signal day's close, 1 = next day's close
    ma_window: int = 60                                         # sector trend filter
    stop: Optional[float] = None                                # trailing stop from peak close; None = no stop
    min_cap: float = 1e12                                       # KRW 1tn
    min_value: float = 5e9                                      # KRW 5bn average daily trading value
    skip_days: int = 5                                          # skip most recent week in momentum
    max_weight: float = 0.15                                    # contest single-stock cap
    samsung_weight: Optional[float] = 0.25                      # weight for Samsung Electronics if selected
    buy_cost: float = 0.0010 + 0.0010                           # commission + slippage
    sell_cost: float = 0.0010 + 0.0020 + 0.0010                 # commission + sell tax + slippage
    sector_w: Optional[Tuple[float, ...]] = None                # None -> default by n_sectors

    def sector_weights(self):
        if self.sector_w is not None:
            return self.sector_w
        return {1: (1.0,), 2: (0.55, 0.45), 3: (0.40, 0.35, 0.25), 4: (0.30, 0.27, 0.23, 0.20)}[self.n_sectors]

    def label(self):
        lw = "/".join(f"{w:.2f}" for w in self.lookback_w)
        st = "none" if self.stop is None else f"{int(self.stop * 100)}%"
        return f"LB[{lw}]_S{self.n_sectors}_stop{st}"


BASE = Params()

# The version presented on 2026-09-29, kept for comparison: 12% trailing stop, trades at the next day's close.
ORIGINAL = Params(stop=0.12, exec_lag=1)

# Parameter grid used for the overfitting check (3 x 3 x 4 = 36 runs).
GRID_LOOKBACK = [(0.2, 0.4, 0.4), (1 / 3, 1 / 3, 1 / 3), (0.0, 0.5, 0.5)]
GRID_SECTORS = [2, 3, 4]
GRID_STOPS = [0.08, 0.12, 0.15, None]
