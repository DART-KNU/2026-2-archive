"""Step 1: download and cache all data.

    py 01_download_data.py            first run; re-runs resume where they stopped
    py 01_download_data.py --update   refresh recent prices before generating a live signal
    py 01_download_data.py --static   industry classification without KRX login (FinanceDataReader)

KRX login (KRX_ID / KRX_PW) is needed for the KOSPI index, weekly snapshots and industry classification.
Adjusted stock prices do not need it. If KRX is unavailable, cached files are used and the KRX steps
are retried on the next run.
"""
import sys
import time

import data
from config import DATA_DIR

update = "--update" in sys.argv
t0 = time.time()
DATA_DIR.mkdir(exist_ok=True)

try:
    import pykrx  # noqa: F401  (logs in to KRX on import when KRX_ID / KRX_PW are set)
except Exception as e:  # noqa: BLE001
    sys.exit(f"pykrx could not start ({e}).\nKRX login failed: clear the login variables and run again:\n"
             "   $env:KRX_ID=$null; $env:KRX_PW=$null")

kospi = data.load_kospi(refresh=True)
days = kospi.index
print(f"[calendar] {len(days)} trading days: {days[0]:%Y-%m-%d} ~ {days[-1]:%Y-%m-%d}")

wdates = data.weekly_dates(days)
data.download_snapshots(wdates)
cap, val = data.load_snapshots(wdates)

# Prices first (no KRX login needed), for every stock that ever passes the cap / trading-value filter.
tickers = data.candidate_tickers(cap, val)
data.download_prices(tickers, update=update)

mdates = data.membership_dates(days)
if "--static" in sys.argv:
    data.download_static_classification()
else:
    data.download_membership(mdates)
try:
    snaps = data.load_membership()
    missing = [] if len(snaps) == 1 and min(snaps).year < 2000 else [d for d in mdates if d not in snaps]
    latest = snaps[max(snaps)]
    print(f"[sectors] {latest['sector'].nunique()} sectors in the latest classification:")
    print("   " + ", ".join(sorted(latest["sector"].unique())))
    if missing:
        print(f"   ! {len(missing)} classification snapshots missing - run this script again later (KRX login)")
    else:
        print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Next:  py 02_run_backtest.py")
except RuntimeError:
    print("\n! Industry classification not downloaded yet (needs KRX login). Run this script again later.")
