"""Step 3 (contest use): target portfolio from the latest close.
Run  python 01_download_data.py --update  first, then  python 03_live_signal.py"""
import pandas as pd

import data
from config import BASE, RESULTS_DIR
from strategy import make_signal

_02 = __import__("02_run_backtest")

md, kospi = _02.load_market_data()
i = len(md.days) - 1
held = []   # put the sectors you currently hold here to apply the buffer rule, e.g. ["KOSPI:전기전자"]
sig = make_signal(md, i, BASE, held)
names = data.ticker_names(list(sig.weights))

print(f"Signal date: {sig.date:%Y-%m-%d}\n")
print("Top sector scores:")
print(sig.sector_scores.sort_values(ascending=False).head(8).round(4).to_string(), "\n")
rows = []
for s in sig.sectors:
    for t in sig.picks[s]:
        rows.append({"sector": s, "ticker": t, "name": names.get(t, ""), "weight": round(sig.weights[t], 4)})
    rows.append({"sector": s, "ticker": "(backups)", "name": ", ".join(sig.backups[s][:3]), "weight": None})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
RESULTS_DIR.mkdir(exist_ok=True)
df.to_csv(RESULTS_DIR / f"live_signal_{sig.date:%Y%m%d}.csv", index=False, encoding="utf-8-sig")
