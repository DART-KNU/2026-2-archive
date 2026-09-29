"""Step 0: quick connectivity check for pykrx (takes ~10 seconds)."""
import sys

print("Python", sys.version.split()[0])
try:
    from pykrx import stock
    import pykrx
    print("pykrx", getattr(pykrx, "__version__", "?"))
except ImportError:
    sys.exit("pykrx is not installed. Run:  pip install -r requirements.txt")

ok = True
tests = [
    ("KOSPI index", lambda: stock.get_index_ohlcv("20260901", "20260925", "1001")),
    ("Market cap snapshot", lambda: stock.get_market_cap("20260925", market="ALL")),
    ("Adjusted prices (Samsung)", lambda: stock.get_market_ohlcv("20260901", "20260925", "005930", adjusted=True)),
    ("Sector index list", lambda: stock.get_index_ticker_list("20260925", market="KOSPI")),
]
for name, fn in tests:
    try:
        out = fn()
        n = len(out)
        print(f"[{'OK' if n else 'EMPTY'}] {name}: {n} rows")
        ok &= n > 0
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {e}")
        ok = False

if ok:
    codes = stock.get_index_ticker_list("20260925", market="KOSPI")
    print("\nKOSPI indices:", ", ".join(f"{c} {stock.get_index_ticker_name(c)}" for c in codes[:40]))
    print("\nAll good. Next:  python 01_download_data.py")
else:
    print("\nSomething failed. Copy this whole output and send it to Claude.")
