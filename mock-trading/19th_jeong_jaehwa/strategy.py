"""Sector Relative-Strength Momentum: signal generation, daily backtest engine, and performance metrics."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import Params, SAMSUNG


# ====================================================================== data container
class MarketData:
    """Aligns all inputs on the trading-day calendar and precomputes signal inputs."""

    def __init__(self, kospi, sec_px, px, cap, val, membership, value_window=4):
        self.days = pd.DatetimeIndex(kospi.index)
        self.kospi = kospi.astype(float)
        self.sec_px = sec_px.reindex(self.days).ffill()
        # Stock prices: fill short gaps (trading halts) but never beyond the last traded day (delisting).
        px = px.reindex(self.days)
        last = px.apply(pd.Series.last_valid_index)
        filled = px.ffill(limit=10)
        for t in px.columns:
            if pd.notna(last[t]):
                filled.loc[filled.index > last[t], t] = np.nan
        self.px = filled
        self.tickers = list(px.columns)
        self.tix = {t: j for j, t in enumerate(self.tickers)}
        self.P = self.px.values
        self.last_pos = np.array([self.days.get_loc(last[t]) if pd.notna(last[t]) else -1 for t in self.tickers])
        self.hi52 = self.px.rolling(252, min_periods=126).max().values

        self.K = self.kospi.values
        self.sectors = list(self.sec_px.columns)
        self.S = self.sec_px.values
        self.S_ma = {}

        # Weekly eligibility (cap and 4-week average trading value), carried to each day by as-of lookup.
        self.cap = cap.sort_index()
        self.avg_val = val.sort_index().rolling(value_window, min_periods=1).mean()
        self.membership = dict(sorted(membership.items()))
        self._m_dates = list(self.membership.keys())

    def sector_ma(self, window):
        if window not in self.S_ma:
            self.S_ma[window] = self.sec_px.rolling(window).mean().values
        return self.S_ma[window]

    def sector_map(self, date):
        """Ticker -> sector from the latest membership snapshot on or before date."""
        prior = [d for d in self._m_dates if d <= date]
        d = prior[-1] if prior else self._m_dates[0]
        df = self.membership[d]
        return dict(zip(df["ticker"], df["sector"]))

    def eligible(self, date, p: Params):
        ci = self.cap.index.searchsorted(date, side="right") - 1
        if ci < 0:
            return set()
        cap = self.cap.iloc[ci]
        val = self.avg_val.iloc[ci]
        ok = (cap >= p.min_cap) & (val >= p.min_value)
        return set(ok.index[ok.fillna(False)])


# ====================================================================== signal
@dataclass
class Signal:
    date: pd.Timestamp
    sectors: list          # ordered, strongest first
    picks: dict            # sector -> list of tickers
    backups: dict          # sector -> ordered list of next-ranked tickers
    weights: dict          # ticker -> target weight
    sector_scores: pd.Series


def _cap_weights(w, cap, exempt=()):
    w = dict(w)
    for _ in range(20):
        over = {t: v for t, v in w.items() if v > cap + 1e-12 and t not in exempt}
        if not over:
            break
        excess = sum(v - cap for v in over.values())
        for t in over:
            w[t] = cap
        free = [t for t in w if w[t] < cap - 1e-12 and t not in exempt]
        tot = sum(w[t] for t in free)
        if not free or tot <= 0:
            break  # leftover stays in cash
        for t in free:
            w[t] += excess * w[t] / tot
    return w


def make_signal(md: MarketData, i: int, p: Params, held_sectors=(), held_stocks=()):
    date = md.days[i]
    lags = (21, 63, 126)
    need = max(lags[-1], p.ma_window, p.skip_days + 126) + 1
    if i < need:
        return None

    # ---- step 1: sector score and trend filter
    S, K = md.S, md.K
    score = np.zeros(len(md.sectors))
    for w, k in zip(p.lookback_w, lags):
        score += w * ((S[i] / S[i - k] - 1) - (K[i] / K[i - k] - 1))
    ma = md.sector_ma(p.ma_window)[i]
    trend_ok = S[i] > ma
    sec_scores = pd.Series(score, index=md.sectors)
    valid = sec_scores[np.isfinite(score) & trend_ok].sort_values(ascending=False, kind="mergesort")

    # ---- step 2 inputs: stock momentum, 52-week-high proximity
    smap = md.sector_map(date)
    elig = md.eligible(date, p)
    P, sk = md.P, p.skip_days
    rows = []
    for t in sorted(elig):
        j = md.tix.get(t)
        s = smap.get(t)
        if j is None or s is None or s not in valid.index:
            continue
        p0, p3, p6, hi = P[i, j], P[i - sk - 63, j], P[i - sk - 126, j], md.hi52[i, j]
        pk = P[i - sk, j]
        if not np.all(np.isfinite([p0, p3, p6, hi, pk])):
            continue
        rows.append((t, s, pk / p3 - 1, pk / p6 - 1, p0 / hi))
    if not rows:
        return Signal(date, [], {}, {}, {}, sec_scores)
    st = pd.DataFrame(rows, columns=["ticker", "sector", "r3", "r6", "hi"])
    st["score"] = st.groupby("sector")[["r3", "r6", "hi"]].rank(pct=True).sum(axis=1)

    counts = st["sector"].value_counts()
    ranked = [s for s in valid.index if counts.get(s, 0) >= 2]

    # ---- sector selection with buffer
    keep = [s for s in held_sectors if s in ranked[: p.n_sectors + p.buffer]]
    chosen = list(keep)
    for s in ranked:
        if len(chosen) >= p.n_sectors:
            break
        if s not in chosen:
            chosen.append(s)
    chosen = sorted(chosen, key=lambda s: -sec_scores[s])

    picks, backups, weights = {}, {}, {}
    sw = p.sector_weights()
    for rank, s in enumerate(chosen):
        g = st[st["sector"] == s].sort_values(["score", "ticker"], ascending=[False, True])["ticker"].tolist()
        # stock buffer: a held stock stays while it ranks within n_stocks + stock_buffer in its sector
        keep_t = [t for t in g[: p.n_stocks + p.stock_buffer] if t in held_stocks][: p.n_stocks]
        chosen_t = keep_t + [t for t in g if t not in keep_t][: p.n_stocks - len(keep_t)]
        chosen_t = sorted(chosen_t, key=g.index)
        picks[s], backups[s] = chosen_t, [t for t in g if t not in chosen_t]
        for t in picks[s]:
            weights[t] = sw[rank] / len(picks[s])

    if p.samsung_weight and SAMSUNG in weights:
        others = {t: v for t, v in weights.items() if t != SAMSUNG}
        tot = sum(others.values())
        weights = {t: v * (1 - p.samsung_weight) / tot for t, v in others.items()} if tot > 0 else {}
        weights[SAMSUNG] = p.samsung_weight
    weights = _cap_weights(weights, p.max_weight, exempt=(SAMSUNG,))
    return Signal(date, chosen, picks, backups, weights, sec_scores)


# ====================================================================== backtest
@dataclass
class Result:
    params: Params
    nav: pd.Series
    turnover: pd.Series        # daily traded notional / NAV (buys + sells)
    n_holdings: pd.Series
    trades: pd.DataFrame
    signals: list


def run_backtest(md: MarketData, p: Params, start, end=None, keep_signals=False, verbose=False):
    days = md.days
    i0 = days.searchsorted(pd.Timestamp(start))
    i1 = len(days) - 1 if end is None else days.searchsorted(pd.Timestamp(end), side="right") - 1
    week_end = set(pd.DatetimeIndex(pd.Series(days, index=days).groupby(days.to_period("W-SUN")).max().values))
    P = md.P

    cash, pos, peak, entry = 1.0, {}, {}, {}
    pending_target, pending_repl = None, []
    signal, held_sectors, stopped = None, [], set()
    week_no = 0
    nav_rec, to_rec, nh_rec, trade_log, sig_log = [], [], [], [], []

    def price(t, i):
        return P[i, md.tix[t]]

    def rebalance(i, d, target):
        """Trade to target weights at day i's close. Returns traded notional."""
        nonlocal cash
        traded = 0.0
        nav = cash + sum(pos.values())
        tgt = {t: w for t, w in target.items() if np.isfinite(price(t, i))}
        # rebalance band: a kept holding is only resized when it drifts more than p.band from target
        for t in list(pos):
            if t in tgt and abs(pos[t] / nav - tgt[t]) <= p.band:
                tgt[t] = pos[t] / nav
        for t in list(pos):                                   # sells first
            cur = pos[t]
            want = tgt.get(t, 0.0) * nav
            if want < cur - 1e-12 and np.isfinite(price(t, i)):
                q = cur - want
                pos[t] = want; cash += q * (1 - p.sell_cost); traded += q
                trade_log.append((d, t, "sell", -q))
                if pos[t] <= 1e-12:
                    pos.pop(t); peak.pop(t, None)
        for t, w in tgt.items():                              # then buys
            want = w * nav
            cur = pos.get(t, 0.0)
            if want > cur + 1e-12:
                q = min(want - cur, cash / (1 + p.buy_cost))
                if q <= 0:
                    continue
                cash -= q * (1 + p.buy_cost); traded += q
                if t not in pos:
                    peak[t] = price(t, i); entry[t] = i
                pos[t] = cur + q
                trade_log.append((d, t, "buy", q))

        return traded


    for i in range(i0, i1 + 1):
        d = days[i]
        traded = 0.0

        # 1) mark to market; liquidate delisted names
        for t in list(pos):
            j = md.tix[t]
            a, b = P[i - 1, j], P[i, j]
            if np.isfinite(a) and np.isfinite(b):
                pos[t] *= b / a
            if i > md.last_pos[j]:
                v = pos.pop(t); cash += v * (1 - p.sell_cost); traded += v
                peak.pop(t, None); trade_log.append((d, t, "delist", -v))

        # 2) rebalance decided at the previous signal (exec_lag = 1)
        if pending_target is not None:
            traded += rebalance(i, d, pending_target)
            pending_target, stopped = None, set()

        # 3) replacements for yesterday's stop-outs
        for sector, amount in pending_repl:
            cands = signal.backups.get(sector, []) if signal else []
            for t in cands:
                if t in pos or t in stopped or not np.isfinite(price(t, i)):
                    continue
                q = min(amount, cash / (1 + p.buy_cost))
                if q > 0:
                    cash -= q * (1 + p.buy_cost); traded += q
                    pos[t] = q; peak[t] = price(t, i); entry[t] = i
                    trade_log.append((d, t, "replace", q))
                break
        pending_repl = []

        # 4) trailing stops on the close
        if p.stop is not None:
            for t in list(pos):
                px_ = price(t, i)
                if not np.isfinite(px_):
                    continue
                if entry.get(t) != i and px_ < peak[t] * (1 - p.stop):
                    v = pos.pop(t); cash += v * (1 - p.sell_cost); traded += v
                    peak.pop(t, None); stopped.add(t)
                    trade_log.append((d, t, "stop", -v))
                    sector = next((s for s, ts in (signal.picks.items() if signal else []) if t in ts), None)
                    if sector is not None:
                        pending_repl.append((sector, v * (1 - p.sell_cost)))
                else:
                    peak[t] = max(peak[t], px_)
        else:
            for t in pos:
                if np.isfinite(price(t, i)):
                    peak[t] = max(peak[t], price(t, i))

        nav = cash + sum(pos.values())
        nav_rec.append(nav); to_rec.append(traded / nav if nav > 0 else 0); nh_rec.append(len(pos))

        # 5) new signal at the week's last trading day (every p.rebalance_weeks weeks);
        #    executed at the same close (exec_lag = 0) or the next trading day's close (exec_lag = 1)
        is_signal_day = (d in week_end) if p.rebalance_weeks > 0 else True   # rebalance_weeks = 0 -> daily
        if is_signal_day and i < i1:
            week_no += 1
        if is_signal_day and i < i1 and (p.rebalance_weeks == 0 or (week_no - 1) % p.rebalance_weeks == 0):
            sig = make_signal(md, i, p, held_sectors, set(pos))
            if sig is not None:
                signal = sig
                held_sectors = sig.sectors
                if keep_signals:
                    sig_log.append(sig)
                if p.exec_lag == 0:
                    traded += rebalance(i, d, sig.weights)
                    stopped = set()
                    nav = cash + sum(pos.values())
                    nav_rec[-1] = nav; to_rec[-1] = traded / nav if nav > 0 else 0; nh_rec[-1] = len(pos)
                else:
                    pending_target = sig.weights
        if verbose and i % 250 == 0:
            print(f"   {d:%Y-%m-%d} NAV {nav:.3f}")

    idx = days[i0:i1 + 1]
    trades = pd.DataFrame(trade_log, columns=["date", "ticker", "action", "notional"])
    return Result(p, pd.Series(nav_rec, idx), pd.Series(to_rec, idx), pd.Series(nh_rec, idx), trades, sig_log)


# ====================================================================== metrics
def perf_stats(nav, bench=None, turnover=None):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    out = {
        "CAGR": nav.iloc[-1] ** (1 / yrs) - 1,
        "Volatility": r.std() * np.sqrt(252),
        "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
        "MaxDrawdown": (nav / nav.cummax() - 1).min(),
        "TotalReturn": nav.iloc[-1] / nav.iloc[0] - 1,
    }
    wk = nav.resample("W-FRI").last().pct_change(fill_method=None).dropna()
    out["WeeklyHitRate"] = (wk > 0).mean()
    if bench is not None:
        b = bench.reindex(nav.index).ffill()
        br = b.pct_change().dropna()
        out["Bench_CAGR"] = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
        ex = r - br.reindex(r.index)
        out["ExcessCAGR"] = out["CAGR"] - out["Bench_CAGR"]
        out["InfoRatio"] = ex.mean() / ex.std() * np.sqrt(252)
    if turnover is not None:
        # contest formula: (buys + sells) / avg NAV * 0.5, summed per week
        wto = turnover.resample("W-FRI").sum() * 0.5
        wto = wto[wto.index >= nav.index[0] + pd.Timedelta(days=7)]
        out["AvgWeeklyTurnover"] = wto.mean()
        out["WeeksTurnover>=5%"] = (wto >= 0.05).mean()
    return out


def oct_nov_returns(nav, bench):
    rows = []
    for y in sorted(set(nav.index.year)):
        a = nav[(nav.index <= f"{y}-09-30")]
        b = nav[(nav.index <= f"{y}-11-30")]
        if a.empty or b.empty or a.index[-1].year != y or b.index[-1].month < 11:
            continue
        ka, kb = bench.asof(a.index[-1]), bench.asof(b.index[-1])
        rows.append({"Year": y, "Strategy": b.iloc[-1] / a.iloc[-1] - 1, "KOSPI": kb / ka - 1})
    df = pd.DataFrame(rows).set_index("Year")
    df["Excess"] = df["Strategy"] - df["KOSPI"]
    return df


def deflated_sharpe(returns_by_trial: pd.DataFrame, chosen: str):
    """Bailey & Lopez de Prado (2014). Inputs are daily returns (non-annualised SR)."""
    sr = returns_by_trial.mean() / returns_by_trial.std()
    n = returns_by_trial.shape[1]
    gamma = 0.5772156649
    sr0 = np.sqrt(sr.var(ddof=1)) * ((1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * np.e)))
    r = returns_by_trial[chosen].dropna()
    T = len(r)
    skew = r.skew()
    kurt = r.kurt() + 3   # non-excess
    s = sr[chosen]
    z = (s - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * s + (kurt - 1) / 4 * s ** 2)
    return {"SR_daily": s, "SR_annual": s * np.sqrt(252), "SR0_daily": sr0, "SR0_annual": sr0 * np.sqrt(252),
            "N_trials": n, "T": T, "skew": skew, "kurtosis": kurt, "DSR": norm.cdf(z)}
