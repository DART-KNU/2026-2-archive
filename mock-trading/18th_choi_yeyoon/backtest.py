"""
일별 시뮬레이션 엔진
  - 매수/매도는 다음 날 시가 체결 (신호는 종가에 확정)
  - 비용: 매수 10bp, 매도 30bp
  - 대회 규정: 종목·섹터·소형주 한도, 주간 회전율 5% 점검
"""
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C
from signals import eligible_mask, stage1_candidates


def simulate(P, ev, k200, start, end, contest=False, stage1=False, verbose=False):
    dates = P["dates"]
    i0 = int(dates.searchsorted(pd.Timestamp(start)))
    i1 = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    cols = P["cols"]
    col_ix = {t: j for j, t in enumerate(cols)}
    OPEN, CLOSE, RET = P["open"].values, P["close"].values, P["ret"].values
    MCAP, VOL = P["mcap"].values, P["vol60"].values
    SEC = P["sector"].values
    kv = k200.reindex(dates).ffill()
    k_ma20 = kv.rolling(20).mean()
    k_vol = kv.pct_change().rolling(20).std()
    k_vol_hi = k_vol.rolling(252, min_periods=120).quantile(0.8)

    # 이 기간에 매수 가능한 신호: t1(신호 확정일)이 [i0-1, i1-1]
    evw = ev[(ev.t1_idx >= i0 - 1) & (ev.t1_idx <= i1 - 1)]
    by_t1 = defaultdict(list)
    for e in evw.itertuples(index=False):
        by_t1[int(e.t1_idx)].append(e)

    cash = C.INIT_CAPITAL
    pos = {}                        # ticker → dict(val, cum, until, kind, tw)
    buys, sells, trims = [], set(), {}
    nav_hist, gross_hist, hhi_hist, npos_hist = [], [], [], []
    pnl = defaultdict(float)
    week_trade, week_navs = defaultdict(float), defaultdict(list)
    trades = []
    peak, dd_mode = C.INIT_CAPITAL, False

    def sector_w(nav):
        s = defaultdict(float)
        for t, p in pos.items():
            s[SEC[col_ix[t]]] += p["val"] / nav
        return s

    def small_w(nav, i):
        return sum(p["val"] for t, p in pos.items() if MCAP[i, col_ix[t]] < C.SMALLCAP_LINE) / nav

    def size_new(t, i, nav, room, secw, smw, med_vol, base=C.BASE_W, cap_default=C.MAX_W):
        j = col_ix[t]
        v = VOL[i, j]
        w = base * (med_vol / v) if (np.isfinite(v) and v > 0) else base
        w = float(np.clip(w, C.MIN_W, C.NAME_CAPS.get(t, cap_default)))
        w = min(w, C.SECTOR_CAP - secw[SEC[j]], room)
        if MCAP[i, j] < C.SMALLCAP_LINE:
            w = min(w, C.SMALLCAP_CAP - smw)
        return w

    # Stage 1: 시작일 시가에 선취매
    if stage1:
        nav0 = cash
        secw, smw, room = defaultdict(float), 0.0, C.S1_GROSS
        elig = eligible_mask(P, i0 - 1)
        med_vol = np.nanmedian(np.where(elig, VOL[i0 - 1], np.nan))
        for t, sc in stage1_candidates(P, ev, i0, dates[i0].year)[:C.S1_MAX_POS]:
            w = size_new(t, i0 - 1, nav0, room, secw, smw, med_vol, base=C.S1_W, cap_default=C.S1_W)
            if w < C.MIN_W * 0.5:
                continue
            buys.append((t, w, "S1", sc, i1 + 1))
            room -= w
            secw[SEC[col_ix[t]]] += w
            if MCAP[i0 - 1, col_ix[t]] < C.SMALLCAP_LINE:
                smw += w

    for i in range(i0, i1 + 1):
        d = dates[i]
        wk = tuple(d.isocalendar()[:2])

        # ── 시가: 매도 / 비중축소 ─────────────────────────
        for t in list(sells) + list(trims):
            if t not in pos:
                sells.discard(t); trims.pop(t, None)
                continue
            j = col_ix[t]
            o, c, r = OPEN[i, j], CLOSE[i, j], RET[i, j]
            if not (o > 0 and c > 0 and np.isfinite(r)):
                continue                                     # 거래정지 → 다음 날 재시도
            ratio = o * (1 + r) / c                          # 전일 종가 → 오늘 시가 (보정 수익률)
            frac = 1.0 if t in sells else trims[t]
            v_old = pos[t]["val"]
            v = v_old * ratio * frac
            cost = v * C.SELL_COST
            pnl[t] += (v_old * ratio - v_old) * frac - cost
            cash += v - cost
            week_trade[wk] += v
            trades.append((d, t, "SELL" if frac == 1.0 else "TRIM", v))
            if frac == 1.0:
                del pos[t]
            else:
                pos[t]["val"] = v_old * (1 - frac)          # 남은 부분은 아래 종가 평가에서 오늘 수익률 반영
            sells.discard(t); trims.pop(t, None)

        # ── 보유 종목 평가 (전일 종가 → 오늘 종가) ─────────
        for t, p in list(pos.items()):
            r = RET[i, col_ix[t]]
            if not np.isfinite(r):
                p["miss"] += 1
                r = 0.0
                if p["miss"] >= 5:                           # 5일 연속 데이터 없음 → 상폐 가정, 마지막 가치로 정리
                    cash += p["val"]
                    trades.append((d, t, "DELIST", p["val"]))
                    del pos[t]
                    continue
            else:
                p["miss"] = 0
            nv = p["val"] * (1 + r)
            pnl[t] += nv - p["val"]
            p["val"] = nv
            p["cum"] = (1 + p["cum"]) * (1 + r) - 1

        # ── 시가: 매수 ─────────────────────────────────
        nav_est = cash + sum(p["val"] for p in pos.values())
        for t, w, kind, sc, until in sorted(buys, key=lambda x: -x[3]):
            j = col_ix[t]
            o, c = OPEN[i, j], CLOSE[i, j]
            if not (o > 0 and c > 0):
                continue
            amt = min(w * nav_est, cash / (1 + C.BUY_COST))
            if amt < 0.002 * nav_est:
                continue
            cost = amt * C.BUY_COST
            cash -= amt + cost
            v = amt * c / o                                  # 시가 매수 → 오늘 종가 평가
            pnl[t] += v - amt - cost
            week_trade[wk] += amt
            trades.append((d, t, "BUY-" + kind, amt))
            if t in pos:                                     # 리밸런싱 추가매수
                pos[t]["val"] += v
            else:
                pos[t] = dict(val=v, cum=c / o - 1, until=until, kind=kind, tw=w, miss=0)
        buys = []

        # ── 종가: 기록 ─────────────────────────────────
        nav = cash + sum(p["val"] for p in pos.values())
        gross = (nav - cash) / nav
        nav_hist.append(nav); gross_hist.append(gross); npos_hist.append(len(pos))
        hhi_hist.append(sum((p["val"] / nav) ** 2 for p in pos.values()))
        week_navs[wk].append(nav)
        peak = max(peak, nav)
        if i == i1:
            break

        # ── 종가: 청산 조건 (손절 / 보유기간 만료) ─────────
        for t, p in pos.items():
            if p["cum"] <= C.STOP_LOSS or i + 1 >= p["until"]:
                sells.add(t)

        # ── 리스크 오버레이 → 총 노출 상한 ──────────────────
        in_s1 = contest and (d.month, d.day) < C.S1_END
        cap = C.S1_GROSS if in_s1 else C.S2_GROSS
        if len(nav_hist) >= 10:
            rr = pd.Series(nav_hist[-21:]).pct_change().dropna()
            g_avg = max(np.mean(gross_hist[-20:]), 0.2)
            vol_full = rr.std() * np.sqrt(252) / g_avg       # 100% 투자 시 변동성으로 환산
            if vol_full > 0:
                cap = min(cap, C.VOL_TARGET / vol_full)
        if (kv.iloc[i] < k_ma20.iloc[i]) and (k_vol.iloc[i] > k_vol_hi.iloc[i]):
            cap = min(cap, C.REGIME_EXPOSURE)
        dd = nav / peak - 1
        if dd <= C.DD_TRIGGER:
            dd_mode = True
        elif dd >= C.DD_RECOVER:
            dd_mode = False
        if dd_mode:
            cap = min(cap, C.DD_EXPOSURE)

        live = {t: p for t, p in pos.items() if t not in sells}
        live_gross = sum(p["val"] for p in live.values()) / nav
        if live_gross > cap + C.TRIM_BAND:                   # 비례 축소
            f = 1 - cap / live_gross
            for t in live:
                trims[t] = f

        # ── 종가: 신규 신호 (t+1 확정 → 내일 시가 매수) ─────
        allow_new = not (contest and (d.month, d.day) >= C.NO_ENTRY_AFTER)
        elig = eligible_mask(P, i)
        cands = []
        for e in by_t1.get(i, []):
            t = e.ticker
            j = col_ix.get(t)
            if j is None:
                continue
            ok = bool(e.passed) and elig[j]
            if t in pos and pos[t]["kind"] == "S1":           # Stage1 종목의 3Q 발표 → 재심사
                if ok:
                    pos[t].update(kind="PEAD", until=i + 1 + C.HOLD_DAYS, cum=0.0)
                else:
                    sells.add(t)
                continue
            if ok and allow_new and t not in pos:
                cands.append(e)

        if cands:
            med_vol = np.nanmedian(np.where(elig, VOL[i], np.nan))
            secw, smw = sector_w(nav), small_w(nav, i)
            room = cap - live_gross
            n_after = len(live)
            for e in sorted(cands, key=lambda x: -x.score):
                if n_after >= C.MAX_POS or room < C.MIN_W * 0.5:
                    break
                w = size_new(e.ticker, i, nav, room, secw, smw, med_vol)
                if w < C.MIN_W * 0.5:
                    continue
                buys.append((e.ticker, w, "PEAD", e.score, i + 1 + C.HOLD_DAYS))
                room -= w; n_after += 1
                secw[SEC[col_ix[e.ticker]]] += w
                if MCAP[i, col_ix[e.ticker]] < C.SMALLCAP_LINE:
                    smw += w

        # ── 대회 모드: 목요일 회전율 점검 → 금요일 리밸런싱 ─
        if contest and d.weekday() == 3:
            avg_nav = np.mean(week_navs[wk])
            need = 2 * C.TURNOVER_TOPUP * avg_nav - week_trade[wk]      # 금요일에 더 거래해야 할 금액
            need -= sum(pos[t]["val"] for t in sells if t in pos)          # 이미 예정된 매도
            need -= sum(w * nav for _, w, *_ in buys)                      # 이미 예정된 매수
            if need > 0:
                # ① 목표 비중에서 벗어난 만큼 되돌리기
                for t, p in live.items():
                    if t in trims:
                        continue
                    drift = p["val"] / nav - p["tw"]
                    if drift > 0.002:
                        trims[t] = drift / (p["val"] / nav); need -= drift * nav
                    elif drift < -0.002:
                        buys.append((t, -drift, p["kind"], 0.0, p["until"])); need += drift * nav
                # ② 아직 부족하고 Stage1 보유 중이면: Stage1 후보 재선정 → 조건 빠진 종목 교체
                if need > 0 and any(p["kind"] == "S1" for p in live.values()):
                    fresh = stage1_candidates(P, ev, i + 1, d.year)
                    fresh_set = {t for t, _ in fresh}
                    new_names = [x for x in fresh if x[0] not in pos]
                    for t, p in sorted(live.items(), key=lambda kv: kv[1]["cum"]):
                        if need <= 0 or not new_names:
                            break
                        if p["kind"] == "S1" and t not in fresh_set and t not in trims:
                            sells.add(t)
                            nt, sc = new_names.pop(0)
                            buys.append((nt, p["tw"], "S1", sc, i1 + 1))
                            need -= 2 * p["val"]

    idx = dates[i0:i0 + len(nav_hist)]
    nav_s = pd.Series(nav_hist, index=idx, name="nav")
    wk_to = pd.Series({k: week_trade[k] / np.mean(v) * 0.5 for k, v in week_navs.items()})
    return dict(nav=nav_s,
                gross=pd.Series(gross_hist, index=idx),
                hhi=pd.Series(hhi_hist, index=idx),
                npos=pd.Series(npos_hist, index=idx),
                pnl=pd.Series(pnl).sort_values(ascending=False),
                weekly_turnover=wk_to,
                trades=pd.DataFrame(trades, columns=["date", "ticker", "side", "amount"]))


# ════════════════════════════════════════════════════════════
def metrics(res, k200, rf=0.0):
    nav = res["nav"]
    r = nav.pct_change().dropna()
    n = len(r)
    tot = nav.iloc[-1] / nav.iloc[0] - 1 if n else np.nan
    yrs = n / 252
    b = k200.reindex(nav.index).ffill()
    br = b.pct_change().dropna()
    btot = b.iloc[-1] / b.iloc[0] - 1 if n else np.nan
    sd = r.std()
    dn = r[r < 0].std()
    pnl = res["pnl"]
    gains = pnl[pnl > 0]
    top3 = gains.head(3).sum() / gains.sum() if len(gains) else np.nan   # 이익 종목 합계 중 상위 3종목 비중
    wk = res["weekly_turnover"]
    # 대회 기간은 첫 주/마지막 주가 부분 주라서 그대로 집계 (대회도 동일 기준)
    beta = np.cov(r, br.reindex(r.index).fillna(0))[0, 1] / br.var() if n > 5 and br.var() > 0 else np.nan
    return {
        "Total Return": tot,
        "Benchmark (KOSPI200)": btot,
        "Excess": tot - btot,
        "CAGR": (1 + tot) ** (1 / yrs) - 1 if yrs >= 0.9 else np.nan,
        "Ann. Vol": sd * np.sqrt(252),
        "Sharpe": (r.mean() - rf / 252) / sd * np.sqrt(252) if sd > 0 else np.nan,
        "Sortino": (r.mean() - rf / 252) / dn * np.sqrt(252) if dn and dn > 0 else np.nan,
        "MDD": (nav / nav.cummax() - 1).min(),
        "Beta": beta,
        "Daily Win Rate": (r > 0).mean(),
        "Avg Gross": res["gross"].mean(),
        "Avg Positions": res["npos"].mean(),
        "Top3 PnL Share": top3,
        "Profitable Names": int((pnl > 0).sum()),
        "Avg HHI": res["hhi"].mean(),
        "Avg Weekly Turnover": wk.mean(),
        "Weeks < 5%": int((wk < C.TURNOVER_MIN).sum()),
    }
