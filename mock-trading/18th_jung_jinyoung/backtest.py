"""주간 리밸런싱 시뮬레이션: 화요일 종가 신호 → 수요일 종가 체결."""
from __future__ import annotations

import numpy as np
import pandas as pd

from allocate import allocate, allocate_three, swap_lowest
from signals import sector_momentum
from regime import aggressiveness
from allocate import check_portfolio, weekly_turnover
from signals import GICS_ORDER, market_sector_weights


def schedule(dates: pd.DatetimeIndex, start, end, sig_wd=1, trade_wd=2) -> list[tuple]:
    """주마다 (신호일, 체결일). 신호일 = 화요일 이하 마지막 거래일, 체결일 = 그 주 수요일 이상 첫 거래일."""
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    out = []
    weeks = pd.Series(dates, index=dates).groupby(dates.to_period("W-SUN"))
    for wk, ds in weeks:
        mon = wk.start_time.normalize()
        tue, wed = mon + pd.Timedelta(days=sig_wd), mon + pd.Timedelta(days=trade_wd)
        trade = ds[ds >= wed]
        sig = dates[dates <= tue]
        if trade.empty or sig.empty:
            continue
        out.append((sig[-1], trade.iloc[0]))
    return out


def snapshot(pn, S) -> pd.DataFrame:
    """신호일 S 기준 매수 가능 종목 스냅샷 (S까지의 데이터만)."""
    d = pn.d
    e = d["elig"].loc[S]
    codes = e.index[e.to_numpy()]
    sec = d["sec"].loc[S, codes]
    m_all, s_all = pn.mcap.loc[S], d["sec"].loc[S]
    total = m_all[(s_all >= 0) & m_all.notna()].sum()     # 시장 전체(섹터 있는 보통주) 시총
    return pd.DataFrame({
        "mkt_share": pn.mcap.loc[S, codes] / total,
        "name": pn.stocks.loc[codes, "name"],
        "sector": [GICS_ORDER[i] for i in sec],
        "mcap": pn.mcap.loc[S, codes],
        "mom": d["mom61"].loc[S, codes],
        "vol": d["vol60"].loc[S, codes],
        "roe": d["roe"].loc[S, codes],
    }, index=codes)


def trend_ok(mkt, S, n) -> bool:
    """S까지의 시장 누적지수가 n일 이동평균 이상이면 True (상승 추세)."""
    lvl = (1 + mkt.loc[:S].fillna(0)).cumprod()
    return bool(lvl.iloc[-1] >= lvl.iloc[-n:].mean())


def signal_at(pn, sr, mkt, p_turb, S, cfg, a_fixed=None, held=()):
    """신호일 S의 목표 포트폴리오. 반환 (비중, info)."""
    rg = cfg["regime"]
    # 레짐 확률: S 당일 값이 없으면(데이터만 추가된 경우) S 이전 가장 최근 값 — 미래 값은 쓰지 않음
    p = float(p_turb.asof(S)) if p_turb is not None and len(p_turb) else np.nan
    a = float(aggressiveness(p, rg["a_full"], rg["a_zero"])) if a_fixed is None else float(a_fixed)
    up = trend_ok(mkt, S, rg["trend_ma"]) if rg.get("trend_filter") else False
    if a_fixed is None and up:
        a = 1.0          # 추세 조건: 시장이 이동평균 위면 브레이크를 걸지 않음
    mom = sector_momentum(sr, mkt, S, cfg)
    mkt_w = market_sector_weights(pn.mcap.loc[S], pn.d["sec"].loc[S])
    snap = snapshot(pn, S)
    sat = cfg["allocate"].get("satellite_weight")
    if cfg["allocate"].get("anchor_sleeve") and a_fixed is None:
        # 3-슬리브: 앵커(고정 보유) + 코어(앵커 제외 시장 섹터 비중, 고정 N종목) + 위성(고정 M종목)
        anc = cfg["allocate"]["anchor_sleeve"]
        m_all = pn.mcap.loc[S].drop(index=[c for c in anc if c in pn.mcap.columns])
        mkt_w_ex = market_sector_weights(m_all, pn.d["sec"].loc[S].reindex(m_all.index))
        w_core, w_sat, anchors, info, info_sat = allocate_three(snap, mom, mkt_w_ex, cfg, held)
        core_w = 1 - sat - sum(anchors.values())
        w = w_core.mul(core_w).add(w_sat.mul(sat), fill_value=0.0).add(pd.Series(anchors), fill_value=0.0)
        info.update(w_core=w_core, w_sat=w_sat, info_sat=info_sat, anchors=anchors, core_w=core_w, mkt_w_ex=mkt_w_ex)
        sec_of = pd.Series([GICS_ORDER[i] if i >= 0 else None for i in pn.d["sec"].loc[S]], index=pn.mcap.columns)
        info["sector_target"] = w.groupby(sec_of.reindex(w.index)).sum().reindex(GICS_ORDER).fillna(0.0)
        a = sat
    elif sat is not None and a_fixed is None:
        # 코어-위성: 코어(a=0: 시장 섹터 비중 + 방어 점수) (1-sat) + 위성(a=1: 모멘텀 집중 + 공격 점수) sat.
        # 두 포트폴리오가 각각 모든 한도를 만족하므로 가중합도 만족한다(한도가 모두 선형 상한).
        mp = cfg["allocate"].get("min_weight", 0.0)          # 전체 포트폴리오 기준 최소 편입 비중
        w_core, info = allocate(snap, mom, 0.0, mkt_w, cfg, held, min_w=mp / (1 - sat) if sat < 1 else 0.0, scale=1 - sat)
        sat_score = "defensive" if cfg["allocate"].get("satellite_score", "attack") == "defensive" else None
        w_sat, info_sat = allocate(snap, mom, 1.0, mkt_w, cfg, held, score_mode=sat_score, min_w=mp / sat if sat > 0 else 0.0, scale=max(sat, 1e-9))
        w = w_core.mul(1 - sat).add(w_sat.mul(sat), fill_value=0.0)
        info["sector_target"] = (1 - sat) * info["sector_target"] + sat * info_sat["sector_target"]
        info.update(w_core=w_core, w_sat=w_sat, info_sat=info_sat)
        a = sat
    else:
        w, info = allocate(snap, mom, a, mkt_w, cfg, held, min_w=cfg["allocate"].get("min_weight", 0.0))
    info.update(p_turb=p, a=a, mom=mom, mkt_w=mkt_w, trend_up=up)
    return w, info


def run(pn, sr, mkt, p_turb, cfg, a_fixed=None, label="strategy", verbose=False):
    al = cfg["allocate"]
    if a_fixed is None and al.get("satellite_weight") is not None and (al.get("core_hold_weeks") or al.get("anchor_sleeve")):
        return run_sleeves(pn, sr, mkt, p_turb, cfg, label, verbose)
    bt = cfg["backtest"]
    cost_buy = bt["commission"] + bt["slippage"]
    cost_sell = bt["commission"] + bt["slippage"] + bt["sell_tax"]
    dates = pn.price.index
    codes = pn.price.columns
    col = {c: i for i, c in enumerate(codes)}
    px = pn.price.to_numpy()
    ret = np.nan_to_num(pn.d["ret"].to_numpy(), nan=0.0)
    halt = pn.d["halt"].to_numpy() | np.isnan(px)
    sched = schedule(dates, bt["start"], bt["end"], bt["signal_weekday"], bt["trade_weekday"])
    trade_on = {T: S for S, T in sched}
    t0 = dates.get_loc(sched[0][1])

    h = np.zeros(len(codes))
    cash = 1.0
    nav_hist, trade_hist, logs, delists = [], [], [], []
    last_trade_val = {}
    for ti in range(t0, len(dates)):
        t = dates[ti]
        if ti > t0:
            h = h * (1 + ret[ti])
        buy = sell = 0.0
        T_S = trade_on.get(t)
        if T_S is not None:
            S = T_S
            w, info = signal_at(pn, sr, mkt, p_turb, S, cfg, a_fixed, held=[codes[i] for i in np.nonzero(h > 0)[0]])
            nav = h.sum() + cash
            frozen = (h > 0) & halt[ti]
            frozen_val = h[frozen].sum()

            def plan(w):
                w = w[[not halt[ti, col[c]] for c in w.index]]      # 정지 종목은 신규 매수 불가
                tv = np.zeros(len(codes))
                idx = [col[c] for c in w.index]
                tv[idx] = w.to_numpy() / w.sum() * (nav - frozen_val)
                tv[frozen] = h[frozen]
                return tv

            tv = plan(w)
            est = 0.5 * np.abs(tv - h).sum() / nav * 100
            swaps, added, removed = 0, set(), set()
            while est < bt["turnover_target"] * 100 and swaps < 10:
                w2 = swap_lowest(w, info, cfg, exclude=added, banned=removed)
                if w2 is None:
                    break
                added |= set(w2.index) - set(w.index)
                removed |= set(w.index) - set(w2.index)
                w, swaps = w2, swaps + 1
                tv = plan(w)
                est = 0.5 * np.abs(tv - h).sum() / nav * 100
            diff = tv - h
            buy, sell = diff[diff > 0].sum(), -diff[diff < 0].sum()
            cost = buy * cost_buy + sell * cost_sell
            h = tv
            if h.sum() > 0:
                h = h * (1 - cost / h.sum())
            cash = nav - tv.sum()
            snapi = info["snap"]
            errs = check_portfolio(w / w.sum(), snapi["sector"].reindex(w.index), snapi["mcap"].reindex(w.index), info["mkt_w"], cfg)
            held_sec = pd.Series(h, index=codes)[h > 0]
            logs.append({"signal": S, "trade": t, "p_turb": info["p_turb"], "a": info["a"], "n": int((h > 0).sum()),
                         "est_turnover": est, "swaps": swaps, "cost": cost, "violations": "; ".join(errs),
                         **{f"sec_{g}": v for g, v in info["sector_target"].items()},
                         **{f"mom_{g}": v for g, v in info["mom"].items()}})
            last_trade_val = {i: h[i] for i in np.nonzero(h > 0)[0]}
            if verbose and len(logs) % 50 == 0:
                print(f"  [{label}] {t.date()} a={info['a']:.2f} n={logs[-1]['n']} est_to={est:.1f}%")
        # 상장폐지: 다음 거래일에 가격이 없으면 오늘 종가(정리매매 마지막 가격)로 청산
        if ti + 1 < len(dates):
            gone = np.nonzero((h > 0) & ~np.isnan(px[ti]) & np.isnan(px[ti + 1]))[0]
            for i in gone:
                v0 = last_trade_val.get(i, np.nan)
                val = h[i] if bt["delist_mode"] == "last_price" else 0.0
                delists.append({"date": t, "code": codes[i], "name": pn.stocks.at[codes[i], "name"],
                                "weight": h[i] / (h.sum() + cash), "ret_since_trade": h[i] / v0 - 1 if v0 else np.nan,
                                "loss_nav": (val - v0) / (h.sum() + cash) if v0 else np.nan})
                sell += val
                cash += val * (1 - cost_sell)
                h[i] = 0.0
        nav_hist.append(h.sum() + cash)
        trade_hist.append((buy, sell))
    idx = dates[t0:]
    nav = pd.Series(nav_hist, index=idx, name=label)
    tr = pd.DataFrame(trade_hist, index=idx, columns=["buy", "sell"])
    wk = pd.DataFrame({"buy": tr.buy, "sell": tr.sell, "nav": nav}).groupby(idx.to_period("W-SUN"))
    turn = wk.apply(lambda g: weekly_turnover(g.buy.sum(), g.sell.sum(), g.nav.mean()))
    return {"nav": nav, "log": pd.DataFrame(logs), "turnover": turn, "delists": pd.DataFrame(delists)}


def sleeve_trade(pn, sr, mkt, p_turb, cfg, S, halt_row, hc, hs, ha, cash, wk_since):
    """매매일 1회의 목표 금액 계산 (백테스트 run_sleeves와 실전 run_live가 공용).
    hc/hs/ha: 현재 코어/위성/앵커 보유 금액(종목 순서 = pn.price.columns), cash: 현금, halt_row: 매매 불가 여부.
    반환: 목표 금액 tc/ts/ta(비용 반영 전), 매수·매도 금액, 비용, 로그용 정보."""
    bt, al = cfg["backtest"], cfg["allocate"]
    sat_w, hold_wk = al["satellite_weight"], al["core_hold_weeks"]
    band = al.get("anchor_band", 0.05)
    cost_buy = bt["commission"] + bt["slippage"]
    cost_sell = bt["commission"] + bt["slippage"] + bt["sell_tax"]
    codes = pn.price.columns
    col = {c: i for i, c in enumerate(codes)}
    N = len(codes)

    def _plan(h_cur, w, value):
        """슬리브 목표 금액. 정지 종목은 현 보유 유지, 신규 매수 불가."""
        frozen = (h_cur > 0) & halt_row
        w = w[[not halt_row[col[c]] for c in w.index]]
        tv = np.zeros(N)
        if len(w):
            tv[[col[c] for c in w.index]] = w.to_numpy() / w.sum() * (value - h_cur[frozen].sum())
        tv[frozen] = h_cur[frozen]
        return tv

    held = [codes[i] for i in np.nonzero(hc + hs + ha > 0)[0]]
    _, info = signal_at(pn, sr, mkt, p_turb, S, cfg, None, held=held)
    w_core, w_sat, info_sat = info["w_core"], info["w_sat"], info["info_sat"]
    nav = hc.sum() + hs.sum() + ha.sum() + cash
    anchors = info.get("anchors", {})
    core_w = info.get("core_w", 1 - sat_w)
    ta = ha.copy()
    if anchors:
        w_now = {c: ha[col[c]] / nav for c in anchors}
        if ha.sum() <= 0 or any(abs(w_now[c] - t_) > band for c, t_ in anchors.items()):
            ta = _plan(ha, pd.Series(anchors), sum(anchors.values()) * nav)
    sec_S = pd.Series([GICS_ORDER[i] if i >= 0 else None for i in pn.d["sec"].loc[S]], index=codes)
    mcap_S = pn.mcap.loc[S]
    cur_core = pd.Series(hc, index=codes)[hc > 0]

    def build(refresh, w_sat_, w_core_):
        if refresh:
            tc_ = _plan(hc, w_core_, core_w * nav)
            return tc_, _plan(hs, w_sat_, nav - tc_.sum() - ta.sum())
        # 코어 유지(보유 그대로, 교체가 있으면 그 비중으로), 위성만 목표로. 위성 슬리브 가치 = 현재 위성 + 현금
        tc = base_core.copy() if w_core_ is None else _plan(base_core, w_core_, base_core.sum())
        return tc, _plan(hs, w_sat_, nav - tc.sum() - ta.sum())

    def violations(tc, ts):
        tot = (tc + ts + ta).sum()
        comb = pd.Series((tc + ts + ta) / max(tot, 1e-12), index=codes)
        comb = comb[comb > 1e-9]
        return check_portfolio(comb, sec_S.reindex(comb.index), mcap_S.reindex(comb.index), info["mkt_w"], cfg)

    base_core = hc
    if al.get("core_mode", "block") == "names":
        # 종목 단위 교체: 코어는 항상 목표 종목 수(core_n). 목표에서 빠진 보유 종목(점수 낮은 순)을
        # 목표 신규 종목(점수 높은 순)으로 통째로 교체. 최소 core_k_min개, 회전율 목표 미달·한도 위반이면 더 교체.
        score = info["snap"]["score"]
        n_tgt = len(w_core)
        core_val = nav - ta.sum() - sat_w * nav        # 위성은 항상 sat_w 고정, 앵커 등락은 코어가 흡수
        H = [codes[i] for i in np.nonzero(hc > 0)[0]]
        T = set(w_core.index)
        D = sorted([c for c in H if c not in T and not halt_row[col[c]]], key=lambda c: score.get(c, -np.inf))
        A = sorted([c for c in T if c not in H and not halt_row[col[c]]], key=lambda c: -score[c])
        extra = max(len(H) - n_tgt, 0)                 # 보유가 목표보다 많으면 추가 매도
        fill = max(n_tgt - len(H), 0)                  # 모자라면(첫 주·상폐 후) 추가 매수
        k_min = min(al.get("core_k_min", 2), len(D), len(A))
        reason = ""
        for k in range(k_min, max(len(D), len(A)) + 1):
            drop, add = D[: k + extra], A[: k + fill]
            tc = hc.copy()
            tc[[col[c] for c in drop]] = 0.0
            for c in add:
                tc[col[c]] = w_core[c] * core_val
            if tc.sum() > 0:
                tc = tc * core_val / tc.sum()          # 코어 슬리브 금액을 정확히 맞춤 (상대 비중은 유지)
            ts = _plan(hs, w_sat, nav - tc.sum() - ta.sum())
            est = 0.5 * np.abs((tc + ts + ta) - (hc + hs + ha)).sum() / nav * 100
            errs_k = violations(tc, ts)
            reason = f"k={len(drop)}/{len(add)}"
            if est >= bt["turnover_target"] * 100 and not errs_k:
                break
        if violations(tc, ts):                        # 그래도 한도 위반이면 코어 전체를 목표로
            tc = _plan(hc, w_core, core_val)
            ts = _plan(hs, w_sat, nav - tc.sum() - ta.sum())
            reason = "full"
            if violations(tc, ts) and anchors:          # 앵커 쏠림까지 원인이면 앵커도 목표로
                ta = _plan(ha, pd.Series(anchors), sum(anchors.values()) * nav)
                tc = _plan(hc, w_core, nav - ta.sum() - sat_w * nav)
                ts = _plan(hs, w_sat, nav - tc.sum() - ta.sum())
                reason = "full+anchor"
        base_core = tc
        refresh = False
    elif al.get("core_mode", "block") == "rolling":
        # 순환 갱신: 코어를 목표 쪽으로 매주 f만큼 이동 (기본 1/hold_wk). 회전율 목표 미달이거나
        # 합산 한도 위반이면 f를 키운다 → 규정상 의무 매매를 코어 최신화에 사용. f=1이면 전체 갱신.
        tgt_c = _plan(hc, w_core, core_w * nav)
        f0 = 1.0 / hold_wk if hc.sum() > 0 else 1.0
        min_pos = al.get("min_position", 0.005) * nav
        for f in np.unique(np.r_[np.linspace(f0, 1.0, 25), 1.0]):
            tc = hc + f * (tgt_c - hc)
            # 목표에서 빠진 종목의 잔량이 min_position 미만이면 전량 매도, 그 금액은 코어 목표 종목에 비례 배분
            dust = (tgt_c <= 0) & (tc > 0) & (tc < min_pos) & ~halt_row
            if dust.any() and tgt_c.sum() > 0:
                freed = tc[dust].sum()
                tc[dust] = 0.0
                tc += freed * tgt_c / tgt_c.sum()
            ts = _plan(hs, w_sat, nav - tc.sum() - ta.sum())
            est = 0.5 * np.abs((tc + ts + ta) - (hc + hs + ha)).sum() / nav * 100
            errs_f = violations(tc, ts)
            if errs_f and anchors and ta is not None and np.allclose(ta, ha):
                # 앵커가 한도 위반 원인이면 앵커를 목표로 되돌림
                ta = _plan(ha, pd.Series(anchors), sum(anchors.values()) * nav)
                ts = _plan(hs, w_sat, nav - tc.sum() - ta.sum())
                est = 0.5 * np.abs((tc + ts + ta) - (hc + hs + ha)).sum() / nav * 100
                errs_f = violations(tc, ts)
            if est >= bt["turnover_target"] * 100 and not errs_f:
                break
        refresh, reason = f >= 1.0 - 1e-9, f"f={f:.2f}"
    else:
        refresh = wk_since >= hold_wk
        reason = "schedule" if refresh else ""
        tc, ts = build(refresh, w_sat, w_core if refresh else None)
        errs = violations(tc, ts)
        if errs and not refresh:                  # 유지 중인 코어가 합산 한도를 넘으면 즉시 갱신
            refresh, reason = True, "limit"
            tc, ts = build(True, w_sat, w_core)
    bc = pd.Series(base_core, index=codes)[base_core > 0]
    core_cur = w_core if refresh else (bc / bc.sum() if len(bc) else bc)
    core_changed = None
    est = 0.5 * np.abs((tc + ts + ta) - (hc + hs + ha)).sum() / nav * 100
    swaps, added, removed = 0, set(), set()
    while est < bt["turnover_target"] * 100 and swaps < 15:
        w2, target = swap_lowest(w_sat, info_sat, cfg, exclude=added, banned=removed), "sat"
        if w2 is None:
            w2, target = swap_lowest(core_cur, info, cfg, exclude=added, banned=removed), "core"
            if w2 is None:
                break
        old = w_sat if target == "sat" else core_cur
        new_in = set(w2.index) - set(old.index)
        swaps += 1
        cand = (w2, core_cur) if target == "sat" else (w_sat, w2)
        tc2, ts2 = build(refresh, cand[0], cand[1] if (refresh or target == "core") else core_changed)
        if violations(tc2, ts2):             # 합산 포트폴리오 기준 한도 위반이면 이 교체는 취소
            removed |= new_in
            continue
        added |= new_in
        removed |= set(old.index) - set(w2.index)
        if target == "sat":
            w_sat = w2
        else:
            core_cur = core_changed = w2
        tc, ts = tc2, ts2
        est = 0.5 * np.abs((tc + ts + ta) - (hc + hs + ha)).sum() / nav * 100
    errs = violations(tc, ts)
    diff = (tc - hc) + (ts - hs) + (ta - ha)
    buy, sell = diff[diff > 0].sum(), -diff[diff < 0].sum()
    cost = buy * cost_buy + sell * cost_sell
    return {"tc": tc, "ts": ts, "ta": ta, "buy": buy, "sell": sell, "cost": cost, "refresh": refresh,
            "reason": reason, "est": est, "swaps": swaps, "errs": errs, "info": info, "nav": nav,
            "w_core": w_core, "w_sat": w_sat}


def run_sleeves(pn, sr, mkt, p_turb, cfg, label="strategy", verbose=False):
    """코어-위성 분리 운용: 위성은 매주 목표로 회전, 코어는 core_hold_weeks마다(또는 합산 한도 위반 시) 갱신하고
    그 사이에는 보유 유지. 회전율이 목표 미만이면 위성 → 코어 순으로 최하위 종목 교체.
    anchor_sleeve가 있으면 앵커(삼성·하이닉스 등)를 목표 비중으로 사서 보유하고, 목표에서 anchor_band 넘게
    벗어나거나 합산 한도를 위반할 때만 목표로 되돌린다."""
    bt, al = cfg["backtest"], cfg["allocate"]
    sat_w, hold_wk = al["satellite_weight"], al["core_hold_weeks"]
    cost_buy = bt["commission"] + bt["slippage"]
    cost_sell = bt["commission"] + bt["slippage"] + bt["sell_tax"]
    dates, codes = pn.price.index, pn.price.columns
    col = {c: i for i, c in enumerate(codes)}
    px = pn.price.to_numpy()
    ret = np.nan_to_num(pn.d["ret"].to_numpy(), nan=0.0)
    halt = pn.d["halt"].to_numpy() | np.isnan(px)
    sched = schedule(dates, bt["start"], bt["end"], bt["signal_weekday"], bt["trade_weekday"])
    trade_on = {T: S for S, T in sched}
    t0 = dates.get_loc(sched[0][1])
    N = len(codes)
    hc, hs, ha = np.zeros(N), np.zeros(N), np.zeros(N)   # 코어 / 위성 / 앵커 보유 금액
    band = al.get("anchor_band", 0.05)
    cash = 1.0
    wk_since = hold_wk                            # 첫 주는 코어 신규 매수
    nav_hist, trade_hist, logs, delists = [], [], [], []
    last_val = {}

    for ti in range(t0, len(dates)):
        t = dates[ti]
        if ti > t0:
            hc, hs, ha = hc * (1 + ret[ti]), hs * (1 + ret[ti]), ha * (1 + ret[ti])
        buy = sell = 0.0
        S = trade_on.get(t)
        if S is not None:
            r = sleeve_trade(pn, sr, mkt, p_turb, cfg, S, halt[ti], hc, hs, ha, cash, wk_since)
            info, nav, est, swaps, reason, errs = r["info"], r["nav"], r["est"], r["swaps"], r["reason"], r["errs"]
            tc, ts, ta, buy, sell, cost = r["tc"], r["ts"], r["ta"], r["buy"], r["sell"], r["cost"]
            tot = tc.sum() + ts.sum() + ta.sum()
            cash = nav - tot
            if tot > 0:
                tc, ts, ta = tc * (1 - cost / tot), ts * (1 - cost / tot), ta * (1 - cost / tot)
            hc, hs, ha = tc, ts, ta
            wk_since = 1 if r["refresh"] else wk_since + 1
            logs.append({"signal": S, "trade": t, "p_turb": info["p_turb"], "a": sat_w, "n": int((hc + hs + ha > 0).sum()), "n_core": int((hc > 0).sum()), "n_sat": int((hs > 0).sum()),
                         "anchor_share": ha.sum() / nav,
                         "core_refresh": reason, "est_turnover": est, "swaps": swaps, "cost": cost,
                         "core_share": hc.sum() / nav, "violations": "; ".join(errs),
                         **{f"sec_{g}": v for g, v in info["sector_target"].items()},
                         **{f"mom_{g}": v for g, v in info["mom"].items()}})
            last_val = {i: hc[i] + hs[i] + ha[i] for i in np.nonzero(hc + hs + ha > 0)[0]}
            if verbose and len(logs) % 50 == 0:
                print(f"  [{label}] {t.date()} n={logs[-1]['n']} est_to={est:.1f}% refresh={reason}")
        if ti + 1 < len(dates):
            gone = np.nonzero((hc + hs + ha > 0) & ~np.isnan(px[ti]) & np.isnan(px[ti + 1]))[0]
            for i in gone:
                tot_nav = hc.sum() + hs.sum() + ha.sum() + cash
                v = hc[i] + hs[i] + ha[i]
                val = v if bt["delist_mode"] == "last_price" else 0.0
                v0 = last_val.get(i, np.nan)
                delists.append({"date": t, "code": codes[i], "name": pn.stocks.at[codes[i], "name"], "weight": v / tot_nav,
                                "ret_since_trade": v / v0 - 1 if v0 else np.nan,
                                "loss_nav": (val - v0) / tot_nav if v0 else np.nan})
                sell += val
                cash += val * (1 - cost_sell)
                hc[i] = hs[i] = ha[i] = 0.0
        nav_hist.append(hc.sum() + hs.sum() + ha.sum() + cash)
        trade_hist.append((buy, sell))
    idx = dates[t0:]
    nav = pd.Series(nav_hist, index=idx, name=label)
    tr = pd.DataFrame(trade_hist, index=idx, columns=["buy", "sell"])
    wk = pd.DataFrame({"buy": tr.buy, "sell": tr.sell, "nav": nav}).groupby(idx.to_period("W-SUN"))
    turn = wk.apply(lambda g: weekly_turnover(g.buy.sum(), g.sell.sum(), g.nav.mean()))
    return {"nav": nav, "log": pd.DataFrame(logs), "turnover": turn, "delists": pd.DataFrame(delists)}
