"""섹터·종목 배분과 한도 조정 + 대회 규칙 점검(한도·회전율). 백테스트와 실전이 같이 사용."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from signals import GICS_ORDER


# ────────────────────────────── allocate ──────────────────────────────


EPS = 1e-9


def stock_cap(code: str, cfg) -> float:
    return cfg["limits"]["stock_exceptions"].get(code, cfg["limits"]["stock_max"])


def sector_caps(mkt_w: pd.Series, cfg) -> pd.Series:
    L = cfg["limits"]
    cap = np.where(mkt_w <= L["sector_small_thresh"], L["sector_small_cap"], L["sector_mult"] * mkt_w)
    return pd.Series(np.minimum(cap, 1.0), index=mkt_w.index)


def waterfill(desired: pd.Series, caps: pd.Series, total: float = 1.0) -> pd.Series:
    """desired 비율대로 total을 배분하되 caps를 넘는 몫은 여유 있는 곳에 비례 재배분."""
    w = pd.Series(0.0, index=desired.index)
    free = (caps > EPS) & (desired > 0)
    remain = total
    for _ in range(len(w) + 1):
        if remain <= EPS or not free.any():
            break
        share = desired[free] / desired[free].sum() * remain
        room = caps[free] - w[free]
        add = np.minimum(share, room)
        w[free] += add
        remain -= add.sum()
        free = free & (caps - w > EPS)
    return w


def concentrated(mom: pd.Series, caps: pd.Series, cfg) -> pd.Series:
    """모멘텀 상위 섹터부터 한도까지 채움 (합이 100%가 될 때까지 다음 순위로)."""
    a = cfg["allocate"]
    caps = caps.copy()
    if a.get("it_cap"):
        caps["정보기술"] = min(caps["정보기술"], a["it_cap"])
    order = mom.reindex(caps.index).sort_values(ascending=False, na_position="last").index
    w, left = pd.Series(0.0, index=caps.index), 1.0
    for g in order:
        if left <= EPS:
            break
        w[g] = min(caps[g], left)
        left -= w[g]
    return w


def stock_scores(snap: pd.DataFrame, a: float, mode: str = "blend") -> pd.Series:
    """섹터 내 백분위 순위. 공격 = 6-1 모멘텀 + ROE, 방어 = (−60일 변동성) + ROE.
    mode="blend": a로 공격·방어 혼합 (SPEC 기본), "defensive": a와 무관하게 방어 점수만."""
    if mode == "defensive":
        a = 0.0
    g = snap.groupby("sector")
    r_mom = g["mom"].rank(pct=True).fillna(0.5)
    r_roe = g["roe"].rank(pct=True).fillna(0.5)
    r_vol = g["vol"].rank(pct=True, ascending=False).fillna(0.5)
    return a * (r_mom + r_roe) + (1 - a) * (r_vol + r_roe)


def _fill_sector(ranked: list[str], forced: list[str], w_s: float, cfg, held=(), mcap=None, share=None,
                 min_w: float = 0.0, scale: float = 1.0) -> pd.Series:
    """섹터 비중 w_s를 종목에 배분.
    - 앵커(anchor=market): 강제 편입 종목은 시장 시총 비중(share)으로 고정(종목 한도 이내), 나머지만 선별 종목에 동일가중.
    - min_w: 종목당 최소 비중. 이보다 작아지지 않게 선별 종목 수를 줄인다."""
    al = cfg["allocate"]
    cap0 = cfg["limits"]["stock_max"]
    # 비강제 종목 상한(전체 포트폴리오 기준 noncore_cap → 슬리브 기준으로 환산)
    ncap = al.get("noncore_cap")
    ncap_s = ncap / scale if ncap else None
    capf = lambda c: stock_cap(c, cfg) if (c in forced or not ncap_s) else min(stock_cap(c, cfg), ncap_s)
    anchors = {}
    if al.get("anchor") == "market" and share is not None and forced:
        anchors = {c: min(stock_cap(c, cfg), float(share.get(c, 0.0))) for c in forced}
        tot = sum(anchors.values())
        if tot > w_s:
            anchors = {c: v * w_s / tot for c, v in anchors.items()}
    rest_w = w_s - sum(anchors.values())
    pool = [c for c in ranked if c not in anchors]
    lead = [] if anchors else list(forced)           # 앵커가 없으면 기존처럼 강제 편입 종목도 동일가중 대상
    n_max = al["max_stocks_per_sector"] - len(anchors)
    n = int(np.clip(math.ceil(rest_w / cap0 - EPS) + 1, 1, max(n_max, 1)))
    if ncap_s and al.get("noncore_mode", "overflow") == "spread":
        n = max(n, int(math.ceil(rest_w / ncap_s - EPS)))       # 상한을 지키도록 종목 수를 늘림 (5종목 제한 해제)
    if min_w > 0:
        n = max(min(n, int(rest_w / min_w + EPS)), 1)
    if rest_w <= EPS:
        return pd.Series(anchors, dtype=float)
    # 버퍼: 기존 보유 종목은 섹터 내 순위가 buffer_mult × n 안이면 유지 (순위 흔들림에 의한 교체 방지)
    bm = al.get("buffer_mult")
    keep = []
    if bm and held:
        keep_rank = max(int(math.ceil(bm * n)), n + 1)
        keep = [c for c in pool[:keep_rank] if c in held and c not in lead][: max(n - len(lead), 0)]
    chosen = lead + keep
    chosen += [c for c in pool if c not in chosen][: max(n - len(chosen), 0)]
    rest = [c for c in pool if c not in chosen]
    while True:
        caps = pd.Series({c: capf(c) for c in chosen}, dtype=float)
        if caps.sum() >= rest_w - EPS or not rest:
            break
        chosen.append(rest.pop(0))          # 종목 한도 초과 → 같은 섹터 다음 순위 종목 추가
    if not chosen:                           # 선별 종목이 없으면 앵커에 (한도 내) 몰아줌
        a = pd.Series(anchors, dtype=float)
        return a + waterfill(a.clip(lower=EPS), pd.Series({c: stock_cap(c, cfg) for c in a.index}) - a, rest_w)
    desired = pd.Series(1.0 / len(chosen), index=chosen)
    if al.get("intra_weight", "equal") == "half_cap" and mcap is not None:
        # 섹터 안 비중 = 동일가중 50% + 선택 종목 내 시총가중 50%
        m = mcap.reindex(chosen).fillna(0.0)
        if m.sum() > 0:
            desired = 0.5 * desired + 0.5 * m / m.sum()
    w = waterfill(desired, caps, rest_w)
    left = rest_w - w.sum()
    if anchors:
        a = pd.Series(anchors, dtype=float)
        if left > EPS:                       # 선별 종목 한도가 모자라면 남는 몫은 앵커로
            a = a + waterfill(a.clip(lower=EPS), pd.Series({c: stock_cap(c, cfg) for c in a.index}) - a, left)
        w = pd.concat([a, w])
    return w


def allocate(snap: pd.DataFrame, mom: pd.Series, a: float, mkt_w: pd.Series, cfg, held=(), score_mode=None,
             min_w: float = 0.0, scale: float = 1.0):
    """snap: index=code, 열 sector, mcap, mom, vol, roe (매수 가능 종목만).
    반환: (종목 비중 Series, 정보 dict)."""
    L = cfg["limits"]
    snap = snap.copy()
    snap["score"] = stock_scores(snap, a, score_mode or cfg["allocate"].get("stock_score", "blend"))
    rule_cap = sector_caps(mkt_w, cfg)
    # 섹터가 담을 수 있는 최대 비중 = 섹터 내 종목 한도 합 (비강제 종목은 noncore_cap을 슬리브 기준으로 환산해 적용)
    ncap = cfg["allocate"].get("noncore_cap")
    forced_all = {c for v in (cfg["allocate"].get("core", {}) or {}).values() for c in v}
    eff_cap = lambda c: stock_cap(c, cfg) if (c in forced_all or not ncap) else min(stock_cap(c, cfg), ncap / scale)
    capacity = snap.groupby("sector").apply(lambda g: sum(eff_cap(c) for c in g.index)).reindex(GICS_ORDER).fillna(0.0)
    caps = np.minimum(rule_cap, capacity)

    w_neu = waterfill(mkt_w.where(caps > EPS, 0.0), caps)
    w_conc = concentrated(mom.where(caps > EPS), caps, cfg)
    target = a * w_conc + (1 - a) * w_neu
    if min_w > 0:
        # 최소 편입 비중 미만 섹터는 제외하고, 그 몫을 편입 섹터에 한도 내에서 비례 배분
        small = (target > EPS) & (target < min_w)
        if small.any():
            freed = target[small].sum()
            target[small] = 0.0
            room = (caps - target).where(target > EPS, 0.0).clip(lower=0.0)
            add = waterfill(target, room, freed)
            target = target + add
            left = freed - add.sum()
            if left > EPS:   # 편입 섹터가 모두 한도면(집중 포트폴리오) 제외 섹터 말고 다른 섹터에 시장 비중 비례로
                room2 = (caps - target).where(~small, 0.0).clip(lower=0.0)
                target = target + waterfill(mkt_w.where(room2 > EPS, 0.0), room2, left)
    assert (target <= caps + 1e-7).all(), "섹터 목표 비중이 한도 초과"
    assert abs(target.sum() - 1) < 1e-6, f"섹터 비중 합 {target.sum()}"

    core = cfg["allocate"].get("core", {}) or {}
    w = {}
    ranked_by = {}
    for g, w_s in target.items():
        sub = snap[snap.sector == g].sort_values("score", ascending=False)
        ranked_by[g] = list(sub.index)
        if w_s <= EPS:
            continue
        forced = [c for c in core.get(g, []) if c in sub.index]
        share = snap["mkt_share"] if "mkt_share" in snap else None
        w.update(_fill_sector(list(sub.index), forced, w_s, cfg, set(held), snap["mcap"], share, min_w, scale).to_dict())
    w = pd.Series(w)
    w = w[w > EPS]
    w = fix_smallcap(w, snap, ranked_by, core, cfg)
    info = {"sector_target": target, "w_conc": w_conc, "w_neu": w_neu, "caps": caps, "rule_caps": rule_cap,
            "ranked": ranked_by, "snap": snap, "core": core}
    return w, info


def fix_smallcap(w: pd.Series, snap: pd.DataFrame, ranked_by: dict, core: dict, cfg, lim=None) -> pd.Series:
    """시총 1조 미만 합산 30%(또는 lim) 초과 → 점수 낮은 소형주부터 같은 섹터 대형주로 교체."""
    L = cfg["limits"]
    line = L["smallcap_mcap"]
    lim = L["smallcap_max"] if lim is None else lim
    is_small = lambda c: snap.at[c, "mcap"] < line
    core_all = {c for v in core.values() for c in v}
    tried = set()
    while w[[c for c in w.index if is_small(c)]].sum() > lim + EPS:
        smalls = [c for c in w.index if is_small(c) and c not in tried and c not in core_all]
        if not smalls:
            break
        c = min(smalls, key=lambda x: snap.at[x, "score"])
        tried.add(c)
        g = snap.at[c, "sector"]
        cand = [k for k in ranked_by[g] if k not in w.index and not is_small(k)]
        if cand:
            new = cand[0]
            w[new] = w.pop(c)
    # 대형주로 교체할 후보가 없으면: 소형주 비중을 줄이고 같은 섹터 → 전체 대형 보유 종목으로 이동
    small_idx = [c for c in w.index if is_small(c)]
    excess = w[small_idx].sum() - lim
    if excess > EPS:
        w[small_idx] *= lim / w[small_idx].sum()
        large = [c for c in w.index if not is_small(c)]
        caps = pd.Series({c: stock_cap(c, cfg) for c in large})
        w[large] += waterfill(w[large].clip(lower=EPS), caps - w[large], excess)
    return w


def swap_lowest(w: pd.Series, info: dict, cfg, exclude=(), banned=()) -> pd.Series | None:
    """회전율 보충: 점수 최하위 보유 종목(강제 편입 제외)을 같은 섹터 차순위 종목으로 교체.
    exclude: 교체 대상에서 뺄 종목(이미 새로 넣은 종목), banned: 다시 넣지 않을 종목(이미 뺀 종목)."""
    snap = info["snap"]
    core_all = {c for v in info["core"].values() for c in v}
    line, lim = cfg["limits"]["smallcap_mcap"], cfg["limits"]["smallcap_max"]
    held = [c for c in w.index if c in snap.index and c not in core_all and c not in exclude]
    for c in sorted(held, key=lambda x: snap.at[x, "score"]):
        g = snap.at[c, "sector"]
        small_sum = w[[k for k in w.index if k in snap.index and snap.at[k, "mcap"] < line]].sum()
        for new in info["ranked"][g]:
            if new in w.index or new in banned:
                continue
            if snap.at[new, "mcap"] < line and snap.at[c, "mcap"] >= line and small_sum + w[c] > lim + EPS:
                continue
            if w[c] > stock_cap(new, cfg) + EPS:
                continue
            out = w.drop(c)
            out[new] = w[c]
            return out
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3-슬리브 구조: 앵커(삼성·하이닉스 고정 보유) + 코어(고정 N종목) + 위성(고정 M종목)
# ─────────────────────────────────────────────────────────────────────────────
def allocate_slots(sector_w: pd.Series, n: int, avail: pd.Series) -> pd.Series:
    """섹터 비중에 비례해 n개 종목 슬롯을 최대잔여법으로 배정 (섹터별 가능 종목 수 이내)."""
    w = sector_w[sector_w > EPS]
    w = w / w.sum()
    quota = w * n
    slots = np.floor(quota).astype(int).clip(upper=avail.reindex(w.index).fillna(0).astype(int))
    rem = (quota - slots).sort_values(ascending=False)
    while slots.sum() < n:
        room = [g for g in rem.index if slots[g] < avail.get(g, 0)]
        if not room:
            break
        g = max(room, key=lambda x: rem[x])
        slots[g] += 1
        rem[g] -= 1
    return slots[slots > 0]


def _pick(ranked: list[str], k: int, held, bm) -> list[str]:
    """섹터 내 상위 k종목. 버퍼: 기존 보유 종목은 순위가 bm×k 안이면 유지."""
    keep = []
    if bm and held:
        keep = [c for c in ranked[: max(int(math.ceil(bm * k)), k + 1)] if c in held][:k]
    return keep + [c for c in ranked if c not in keep][: k - len(keep)]


def allocate_three(snap: pd.DataFrame, mom: pd.Series, mkt_w_ex: pd.Series, cfg, held=()):
    """반환: (w_core, w_sat, anchors, info_core, info_sat). w_core·w_sat은 슬리브 내 비중(합 1).
    - 코어: 앵커 종목을 뺀 시장의 섹터 비중(mkt_w_ex)에 core_n개 슬롯을 비례 배정, 섹터 안 방어 점수(저변동+ROE) 상위,
            섹터 비중은 슬롯에 균등 분배.
    - 위성: 앵커를 뺀 종목으로 모멘텀 집중 섹터 비중을 만들고 sat_n개 슬롯 배정, 공격 점수(6-1 모멘텀+ROE) 상위, 동일가중."""
    al, L = cfg["allocate"], cfg["limits"]
    anchors = dict(al["anchor_sleeve"])
    sat_w = al["satellite_weight"]
    core_w = 1.0 - sat_w - sum(anchors.values())
    bm = al.get("buffer_mult")
    held = set(held)
    ex = snap.drop(index=[c for c in anchors if c in snap.index])
    avail = ex.groupby("sector").size()

    # 코어
    sc = ex.copy()
    sc["score"] = stock_scores(sc, 0.0)
    ranked_c = {g: list(s.sort_values("score", ascending=False).index) for g, s in sc.groupby("sector")}
    slots_c = allocate_slots(mkt_w_ex.reindex(avail.index).fillna(0.0), al["core_n"], avail)
    sw = mkt_w_ex.reindex(slots_c.index)
    sw = sw / sw.sum()
    w_core = {}
    for g, k in slots_c.items():
        for c in _pick(ranked_c[g], int(k), held, bm):
            w_core[c] = sw[g] / k
    w_core = pd.Series(w_core)

    # 위성
    ss = ex.copy()
    ss["score"] = stock_scores(ss, 1.0)
    ranked_s = {g: list(s.sort_values("score", ascending=False).index) for g, s in ss.groupby("sector")}
    caps = sector_caps(mkt_w_ex, cfg).reindex(GICS_ORDER).fillna(0.0).where(avail.reindex(GICS_ORDER).fillna(0) > 0, 0.0)
    w_conc = concentrated(mom.reindex(GICS_ORDER).where(caps > EPS), caps, cfg)
    slots_s = allocate_slots(w_conc, al["sat_n"], avail)
    w_sat = {}
    for g, k in slots_s.items():
        for c in _pick(ranked_s[g], int(k), held, bm):
            w_sat[c] = 1.0 / al["sat_n"]
    w_sat = pd.Series(w_sat)
    w_sat = w_sat / w_sat.sum()

    # 소형주 한도: 위성 소형주 몫을 뺀 나머지 한도 안으로 코어를 맞춤
    line = L["smallcap_mcap"]
    sat_small = w_sat[snap.loc[w_sat.index, "mcap"] < line].sum() * sat_w
    lim_core = max(L["smallcap_max"] - sat_small, 0.0) / core_w
    w_core = fix_smallcap(w_core, sc, ranked_c, {}, cfg, lim=lim_core)

    info_core = {"snap": sc, "ranked": ranked_c, "core": {}, "sector_target": sw.reindex(GICS_ORDER).fillna(0.0),
                 "w_neu": sw, "w_conc": w_conc, "slots": slots_c}
    info_sat = {"snap": ss, "ranked": ranked_s, "core": {}, "sector_target": w_conc, "slots": slots_s}
    return w_core, w_sat, anchors, info_core, info_sat


# ────────────────────────────── rules ──────────────────────────────


TOL = 1e-6


def check_portfolio(w: pd.Series, sector: pd.Series, mcap: pd.Series, mkt_w: pd.Series, cfg) -> list[str]:
    """w: 종목 비중(합 1), sector/mcap: 종목별 섹터·시총, mkt_w: 시장 섹터 비중. 위반 목록 반환."""
    L = cfg["limits"]
    errs = []
    if abs(w.sum() - 1) > 1e-4:
        errs.append(f"비중 합 {w.sum():.4f} != 1")
    if (w < -TOL).any():
        errs.append("음수 비중(공매도)")
    for c, x in w.items():
        if x > stock_cap(c, cfg) + TOL:
            errs.append(f"종목한도 초과 {c} {x:.2%} > {stock_cap(c, cfg):.0%}")
    caps = sector_caps(mkt_w, cfg)
    by = w.groupby(sector.reindex(w.index)).sum()
    for g, x in by.items():
        if x > caps.get(g, 0) + TOL:
            errs.append(f"섹터한도 초과 {g} {x:.2%} > {caps.get(g, 0):.2%}")
    small = w[mcap.reindex(w.index) < L["smallcap_mcap"]].sum()
    if small > L["smallcap_max"] + TOL:
        errs.append(f"시총 1조 미만 합산 {small:.2%} > {L['smallcap_max']:.0%}")
    return errs


def weekly_turnover(buy_amt: float, sell_amt: float, avg_aum: float) -> float:
    """주간 회전율(%) = (매수총액 + 매도총액) ÷ 주간 평균 운용금액 × 0.5 × 100."""
    return (buy_amt + sell_amt) / avg_aum * 0.5 * 100
