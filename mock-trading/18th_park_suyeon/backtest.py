"""박스권 돌파형 52주 신고가 전략 백테스트 (DART 2026-2 / 타임폴리오 Road to Fund Manager 대회용).

규칙 (t일 매매, 신호는 t-1일 종가까지의 정보로만 계산 → 미래참조 없음)
  시장 필터   : KOSPI > MA60 이고 MA60 우상향(MA60[t-1] > MA60[t-6])
  스크리닝 1  : 종가 >= 52주 고가(직전 250일 고가 최대) × 95%, 아직 돌파 전
  스크리닝 2  : 52주 고가가 만들어진 뒤 20~120일 횡보, 그 기간 변동폭 40% 이내
  유동성      : 시총 >= 1,000억, 5일 평균 거래대금 > 30억 (대회 매수가능 조건)
  매수        : t일 장중 52주 고가 돌파 시 STOP 매수(돌파가 체결). 시가가 돌파가 위 = 갭상승 → 제외
  손절/트레일 : 매수가 -8% / 보유 중 최고가 대비 -15%
  1차 익절    : +20%에서 절반 매도, 이후 트레일링 -15% 또는 매수가 이탈 시 전량 매도
  시간 청산   : 20거래일 동안 1차 익절이 없으면 전량 매도
  포트폴리오  : 최대 20종목 × 5%, 동일 섹터 최대 6종목, 시총 1조 미만 최대 6종목
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"


@dataclass
class Params:
    name: str = "A_hard_filter"
    start: str = "2017-01-01"
    end: str | None = None
    lookback: int = 250            # 52주 = 250거래일
    near_high: float = 0.95        # 스크리닝 1
    base_min: int = 20             # 스크리닝 2: 횡보 기간
    base_max: int = 120
    base_range: float = 0.40       # 스크리닝 2: 변동폭
    min_mcap: float = 1_000e8      # 원
    min_tv5: float = 30e8          # 원
    use_market_filter: bool = True
    weight_on: float = 0.05        # 시장 필터 ON일 때 종목당 비중
    weight_off: float = 0.0        # 필터 OFF일 때 (0이면 신규매수 중단 = A안, 0.025면 B안)
    max_positions: int = 20
    max_per_sector: int = 6
    max_small: int = 6             # 시총 1조 미만 종목 수 한도 (≈30%)
    small_cap: float = 1e12
    stop_loss: float = 0.08
    trail: float = 0.15
    tp1: float = 0.20
    tp1_frac: float = 0.5
    time_exit: int = 20
    buy_cost: float = 0.001 + 0.001        # 수수료 10bp + 슬리피지 10bp
    sell_cost: float = 0.001 + 0.002 + 0.001  # 수수료 10bp + 매도세 20bp + 슬리피지 10bp
    rank_by: str = "tight"         # 후보 우선순위: tight(변동폭 작은 순) / mom(6개월 수익률 큰 순)
    filter_mode: str = "ma60_slope"  # ma60_slope(MA60 위+우상향) / ma60(MA60 위) / ma20 / none
    chase_limit: float = 0.07      # 돌파일 종가가 돌파가 대비 이 이상 뛰면 추격 금지
    entry_mode: str = "breakout"   # breakout(돌파일 매수) / near(셋업 통과 즉시 매수, 돌파 대기 없음)


# 최종안: 횡보 10~120일, 시간청산 40일, 시장 필터는 KOSPI > MA60만 확인
FINAL = Params(name="Final", base_min=10, time_exit=40, filter_mode="ma60")
WATCH_NEAR = 0.90   # 관심 종목(1단계) 기준: 52주 최고 종가 90% 이내


@dataclass
class Position:
    code: str
    entry_i: int
    entry_px: float
    units: float
    peak: float
    tp_done: bool = False
    last_px: float = np.nan
    nan_days: int = 0
    units0: float = np.nan      # 최초 매수 수량 (부분 매도 비중 계산용)


def market_filter(kospi: pd.Series, mode: str = "ma60_slope") -> pd.Series:
    if mode == "none":
        return pd.Series(True, index=kospi.index)
    n = 20 if mode == "ma20" else 60
    ma = kospi.rolling(n).mean()
    on = kospi > ma
    return on & (ma > ma.shift(5)) if mode == "ma60_slope" else on


def run(data: dict[str, pd.DataFrame], kospi: pd.Series, sector: pd.Series, p: Params):
    o, h, l, c = (data[k] for k in ("open", "high", "low", "close"))
    mcap, tv = data["mcap"], data["tv"]
    dates = c.index
    codes = c.columns
    O, H, L, C = (x.to_numpy(dtype="float64") for x in (o, h, l, c))
    MC = mcap.reindex(index=dates, columns=codes).to_numpy(dtype="float64")
    TV5 = tv.reindex(index=dates, columns=codes).rolling(5, min_periods=5).mean().to_numpy(dtype="float64")
    H52 = h.rolling(p.lookback, min_periods=p.lookback // 2).max().to_numpy(dtype="float64")  # 당일 포함, t-1 행을 t의 기준으로 사용
    MOM = (c / c.shift(120) - 1).to_numpy(dtype="float64")
    mkt = market_filter(kospi.reindex(dates).ffill()).to_numpy()
    sec = sector.reindex(codes).fillna("NA").to_numpy()
    is_common = np.array([str(x).endswith("0") for x in codes])

    start_i = int(np.searchsorted(dates, pd.Timestamp(p.start)))
    end_i = len(dates) if p.end is None else int(np.searchsorted(dates, pd.Timestamp(p.end), side="right"))

    cash, positions = 1.0, {}
    nav_hist, trades, turnover_log, cand_log = [], [], [], []

    def sell(pos: Position, frac: float, px: float, i: int, reason: str):
        nonlocal cash
        units = pos.units * frac
        proceeds = units * px * (1 - p.sell_cost)
        cash += proceeds
        pos.units -= units
        trades.append(dict(code=pos.code, entry_date=dates[pos.entry_i], exit_date=dates[i], entry_px=pos.entry_px,
                           exit_px=px, frac=frac, ret=px / pos.entry_px * (1 - p.sell_cost) / (1 + p.buy_cost) - 1,
                           days=i - pos.entry_i, reason=reason, value=units * px, w=units / pos.units0))
        turnover_log.append((dates[i], units * px))

    for i in range(start_i, end_i):
        # ---------- 1) 보유 종목 청산 ----------
        for code in list(positions):
            pos = positions[code]
            j = codes.get_loc(code)
            op, hi, lo, cl = O[i, j], H[i, j], L[i, j], C[i, j]
            if np.isnan(cl):
                pos.nan_days += 1
                if pos.nan_days >= 5:            # 거래정지/상장폐지 → 마지막 가격으로 정리
                    sell(pos, 1.0, pos.last_px, i, "delisted/halt")
                    del positions[code]
                continue
            pos.nan_days = 0
            op = cl if np.isnan(op) else op
            hi = cl if np.isnan(hi) else hi
            lo = cl if np.isnan(lo) else lo
            if i > pos.entry_i:
                stop = max(pos.entry_px * (1 - p.stop_loss), pos.peak * (1 - p.trail))
                if pos.tp_done:
                    stop = max(pos.entry_px, pos.peak * (1 - p.trail))
                if op <= stop:
                    sell(pos, 1.0, op, i, "stop_gap"); del positions[code]; continue
                if lo <= stop:
                    reason = "trail" if pos.peak * (1 - p.trail) >= stop - 1e-9 else ("breakeven" if pos.tp_done else "stop")
                    sell(pos, 1.0, stop, i, reason); del positions[code]; continue
                if not pos.tp_done and hi >= pos.entry_px * (1 + p.tp1):
                    sell(pos, p.tp1_frac, max(op, pos.entry_px * (1 + p.tp1)), i, "tp1")
                    pos.tp_done = True
            pos.peak = max(pos.peak, hi)
            pos.last_px = cl
            if not pos.tp_done and i - pos.entry_i >= p.time_exit:
                sell(pos, 1.0, cl, i, "time"); del positions[code]; continue

        # ---------- 2) 신규 진입 (t-1 정보로 후보 선정, t일 장중 돌파 시 체결) ----------
        k = i - 1
        nav_prev = nav_hist[-1][1] if nav_hist else 1.0
        w = p.weight_on if (not p.use_market_filter or mkt[k]) else p.weight_off
        free = p.max_positions - len(positions)
        if w > 0 and free > 0 and k >= p.lookback:
            pivot = H52[k]
            ok = (is_common & (C[k] >= pivot * p.near_high) & (C[k] < pivot)
                  & (MC[k] >= p.min_mcap) & (TV5[k] > p.min_tv5) & ~np.isnan(pivot))
            idx = np.flatnonzero(ok)
            cands = []
            for j in idx:
                if codes[j] in positions:
                    continue
                win_h = H[k - p.lookback + 1:k + 1, j]
                if np.isnan(win_h).all():
                    continue
                peak_pos = int(np.nanargmax(win_h))
                base_len = len(win_h) - 1 - peak_pos
                if not (p.base_min <= base_len <= p.base_max):
                    continue
                base_low = np.nanmin(L[k - base_len:k + 1, j])
                rng = 1 - base_low / pivot[j]
                if rng > p.base_range:
                    continue
                cands.append((j, rng, MOM[k, j]))
            cand_log.append((dates[i], len(cands), bool(mkt[k])))
            cands.sort(key=(lambda x: x[1]) if p.rank_by == "tight" else (lambda x: -np.nan_to_num(x[2])))
            n_sec = pd.Series([sec[codes.get_loc(cd)] for cd in positions]).value_counts().to_dict()
            n_small = sum(MC[k, codes.get_loc(cd)] < p.small_cap for cd in positions)
            placed = 0
            for j, rng, _ in cands:
                if placed >= free:
                    break
                s = sec[j]
                small = MC[k, j] < p.small_cap
                if n_sec.get(s, 0) >= p.max_per_sector and s != "NA":
                    continue
                if small and n_small >= p.max_small:
                    continue
                placed += 1                         # STOP 매수 주문을 건 종목 수
                op, hi = O[i, j], H[i, j]
                trig = pivot[j]
                if np.isnan(op) or np.isnan(hi) or op > trig or hi <= trig:
                    continue                        # 갭상승 제외 / 미돌파
                value = min(w * nav_prev, cash)
                if value <= 0:
                    break
                px = trig * (1 + 1e-9)
                units = value * (1 - p.buy_cost) / px
                cash -= value
                positions[codes[j]] = Position(codes[j], i, px, units, peak=max(px, hi), last_px=C[i, j], units0=units)
                n_sec[s] = n_sec.get(s, 0) + 1
                n_small += int(small)
                turnover_log.append((dates[i], value))

        # ---------- 3) 평가 ----------
        mv = 0.0
        for code, pos in positions.items():
            px = C[i, codes.get_loc(code)]
            mv += pos.units * (pos.last_px if np.isnan(px) else px)
        nav_hist.append((dates[i], cash + mv, len(positions), mv / (cash + mv)))

    nav = pd.DataFrame(nav_hist, columns=["date", "nav", "n_pos", "exposure"]).set_index("date")
    tr = pd.DataFrame(trades)
    to = pd.DataFrame(turnover_log, columns=["date", "value"]).set_index("date")["value"] if turnover_log else pd.Series(dtype=float)
    cl = pd.DataFrame(cand_log, columns=["date", "n_cand", "mkt_on"]).set_index("date")
    return nav, tr, to, cl


def run_close(data: dict[str, pd.DataFrame], kospi: pd.Series, sector: pd.Series, p: Params):
    """종가 기준 버전: 장 마감 후 판단하는 실전 운용과 동일 (고가/저가/시가 데이터 불필요).

    t일 종가가 직전 52주 최고 종가(t-1까지)를 돌파하고, 셋업 조건(t-1 기준)을 만족하면 t일 종가에 매수.
    돌파일 종가가 돌파가 대비 chase_limit 초과로 뛰었으면 추격 금지(장중 '갭상승 제외' 규칙을 종가 기준으로 대체).
    손절·트레일링·익절·시간청산 모두 종가로 판단해 당일 종가에 매도.
    """
    c = data["close"]
    dates, codes = c.index, c.columns
    C = c.to_numpy(dtype="float64")
    MC = data["mcap"].reindex(index=dates, columns=codes).to_numpy(dtype="float64")
    TV5 = data["tv5"].reindex(index=dates, columns=codes).to_numpy(dtype="float64")
    HC = c.rolling(p.lookback, min_periods=p.lookback // 2).max().to_numpy(dtype="float64")
    mkt = market_filter(kospi.reindex(dates).ffill(), p.filter_mode).to_numpy()
    sec = sector.reindex(codes).fillna("NA").to_numpy()
    is_common = np.array([str(x).endswith("0") for x in codes])
    col = {cd: j for j, cd in enumerate(codes)}
    chase_limit = p.chase_limit

    start_i = int(np.searchsorted(dates, pd.Timestamp(p.start)))
    end_i = len(dates) if p.end is None else int(np.searchsorted(dates, pd.Timestamp(p.end), side="right"))
    cash, positions = 1.0, {}
    nav_hist, trades, turnover_log, cand_log = [], [], [], []

    def sell(pos, frac, px, i, reason):
        nonlocal cash
        units = pos.units * frac
        cash += units * px * (1 - p.sell_cost)
        pos.units -= units
        trades.append(dict(code=pos.code, entry_date=dates[pos.entry_i], exit_date=dates[i], entry_px=pos.entry_px,
                           exit_px=px, frac=frac, ret=px / pos.entry_px * (1 - p.sell_cost) / (1 + p.buy_cost) - 1,
                           days=i - pos.entry_i, reason=reason, value=units * px, w=units / pos.units0))
        turnover_log.append((dates[i], units * px))

    for i in range(start_i, end_i):
        k = i - 1
        # 1) 청산 (당일 종가 기준)
        for code in list(positions):
            pos = positions[code]
            cl = C[i, col[code]]
            if np.isnan(cl):
                pos.nan_days += 1
                if pos.nan_days >= 5:
                    sell(pos, 1.0, pos.last_px, i, "delisted/halt"); del positions[code]
                continue
            pos.nan_days = 0
            pos.last_px = cl
            if pos.tp_done:
                stop = max(pos.entry_px, pos.peak * (1 - p.trail))
            else:
                stop = max(pos.entry_px * (1 - p.stop_loss), pos.peak * (1 - p.trail))
            if cl <= stop:
                reason = "breakeven" if pos.tp_done and stop == pos.entry_px else (
                    "trail" if stop == pos.peak * (1 - p.trail) else "stop")
                sell(pos, 1.0, cl, i, reason); del positions[code]; continue
            if not pos.tp_done and cl >= pos.entry_px * (1 + p.tp1):
                sell(pos, p.tp1_frac, cl, i, "tp1"); pos.tp_done = True
            pos.peak = max(pos.peak, cl)
            if not pos.tp_done and i - pos.entry_i >= p.time_exit:
                sell(pos, 1.0, cl, i, "time"); del positions[code]

        # 2) 신규 진입: 셋업은 t-1 기준, 돌파 확인은 t일 종가
        nav_prev = nav_hist[-1][1] if nav_hist else 1.0
        w = p.weight_on if (not p.use_market_filter or mkt[k]) else p.weight_off
        free = p.max_positions - len(positions)
        n_cand = 0
        if w > 0 and free > 0 and k >= p.lookback:
            pivot = HC[k]
            trigger = (C[i] > pivot) if p.entry_mode == "breakout" else (C[i] >= pivot * p.near_high)
            ok = (is_common & (C[k] >= pivot * p.near_high) & (C[k] < pivot) & trigger
                  & (C[i] <= pivot * (1 + chase_limit)) & (MC[k] >= p.min_mcap) & (TV5[k] > p.min_tv5))
            cands = []
            for j in np.flatnonzero(ok):
                if codes[j] in positions:
                    continue
                win = C[k - p.lookback + 1:k + 1, j]
                peak_pos = int(np.nanargmax(win))
                base_len = len(win) - 1 - peak_pos
                if not (p.base_min <= base_len <= p.base_max):
                    continue
                rng = 1 - np.nanmin(win[peak_pos:]) / pivot[j]
                if rng > p.base_range:
                    continue
                cands.append((j, rng))
            n_cand = len(cands)
            cands.sort(key=lambda x: x[1])
            n_sec = pd.Series([sec[col[cd]] for cd in positions]).value_counts().to_dict()
            n_small = sum(MC[k, col[cd]] < p.small_cap for cd in positions)
            for j, _ in cands:
                if len(positions) >= p.max_positions:
                    break
                s, small = sec[j], MC[k, j] < p.small_cap
                if (s != "NA" and n_sec.get(s, 0) >= p.max_per_sector) or (small and n_small >= p.max_small):
                    continue
                value = min(w * nav_prev, cash)
                if value <= 1e-9:
                    break
                px = C[i, j]
                cash -= value
                u = value * (1 - p.buy_cost) / px
                positions[codes[j]] = Position(codes[j], i, px, u, peak=px, last_px=px, units0=u)
                n_sec[s] = n_sec.get(s, 0) + 1
                n_small += int(small)
                turnover_log.append((dates[i], value))
        cand_log.append((dates[i], n_cand, bool(mkt[k])))

        mv = sum(pos.units * (C[i, col[cd]] if not np.isnan(C[i, col[cd]]) else pos.last_px) for cd, pos in positions.items())
        nav_hist.append((dates[i], cash + mv, len(positions), mv / (cash + mv)))

    nav = pd.DataFrame(nav_hist, columns=["date", "nav", "n_pos", "exposure"]).set_index("date")
    tr = pd.DataFrame(trades)
    to = pd.DataFrame(turnover_log, columns=["date", "value"]).set_index("date")["value"]
    cl = pd.DataFrame(cand_log, columns=["date", "n_cand", "mkt_on"]).set_index("date")
    return nav, tr, to, cl


def position_returns(tr: pd.DataFrame) -> pd.Series:
    """포지션 단위 수익률 = 각 매도분 수익률을 최초 수량 대비 비중(w)으로 가중합."""
    return (tr["ret"] * tr["w"]).groupby([tr["code"], tr["entry_date"]]).sum() / tr["w"].groupby([tr["code"], tr["entry_date"]]).sum()


def stats(nav: pd.DataFrame, tr: pd.DataFrame, to: pd.Series, bench: pd.Series, rf: float = 0.03) -> dict:
    r = nav["nav"].pct_change().dropna()
    yrs = len(r) / 252
    cagr = nav["nav"].iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - rf) / vol if vol > 0 else np.nan
    dd = nav["nav"] / nav["nav"].cummax() - 1
    b = bench.reindex(nav.index).ffill()
    br = b.pct_change().dropna()
    bdd = b / b.cummax() - 1
    # 주간 회전율 = (매수+매도)/평균순자산 × 0.5 (대회 산식)
    wk_to = (to.resample("W-FRI").sum() / nav["nav"].resample("W-FRI").mean() * 0.5).reindex(
        nav["nav"].resample("W-FRI").mean().index).fillna(0)
    full_exit = tr[tr["frac"] == 1.0] if len(tr) else tr
    return dict(
        CAGR=cagr, Vol=vol, Sharpe=sharpe, MDD=dd.min(),
        KOSPI_CAGR=(b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1,
        KOSPI_Sharpe=(br.mean() * 252 - rf) / (br.std() * np.sqrt(252)), KOSPI_MDD=bdd.min(),
        n_trades=int(len(full_exit)), win_rate=float((position_returns(tr) > 0).mean()) if len(tr) else np.nan,
        avg_ret=float(position_returns(tr).mean()) if len(tr) else np.nan,
        avg_days=float(full_exit["days"].mean()) if len(full_exit) else np.nan,
        avg_exposure=float(nav["exposure"].mean()), avg_positions=float(nav["n_pos"].mean()),
        weekly_turnover_mean=float(wk_to.mean()), weeks_below_5pct=float((wk_to < 0.05).mean()),
    )


def oct_nov(nav: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    rows = []
    for y in sorted(set(nav.index.year)):
        seg = nav.loc[f"{y}-10-01":f"{y}-11-30", "nav"]
        if len(seg) < 20:
            continue
        prev = nav.loc[:f"{y}-09-30", "nav"]
        base = prev.iloc[-1] if len(prev) else seg.iloc[0]
        b = bench.loc[:f"{y}-11-30"]
        bprev = b.loc[:f"{y}-09-30"].iloc[-1]
        r = seg.pct_change().dropna()
        rows.append(dict(year=y, strategy=seg.iloc[-1] / base - 1, kospi=b.iloc[-1] / bprev - 1,
                         vol=r.std() * np.sqrt(252), mdd=(seg / seg.cummax() - 1).min()))
    return pd.DataFrame(rows).set_index("year")


def save(p: Params, nav, tr, to, cl, bench):
    d = OUT / p.name
    d.mkdir(parents=True, exist_ok=True)
    nav.to_csv(d / "nav.csv")
    tr.to_csv(d / "trades.csv", index=False)
    cl.to_csv(d / "candidates.csv")
    st = stats(nav, tr, to, bench)
    on = oct_nov(nav, bench)
    on.to_csv(d / "oct_nov.csv")
    (d / "stats.json").write_text(json.dumps({"params": asdict(p), "stats": st}, ensure_ascii=False, indent=2, default=str),
                                  encoding="utf-8")
    return st, on
