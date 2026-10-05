"""
3차 전략 (사전 등록: preregistration_v3.md)
  저변동성 인핸스드 인덱스 + PEAD 오버레이 틸트

  방향 전환: '코스피200을 이긴다'(불가능) → '통제된 위험 안에서 시장을 따라가되
  저변동성으로 샤프를 올리고, 실적 시즌 PEAD로 얇게 틸트한다'(A+ 운용능력 평가 지향)

  핵심 수정 (진단 반영, 새 데이터 없음 — 이미 받은 가격·재무만 사용):
   ① 리스크 오버레이 제거(변동성 타기팅·레짐 필터) → 드로우다운 −15% 브레이크만
   ② 종목·섹터 상한 초과분을 현금으로 두지 않고 재배분 → 항상 만투
   ③ 반도체를 시장 비중 근처까지 허용(삼성전자 20%, SK하이닉스 12%)
   ④ 저변동성 팩터로 위험 대비 수익 개선 + 포트 베타 0.9~1.0 타기팅

  python run_v3.py          # 실제 데이터
  python run_v3.py --demo   # 합성 데이터로 동작 확인
"""
import argparse
import itertools
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import config as C
from backtest import metrics
from run_backtest import fmt, plot_full, plot_window
from signals import build_events, eligible_mask, prepare_panels
from validate import cscv_pbo, expected_max_sr, implied_n, plots, psr

V3 = dict(
    N_CORE=40,
    CAP={"005930": 0.20, "000660": 0.12}, CAP_NAME=0.05,
    BETA_LO=0.90, BETA_HI=1.00,
    PEAD_WINDOW=20, PEAD_MAX_TILT=0.01,     # 코어 종목 비중에 최대 +1%p 얹기
    DD_TRIGGER=-0.15, DD_EXPOSURE=0.70, DD_RECOVER=-0.08,
    TRADE_BAND=0.005,
    SLIP_LARGE=0.0005,   # 대형주 5bp
    SLIP_SMALL=0.0015,   # 중형주 15bp
    LARGE_LINE=1e12,     # 시총 1조 이상 = 대형주
)
GRID_LOWVOL = [0.3, 0.5, 0.7]     # 저변동성 가중 강도
GRID_TILT = [0.0, 0.15]           # PEAD 틸트 비중
DESIGNATED = (0.5, 0.15)
PRIOR_TRIALS = 36                 # 1차 27 + 2차 9 (누적)


def name_cap(t):
    return V3["CAP"].get(t, V3["CAP_NAME"])


def redistribute(w, total):
    """종목 상한을 지키면서 합이 total이 되도록 반복 재배분 (초과분을 현금으로 두지 않음)"""
    w = {t: v for t, v in w.items() if v > 0}
    if not w:
        return {}
    s = sum(w.values())
    w = {t: v / s * total for t, v in w.items()}
    for _ in range(80):
        over = {t: w[t] - name_cap(t) for t in w if w[t] > name_cap(t) + 1e-12}
        if not over:
            break
        excess = sum(over.values())
        for t in over:
            w[t] = name_cap(t)
        free = {t: v for t, v in w.items() if w[t] < name_cap(t) - 1e-12}
        fs = sum(free.values())
        if fs <= 0:
            break
        for t in free:
            w[t] += excess * w[t] / fs
    return w


class Engine:
    def __init__(self, P, ev, k200):
        self.P = P
        self.dates = P["dates"]
        self.cols = P["cols"]
        self.col_ix = {t: j for j, t in enumerate(self.cols)}
        self.OPEN, self.CLOSE, self.RET = P["open"].values, P["close"].values, P["ret"].values
        self.MCAP, self.VOL = P["mcap"].values, P["vol60"].values
        self.SEC = P["sector"].values
        # 개별 종목 베타 (120일 롤링, 코스피200 대비)
        kr = k200.reindex(self.dates).pct_change()
        R = P["ret"]
        cov = R.rolling(120, min_periods=80).cov(kr)
        var = kr.rolling(120, min_periods=80).var()
        self.BETA = (cov.div(var, axis=0)).values
        kv = k200.reindex(self.dates).ffill()
        self.kv = kv.values
        self.ev_by_t1 = defaultdict(list)
        for e in ev[ev.passed].itertuples(index=False):
            self.ev_by_t1[int(e.t1_idx)].append(e)
        self._elig = {}
        # 시장 섹터 비중 (대회 규정: 시장 비중의 2배까지) — 시점별 근사
        self.mcap_df = P["mcap"]

    def slip(self, i, t):
        return V3["SLIP_LARGE"] if self.MCAP[i, self.col_ix[t]] >= V3["LARGE_LINE"] else V3["SLIP_SMALL"]
    
    def elig(self, i):
        if i not in self._elig:
            self._elig[i] = eligible_mask(self.P, i)
        return self._elig[i]

    def sector_cap(self, i, sec_of):
        """대회 규정: 섹터 비중 시장의 2배, 시장 5% 이하 섹터는 10%까지"""
        mc = self.MCAP[i]
        el = self.elig(i)
        tot = mc[el].sum()
        caps = {}
        for s in set(sec_of.values()):
            js = [self.col_ix[t] for t, ss in sec_of.items() if ss == s]
            share = mc[js].sum() / tot if tot > 0 else 0
            caps[s] = max(2 * share, 0.10 if share <= 0.05 else 2 * share)
        return caps

    def targets(self, i, lowvol, tilt, cutoff=None):
        el = self.elig(i)
        idx = np.where(el)[0]
        if len(idx) < V3["N_CORE"]:
            return {}
        mc, vol, beta = self.MCAP[i], self.VOL[i], self.BETA[i]
        # 코어 종목 선정: 시총 상위 80 중 저변동성 결합 점수로 40
        big = idx[np.argsort(-mc[idx])][:80]
        v = vol[big]
        z_mc = (np.log(mc[big]) - np.log(mc[big]).mean()) / (np.log(mc[big]).std() + 1e-9)
        z_lv = -(v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
        score = (1 - lowvol) * z_mc + lowvol * z_lv
        core = big[np.argsort(-score)][:V3["N_CORE"]]
        sec_of = {self.cols[j]: self.SEC[j] for j in core}

        # 코어 비중: 시총가중과 동일가중을 저변동성 강도로 섞음
        w_mc = mc[core] / mc[core].sum()
        w_eq = np.full(len(core), 1 / len(core))
        raw = (1 - lowvol) * w_mc + lowvol * w_eq
        w = {self.cols[core[k]]: raw[k] for k in range(len(core))}
        w = redistribute(w, 1.0)

        # 섹터 상한 (대회 규정)
        caps = self.sector_cap(i, sec_of)
        for _ in range(20):
            sec_tot = defaultdict(float)
            for t, val in w.items():
                sec_tot[sec_of[t]] += val
            over = [s for s in sec_tot if sec_tot[s] > caps.get(s, 1) + 1e-9]
            if not over:
                break
            for s in over:
                members = [t for t in w if sec_of[t] == s]
                f = caps[s] / sec_tot[s]
                for t in members:
                    w[t] *= f
            w = redistribute(w, 1.0)          # 깎은 만큼 다른 섹터로 재배분

        # 베타 타기팅: 목표 밴드 밖이면 코어를 현금/지수대용과 섞는 대신
        # 저베타↔고베타 종목으로 살짝 재가중 (새 데이터 없이 개별 베타 사용)
        pb = sum(w[t] * (beta[self.col_ix[t]] if np.isfinite(beta[self.col_ix[t]]) else 1.0) for t in w)
        if pb > 0 and (pb < V3["BETA_LO"] or pb > V3["BETA_HI"]):
            target = V3["BETA_LO"] if pb < V3["BETA_LO"] else V3["BETA_HI"]
            bt = np.array([beta[self.col_ix[t]] if np.isfinite(beta[self.col_ix[t]]) else 1.0 for t in w])
            names = list(w)
            wv = np.array([w[t] for t in names])
            tilt_dir = np.sign(target - pb) * (bt - bt.mean())
            wv = np.clip(wv * (1 + 0.5 * tilt_dir), 0, None)
            wv = wv / wv.sum()
            w = redistribute({names[k]: wv[k] for k in range(len(names))}, 1.0)

        # PEAD 오버레이 틸트: 코어 안에 있는 최근 신호 종목 비중을 +up 하고 재정규화
        if tilt > 0:
            active = set()
            for t1 in range(max(0, i - V3["PEAD_WINDOW"] + 1), i + 1):
                for e in self.ev_by_t1.get(t1, []):
                    if cutoff is not None and self.dates[t1] > cutoff:
                        continue
                    if e.ticker in w:
                        active.add(e.ticker)
            if active:
                up = tilt / len(active)
                for t in active:
                    w[t] = min(w[t] + up, name_cap(t))
                w = redistribute(w, 1.0)
        return w

    def run(self, start, end, lowvol, tilt, contest=False):
        d = self.dates
        i0 = int(d.searchsorted(pd.Timestamp(start)))
        i1 = int(d.searchsorted(pd.Timestamp(end), side="right")) - 1
        cutoff = pd.Timestamp(d[i1].year, *C.NO_ENTRY_AFTER) if contest else None
        cash, pos = C.INIT_CAPITAL, {}
        miss = defaultdict(int)
        nav_hist, gross_hist, hhi_hist, npos_hist = [], [], [], []
        pnl = defaultdict(float)
        week_trade, week_navs = defaultdict(float), defaultdict(list)
        trades = []
        peak, dd_mode = C.INIT_CAPITAL, False

        def apply_dd(w):
            return {t: v * (V3["DD_EXPOSURE"] if dd_mode else 1.0) for t, v in w.items()}

        pending = apply_dd(self.targets(i0 - 1, lowvol, tilt, cutoff)) if i0 > 0 else None
        for i in range(i0, i1 + 1):
            wk = tuple(d[i].isocalendar()[:2])
            nav_open = cash + sum(pos.values())
            flows = defaultdict(float)
            new_pos = {}
            touched = set(pos) | (set(pending) if pending else set())
            for t in touched:
                j = self.col_ix[t]
                o, c, r = self.OPEN[i, j], self.CLOSE[i, j], self.RET[i, j]
                v_prev = pos.get(t, 0.0)
                if not (o > 0 and c > 0 and np.isfinite(r)):
                    if v_prev > 0:
                        miss[t] += 1
                        if miss[t] >= 5:
                            cash += v_prev; trades.append((d[i], t, "DELIST", v_prev)); continue
                        new_pos[t] = v_prev * (1 + (r if np.isfinite(r) else 0.0)); pnl[t] += new_pos[t] - v_prev
                    continue
                miss[t] = 0
                v_open = v_prev * o * (1 + r) / c if v_prev > 0 else 0.0
                if pending is not None:
                    tgt = pending.get(t, 0.0) * nav_open
                    diff = tgt - v_open
                    if diff < -V3["TRADE_BAND"] * nav_open or (tgt == 0 and v_open > 0):
                        sell = min(-diff, v_open); cost = sell * (C.SELL_COST + self.slip(i, t))
                        cash += sell - cost; v_open -= sell; flows[t] += sell - cost
                        week_trade[wk] += sell; trades.append((d[i], t, "SELL", sell))
                new_pos[t] = (v_open, c / o, v_prev)
            if pending is not None:
                for t, wt in sorted(pending.items(), key=lambda x: -x[1]):
                    if t not in new_pos or not isinstance(new_pos[t], tuple):
                        continue
                    v_open, ratio, v_prev = new_pos[t]
                    diff = wt * nav_open - v_open
                    if diff > V3["TRADE_BAND"] * nav_open:
                        sl = self.slip(i, t)
                        amt = min(diff, cash / (1 + C.BUY_COST + sl))
                        if amt <= 0:
                            continue
                        cost = amt * (C.BUY_COST + sl)
                        cash -= amt + cost; flows[t] -= amt + cost
                        week_trade[wk] += amt; trades.append((d[i], t, "BUY", amt))
                        new_pos[t] = (v_open + amt, ratio, v_prev)
            pos = {}
            for t, v in new_pos.items():
                if isinstance(v, tuple):
                    v_open, ratio, v_prev = v; vc = v_open * ratio
                    pnl[t] += vc - v_prev + flows[t]
                    if vc > 1:
                        pos[t] = vc
                elif v > 1:
                    pos[t] = v
            nav = cash + sum(pos.values())
            nav_hist.append(nav); gross_hist.append((nav - cash) / nav); npos_hist.append(len(pos))
            hhi_hist.append(sum((v / nav) ** 2 for v in pos.values()))
            week_navs[wk].append(nav); peak = max(peak, nav)
            dd = nav / peak - 1
            dd_mode = True if dd <= V3["DD_TRIGGER"] else (False if dd >= V3["DD_RECOVER"] else dd_mode)
            first_of_week = (i == i0) or (tuple(d[i - 1].isocalendar()[:2]) != wk)
            pending = apply_dd(self.targets(i, lowvol, tilt, cutoff)) if (first_of_week and i < i1) else None

        idx = d[i0:i0 + len(nav_hist)]
        return dict(nav=pd.Series(nav_hist, index=idx, name="nav"),
                    gross=pd.Series(gross_hist, index=idx), hhi=pd.Series(hhi_hist, index=idx),
                    npos=pd.Series(npos_hist, index=idx),
                    pnl=pd.Series(pnl).sort_values(ascending=False),
                    weekly_turnover=pd.Series({k: week_trade[k] / np.mean(v) * 0.5 for k, v in week_navs.items()}),
                    trades=pd.DataFrame(trades, columns=["date", "ticker", "side", "amount"]))


def label(lv, tl):
    return f"LV{int(lv * 100)}_T{int(tl * 100)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--skip-grid", action="store_true")
    args = ap.parse_args()
    if args.demo:
        from demo_data import make_demo
        panels, quarters, k200, sectors = make_demo()
        years, start, end = list(range(2018, 2025)), "2018-01-01", "2025-12-31"
        out = os.path.join(C.OUT_DIR, "demo", "v3")
        prior_paths = [os.path.join(C.OUT_DIR, "demo", "validation", "grid_daily_returns.csv"),
                       os.path.join(C.OUT_DIR, "demo", "v2", "grid_daily_returns.csv")]
    else:
        from data_loader import load_all
        panels, quarters, k200, sectors = load_all()
        years, start, end = C.CONTEST_YEARS, C.BACKTEST_START, C.END
        out = os.path.join(C.OUT_DIR, "v3")
        prior_paths = [os.path.join(C.OUT_DIR, "validation", "grid_daily_returns.csv"),
                       os.path.join(C.OUT_DIR, "v2", "grid_daily_returns.csv")]
    os.makedirs(out, exist_ok=True)

    print("[1/4] 신호 계산")
    P = prepare_panels(panels, sectors)
    ev = build_events(quarters, P)
    eng = Engine(P, ev, k200)
    lv, tl = DESIGNATED

    print("[2/4] 지정 설정 전체기간 백테스트")
    full = eng.run(start, end, lv, tl)
    mf = metrics(full, k200)
    pd.Series(fmt(mf)).to_csv(os.path.join(out, "summary_full.csv"), header=["value"], encoding="utf-8-sig")
    full["nav"].to_csv(os.path.join(out, "full_nav.csv"), encoding="utf-8-sig")
    plot_full(full, k200, out)

    print("[3/4] 연도별 10/1~11/30 대회 모드")
    rows = {}
    for y in years:
        res = eng.run(f"{y}-10-01", f"{y}-11-30", lv, tl, contest=True)
        m = metrics(res, k200)
        rows[y] = {"Strategy": m["Total Return"], "KOSPI200": m["Benchmark (KOSPI200)"], "Excess": m["Excess"],
                   "Sharpe": m["Sharpe"], "MDD": m["MDD"], "Avg Positions": m["Avg Positions"],
                   "Top3 PnL Share": m["Top3 PnL Share"], "Weeks < 5%": m["Weeks < 5%"]}
    tbl = pd.DataFrame(rows).T; tbl.index.name = "Year"
    tbl.to_csv(os.path.join(out, "window_table.csv"), encoding="utf-8-sig")
    plot_window(tbl, out)

    print("\n══════ 3차 전략 전체기간 요약 (지정 설정 LV50_T15) ══════")
    for k, v in fmt(mf).items():
        print(f"  {k:<22}{v}")
    show = tbl.copy()
    for c in ["Strategy", "KOSPI200", "Excess", "MDD", "Top3 PnL Share"]:
        show[c] = (show[c] * 100).map(lambda v: f"{v:6.2f}%" if pd.notna(v) else "   n/a")
    show["Sharpe"] = show["Sharpe"].map(lambda v: f"{v:5.2f}" if pd.notna(v) else "  n/a")
    show["Avg Positions"] = show["Avg Positions"].map(lambda v: f"{v:4.1f}")
    print("\n══════ 10~11월 구간 (연도별) ══════")
    print(show.to_string())
    print(f"\n  초과수익 연도: {(tbl.Excess > 0).sum()} / {len(tbl)}")

    if args.skip_grid:
        return
    print(f"\n[4/4] 사전 등록된 6개 설정 + PBO + DSR (누적 시도 N={PRIOR_TRIALS + 6})")
    rets = {}
    for lv_, tl_ in itertools.product(GRID_LOWVOL, GRID_TILT):
        r = eng.run(start, end, lv_, tl_)
        rets[label(lv_, tl_)] = r["nav"].pct_change().fillna(0.0)
        print(f"  {label(lv_, tl_):<10} 총수익 {r['nav'].iloc[-1] / r['nav'].iloc[0] - 1:8.2%}")
    M = pd.DataFrame(rets)
    M.to_csv(os.path.join(out, "grid_daily_returns.csv"), encoding="utf-8-sig")
    des = label(*DESIGNATED)
    sr = (M.mean() / M.std(ddof=1) * math.sqrt(252)).sort_values(ascending=False)
    pbo, logits, pairs = cscv_pbo(M)

    allM = M
    for pp in prior_paths:
        if os.path.exists(pp):
            allM = allM.join(pd.read_csv(pp, index_col=0, parse_dates=True), how="inner", rsuffix="_p")
    daily_sr = allM.mean() / allM.std(ddof=1)
    var_sr = daily_sr.var(ddof=1)
    n_total = PRIOR_TRIALS + len(GRID_LOWVOL) * len(GRID_TILT)
    sr0 = expected_max_sr(var_sr, n_total)
    p_dsr, sr_d, g3, g4, T = psr(M[des], sr0)
    p0 = psr(M[des], 0.0)[0]
    rho, n_hat = implied_n(allM)
    p_dsr_b = psr(M[des], expected_max_sr(var_sr, n_hat))[0]
    plots(M, des, logits, pairs, pbo, out)

    summ = {"지정 설정": des, "지정 설정 순위 (6개 중)": list(sr.index).index(des) + 1,
            "지정 설정 연환산 샤프": round(sr_d * math.sqrt(252), 3), "PSR (샤프>0 확률)": round(p0, 4),
            "왜도": round(g3, 3), "첨도": round(g4, 3), "표본 길이 T": T,
            "PBO (6개, 참고용)": round(pbo, 4), "Prob[OOS 샤프<0]": round(float((pairs[:, 1] < 0).mean()), 4),
            f"DSR (N={n_total}, 누적)": round(p_dsr, 4), f"SR0 기준선 N={n_total} (연환산)": round(sr0 * math.sqrt(252), 3),
            f"DSR (상관 반영 N={n_hat:.1f})": round(p_dsr_b, 4), "시도 간 평균 상관": round(rho, 3)}
    pd.Series(summ).to_csv(os.path.join(out, "validation_summary.csv"), header=["value"], encoding="utf-8-sig")
    print("\n══════ Backtest Validity (3차) ══════")
    for k, v in summ.items():
        print(f"  {k:<30}{v}")
    print(f"\n결과 저장 위치: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
