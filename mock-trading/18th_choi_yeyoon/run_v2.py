"""
2차 전략 (사전 등록: preregistration_v2.md)
  분산형 대형주 코어 + 전술 틸트 (리더 섹터 모멘텀 · PEAD · 단기 리버설)

  python run_v2.py          # 실제 데이터
  python run_v2.py --demo   # 합성 데이터로 동작 확인
결과: results/v2/
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

# ─── 사전 등록된 고정값 (문헌 기본값 / 대회 규정 버퍼) ─────────────
V2 = dict(
    N_CORE=25,                  # 코어: 시가총액 상위 25종목, 상한 적용 시총가중
    CAP={"005930": 0.10, "000660": 0.08}, CAP_NAME=0.05,
    SECTOR_CAP=0.35, SMALL_CAP=0.20,
    MOM_SKIP=5, MOM_TOP_SECTORS=3, MOM_NAMES=10,          # Jegadeesh & Titman (1993)
    PEAD_WINDOW=20, PEAD_NAME_W=0.025,                    # 1차 신호 그대로, 20거래일 보유
    REV_LOOKBACK=5, REV_NAMES=5,                          # Lehmann (1990), 1주 보유
    VOL_TARGET=0.20, REGIME_EXPOSURE=0.80,
    DD_TRIGGER=-0.07, DD_EXPOSURE=0.60, DD_RECOVER=-0.04,
    TRADE_BAND=0.005,                                     # 목표와 0.5%p 이상 차이날 때만 거래
)
GRID_CORE = [0.5, 0.6, 0.7]          # 코어 비중
GRID_MOM = [63, 126, 252]            # 모멘텀 기간 (3·6·12개월)
DESIGNATED = (0.6, 126)
PRIOR_TRIALS = 27                    # 1차 시도 수 (DSR의 N에 누적)


def name_cap(t):
    return V2["CAP"].get(t, V2["CAP_NAME"])


def cap_weights(raw, total):
    """시총 비례 가중을 종목 상한에 맞춰 잘라내고, 남는 비중은 나머지에 비례 배분"""
    w = {t: v for t, v in raw.items() if v > 0}
    if not w or total <= 0:
        return {}
    s = sum(w.values())
    w = {t: v / s * total for t, v in w.items()}
    fixed = set()
    for _ in range(60):
        over = [t for t, v in w.items() if t not in fixed and v > name_cap(t) + 1e-12]
        if not over:
            break
        excess = sum(w[t] - name_cap(t) for t in over)
        for t in over:
            w[t] = name_cap(t)
            fixed.add(t)
        free = {t: v for t, v in w.items() if t not in fixed}
        fs = sum(free.values())
        if fs <= 0:
            break
        for t, v in free.items():
            w[t] = v + excess * v / fs
    return w


class Engine:
    def __init__(self, P, ev, k200):
        self.P = P
        self.dates = P["dates"]
        self.cols = P["cols"]
        self.col_ix = {t: j for j, t in enumerate(self.cols)}
        self.OPEN, self.CLOSE, self.RET = P["open"].values, P["close"].values, P["ret"].values
        self.MCAP = P["mcap"].values
        self.SEC = P["sector"].values
        self.CUM = np.log1p(P["ret"].fillna(0).clip(lower=-0.99)).cumsum().values
        self.MA60 = P["close"].rolling(60, min_periods=40).mean().values
        kv = k200.reindex(self.dates).ffill()
        self.kv, self.k_ma20 = kv.values, kv.rolling(20).mean().values
        kvol = kv.pct_change().rolling(20).std()
        self.kvol, self.kvol_hi = kvol.values, kvol.rolling(252, min_periods=120).quantile(0.8).values
        self.ev_by_t1 = defaultdict(list)
        for e in ev[ev.passed].itertuples(index=False):
            self.ev_by_t1[int(e.t1_idx)].append(e)
        self._elig = {}

    def elig(self, i):
        if i not in self._elig:
            self._elig[i] = eligible_mask(self.P, i)
        return self._elig[i]

    # ── 목표 비중 계산 (i일 종가 기준) ─────────────────────────
    def targets(self, i, core_w, mom_l, pead_cutoff=None):
        el = self.elig(i)
        idx = np.where(el)[0]
        if len(idx) < V2["N_CORE"]:
            return {}
        mc = self.MCAP[i]
        rest = 1 - core_w
        mom_w, pead_w, rev_w = rest * 0.5, rest * 0.25, rest * 0.25
        w = defaultdict(float)
        unused = 0.0

        # ② 리더 섹터 모멘텀
        k = V2["MOM_SKIP"]
        top_secs = []
        if i - k - mom_l >= 0:
            r = np.exp(self.CUM[i - k] - self.CUM[i - k - mom_l]) - 1
            df = pd.DataFrame({"sec": self.SEC[idx], "r": r[idx], "j": idx}).dropna()
            sec_r = df.groupby("sec")["r"].mean()
            top_secs = list(sec_r[df.groupby("sec").size() >= 3].sort_values(ascending=False).index[:V2["MOM_TOP_SECTORS"]])
            pick = df[df.sec.isin(top_secs)].sort_values("r", ascending=False).head(V2["MOM_NAMES"])
            if len(pick):
                for j in pick.j:
                    w[self.cols[j]] += mom_w / len(pick)
            else:
                unused += mom_w
        else:
            unused += mom_w

        # ④ 단기 리버설 (리더 섹터 안에서 1주 급락 + 60일선 위)
        if top_secs and i - V2["REV_LOOKBACK"] >= 0:
            r5 = np.exp(self.CUM[i] - self.CUM[i - V2["REV_LOOKBACK"]]) - 1
            df = pd.DataFrame({"sec": self.SEC[idx], "r5": r5[idx], "j": idx,
                               "up": self.CLOSE[i, idx] > self.MA60[i, idx]})
            df = df[df.sec.isin(top_secs)]
            df["rel"] = df.r5 - df.groupby("sec").r5.transform("mean")
            pick = df[df.up].sort_values("rel").head(V2["REV_NAMES"])
            if len(pick):
                for j in pick.j:
                    w[self.cols[j]] += rev_w / len(pick)
            else:
                unused += rev_w
        else:
            unused += rev_w

        # ③ PEAD 틸트 (최근 20거래일 안에 신호가 확정된 종목)
        active = []
        for t1 in range(max(0, i - V2["PEAD_WINDOW"] + 1), i + 1):
            for e in self.ev_by_t1.get(t1, []):
                if pead_cutoff is not None and self.dates[t1] > pead_cutoff:
                    continue
                j = self.col_ix.get(e.ticker)
                if j is not None and el[j]:
                    active.append(e.ticker)
        active = list(dict.fromkeys(active))
        if active:
            per = min(V2["PEAD_NAME_W"], pead_w / len(active))
            for t in active:
                w[t] += per
            unused += pead_w - per * len(active)
        else:
            unused += pead_w

        # ① 코어 (남은 슬리브 비중까지 흡수 → 현금 과다 방지)
        order = idx[np.argsort(-mc[idx])][:V2["N_CORE"]]
        for t, v in cap_weights({self.cols[j]: mc[j] for j in order}, core_w + unused).items():
            w[t] += v

        # 합산 후 대회 한도 (초과분은 현금)
        for t in list(w):
            w[t] = min(w[t], name_cap(t))
        sec_tot = defaultdict(float)
        for t, v in w.items():
            sec_tot[self.SEC[self.col_ix[t]]] += v
        for s, tot in sec_tot.items():
            if tot > V2["SECTOR_CAP"]:
                f = V2["SECTOR_CAP"] / tot
                for t in w:
                    if self.SEC[self.col_ix[t]] == s:
                        w[t] *= f
        small = [t for t in w if mc[self.col_ix[t]] < C.SMALLCAP_LINE]
        st = sum(w[t] for t in small)
        if st > V2["SMALL_CAP"]:
            for t in small:
                w[t] *= V2["SMALL_CAP"] / st
        return dict(w)

    # ── 시뮬레이션 ─────────────────────────────────────────
    def run(self, start, end, core_w, mom_l, contest=False):
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

        def overlay(i, w):
            g = 1.0
            if len(nav_hist) >= 20:
                rr = pd.Series(nav_hist[-61:]).pct_change().dropna()
                ga = max(np.mean(gross_hist[-60:]), 0.3)
                vf = rr.std() * math.sqrt(252) / ga
                if vf > 0:
                    g = min(g, V2["VOL_TARGET"] / vf)
            if self.kv[i] < self.k_ma20[i] and self.kvol[i] > self.kvol_hi[i]:
                g = min(g, V2["REGIME_EXPOSURE"])
            if dd_mode:
                g = min(g, V2["DD_EXPOSURE"])
            return {t: v * g for t, v in w.items()}

        pending = overlay(i0 - 1, self.targets(i0 - 1, core_w, mom_l, cutoff)) if i0 > 0 else None
        for i in range(i0, i1 + 1):
            wk = tuple(d[i].isocalendar()[:2])
            nav_open = cash + sum(pos.values())
            flows = defaultdict(float)
            touched = set(pos) | (set(pending) if pending else set())
            new_pos = {}
            for t in touched:
                j = self.col_ix[t]
                o, c, r = self.OPEN[i, j], self.CLOSE[i, j], self.RET[i, j]
                v_prev = pos.get(t, 0.0)
                tradable = o > 0 and c > 0 and np.isfinite(r)
                if not tradable:
                    if v_prev > 0:
                        miss[t] += 1
                        if miss[t] >= 5:                      # 상폐 가정
                            cash += v_prev
                            trades.append((d[i], t, "DELIST", v_prev))
                            continue
                        new_pos[t] = v_prev * (1 + (r if np.isfinite(r) else 0.0))
                        pnl[t] += new_pos[t] - v_prev
                    continue
                miss[t] = 0
                v_open = v_prev * o * (1 + r) / c if v_prev > 0 else 0.0
                if pending is not None:
                    tgt = pending.get(t, 0.0) * nav_open
                    diff = tgt - v_open
                    if diff < -V2["TRADE_BAND"] * nav_open or (tgt == 0 and v_open > 0):
                        sell = min(-diff, v_open)
                        cost = sell * C.SELL_COST
                        cash += sell - cost
                        v_open -= sell
                        flows[t] += sell - cost
                        week_trade[wk] += sell
                        trades.append((d[i], t, "SELL", sell))
                new_pos[t] = (v_open, c / o, v_prev)
            if pending is not None:                              # 매수는 매도 후
                for t, wt in sorted(pending.items(), key=lambda x: -x[1]):
                    if t not in new_pos or not isinstance(new_pos[t], tuple):
                        continue
                    v_open, ratio, v_prev = new_pos[t]
                    diff = wt * nav_open - v_open
                    if diff > V2["TRADE_BAND"] * nav_open:
                        amt = min(diff, cash / (1 + C.BUY_COST))
                        if amt <= 0:
                            continue
                        cost = amt * C.BUY_COST
                        cash -= amt + cost
                        flows[t] -= amt + cost
                        week_trade[wk] += amt
                        trades.append((d[i], t, "BUY", amt))
                        new_pos[t] = (v_open + amt, ratio, v_prev)
            pos = {}
            for t, v in new_pos.items():
                if isinstance(v, tuple):
                    v_open, ratio, v_prev = v
                    vc = v_open * ratio
                    pnl[t] += vc - v_prev + flows[t]
                    if vc > 1:
                        pos[t] = vc
                elif v > 1:
                    pos[t] = v
            nav = cash + sum(pos.values())
            nav_hist.append(nav)
            gross_hist.append((nav - cash) / nav)
            npos_hist.append(len(pos))
            hhi_hist.append(sum((v / nav) ** 2 for v in pos.values()))
            week_navs[wk].append(nav)
            peak = max(peak, nav)
            dd = nav / peak - 1
            dd_mode = True if dd <= V2["DD_TRIGGER"] else (False if dd >= V2["DD_RECOVER"] else dd_mode)

            # 매주 첫 거래일 종가에 목표 계산 → 다음 날 시가 체결
            first_of_week = (i == i0) or (tuple(d[i - 1].isocalendar()[:2]) != wk)
            pending = overlay(i, self.targets(i, core_w, mom_l, cutoff)) if (first_of_week and i < i1) else None

        idx = d[i0:i0 + len(nav_hist)]
        return dict(nav=pd.Series(nav_hist, index=idx, name="nav"),
                    gross=pd.Series(gross_hist, index=idx), hhi=pd.Series(hhi_hist, index=idx),
                    npos=pd.Series(npos_hist, index=idx),
                    pnl=pd.Series(pnl).sort_values(ascending=False),
                    weekly_turnover=pd.Series({k: week_trade[k] / np.mean(v) * 0.5 for k, v in week_navs.items()}),
                    trades=pd.DataFrame(trades, columns=["date", "ticker", "side", "amount"]))


def label(core, mom):
    return f"C{int(core * 100)}_M{mom}"


# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--skip-grid", action="store_true", help="9개 격자·PBO·DSR 생략")
    args = ap.parse_args()
    if args.demo:
        from demo_data import make_demo
        panels, quarters, k200, sectors = make_demo()
        years, start, end = list(range(2017, 2025)), "2017-01-01", "2025-12-31"
        out = os.path.join(C.OUT_DIR, "demo", "v2")
        prior_path = os.path.join(C.OUT_DIR, "demo", "validation", "grid_daily_returns.csv")
    else:
        from data_loader import load_all
        panels, quarters, k200, sectors = load_all()
        years, start, end = C.CONTEST_YEARS, C.BACKTEST_START, C.END
        out = os.path.join(C.OUT_DIR, "v2")
        prior_path = os.path.join(C.OUT_DIR, "validation", "grid_daily_returns.csv")
    os.makedirs(out, exist_ok=True)

    print("[1/4] 신호 계산")
    P = prepare_panels(panels, sectors)
    ev = build_events(quarters, P)
    eng = Engine(P, ev, k200)

    print("[2/4] 지정 설정 전체기간 백테스트")
    core, mom = DESIGNATED
    full = eng.run(start, end, core, mom)
    mf = metrics(full, k200)
    pd.Series(fmt(mf)).to_csv(os.path.join(out, "summary_full.csv"), header=["value"], encoding="utf-8-sig")
    full["nav"].to_csv(os.path.join(out, "full_nav.csv"), encoding="utf-8-sig")
    full["trades"].to_csv(os.path.join(out, "full_trades.csv"), index=False, encoding="utf-8-sig")
    plot_full(full, k200, out)

    print("[3/4] 연도별 10/1~11/30 대회 모드")
    rows = {}
    for y in years:
        res = eng.run(f"{y}-10-01", f"{y}-11-30", core, mom, contest=True)
        m = metrics(res, k200)
        rows[y] = {"Strategy": m["Total Return"], "KOSPI200": m["Benchmark (KOSPI200)"], "Excess": m["Excess"],
                   "Sharpe": m["Sharpe"], "MDD": m["MDD"], "Avg Positions": m["Avg Positions"],
                   "Top3 PnL Share": m["Top3 PnL Share"], "Weeks < 5%": m["Weeks < 5%"]}
    tbl = pd.DataFrame(rows).T
    tbl.index.name = "Year"
    tbl.to_csv(os.path.join(out, "window_table.csv"), encoding="utf-8-sig")
    plot_window(tbl, out)

    print("\n══════ 2차 전략 전체기간 요약 (지정 설정 C60_M126) ══════")
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
    print(f"\n[4/4] 사전 등록된 9개 설정 + PBO + DSR (누적 시도 N={PRIOR_TRIALS + 9})")
    rets = {}
    for c_, m_ in itertools.product(GRID_CORE, GRID_MOM):
        r = eng.run(start, end, c_, m_)
        rets[label(c_, m_)] = r["nav"].pct_change().fillna(0.0)
        print(f"  {label(c_, m_):<10} 총수익 {r['nav'].iloc[-1] / r['nav'].iloc[0] - 1:8.2%}")
    M = pd.DataFrame(rets)
    M.to_csv(os.path.join(out, "grid_daily_returns.csv"), encoding="utf-8-sig")
    des = label(*DESIGNATED)
    sr = (M.mean() / M.std(ddof=1) * math.sqrt(252)).sort_values(ascending=False)
    pbo, logits, pairs = cscv_pbo(M)

    # DSR: 1차 27개 + 2차 9개의 샤프 분산, N=36
    allM = M
    if os.path.exists(prior_path):
        prior = pd.read_csv(prior_path, index_col=0, parse_dates=True)
        allM = M.join(prior, how="inner")
        print(f"  1차 시도 {prior.shape[1]}개의 수익률을 DSR 분산 계산에 포함")
    else:
        print("  ※ 1차 validate.py 결과가 없어 2차 9개로만 분산 계산 (N은 36으로 유지)")
    daily_sr = allM.mean() / allM.std(ddof=1)
    var_sr = daily_sr.var(ddof=1)
    n_total = PRIOR_TRIALS + len(GRID_CORE) * len(GRID_MOM)
    sr0 = expected_max_sr(var_sr, n_total)
    p_dsr, sr_d, g3, g4, T = psr(M[des], sr0)
    p0 = psr(M[des], 0.0)[0]
    rho, n_hat = implied_n(allM)
    sr0b = expected_max_sr(var_sr, n_hat)
    p_dsr_b = psr(M[des], sr0b)[0]
    plots(M, des, logits, pairs, pbo, out)

    summ = {"지정 설정": des, "지정 설정 순위 (9개 중)": list(sr.index).index(des) + 1,
            "지정 설정 연환산 샤프": round(sr_d * math.sqrt(252), 3), "PSR (샤프>0 확률)": round(p0, 4),
            "왜도": round(g3, 3), "첨도": round(g4, 3), "표본 길이 T": T,
            "PBO (9개, 참고용)": round(pbo, 4), "Prob[OOS 샤프<0]": round(float((pairs[:, 1] < 0).mean()), 4),
            f"DSR (N={n_total}, 누적)": round(p_dsr, 4), f"SR0 기준선 N={n_total} (연환산)": round(sr0 * math.sqrt(252), 3),
            f"DSR (상관 반영 N={n_hat:.1f})": round(p_dsr_b, 4), "시도 간 평균 상관": round(rho, 3)}
    pd.Series(summ).to_csv(os.path.join(out, "validation_summary.csv"), header=["value"], encoding="utf-8-sig")
    print("\n══════ Backtest Validity (2차) ══════")
    for k, v in summ.items():
        print(f"  {k:<30}{v}")
    print("  ※ PBO는 설정이 9개뿐이라 해상도가 낮음 (논문: N이 10보다 충분히 커야 정밀) → DSR을 주 지표로")
    print(f"\n결과 저장 위치: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
