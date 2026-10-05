"""
과최적화 검증 (Backtest Validity)
  1) 사전 등록된 27개 설정을 전부 백테스트 (보유기간 × 상위비율 × 손절)
  2) PBO  — Bailey, Borwein, López de Prado & Zhu (2015), CSCV 방식
  3) DSR  — Bailey & López de Prado (2014), 시도 횟수·왜도·첨도·표본 길이 반영

  python validate.py          # 실제 데이터
  python validate.py --demo   # 합성 데이터로 동작 확인
결과: results/validation/
"""
import argparse
import itertools
import math
import os
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from backtest import simulate
from signals import build_events, prepare_panels

# ─── 사전 등록된 격자 (preregistration.md 와 동일해야 함) ──────────
GRID_HOLD = [10, 20, 30]
GRID_TOP = [0.10, 0.20, 0.30]
GRID_STOP = [-0.05, -0.08, None]          # None = 손절 없음
DESIGNATED = (20, 0.20, -0.08)            # 백테스트 전에 지정한 설정
CSCV_S = 16                                # 기간을 16조각 → 12,870개 조합

N01 = NormalDist()
EMC = 0.5772156649                         # 오일러-마스케로니 상수


def label(h, t, s):
    return f"H{h}_T{int(t * 100)}_S{'none' if s is None else int(-s * 100)}"


# ════════════════════════════════════════════════════════════
def run_grid(P, ev, k200, start, end):
    base = (C.HOLD_DAYS, C.TOP_PCT, C.STOP_LOSS)
    rets = {}
    combos = list(itertools.product(GRID_HOLD, GRID_TOP, GRID_STOP))
    for k, (h, t, s) in enumerate(combos, 1):
        C.HOLD_DAYS, C.TOP_PCT, C.STOP_LOSS = h, t, (-9.99 if s is None else s)
        e = ev.copy()
        e["passed"] = (e["pScore"] >= 1 - t) & (e["sue"] > 0) & (e["ear"] > 0)
        res = simulate(P, e, k200, start, end, contest=False, stage1=False)
        rets[label(h, t, s)] = res["nav"].pct_change().fillna(0.0)
        print(f"  [{k:2d}/27] {label(h, t, s):<16} 총수익 {res['nav'].iloc[-1] / res['nav'].iloc[0] - 1:7.2%}")
    C.HOLD_DAYS, C.TOP_PCT, C.STOP_LOSS = base
    return pd.DataFrame(rets)


# ─── PBO (CSCV) ─────────────────────────────────────────────
def cscv_pbo(M, S=CSCV_S):
    """M: (T × N) 일별 수익률. 반환: PBO, 로짓 배열, (IS 최고 샤프, 그 설정의 OOS 샤프) 쌍"""
    X = M.values
    T, N = X.shape
    T = (T // S) * S
    X = X[-T:]                                      # 앞부분을 잘라 S로 나눠떨어지게
    blocks = np.array_split(np.arange(T), S)
    s1 = np.array([X[b].sum(0) for b in blocks])    # 조각별 합
    s2 = np.array([(X[b] ** 2).sum(0) for b in blocks])
    n_b = np.array([len(b) for b in blocks])

    def sharpe(idx):
        n = n_b[list(idx)].sum()
        mu = s1[list(idx)].sum(0) / n
        var = s2[list(idx)].sum(0) / n - mu ** 2
        return mu / np.sqrt(np.maximum(var, 1e-18))

    logits, pairs = [], []
    all_idx = set(range(S))
    for comb in itertools.combinations(range(S), S // 2):
        is_sr = sharpe(comb)
        oos_sr = sharpe(tuple(all_idx - set(comb)))
        n_star = int(np.argmax(is_sr))
        rank = (oos_sr < oos_sr[n_star]).sum() + 1 + 0.5 * ((oos_sr == oos_sr[n_star]).sum() - 1)
        w = rank / (N + 1)
        logits.append(math.log(w / (1 - w)))
        pairs.append((is_sr[n_star] * math.sqrt(252), oos_sr[n_star] * math.sqrt(252)))
    logits = np.array(logits)
    return float((logits <= 0).mean()), logits, np.array(pairs)


# ─── DSR ────────────────────────────────────────────────────
def expected_max_sr(var_sr, n):
    if n <= 1:
        return 0.0
    return math.sqrt(var_sr) * ((1 - EMC) * N01.inv_cdf(1 - 1 / n) + EMC * N01.inv_cdf(1 - 1 / (n * math.e)))


def psr(r, sr0):
    """Probabilistic Sharpe Ratio: 진짜 샤프가 sr0보다 클 확률 (비연환산 기준)"""
    r = np.asarray(r)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3 = ((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3
    g4 = ((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return N01.cdf(z), sr, g3, g4, T


def implied_n(M):
    """시도 간 평균 상관으로 '독립 시도 수' 추정: N = ρ + (1-ρ)·M  (DSR 논문 부록 3)"""
    c = M.corr().values
    m = c.shape[0]
    rho = (c.sum() - m) / (m * (m - 1))
    return rho, rho + (1 - rho) * m


def dsr_report(M, designated):
    daily_sr = M.mean() / M.std(ddof=1)
    var_sr = daily_sr.var(ddof=1)
    rho, n_hat = implied_n(M)
    out = {}
    for name, n in [("N=27 (보수적)", M.shape[1]), (f"N={n_hat:.1f} (상관 반영)", n_hat)]:
        sr0 = expected_max_sr(var_sr, n)
        p, sr, g3, g4, T = psr(M[designated], sr0)
        out[name] = dict(DSR=p, SR0_annual=sr0 * math.sqrt(252))
    p0, sr, g3, g4, T = psr(M[designated], 0.0)
    base = dict(SR_annual=sr * math.sqrt(252), PSR_vs_0=p0, skew=g3, kurtosis=g4, T=T,
                avg_corr=rho, implied_N=n_hat, var_SR_trials_annual=var_sr * 252)
    return base, out


# ─── 그래프 ─────────────────────────────────────────────────
def plots(M, designated, logits, pairs, pbo, out):
    sr = (M.mean() / M.std(ddof=1) * math.sqrt(252)).sort_values()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#E08A2E" if c == designated else "#2F5E9E" for c in sr.index]
    ax.bar(range(len(sr)), sr.values, color=colors)
    ax.set_xticks(range(len(sr))); ax.set_xticklabels(sr.index, rotation=75, fontsize=8)
    ax.set_title("Annualized Sharpe of all 27 pre-registered configs (orange = designated)")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "grid_sharpe.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(pairs[:, 0], pairs[:, 1], s=4, alpha=0.3, color="#8B1E1E")
    b, a = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
    xs = np.linspace(pairs[:, 0].min(), pairs[:, 0].max(), 10)
    ax.plot(xs, a + b * xs, color="black", lw=1)
    ax.set_xlabel("SR IS (annualized)"); ax.set_ylabel("SR OOS (annualized)")
    ax.set_title(f"OOS Performance Degradation  |  Prob[SR OOS<0] = {(pairs[:, 1] < 0).mean():.2f}")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(out, "cscv_degradation.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(logits, bins=50, color="#3A8E3A", edgecolor="black", density=True)
    ax.axvline(0, color="red", lw=1)
    ax.set_xlabel("Logit"); ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of Rank Logits  |  PBO = {pbo:.2f}")
    fig.tight_layout(); fig.savefig(os.path.join(out, "cscv_logits.png"), dpi=160); plt.close(fig)


# ════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        from demo_data import make_demo
        panels, quarters, k200, sectors = make_demo()
        start, end, out = "2017-01-01", "2025-12-31", os.path.join(C.OUT_DIR, "demo", "validation")
    else:
        from data_loader import load_all
        panels, quarters, k200, sectors = load_all()
        start, end, out = C.BACKTEST_START, C.END, os.path.join(C.OUT_DIR, "validation")
    os.makedirs(out, exist_ok=True)

    print("[1/4] 신호 계산")
    P = prepare_panels(panels, sectors)
    ev = build_events(quarters, P)
    print("[2/4] 사전 등록된 27개 설정 백테스트")
    M = run_grid(P, ev, k200, start, end)
    M.to_csv(os.path.join(out, "grid_daily_returns.csv"), encoding="utf-8-sig")
    designated = label(*DESIGNATED)

    sr = (M.mean() / M.std(ddof=1) * math.sqrt(252)).sort_values(ascending=False)
    rank = list(sr.index).index(designated) + 1

    print(f"[3/4] CSCV / PBO 계산 (S={CSCV_S}, 조합 {math.comb(CSCV_S, CSCV_S // 2):,}개)")
    pbo, logits, pairs = cscv_pbo(M)

    print("[4/4] DSR 계산")
    base, dsr = dsr_report(M, designated)
    plots(M, designated, logits, pairs, pbo, out)

    rows = {"지정 설정": designated, "지정 설정 순위 (27개 중)": rank,
            "지정 설정 연환산 샤프": round(base["SR_annual"], 3),
            "PSR (진짜 샤프 > 0 확률)": round(base["PSR_vs_0"], 4),
            "왜도": round(base["skew"], 3), "첨도": round(base["kurtosis"], 3), "표본 길이 T(일)": base["T"],
            "시도 간 평균 상관": round(base["avg_corr"], 3),
            "PBO": round(pbo, 4), "Prob[OOS 샤프 < 0]": round(float((pairs[:, 1] < 0).mean()), 4)}
    for k, v in dsr.items():
        rows[f"DSR {k}"] = round(v["DSR"], 4)
        rows[f"SR0 기준선 {k} (연환산)"] = round(v["SR0_annual"], 3)
    pd.Series(rows).to_csv(os.path.join(out, "validation_summary.csv"), header=["value"], encoding="utf-8-sig")
    sr.rename("annual_sharpe").to_csv(os.path.join(out, "grid_sharpe.csv"), encoding="utf-8-sig")

    print("\n══════ Backtest Validity ══════")
    for k, v in rows.items():
        print(f"  {k:<34}{v}")
    print("\n  해석 가이드")
    print("  · PBO  < 0.5 이면 IS 1등이 OOS에서도 중간 이상인 경우가 더 많음 (낮을수록 좋음)")
    print("  · DSR ≥ 0.95 이면 27번 시도를 감안해도 95% 신뢰수준에서 유의")
    print(f"\n결과 저장 위치: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
