"""백테스트 실행 → outputs/ 와 outputs/presentation/."""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import backtest, report
from data_io import load_config
from regime import run_msgarch, walk_forward_regime
from signals import sector_returns
from signals import load_panels, prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-regime", action="store_true", help="MS-GARCH 워크포워드 재추정")
    args = ap.parse_args()
    cfg = load_config()
    P, O = cfg["paths"]["processed"], cfg["paths"]["outputs"]
    PR = O / "presentation"
    PR.mkdir(parents=True, exist_ok=True)

    print("데이터 준비…")
    pn = prepare(load_panels(cfg), cfg)
    sr, mkt = sector_returns(pn.d["ret"], pn.mcap, pn.d["sec"], pn.d["elig"])

    rp = P / "regime_wf.parquet"
    years = range(pd.Timestamp(cfg["backtest"]["start"]).year, pd.Timestamp(cfg["backtest"]["end"]).year + 1)
    if args.refit_regime or not rp.exists():
        print("MS-GARCH 워크포워드 추정…")
        p_turb, log = walk_forward_regime(mkt, years, P / "regime", cfg)
        p_turb.to_frame().to_parquet(rp)
        log.to_csv(P / "regime_log.csv", index=False)
    p_turb = pd.read_parquet(rp)["p_turb"]
    last = mkt.dropna().index[-1]
    if p_turb.index[-1] < last:
        # 데이터가 추가됨(예: KRX 주간 갱신) → 올해 추정 파라미터를 고정한 채 새 날짜만 필터링해 이어 붙임
        y = last.year
        r = (100 * mkt).dropna()
        r = r[r.index.year <= y]
        p_new, _ = run_msgarch(r, r.index[r.index.year < y][-1], P / "regime", cfg, tag=str(y), filter_only=True)
        p_turb = pd.concat([p_turb[p_turb.index.year < y], p_new[p_new.index.year == y]])
        p_turb.to_frame().to_parquet(rp)
        print(f"레짐 확률을 {last.date()}까지 연장했습니다 (파라미터 고정, 필터링만).")
    pd.read_csv(P / "regime_log.csv").to_csv(O / "regime_log.csv", index=False)

    cfg_regime = copy.deepcopy(cfg)
    cfg_regime["allocate"]["satellite_weight"] = None      # SPEC 원안: a = f(p_turb)로 코어·위성 혼합
    runs = {}
    for label, c, a_fixed in [("Strategy", cfg, None), ("Core only (a=0)", cfg, 0.0), ("Satellite only (a=1)", cfg, 1.0),
                              ("Regime blend (SPEC)", cfg_regime, None)]:
        print(f"백테스트: {label}")
        runs[label] = backtest.run(pn, sr, mkt, p_turb, c, a_fixed=a_fixed, label=label, verbose=True)

    idx = runs["Strategy"]["nav"].index
    m = mkt.reindex(idx).fillna(0.0)
    m.iloc[0] = 0.0
    navs = {"Strategy": runs["Strategy"]["nav"], "Market": (1 + m).cumprod().rename("Market"),
            "Core only (a=0)": runs["Core only (a=0)"]["nav"], "Satellite only (a=1)": runs["Satellite only (a=1)"]["nav"]}
    navs_all = {**navs, "Regime blend (SPEC)": runs["Regime blend (SPEC)"]["nav"]}
    turns = {k: v["turnover"] for k, v in runs.items()}

    summ = report.summary(navs_all, turns, cfg["backtest"]["turnover_min"] * 100)
    for k, v in runs.items():
        d = v["delists"]
        summ.loc[k, "delist_count"] = len(d)
        summ.loc[k, "delist_loss_nav_%"] = round(d["loss_nav"].sum() * 100, 2) if len(d) else 0.0
        summ.loc[k, "rule_violation_weeks"] = int((v["log"]["violations"] != "").sum())
    win, excess = report.window_stats(navs_all, navs["Market"], 40)
    win = win.drop(index="Market")[["windows", "mean_%", "median_%", "p5_%", "p95_%", "beat_mkt_%", "excess_gt5_%",
                                    "excess_gt10_%", "window_mdd_mean_%", "window_mdd_p5_%", "window_mdd_worst_%"]]
    summ.to_csv(PR / "summary.csv", encoding="utf-8-sig")
    win.to_csv(PR / "window40.csv", encoding="utf-8-sig")
    report.octnov_table(navs_all).to_csv(O / "octnov.csv", encoding="utf-8-sig")

    log = runs["Regime blend (SPEC)"]["log"].set_index("trade")   # 레짐 그래프: 레짐이 제시한 공격도 a (모니터링용)
    report.plot_cum(navs, PR / "cum_return.png")
    report.plot_regime(p_turb[p_turb.index >= idx[0]], log["a"], pn.index["KOSPI"].reindex(idx), PR / "regime.png",
                       cfg["regime"]["a_full"], cfg["regime"]["a_zero"])
    report.plot_window_hist(excess, PR / "window40.png")

    for k, v in runs.items():
        tag = k.split(" ")[0].lower()
        v["log"].to_csv(O / f"log_{tag}.csv", index=False, encoding="utf-8-sig")
        v["delists"].to_csv(O / f"delists_{tag}.csv", index=False, encoding="utf-8-sig")
        v["turnover"].rename("turnover_%").to_csv(O / f"turnover_{tag}.csv", encoding="utf-8-sig")
    pd.DataFrame(navs_all).to_csv(O / "nav.csv", encoding="utf-8-sig")
    # 레짐 브레이크 작동 구간 (2020-02~04, 2022)
    brk = log[["p_turb", "a"]]
    pd.concat([brk.loc["2020-02":"2020-04"], brk.loc["2022"]]).round(3).to_csv(O / "brake_check.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(summ.to_string())
    print(win.to_string())


if __name__ == "__main__":
    main()
