"""MS-GARCH 레짐 (Rscript 호출 래퍼) + 공격도 a."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

R_SCRIPT = Path(__file__).with_name("regime_msgarch.r")


def run_msgarch(ret_pct: pd.Series, train_end, workdir: Path, cfg, tag: str, filter_only=False) -> tuple[pd.Series, dict]:
    """ret_pct: 퍼센트 단위 시장 수익률. train_end까지 추정하고 전 구간 필터링 확률(p_turb)을 반환."""
    workdir.mkdir(parents=True, exist_ok=True)
    inp, outp, rds = workdir / f"in_{tag}.csv", workdir / f"prob_{tag}.csv", workdir / f"fit_{tag}.rds"
    ret_pct.dropna().rename("ret").rename_axis("date").reset_index().to_csv(inp, index=False)
    args = [cfg["paths"]["rscript"], str(R_SCRIPT), str(inp), str(pd.Timestamp(train_end).date()), str(outp), str(rds),
            str(cfg["regime"]["n_seeds"])] + (["filter_only"] if filter_only else [])
    res = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"MSGARCH 실패 ({tag}):\n{res.stderr[-2000:]}")
    m = re.search(r"LL_MAX=(\S+) LL_RANGE=(\S+) NSEED_OK=(\d+) VOL_CALM=(\S+) VOL_TURB=(\S+)", res.stdout)
    info = dict(zip(["ll_max", "ll_range", "n_ok", "vol_calm", "vol_turb"], map(float, m.groups())))
    p = pd.read_csv(outp, parse_dates=["date"]).set_index("date")["p_turb"]
    return p, info


def walk_forward_regime(mkt_ret: pd.Series, years, workdir: Path, cfg) -> tuple[pd.Series, pd.DataFrame]:
    """매년 전년 말까지로 재추정, 그 해는 파라미터 고정 후 필터링만. 반환: 해당 연도 날짜들의 p_turb."""
    r = (100 * mkt_ret).dropna()
    parts, logs = [], []
    for y in years:
        train_end = r.index[r.index.year < y][-1]
        sub = r[r.index.year <= y]
        p, info = run_msgarch(sub, train_end, workdir, cfg, tag=str(y))
        parts.append(p[p.index.year == y])
        logs.append({"year": y, "train_end": train_end.date(), "n_train": int((sub.index <= train_end).sum()), **info})
        print(f"  레짐 {y}: 학습 ~{train_end.date()}, LL {info['ll_max']:.1f} (시드 범위 {info['ll_range']:.2f}), "
              f"변동성 안정 {info['vol_calm']:.1f}% / 불안 {info['vol_turb']:.1f}%")
    return pd.concat(parts), pd.DataFrame(logs)


def aggressiveness(p_turb, lo: float, hi: float):
    """p ≤ lo → 1, p ≥ hi → 0, 사이는 선형."""
    return np.clip((hi - np.asarray(p_turb, float)) / (hi - lo), 0.0, 1.0)
