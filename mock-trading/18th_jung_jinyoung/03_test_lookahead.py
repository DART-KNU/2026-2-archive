"""미래 참조 테스트: 신호일 S 이후 데이터를 무작위로 바꿔도 S의 신호·비중이 그대로여야 한다.

바꾸는 것: S 이후 가격·시총·거래대금, S 이후 공시 가능한 재무, S 이후 상폐 종목 섹터 분류.
레짐은 S가 속한 해의 저장된 추정(전년 말까지 학습)을 고정하고, 바뀐 시장 수익률로 다시 필터링한다.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest
from data_io import load_config
from regime import run_msgarch
from signals import GICS_ORDER, sector_returns
from signals import load_panels, prepare

S = pd.Timestamp("2022-06-07")


def _signal(pn, cfg, workdir, tag):
    pn = prepare(pn, cfg)
    sr, mkt = sector_returns(pn.d["ret"], pn.mcap, pn.d["sec"], pn.d["elig"])
    r = (100 * mkt).dropna()
    r = r[r.index.year <= S.year]
    fit = cfg["paths"]["processed"] / "regime" / f"fit_{S.year}.rds"
    (workdir / f"fit_{tag}.rds").write_bytes(fit.read_bytes())
    train_end = r.index[r.index.year < S.year][-1]
    p, _ = run_msgarch(r, train_end, workdir, cfg, tag=tag, filter_only=True)
    w, info = backtest.signal_at(pn, sr, mkt, p, S, cfg)
    return w, info, sr, p


def _perturb(pn, rng):
    pn = copy.copy(pn)
    pn.d = {}
    after = pn.price.index > S
    for name in ("price", "mcap", "value"):
        df = getattr(pn, name).copy()
        noise = np.exp(rng.normal(0, 0.3, size=(after.sum(), df.shape[1])))
        df.loc[after] = df.loc[after].to_numpy() * noise
        setattr(pn, name, df)
    fin = pn.fin.copy()
    fut = fin["avail"] > S
    fin.loc[fut, "ni"] = rng.normal(0, 1e11, fut.sum())
    fin.loc[fut, "eq"] = rng.uniform(1e9, 1e12, fut.sum())
    pn.fin = fin
    sm = pn.sec_monthly.copy()
    fut_m = sm.index > S
    vals = sm.loc[fut_m].to_numpy()
    mask = pd.notna(vals)
    vals[mask] = rng.choice(GICS_ORDER[:10], mask.sum())
    sm.loc[fut_m] = vals
    pn.sec_monthly = sm
    return pn


@pytest.mark.slow
def test_no_lookahead(tmp_path):
    cfg = load_config()
    if not (cfg["paths"]["processed"] / "regime" / f"fit_{S.year}.rds").exists():
        pytest.skip("레짐 추정 결과 없음: run_backtest.py 먼저 실행")
    base = load_panels(cfg)
    w0, i0, sr0, p0 = _signal(copy.copy(base), cfg, tmp_path, "base")
    w1, i1, sr1, p1 = _signal(_perturb(base, np.random.default_rng(7)), cfg, tmp_path, "pert")

    # 바뀐 게 실제로 반영됐는지 (S 이후 섹터 수익률은 달라야 함)
    assert not np.allclose(sr0.loc[sr0.index > S].fillna(0), sr1.loc[sr1.index > S].fillna(0))
    # S 이전·당일은 동일
    pd.testing.assert_frame_equal(sr0.loc[:S], sr1.loc[:S])
    assert p0.loc[S] == pytest.approx(p1.loc[S], abs=1e-10)
    assert i0["a"] == pytest.approx(i1["a"])
    pd.testing.assert_series_equal(i0["mom"], i1["mom"])
    pd.testing.assert_series_equal(i0["sector_target"], i1["sector_target"])
    pd.testing.assert_series_equal(w0.sort_index(), w1.sort_index())
