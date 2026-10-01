"""
backtest.py
Dual-Engine Strategy: Sector Risk Parity Core (50%) + Tactical Semiconductor Swing (up to 50%)
- 데이터 로더 (DataGuide 엑셀 형식)
- 백테스트 엔진 (Backtest 클래스)
- 성과 요약 함수 (summarize)
다른 스크립트(run_backtest.py, sensitivity.py, screen.py)에서 import 해서 사용합니다.
"""
import numpy as np
import pandas as pd

SAM, HYN = "005930", "000660"
BUY_COST, SELL_COST = 0.001, 0.003          # 매수 수수료 0.1%, 매도 수수료 0.1% + 매도세 0.2%

PRICE_FILE = "data/수정주가.xlsx"            # DataGuide 시계열: 수정주가(원), 일간
KOSPI_FILE = "data/코스피.xlsx"              # DataGuide 지수: 종가지수(I.001), 일간
UNIVERSE_FILE = "data/대회_유니버스.xlsx"     # 대회 투자가능 종목 파일 (Sector 시트)

# 최종 전략 파라미터
PARAMS = dict(
    core_w=0.50,          # 코어 비중
    semi_w=0.25,          # 반도체 종목당 매수 비중
    n_sectors=8,          # 코어 섹터 수 (8 + 삼성전자·SK하이닉스 = 최대 10종목)
    cap=0.09,             # 코어 종목당 상한 (섹터 10% 한도 - 1%p 버퍼)
    tp=0.05,              # 반도체 익절
    sl=-0.03,             # 반도체 손절
    hold=5,               # 반도체 최대 보유 거래일
    buffer=True,          # 보유 종목이 섹터 2위 이내면 유지
    topup=True,           # 주간 회전율 5% 미달 시 보완 (실전은 매니저 판단, 백테스트는 규칙으로 근사)
    markets=("코스피",),   # 코어 유니버스
)


def read_dataguide(path):
    """DataGuide 시계열 엑셀 → (날짜 × 코드) DataFrame. 9번째 줄이 헤더(코드, 코드명, ..., 날짜들)."""
    raw = pd.read_excel(path, header=None, skiprows=8)
    hdr, body = raw.iloc[0], raw.iloc[1:]
    dates = pd.to_datetime(hdr.iloc[6:].values)
    codes = body.iloc[:, 0].astype(str).str.replace(r"^A", "", regex=True).values
    df = pd.DataFrame(body.iloc[:, 6:].values.astype(float), index=codes, columns=dates).T
    df = df.loc[:, ~df.columns.duplicated()]
    df.index.name = "date"
    return df


def read_universe(path):
    """대회 파일 Sector 시트의 '투자가능 종목' 표 (섹터코드, 섹터명, 종목코드, 종목명, 시장)."""
    s = pd.read_excel(path, sheet_name="Sector", header=None)
    u = s.iloc[5:, 5:10].dropna(how="all")
    u.columns = ["sector_code", "sector", "ticker", "name", "market"]
    u["ticker"] = u["ticker"].astype(str).str[1:]
    return u.reset_index(drop=True)



def load_all(price_file=PRICE_FILE, kospi_file=KOSPI_FILE, universe_file=UNIVERSE_FILE):
    uni = read_universe(universe_file)
    px_all = read_dataguide(price_file)
    px = px_all[[t for t in uni.ticker if t in px_all.columns]]
    kospi = read_dataguide(kospi_file).iloc[:, 0].rename("kospi") if kospi_file else None
    return px, uni, kospi


SECTOR_LIMITS = {"IN": 0.3712, "CD": 0.1102, "Fi": 0.1526, "IT": 1.0}   # 나머지 섹터 10%


def rp_weights(cov, iters=500):
    """Risk parity: 각 종목의 위험 기여도가 같아지도록 반복 계산."""
    n = len(cov)
    w = np.ones(n) / n
    for _ in range(iters):
        rc = w * (cov @ w)
        w = w * (rc.sum() / n / np.maximum(rc, 1e-18)) ** 0.5
        w = w / w.sum()
    return w


def cap_weights(w, total, cap):
    """총합을 total로 맞추고 종목당 cap을 넘는 부분은 나머지에 재분배."""
    w = np.asarray(w, float) * total
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


class Backtest:
    """
    코어 : 120일 샤프 → 섹터별 1등 → 상위 n_sectors개 → risk parity(120일 공분산), 종목당 cap
           화요일 종가 신호 → 수요일 체결 (휴장 시 그 주 수요일 이후 첫 거래일)
    반도체: Close[t-2] > MA20[t-2] & Close[t]/Close[t-2]-1 <= -5% → 다음날 25% 매수
           종가 기준 익절 tp / 손절 sl / hold일 만기
    공통 : 매도 먼저, 매수 나중 / 매수 0.1%, 매도 0.3% / 매일 종목·섹터 한도 점검(0.5%p 버퍼)
           주간 회전율 (매수+매도)/2 < 5% 이면 금요일에 부족분만 교체 (약한 코어 종목 → 같은 섹터 2등)
    체결가는 종가로 가정 (실전은 시간 분할 주문)
    """

    def __init__(self, px, uni, start, end, core_w=0.5, semi_w=0.25, n_sectors=8, cap=0.09, tp=0.05,
                 sl=-0.03, hold=5, use_semis=True, topup=True, buffer=True, markets=("코스피",),
                 lookback=120, trim=True):
        self.px, self.uni = px, uni.set_index("ticker")
        self.dates, self.all_dates = px.loc[start:end].index, px.index
        self.ret = px.pct_change(fill_method=None)
        self.ma20 = px.rolling(20).mean()
        self.p = dict(core_w=core_w, semi_w=semi_w, n_sectors=n_sectors, cap=cap, tp=tp, sl=sl, hold=hold,
                      use_semis=use_semis, topup=topup, buffer=buffer, lookback=lookback, trim=trim)
        cand = self.uni[self.uni.market.isin(markets)].index
        self.core_cands = [t for t in cand if t not in (SAM, HYN) and t in px.columns]

    # ----- 코어 종목 선정 + 비중 -----
    def core_targets(self, sig_date, held=()):
        i, lb = self.all_dates.get_loc(sig_date), self.p["lookback"]
        if i < lb:
            return {}, {}
        win = self.ret.iloc[i - lb + 1:i + 1][self.core_cands]
        win = win.loc[:, win.notna().sum() >= lb - 2]
        sh = (win.mean() / win.std() * np.sqrt(252)).replace([np.inf, -np.inf], np.nan).dropna()
        ranked = pd.DataFrame({"sh": sh, "sec": self.uni.loc[sh.index, "sector_code"]})
        ranked = ranked.sort_values("sh", ascending=False)
        ranked["rk"] = ranked.groupby("sec").cumcount()
        best = ranked[ranked.rk == 0]
        if self.p["buffer"] and held:
            top2 = ranked[ranked.rk <= 1]
            ok_secs = set(best["sec"].iloc[: self.p["n_sectors"] + 2])
            keep = [t for t in held if t in top2.index and top2.loc[t, "sec"] in ok_secs]
            keep_secs = {top2.loc[t, "sec"] for t in keep}
            fill = best[~best.sec.isin(keep_secs)].head(max(self.p["n_sectors"] - len(keep), 0))
            best = pd.concat([top2.loc[keep], fill])
        best = best.sort_values("sh", ascending=False).head(self.p["n_sectors"])
        names = list(best.index)
        second = ranked[ranked.rk == 1]
        runner = {t: second.index[second.sec == best.loc[t, "sec"]][0]
                  for t in names if (second.sec == best.loc[t, "sec"]).any()}
        cov = np.cov(self.ret.iloc[i - lb + 1:i + 1][names].dropna().values.T)
        w = cap_weights(rp_weights(cov), self.p["core_w"], self.p["cap"])
        return dict(zip(names, w)), {"rank": list(best.sort_values("sh").index), "runner": runner}

    def semi_signal(self, t, i):
        if i < 22:
            return False
        c, ma = self.px[t].iloc, self.ma20[t].iloc
        return (c[i - 2] > ma[i - 2]) and (c[i] / c[i - 2] - 1 <= -0.05)

    # ----- 메인 루프 -----
    def run(self):
        P = self.p
        cash, hold, semi_pos, pending = 1.0, {}, {}, []
        nav_hist, trades, weekly, last_meta, self.semi_log = [], [], {}, {}, []
        dates = list(self.dates)
        wkey = lambda d: tuple(d.isocalendar())[:2]
        exec_days, week_last = set(), {}
        for key, grp in pd.Series(dates, index=dates).groupby(lambda d: wkey(d)):
            ds = list(grp)
            wed = [d for d in ds if d.weekday() >= 2]
            if wed:
                exec_days.add(wed[0])
            week_last[key] = ds[-1]

        def trade(t, dv, d, tag):
            nonlocal cash
            if abs(dv) < 1e-9:
                return
            cost = dv * BUY_COST if dv > 0 else -dv * SELL_COST
            hold[t] = hold.get(t, 0.0) + dv
            if abs(hold[t]) < 1e-9:
                hold.pop(t)
            cash -= dv + cost
            w = weekly.setdefault(wkey(d), {"buy": 0.0, "sell": 0.0, "nav0": None})
            w["buy" if dv > 0 else "sell"] += abs(dv)
            trades.append(dict(date=d, ticker=t, value=dv, cost=cost, tag=tag))

        navf = lambda: cash + sum(hold.values())
        for k, d in enumerate(dates):
            i = self.all_dates.get_loc(d)
            for t in list(hold):                                   # 1) 평가
                r = self.ret[t].iloc[i]
                if not np.isnan(r):
                    hold[t] *= 1 + r
            key = wkey(d)
            wrec = weekly.setdefault(key, {"buy": 0.0, "sell": 0.0, "nav0": None})
            if wrec["nav0"] is None:
                wrec["nav0"] = navf()
            if P["use_semis"]:                                     # 2) 반도체 청산
                for t in list(semi_pos):
                    sp = semi_pos[t]
                    sp["days"] += 1
                    rr = self.px[t].iloc[i] / sp["entry_px"] - 1
                    reason = ("TP" if P["tp"] is not None and rr >= P["tp"] else
                              "SL" if rr <= P["sl"] else "EXPIRY" if sp["days"] >= P["hold"] else None)
                    if reason:
                        self.semi_log.append(dict(sp, ticker=t, exit_date=d, ret=rr, reason=reason))
                        trade(t, -hold.get(t, 0.0), d, "semi_exit")
                        semi_pos.pop(t)
            if d in exec_days and k > 0:                           # 3) 코어 리밸런싱
                held = [t for t in hold if t not in semi_pos]
                tgt, meta = self.core_targets(dates[k - 1], held)
                if tgt:
                    last_meta, nav = meta, navf()
                    core_now = {t: v for t, v in hold.items() if t not in semi_pos}
                    diffs = {t: tgt.get(t, 0.0) * nav - core_now.get(t, 0.0) for t in sorted(set(tgt) | set(core_now))}
                    for t, dv in diffs.items():
                        if dv < 0:
                            trade(t, dv, d, "core")
                    buys = {t: dv for t, dv in diffs.items() if dv > 0}
                    tot = sum(buys.values())
                    scale = min(1.0, max(cash, 0) / (1 + BUY_COST) / tot) if tot > 0 else 1.0
                    for t, dv in buys.items():
                        trade(t, dv * scale, d, "core")
            for t in pending:                                      # 4) 반도체 진입
                if t not in semi_pos:
                    nav = navf()
                    amt = min(P["semi_w"] * nav, max(cash, 0) / (1 + BUY_COST))
                    if amt > 0.01 * nav:
                        trade(t, amt, d, "semi_entry")
                        semi_pos[t] = dict(entry_date=d, entry_px=self.px[t].iloc[i], days=0, weight=amt / nav)
            pending = [t for t in (SAM, HYN) if P["use_semis"] and t not in semi_pos and self.semi_signal(t, i)]
            if P["topup"] and d == week_last[key] and last_meta:   # 5) 회전율 보완
                w = weekly[key]
                short = 0.05 - (w["buy"] + w["sell"]) / 2 / w["nav0"]
                if short > 0:
                    amt, done = (short + 0.002) * w["nav0"], 0.0
                    for t in [t for t in last_meta["rank"] if t in hold and t not in semi_pos]:
                        rep = last_meta["runner"].get(t)
                        if rep is None:
                            continue
                        x = min(hold[t], amt - done)
                        trade(t, -x, d, "topup")
                        trade(rep, x, d, "topup")
                        done += x
                        if done >= amt - 1e-9:
                            break
            if P["trim"]:                                          # 6) 한도 점검
                nav = navf()
                for t in list(hold):
                    if t not in semi_pos and t not in (SAM, HYN) and hold[t] / nav > P["cap"] + 0.005:
                        trade(t, -(hold[t] - P["cap"] * nav), d, "trim")
                secs = {}
                for t in hold:
                    secs.setdefault(self.uni.loc[t, "sector_code"], []).append(t)
                for sc, ts in secs.items():
                    lim = SECTOR_LIMITS.get(sc, 0.10) - 0.005
                    tot = sum(hold[t] for t in ts) / nav
                    if sc != "IT" and tot > lim:
                        big = max(ts, key=lambda t: hold[t])
                        trade(big, -(tot - lim) * nav, d, "trim")
            nav = navf()
            secw = {}
            for t, v in hold.items():
                sc = self.uni.loc[t, "sector_code"]
                secw[sc] = secw.get(sc, 0) + v / nav
            semi_w = sum(hold.get(t, 0) for t in semi_pos) / nav
            nav_hist.append(dict(date=d, nav=nav, cash=cash / nav, semi=semi_w, core=1 - cash / nav - semi_w,
                                 max_name=max([v / nav for t, v in hold.items() if t not in (SAM, HYN)] or [0]),
                                 sec_breach=max([w - SECTOR_LIMITS.get(sc, 0.10) for sc, w in secw.items()] or [-1]),
                                 hynix=hold.get(HYN, 0) / nav, samsung=hold.get(SAM, 0) / nav))
        self.nav = pd.DataFrame(nav_hist).set_index("date")
        self.trades = pd.DataFrame(trades)
        wk = pd.DataFrame(weekly).T
        wk["one_way"] = (wk["buy"] + wk["sell"]) / 2 / wk["nav0"]
        self.weekly = wk
        return self


def summarize(b, name):
    s = b.nav["nav"] / b.nav["nav"].iloc[0]
    sl = pd.DataFrame(b.semi_log)
    return dict(strategy=name,
                ret=(s.iloc[-1] - 1) * 100, mdd=(s / s.cummax() - 1).min() * 100,
                vol=s.pct_change().std() * np.sqrt(252) * 100, avg_cash=b.nav.cash.mean() * 100,
                cost=b.trades.cost.sum() * 100, turnover_avg=b.weekly.one_way.mean() * 100,
                weeks_below_5=int((b.weekly.one_way < 0.05 - 1e-9).sum()),
                semi_trades=len(sl), semi_win=(sl.ret > 0).mean() * 100 if len(sl) else np.nan,
                max_name=b.nav.max_name.max() * 100, max_hynix=b.nav.hynix.max() * 100,
                sector_breach_days=int((b.nav.sec_breach > 0).sum()))
