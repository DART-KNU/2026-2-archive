import numpy as np, pandas as pd, beta_backtest as bb
d, names = bb.load_dataguide("0929_data.xlsx")
close, value, mcap = d["close"], d["value"], d["mcap"]
ret = close.pct_change(fill_method=None)
w_prev = mcap.shift(1).where(ret.notna())
rm = (ret * w_prev).sum(axis=1) / w_prev.sum(axis=1); rm.iloc[0] = np.nan
beta = bb.rolling_beta(ret, rm, 60, 40, "all")
t = close.index[-1]
df = pd.DataFrame({
    "종목명": names.reindex(close.columns).values,
    "종가": close.iloc[-1], "시총(억)": mcap.iloc[-1] / 100,
    "20일 평균 거래대금(억)": value.iloc[-20:].mean() / 1e8,
    "5일 평균 거래대금(억)": value.iloc[-5:].mean() / 1e8,
    "베타60": beta.iloc[-1], "변동성60": ret.iloc[-60:].std() * np.sqrt(252),
    "수익률63": close.iloc[-1] / close.iloc[-64] - 1,
    "MA20": close.iloc[-20:].mean(), "MA60": close.iloc[-60:].mean(),
    "상장일수": close.notna().sum()}, index=close.columns)
df.index.name = "코드"
uni = df[(df["시총(억)"] >= 10000) & (df["20일 평균 거래대금(억)"] >= 50) & (df["5일 평균 거래대금(억)"] > 30)
         & df["베타60"].notna() & (df["상장일수"] >= 120) & close.iloc[-1].notna()].copy()
cut = uni["베타60"].quantile(0.6)
pool = uni[uni["베타60"] <= cut].copy()
cand = pool[pool["MA20"] > pool["MA60"]].copy()
cand["점수"] = cand["변동성60"].rank(ascending=False, pct=True) + cand["수익률63"].rank(pct=True)
cand = cand.sort_values("점수", ascending=False)
cand["점수순위"] = range(1, len(cand) + 1)
core = ["A005930", "A000660"]
pick = cand[~cand.index.isin(core)].head(50).copy()
# 비중: 삼성·하이닉스 각 4%, 나머지 92%를 변동성 역가중, 상한 4%
inv = 1 / pick["변동성60"]; w = inv / inv.sum() * 0.92
for _ in range(50):
    over = w > 0.04
    if not over.any(): break
    ex = (w[over] - 0.04).sum(); w[over] = 0.04
    w[~over] += ex * w[~over] / w[~over].sum()
pick["목표비중"] = w
corep = df.loc[core].copy(); corep["목표비중"] = 0.04; corep["점수순위"] = np.nan; corep["점수"] = np.nan
port = pd.concat([corep, pick])
port.insert(0, "구분", ["코어(기본편입)"] * 2 + ["저베타 모멘텀"] * len(pick))
pb = (port["목표비중"] * port["베타60"]).sum()
print(f"기준일 {t.date()} | 유니버스 {len(uni)} | 저베타 풀(β≤{cut:.2f}) {len(pool)} | 추세필터 통과 {len(cand)} | 편입 {len(port)}")
print(f"비중합 {port['목표비중'].sum():.3f} | 포트 베타 {pb:.2f} | Top5 {port['목표비중'].nlargest(5).sum():.3f} | 최대 {port['목표비중'].max():.3f}")
print(f"시장 대비: 유니버스 평균 베타 {uni['베타60'].mean():.2f}")
cols = ["구분","종목명","점수순위","목표비중","베타60","변동성60","수익률63","시총(억)","20일 평균 거래대금(억)","종가","MA20","MA60"]
port = port[cols]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(port.round(3).to_string())
port.to_pickle("screen_port.pkl"); cand.to_pickle("screen_cand.pkl")
open("screen_meta.txt","w").write(f"{t.date()},{len(uni)},{len(pool)},{len(cand)},{cut:.3f},{pb:.3f}")
