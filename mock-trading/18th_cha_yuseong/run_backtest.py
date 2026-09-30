import os,sys
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'python-deps'))
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.linalg as la, osqp, json
from threadpoolctl import threadpool_limits
threadpool_limits(2)
OUT=ROOT/os.environ.get('BL_OUTPUT_DIR','output'); OUT.mkdir(exist_ok=True)
D=np.load(ROOT/'market_data.npz'); dates=pd.to_datetime(D['dates']); codes=D['codes']; names=D['names']
close=D['close']; adj=D['adjusted']; op=D['open']; cap=D['cap']; value=D['value']
T,N=close.shape; lookback=int(os.environ.get('BL_LOOKBACK','480')); delta=2.5; tau=.025; fee=.0015
limits=np.full(N,.15); limits[codes=='A005930']=.40; limits[codes=='A000660']=.30
with np.errstate(divide='ignore',invalid='ignore'):
    rets=adj[1:]/adj[:-1]-1
    aopen=op*adj/close
start=max(lookback+1,int(np.searchsorted(dates,pd.Timestamp(os.environ.get('BL_START_DATE','2025-09-29')))))
params={'lookback_returns':lookback,'covariance':'80% sample covariance + 20% diagonal covariance, annualized 252','delta':delta,'tau':tau,'target_horizon':'12 months assumed; annual Q clipped to [-50%,100%]','confidence':'clip(0.5*n/(n+5)/(1+(sd/target/0.20)^2),0.02,0.50); only n>=2 and valid positive target, sd>=0','Omega':'tau*diag(Sigma)*(1-confidence)/confidence','cost_each_side':fee,'cash_rate':0,'risk_free_rate':0,'stock_budget':.997,'weekly_min_turnover_target':.055,'rebalance':'first observed trading day of each ISO week; signal previous trading close; fill next open','units':'fractional adjusted shares; adjusted open = raw open * adjusted close / raw close','eligibility':f'{lookback} valid past daily returns; positive prior close; prior cap >=100bn KRW; previous 5-day mean traded value >3bn KRW','market_prior':f'all sample stocks with {lookback} complete return observations and valid prior capitalization','views':'eligible stocks only','smallcap_constraint':'prior cap <1trn KRW, total <=29.8% buffer','limits':'Samsung40%, SKHynix30%, other15%, slight cost buffer','limitations':['GICS sector limits NOT modeled','historical warning/management/delisted classifications NOT supplied','current constituent list: survivorship and selection bias','target consensus point-in-time status unverified, assumed historical','cash dividends omitted; adjusted prices handle provider adjustments','fees/slippage/taxes combined into assumed 15bp each side, NOT actual contest schedule','minimum turnover imposed using convex directional restriction when natural turnover below 5.5%; final weekly realized ratio audited','full fills at opening price assumed, no capacity/market impact model','small corrective sales use known opening prices; exact opening fill idealized','no final liquidation; terminal NAV mark-to-market','daily close drawdown only; intraday drawdown unmeasured']}
(OUT/'parameters.json').write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding='utf-8')
states={k:{'units':np.zeros(N),'cash':1.,'nav':[],'trades':[],'holdings':[],'cost':0.,'gross':0.,'daily':[],'constraint':[]} for k in ['BL','Prior']}
models=[]; opt_checks=[]

def solve(mu,cov,upper,small,current,floor=True):
    n=len(mu); A=sp.vstack([sp.eye(n,format='csc'),sp.csc_matrix(np.ones((1,n))),sp.csc_matrix(small.astype(float)[None,:])],format='csc')
    lo=np.r_[np.zeros(n),.997,-np.inf]; hi=np.r_[upper,.997,.298]
    def run(A,lo,hi):
        s=osqp.OSQP(); s.setup(P=sp.csc_matrix(np.triu(delta*cov)),q=-mu,A=A,l=lo,u=hi,verbose=False,eps_abs=2e-7,eps_rel=2e-7,max_iter=20000,polishing=True)
        res=s.solve()
        if res.info.status_val not in (1,2):raise RuntimeError('Optimization '+res.info.status)
        w=np.maximum(res.x,0)
        if max(float(np.max(w-upper)),abs(w.sum()-.997),float(w[small].sum()-.298))>2e-5:raise RuntimeError('Constraint failure')
        return w
    w=run(A,lo,hi)
    if floor and .5*np.abs(w-current).sum()<.055-1e-8:
        direction=np.where(w-current>=0,1.,-1.)
        # A fixed buy/sell direction is a sufficient linear restriction for gross turnover.
        A2=sp.vstack([A,sp.csc_matrix(direction[None,:])],format='csc')
        try:w=run(A2,np.r_[lo,.11+direction@current],np.r_[hi,np.inf])
        except RuntimeError:pass # retained as a disclosed turnover miss, never claim compliance
    return w

def model(j,held):
    hist=rets[j-lookback:j]
    valid=np.all(np.isfinite(hist),axis=0)&np.all(np.isfinite(adj[j-lookback:j+1])&(adj[j-lookback:j+1]>0),axis=0)&np.isfinite(cap[j])&(cap[j]>0)
    # Estimation safety: only past observations; avoid corrupted/ill-conditioned extreme series.
    valid &= np.max(np.abs(np.nan_to_num(hist,nan=10.)),axis=0)<2.
    universe=np.flatnonzero(valid)
    avg=np.mean(value[j-4:j+1],axis=0)
    eligible=valid&(cap[j]>=1e11)&np.isfinite(avg)&(avg>3e9)&np.isfinite(close[j])&(close[j]>0)
    ix=np.flatnonzero(eligible)
    if len(ix)<10:raise RuntimeError('Too few eligible stocks')
    X=hist[:,universe]; X=X-X.mean(axis=0)
    wm=cap[j,universe]/cap[j,universe].sum(); market=X@wm
    pos=np.searchsorted(universe,ix); Y=X[:,pos]; var=np.sum(Y*Y,axis=0)/(lookback-1)*252
    cov=.8*(Y.T@Y)/(lookback-1)*252 + np.diag(.2*var+1e-8)
    prior=delta*(.8*(Y.T@market)/(lookback-1)*252+.2*var*wm[pos])
    tp=D['target'][j,ix]; sd=D['target_sd'][j,ix]; n=D['coverage'][j,ix]
    good=np.isfinite(tp)&(tp>0)&np.isfinite(sd)&(sd>=0)&np.isfinite(n)&(n>=2)
    v=np.flatnonzero(good); mu=prior.copy()
    if len(v):
        conf=np.clip(.5*n[v]/(n[v]+5)/(1+(sd[v]/tp[v]/.20)**2),.02,.5)
        omega=tau*np.diag(cov)[v]*(1-conf)/conf
        Q=np.clip(tp[v]/close[j,ix[v]]-1,-.5,1.)
        mu+=tau*cov[:,v]@la.solve(tau*cov[np.ix_(v,v)]+np.diag(omega),Q-prior[v],assume_a='pos')
    models.append({'signal_date':str(dates[j].date()),'prior_universe':len(universe),'eligible':len(ix),'views':len(v)})
    return ix,cov,prior,mu

lastmarks=np.where(np.isfinite(adj[start-1])&(adj[start-1]>0),adj[start-1],1.)
lastweek=None
for i in range(start,T):
    j=i-1; week=tuple(dates[i].isocalendar()[:2]); weekly=week!=lastweek
    if weekly:
        ix,cov,prior,mu=model(j,None); lastweek=week
        print('model',str(dates[i].date()),len(ix),models[-1]['views'],flush=True)
    tradable=np.isfinite(aopen[i])&(aopen[i]>0)
    opens=np.where(tradable,aopen[i],lastmarks)
    for key,s in states.items():
        opening=s['units']*opens; nav_open=s['cash']+opening.sum(); desired=opening.copy(); reason='hold'
        if weekly:
            # Decisions use prior-close estimates; current open is used for exact notional sizing.
            upper=limits[ix]*(1-.004); small=cap[j,ix]<1e12
            w=solve(mu if key=='BL' else prior,cov,upper,small,opening[ix]/nav_open)
            desired=np.zeros(N); desired[ix]=w*nav_open; reason='weekly'
            opt_checks.append({'date':str(dates[i].date()),'strategy':key,'max_bound_excess':float(np.max(w-upper)),'small_weight':float(w[small].sum()),'stock_sum':float(w.sum())})
        else:
            # Correct excess at next open; cash retained until the next weekly rebalance.
            desired=np.minimum(desired,limits*.996*nav_open)
            small=np.isfinite(cap[j])&(cap[j]<1e12)
            if desired[small].sum()>.298*nav_open:desired[small]*=.298*nav_open/desired[small].sum()
            if np.any(np.abs(desired-opening)>1e-10):reason='limit_trim'
        desired[~tradable]=opening[~tradable]
        sell=np.maximum(opening-desired,0); buy=np.maximum(desired-opening,0)
        sellcash=float(sell.sum()); available=s['cash']+sellcash*(1-fee)
        if buy.sum()>0 and buy.sum()*(1+fee)>available:buy*=max(available,0)/(buy.sum()*(1+fee))
        delta_value=buy-sell; gross=float(buy.sum()+sellcash); cost=fee*gross
        s['cash']+=sellcash-buy.sum()-cost; s['cash']=max(s['cash'],0.); s['units']+=(delta_value/opens)
        if s['cash'] < -1e-8 or np.min(s['units']) < -1e-10:raise RuntimeError('Negative cash/shares')
        s['cost']+=cost; s['gross']+=gross
        for k in np.flatnonzero(np.abs(delta_value)>1e-9):
            s['trades'].append({'date':str(dates[i].date()),'code':codes[k],'name':names[k],'side':'BUY' if delta_value[k]>0 else 'SELL','notional_krw':abs(delta_value[k])*1e9,'cost_krw':abs(delta_value[k])*fee*1e9,'raw_open':float(op[i,k]),'reason':reason})
        marks=np.where(np.isfinite(adj[i])&(adj[i]>0),adj[i],lastmarks)
        holdings=s['units']*marks; nav=float(holdings.sum()+s['cash']); assert np.isfinite(nav) and nav>0; s['nav'].append(nav)
        s['daily'].append({'date':str(dates[i].date()),'nav_krw':nav*1e9,'cash_weight':s['cash']/nav,'gross_traded_krw':gross*1e9,'cost_krw':cost*1e9,'week':str(week)})
        if weekly:
            for k in np.flatnonzero(holdings>1e-8):s['holdings'].append({'date':str(dates[i].date()),'code':codes[k],'name':names[k],'weight':float(holdings[k]/nav)})
        s['constraint'].append({'date':str(dates[i].date()),'close_single_limit_excess':float(np.maximum(holdings/nav-limits,0).max()),'close_smallcap_weight':float(holdings[np.isfinite(cap[i])&(cap[i]<1e12)].sum()/nav)})
    lastmarks=np.where(np.isfinite(adj[i])&(adj[i]>0),adj[i],lastmarks)

results={}; navdf=pd.DataFrame(index=dates[start-1:]);navdf.index.name='date'
for key,s in states.items():
    nav=np.r_[1.,s['nav']]; dd=nav/np.maximum.accumulate(nav)-1; rr=nav[1:]/nav[:-1]-1
    navdf[key]=nav; navdf[key+'_drawdown']=dd
    daily=pd.DataFrame(s['daily']); weekly=daily.groupby('week',sort=False).agg(gross=('gross_traded_krw','sum'),average_nav=('nav_krw','mean'),days=('date','size'))
    weekly['turnover']=.5*weekly['gross']/weekly['average_nav']; weekly.to_csv(OUT/(key+'_weekly_turnover.csv'),encoding='utf-8-sig')
    rolling=nav[42:]/nav[:-42]-1
    results[key]={'total_return':float(nav[-1]-1),'mdd':float(dd.min()),'volatility_annual':float(np.std(rr,ddof=1)*np.sqrt(252)),'final_nav_krw':float(nav[-1]*1e9),'sum_cost_krw':float(s['cost']*1e9),'mean_cash_weight':float(daily.cash_weight.mean()),'weekly_turnover_mean':float(weekly.turnover.mean()),'turnover_miss_weeks':int((weekly.turnover<.05-1e-8).sum()),'weeks':len(weekly),'turnover_miss_excluding_last_partial':int((weekly.iloc[:-1].turnover<.05-1e-8).sum()),'rolling42_median':float(np.median(rolling)),'rolling42_min':float(rolling.min()),'rolling42_max':float(rolling.max()),'rolling42_positive_ratio':float(np.mean(rolling>0))}
    pd.DataFrame(s['trades']).to_csv(OUT/(key+'_trades.csv'),index=False,encoding='utf-8-sig')
    pd.DataFrame(s['holdings']).to_csv(OUT/(key+'_weights.csv'),index=False,encoding='utf-8-sig')
    daily.to_csv(OUT/(key+'_daily.csv'),index=False,encoding='utf-8-sig')
    pd.DataFrame(s['constraint']).to_csv(OUT/(key+'_limits.csv'),index=False,encoding='utf-8-sig')
navdf.to_csv(OUT/'nav_drawdown.csv',encoding='utf-8-sig')
pd.DataFrame(models).to_csv(OUT/'model_log.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(opt_checks).to_csv(OUT/'optimization_checks.csv',index=False,encoding='utf-8-sig')
summary={'start':str(dates[start].date()),'end':str(dates[-1].date()),'trading_days':T-start,'input_stocks':N,'eligible_min':min(m['eligible'] for m in models),'eligible_max':max(m['eligible'] for m in models),'results':results}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True),flush=True)

