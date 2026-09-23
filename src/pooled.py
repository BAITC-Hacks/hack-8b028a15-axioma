"""Cross-product gradient boosting with scale-free lag features.

Training examples end before the forecast origin. No prices, customer identities,
stock snapshots or target-month sales are exposed as future features.
"""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def features(y, target, use_seasonality=True, use_trend=True):
    y=np.asarray(y,dtype=float)
    scale=max(float(y[-6:].mean()),float(y.mean())*.1,1e-6)
    lag=lambda n:float(y[-n]/scale) if len(y)>=n else 0.
    last=y[-12:]
    x=[lag(n) for n in (1,2,3,6,12)]
    x += [float(y[-n:].mean()/scale) for n in (3,6,12)]
    x += [float(last.std()/scale),float(np.mean(last==0)),
          float(np.mean(y[-3:]==0)),min(len(y),48)/48,
          float(np.sin(2*np.pi*target.month/12)),float(np.cos(2*np.pi*target.month/12))]
    if not use_seasonality:x[4]=0.;x[-2:]=[0.,0.]
    if not use_trend:
        x[:4]=[1.]*4;x[5:8]=[1.]*3
    return x,scale


def pooled_forecasts(histories, future, use_seasonality=True, use_trend=True, seasonal=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    series={sku:h.set_index('Дата')['С учётом отсутствия'].astype(float) for sku,h in histories.items()}
    X=[];Y=[]
    for s in series.values():
        y=s.to_numpy()/s.index.days_in_month.to_numpy()
        for i in range(6,len(y)):
            x,scale=features(y[:i],s.index[i],use_seasonality,use_trend)
            X.append(x);Y.append(min(float(y[i]/scale),10.))
    if len(X)<200 or not any(Y):return {}
    model=HistGradientBoostingRegressor(loss='poisson',max_iter=120,
        max_leaf_nodes=15,min_samples_leaf=20,learning_rate=.05,
        l2_regularization=10,early_stopping=False,random_state=42)
    # Bound native threads so a background benchmark does not freeze the UI.
    with threadpool_limits(limits=2):model.fit(np.asarray(X),np.asarray(Y))
    result={sku:[] for sku in series}
    histories_rates={sku:(s.to_numpy()/s.index.days_in_month.to_numpy()).tolist() for sku,s in series.items()}
    months=pd.period_range(min(future).to_period('M'),max(future).to_period('M'),freq='M')
    monthly={sku:{} for sku in series}
    for month in months:
        sku_list=list(series); batch=[];scales=[]
        for sku in sku_list:
            x,scale=features(histories_rates[sku],month.to_timestamp(),use_seasonality,use_trend)
            batch.append(x);scales.append(scale)
        with threadpool_limits(limits=2):predictions=model.predict(np.asarray(batch))
        for sku,pred,scale in zip(sku_list,predictions,scales):
            y=histories_rates[sku]
            # Shrink toward the stable local mean rather than trusting a small panel.
            local_mean=float(np.mean(y[-6:]))
            if use_seasonality and seasonal:
                from .forecasting import predict_rates
                known_dates=pd.date_range(end=month.to_timestamp()-pd.offsets.MonthBegin(1),periods=len(y),freq='MS')
                local_mean=float(predict_rates(y,known_dates,[month.to_timestamp()],'supplier',seasonal)[0])
            rate=max(0,.5*float(pred)*scale+.5*local_mean)
            # A shared positive intercept must not create purchases for dormant
            # products. Confirmed stockouts were compensated before this step.
            if not any(v>0 for v in y[-6:]):rate=0.
            monthly[sku][month]=rate;y.append(rate)
    for sku in result:
        result[sku]=np.array([monthly[sku][d.to_period('M')] for d in future])
    return result
