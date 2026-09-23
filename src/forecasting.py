"""Small auditable forecasting tournament. No services or fitted external models.

Candidates are selected with rolling origins inside the available history.
The outer evaluation in validation.py remains separate from this selection.
Monthly volumes are normalized by calendar days before fitting.
"""
import numpy as np
import pandas as pd

LABELS = {
    'mean3': 'Среднее 3 месяца', 'mean6': 'Среднее 6 месяцев',
    'mean12': 'Среднее 12 месяцев', 'ses': 'Экспоненциальное сглаживание',
    'tsb': 'TSB: вероятность × размер спроса', 'sba': 'Croston–SBA',
    'seasonal': 'Сезонный аналог прошлого года',
    'seasonal_level': 'Сезонный аналог с поправкой уровня',
    'damped': 'Затухающий тренд', 'supplier': 'Профиль сезонности поставщика',
}


def predict_rates(values, dates, future, method, seasonal=None):
    """Daily demand rates, held constant within each future calendar month."""
    y=np.asarray(values,dtype=float)
    if not len(y): return np.zeros(len(future))
    if method.startswith('mean'):
        return np.full(len(future),y[-int(method[4:]):].mean())
    if method=='ses':
        level=y[0]
        for v in y[1:]: level=.2*v+.8*level
        return np.full(len(future),level)
    if method in ('tsb','sba'):
        positive=np.flatnonzero(y>0)
        if not len(positive): return np.zeros(len(future))
        first=int(positive[0]); size=y[first]; interval=float(first+1)
        probability=1/interval; elapsed=1.
        for v in y[first+1:]:
            probability=.2*(v>0)+.8*probability
            if v>0:
                size=.2*v+.8*size
                interval=.2*elapsed+.8*interval
                elapsed=1.
            else: elapsed+=1
        rate=probability*size if method=='tsb' else .9*size/interval
        return np.full(len(future),rate)
    if method=='damped':
        tail=y[-6:]; level=tail[-3:].mean()
        slope=np.polyfit(np.arange(len(tail)),tail,1)[0] if len(tail)>1 else 0.
        months=np.array([(d.year-dates[-1].year)*12+d.month-dates[-1].month for d in future])
        # Damping limits extrapolation; level guards prevent negative demand.
        result=level+slope*(.8*(1-.8**months)/.2)
        return np.clip(result,0,max(2*float(tail.mean()),float(tail.max())))
    if method=='supplier':
        profile={m:max(float((seasonal or {}).get(m,1)),.01) for m in range(1,13)}
        norm=np.mean(list(profile.values())); profile={m:v/norm for m,v in profile.items()}
        base=np.mean([v/profile[d.month] for v,d in zip(y[-6:],dates[-6:])])
        return np.array([base*profile[d.month] for d in future])
    previous={d.month:v for d,v in zip(dates[-12:],y[-12:])}
    ratio=1.
    if method=='seasonal_level' and len(y)>=15:
        old=y[-15:-12].mean()
        if old>0: ratio=np.clip(y[-3:].mean()/old,.5,2.)
    return np.array([previous.get(d.month,y[-6:].mean())*ratio for d in future])


def demand_profile(series):
    values=np.maximum(np.asarray(series,dtype=float),0)
    pos=values[values>0]
    if not len(pos):return 'Нет положительных продаж',float('inf'),0.
    adi=len(values)/len(pos)
    cv2=float((pos.std()/pos.mean())**2)
    kind=('Редкий нерегулярный' if cv2>=.49 else 'Редкий') if adi>=1.32 else ('Нестабильный' if cv2>=.49 else 'Регулярный')
    return kind,adi,cv2


def adaptive_forecast(series, future, use_seasonality=True, use_trend=True, seasonal=None):
    series=series.astype(float).clip(lower=0)
    dates=series.index
    y=series.to_numpy()/dates.days_in_month.to_numpy()
    kind,adi,cv2=demand_profile(series)
    methods=['mean6','mean3','mean12','ses']
    if adi>=1.32:methods+=['tsb','sba']
    if use_trend:methods+=['damped']
    if use_seasonality and len(y)>=18:
        methods+=['seasonal']
        if use_trend:methods+=['seasonal_level']
    if use_seasonality and seasonal:methods+=['supplier']
    origins=list(range(max(6,len(y)-6),len(y)))
    # Seasonal candidates must have a full previous cycle at every origin.
    if any(m.startswith('seasonal') for m in methods):origins=[i for i in origins if i>=12]
    errors={m:[] for m in methods}
    for i in origins:
        for m in methods:
            pred=predict_rates(y[:i],dates[:i],dates[i:i+1],m,seasonal)[0]
            errors[m].append(float(y[i]-pred))
    scores={m:float(np.mean(np.abs(e))) if e else float('inf') for m,e in errors.items()}
    ranked=sorted(methods,key=lambda m:scores[m])
    winner=ranked[0] if origins else 'mean6'
    best=scores[winner]
    # Average near-ties to reduce unstable winner switching; no outer targets.
    chosen=[m for m in ranked if scores[m]<=best*1.05+1e-12][:3] if origins else [winner]
    forecasts=[predict_rates(y,dates,future,m,seasonal) for m in chosen]
    rates=np.maximum(np.mean(forecasts,axis=0),0)
    residual=np.mean([errors[m] for m in chosen],axis=0) if origins else np.array([])
    # A stress scenario, NOT a claimed calibrated prediction interval.
    stress=max(0,float(np.quantile(residual,.9))) if len(residual) else 0.
    denominator=float(np.mean(y[origins])) if origins else 0.
    cv_wape=100*float(np.mean(np.abs(residual)))/denominator if denominator else np.nan
    return rates,dict(method=' + '.join(LABELS[m] for m in chosen),
        demand_type=kind,adi=adi,cv2=cv2,cv_months=len(origins),cv_wape=cv_wape,
        stress_daily=stress,candidates=[dict(method=LABELS[m],mae_daily=scores[m]) for m in ranked])
