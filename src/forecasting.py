"""Small auditable forecasting models with chronological model selection.

All calibration uses observations strictly before the forecast origin. The
outer backtest calls this same function with a truncated series.
"""
from functools import lru_cache
import numpy as np
import pandas as pd

MODEL_NAMES = {'mean3':'Среднее 3 месяца','mean6':'Среднее 6 месяцев',
               'seasonal_naive':'Тот же месяц год назад','seasonal_level':'Сезонность и уровень',
               'ses':'Экспоненциальное сглаживание','auto':'Автовыбор по прошлым периодам',
               'classic':'Исходная модель Axioma','axioma':'Axioma · очищенный спрос'}
CANDIDATES = ('mean3','mean6','ses','seasonal_naive','seasonal_level')
MODEL_NAMES.update(pooled='Общая ML-модель',adaptive='Расширенный автовыбор · редкий спрос',legacy='Исходная модель Axioma')

def seasonal_factors(values, months, supplied=None):
    """Shrink noisy month effects toward one; complete years only."""
    effects={m:[] for m in range(1,13)}
    years=sorted(set(y for y,m in months))
    for y in years:
        indexes=[i for i,(year,_) in enumerate(months) if year==y]
        if len(indexes)!=12:continue
        avg=np.mean(values[indexes])
        if avg<=0:continue
        for i in indexes:effects[months[i][1]].append(values[i]/avg)
    result={}
    for m in effects:
        local=float(np.median(effects[m])) if effects[m] else 1.
        # A single year gives weak evidence; shrink avoids memorizing one-off months.
        weight=min(.8,len(effects[m])/(len(effects[m])+1.5))
        prior=float(supplied.get(m,1.)) if supplied else 1.
        result[m]=np.clip(weight*local+(1-weight)*prior,.35,2.5)
    norm=np.mean(list(result.values()))
    return {m:float(v/norm) for m,v in result.items()}

def _predict(rates, months, target, model, use_seasonality=True, use_trend=True, supplied=None):
    n=len(rates)
    if not n:return 0.
    if model=='mean3':return float(np.mean(rates[-3:]))
    if model=='mean6':return float(np.mean(rates[-6:]))
    if model=='ses':
        level=float(np.mean(rates[:min(3,n)]))
        for value in rates:level=.3*float(value)+.7*level
        return max(0.,level)
    if model=='seasonal_naive':
        if not use_seasonality:return float(np.mean(rates[-6:]))
        index=next((i for i,x in enumerate(months) if x==(target[0]-1,target[1])),None)
        return float(rates[index]) if index is not None else float(np.mean(rates[-6:]))
    factors=seasonal_factors(rates,months,supplied) if use_seasonality else {m:1. for m in range(1,13)}
    de=np.array([r/factors[m] for r,(_,m) in zip(rates,months)])
    level=float(np.mean(de[-6:]))
    trend=1.
    if use_trend and n>=6 and np.mean(de[-6:-3])>0:
        # Damped trend; a single high month is not enough to establish growth.
        previous=float(np.median(de[-6:-3])); recent=de[-3:]
        if previous>0 and (np.all(recent>previous) or np.all(recent<previous)):
            trend=float(np.clip((np.median(recent)/previous)**.5,.75,1.35))
    return max(0.,level*trend*factors[target[1]])

@lru_cache(maxsize=24000)
def _fit_cached(values_tuple, months_tuple, use_seasonality, use_trend, method, prior_tuple):
    rates=np.array(values_tuple,dtype=float);months=list(months_tuple)
    candidates=[m for m in CANDIDATES if use_seasonality or m!='seasonal_naive']
    if len(rates)<18:candidates=[m for m in candidates if m!='seasonal_naive']
    # Historical validation origins, all before the actual forecast origin.
    origins=list(range(max(6,len(rates)-6),len(rates)))
    scores={}; residuals={}
    for model in candidates:
        errors=[]
        for t in origins:
            # Supplier priors may have been calculated later; do not use them in selection.
            pred=_predict(rates[:t],months[:t],months[t],model,use_seasonality,use_trend)
            errors.append(abs(float(rates[t])-pred))
        scores[model]=float(np.mean(errors)) if errors else float('inf')
        residuals[model]=errors
    chosen=min(candidates,key=lambda m:(scores[m],candidates.index(m))) if method=='auto' else method
    if chosen not in CANDIDATES:raise ValueError('Неизвестный метод прогноза')
    err=residuals.get(chosen,[])
    # Descriptive uncertainty from a small calibration window, not a coverage guarantee.
    error_band=float(np.quantile(err,.8)) if err else 0.
    return chosen,scores,error_band,len(origins)

def forecast_series(series, future_dates, method='auto', seasonality=True, trend=True, supplied=None):
    series=series.sort_index().clip(lower=0).astype(float)
    dates=pd.DatetimeIndex(future_dates)
    if len(dates)==0:raise ValueError('Пустой горизонт прогноза')
    if len(series) and series.index.max()>=dates.min().replace(day=1):
        raise ValueError('История должна завершаться до месяца прогноза')
    months=tuple((d.year,d.month) for d in series.index)
    rates=tuple(float(q)/d.days_in_month for d,q in series.items())
    chosen,scores,band,n=_fit_cached(rates,months,seasonality,trend,method,tuple(sorted((supplied or {}).items())))
    monthly={}
    for dt in dates:
        target=(dt.year,dt.month)
        if target not in monthly:monthly[target]=_predict(np.array(rates),list(months),target,chosen,seasonality,trend,supplied)
    daily=np.array([monthly[(d.year,d.month)] for d in dates],dtype=float)
    return {'daily':daily,'model':chosen,'model_name':MODEL_NAMES[chosen],
            'validation_mae_daily':None if not np.isfinite(scores.get(chosen,float('inf'))) else scores[chosen],
            'calibration_months':n,'low':max(0.,float(daily.sum()-band*len(dates))),
            'high':float(daily.sum()+band*len(dates)),'scores':scores}
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

