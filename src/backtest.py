"""Rolling-origin evaluation against observed, unmodified monthly sales.

No future supplier seasonality, growth forecasts, inventory or corrections
from the current file snapshot are used to evaluate a historical forecast.
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from .forecasting import forecast_series, MODEL_NAMES,adaptive_forecast

def evaluate_sales(sales, end='2026-08-01', periods=6, max_skus=None, progress=None,training_by_origin=None,pooled_by_origin=None):
    end=pd.Timestamp(end).replace(day=1)
    if periods<1 or periods>12:raise ValueError('Периодов проверки: 1–12')
    data=sales.copy()
    data['date']=pd.to_datetime(data['date']).dt.to_period('M').dt.to_timestamp()
    data['quantity']=pd.to_numeric(data['quantity'],errors='raise')
    if not np.isfinite(data.quantity).all():raise ValueError('Продажи содержат нечисловые значения')
    grouped=data.groupby(['sku','date']).quantity.sum()
    starts=pd.date_range(end-pd.DateOffset(months=periods-1),end,freq='MS')
    # Choose the evaluation population using data BEFORE the first test month.
    pre=data[data.date<starts.min()].groupby('sku').quantity.sum().sort_values(ascending=False)
    skus=list(pre.index[:max_skus]) if max_skus else list(pre.index)
    rows=[]; skipped=0
    for index,sku in enumerate(skus):
        s=grouped.loc[sku].sort_index()
        observed_dates=set(s.index)
        if s.index.min()>=starts.min() or len(s[s.index<starts.min()])<12:
            skipped+=1;continue
        s=s.reindex(pd.date_range(s.index.min(),end,freq='MS'),fill_value=0)
        for dt in starts:
            if dt not in observed_dates:continue
            history=s[s.index<dt].clip(lower=0)
            future=pd.date_range(dt,periods=dt.days_in_month)
            actual=max(0.,float(s.get(dt,0)))
            # Every model, including its internal selection, sees the same prefix.
            methods=['mean6','seasonal_naive','seasonal_level','auto']+(['axioma'] if training_by_origin is not None else [])+(['adaptive','pooled'] if pooled_by_origin is not None else [])
            for method in methods:
                train=history
                if method in ('axioma','adaptive','pooled'):
                    origin=training_by_origin.get(dt,{})
                    train=origin.get(sku,history).reindex(history.index,fill_value=0).clip(lower=0)
                if method in ('adaptive','pooled'):
                    rates,info=adaptive_forecast(train,future)
                    selected_model=info['method']
                    if method=='pooled' and sku in pooled_by_origin.get(dt,{}):
                        rates=pooled_by_origin[dt][sku];selected_model='HistGradientBoosting + mean'
                    pred=float(rates.sum())
                else:
                    fitted=forecast_series(train,future,method='auto' if method=='axioma' else method)
                    pred=float(fitted['daily'].sum());selected_model=fitted['model']
                rows.append(dict(sku=str(sku),month=dt.strftime('%Y-%m'),method=method,
                                 selected_model=selected_model,actual=actual,prediction=pred,
                                 abs_error=abs(pred-actual),signed_error=pred-actual,
                                 underforecast=max(0,actual-pred),overforecast=max(0,pred-actual)))
        if progress and (index%50==0 or index==len(skus)-1):progress((index+1)/len(skus))
    detail=pd.DataFrame(rows)
    summary=summarize(detail)
    metadata=dict(start=starts.min().strftime('%Y-%m'),end=end.strftime('%Y-%m'),
                  sku_count=int(detail.sku.nunique()) if len(detail) else 0,skipped=skipped,
                  observations=int(len(detail)/(7 if pooled_by_origin is not None else 5 if training_by_origin is not None else 4)),
                  target='Фактические месячные продажи после возвратов, с нижней границей 0',
                  limitation='Продажи не равны скрытому спросу: stockout и разовые проекты могут искажать оценку. Денежная экономия не измерялась.',
                  protocol='Последовательная проверка на прошлом. На каждом шаге обучение и выбор модели используют только более ранние месяцы. Текущие коэффициенты поставщика и остатки не используются.')
    return summary,detail,metadata

def evaluate_dataset(ds,end='2026-08-01',periods=6,max_skus=None,progress=None,include_team_models=True):
    """Evaluate production anomaly cleaning + forecasting, with raw-sales target."""
    from .anomalies import clean_transactions
    from .reconciliation import reconcile_sales
    end=pd.Timestamp(end).replace(day=1)
    starts=pd.date_range(end-pd.DateOffset(months=periods-1),end,freq='MS')
    training={};pooled={} if include_team_models else None;base=ds.sales.groupby(['sku','date']).quantity.sum()
    for dt in starts:
        raw=base[base.index.get_level_values('date')<dt].copy()
        tx=ds.transactions[ds.transactions.date<dt].copy() if not ds.transactions.empty else ds.transactions
        if not tx.empty:
            clean,_=clean_transactions(tx)
            merged,removed,_=reconcile_sales(raw.reset_index(),tx,clean)
            raw=merged.set_index(['sku','date']).quantity
            raw=(raw.clip(lower=0)-removed.reindex(raw.index,fill_value=0)).clip(lower=0)
        else:raw=raw.clip(lower=0)
        per_sku={sku:g.droplevel('sku').sort_index() for sku,g in raw.groupby(level='sku')}
        # Reindex missing internal months just as the operational engine does.
        per_sku={sku:s.reindex(pd.date_range(s.index.min(),dt-pd.offsets.MonthBegin(1),freq='MS'),fill_value=0) for sku,s in per_sku.items()}
        training[dt]=per_sku
        if include_team_models:
            from .pooled import pooled_forecasts
            prepared={sku:pd.DataFrame({'Дата':s.index,'С учётом отсутствия':s.values}) for sku,s in per_sku.items()}
            pooled[dt]=pooled_forecasts(prepared,pd.date_range(dt,periods=dt.days_in_month))
    summary,detail,metadata=evaluate_sales(ds.sales,end,periods,max_skus,progress,training,pooled)
    if not detail.empty:
        units=ds.products.drop_duplicates('sku').set_index('sku')['unit']
        detail['unit']=detail.sku.map(units).fillna('Не указана')
    return summary,detail,metadata

def summarize(detail):
    if detail.empty:return pd.DataFrame(columns=['method','model','wape','mae','bias','underforecast','overforecast','n'])
    out=[]
    for method,g in detail.groupby('method',sort=False):
        actual=float(g.actual.sum())
        grouped=g.groupby('sku').agg(actual=('actual','sum'),error=('abs_error','sum'),prediction=('prediction','sum'))
        valid=grouped[grouped.actual>0]
        ratios=valid.error/valid.actual
        out.append(dict(method=method,model=MODEL_NAMES[method],wape=float(g.abs_error.sum()/actual) if actual else np.nan,
                        mae=float(g.abs_error.mean()),bias=float(g.signed_error.sum()/actual) if actual else np.nan,
                        macro_wape=float(ratios.mean()) if len(ratios) else np.nan,median_wape=float(ratios.median()) if len(ratios) else np.nan,
                        scored_skus=len(valid),zero_actual_skus=int(grouped.actual.eq(0).sum()),zero_actual_with_forecast=int((grouped.actual.eq(0)&grouped.prediction.gt(0)).sum()),
                        underforecast=float(g.underforecast.sum()),overforecast=float(g.overforecast.sum()),n=len(g)))
    return pd.DataFrame(out)
