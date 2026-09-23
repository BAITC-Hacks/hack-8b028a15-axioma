"""Deterministic, inspectable inventory policy. No network or model calls."""
from dataclasses import dataclass, field, replace
import math
import numpy as np
import pandas as pd
from .data import Dataset, number
from .forecasting import adaptive_forecast

@dataclass
class Settings:
    as_of: str = '2026-09-23'
    lead_days: int = 21
    review_days: int = 14
    safety_days: int = 7
    growth_percent: float = 0
    remove_outliers: bool = True
    seasonality: bool = True
    trend: bool = True
    compensate_stockout: bool = True
    approximate_stockout: bool = False
    category_factors: dict = field(default_factory=dict)
    forecast_method: str = 'legacy'
    use_source_growth: bool = False

def clean_transactions(tx):
    if tx.empty:return tx.copy(),pd.DataFrame()
    t=tx.copy()
    t['quantity']=t.quantity.astype(float)
    t['day']=t.date.dt.normalize()
    # Aggregate split lines of the same invoice before detecting large orders.
    docs=t.groupby(['sku','document','day'],as_index=False).agg(quantity=('quantity','sum'),client_id=('client_id','first'))
    out=[]; groups=[]
    for sku,g in docs.groupby('sku'):
        g=g.copy(); pos=g.quantity[g.quantity>0]
        if len(pos)>=8:
            q1,q3=pos.quantile([.25,.75]); median=float(pos.median())
            threshold=max(5*median,float(q3+3*(q3-q1)),1)
            mask=g.quantity>threshold
            for _,r in g[mask].iterrows():out.append(dict(sku=sku,date=r.day,document=r.document,original=r.quantity,regular=median,excluded=r.quantity-median,reason='Крупная разовая накладная'))
            g.loc[mask,'quantity']=median
            # Catch one large client-day split into many smaller invoices.
            known=g[g.client_id.fillna('').ne('')]
            if not known.empty:
                cg=known.groupby(['client_id','day']).quantity.sum()
                positive=cg[cg>0]
                if len(positive)>=8:
                    a,b=positive.quantile([.25,.75]); med=float(positive.median()); cap=max(5*med,float(b+3*(b-a)),1)
                    for (client,day),qty in cg[cg>cap].items():
                        ix=g.client_id.eq(client)&g.day.eq(day)&g.quantity.gt(0)
                        total=g.loc[ix,'quantity'].sum()
                        if total>0:
                            g.loc[ix,'quantity']*=med/total
                            out.append(dict(sku=sku,date=day,document='Группа клиента',original=qty,regular=med,excluded=qty-med,reason='Разовый всплеск клиента за день'))
        groups.append(g)
    result=pd.concat(groups,ignore_index=True).rename(columns={'day':'date'})
    return result,pd.DataFrame(out)

def calculate(ds: Dataset, settings: Settings, _pooled=None):
    cfg=settings; asof=pd.Timestamp(cfg.as_of).normalize(); cutoff=asof.replace(day=1)
    if cfg.forecast_method=='pooled' and _pooled is None:
        from .pooled import pooled_forecasts
        _,prepared,_=calculate(ds,replace(cfg,forecast_method='legacy'))
        future=pd.date_range(asof,periods=cfg.lead_days+cfg.review_days)
        _pooled=pooled_forecasts(prepared,future,cfg.seasonality,cfg.trend and not cfg.use_source_growth,ds.seasonal)
    raw=ds.sales.copy()
    tx=ds.transactions.copy()
    if not tx.empty:
        # Fit the outlier thresholds on the same completed months as the forecast.
        tx=tx[tx.date<cutoff].copy()
        cleaned,anomalies=clean_transactions(tx)
        raw_tx=tx.assign(date=tx.date.dt.to_period('M').dt.to_timestamp()).groupby(['sku','date']).quantity.sum()
        clean_tx=cleaned.assign(date=cleaned.date.dt.to_period('M').dt.to_timestamp()).groupby(['sku','date']).quantity.sum()
        removed=(raw_tx-clean_tx).clip(lower=0)
        # Monthly reports take precedence. Transaction-only products still need
        # their history when other products have a monthly report.
        fallback=raw_tx.reset_index()
        fallback=fallback[~fallback.sku.isin(raw.sku)]
        if not fallback.empty:
            raw=fallback if raw.empty else pd.concat([raw,fallback],ignore_index=True)
    else:removed=pd.Series(dtype=float); anomalies=pd.DataFrame()
    raw=raw[raw.date<cutoff].copy() # Do not use the incomplete current month.
    sales_by_sku={sku:g.groupby('date').quantity.sum().sort_index() for sku,g in raw.groupby('sku')}
    histories={}; results=[]
    for _,p in ds.products.iterrows():
        sku=p.sku
        s=sales_by_sku.get(sku,pd.Series(dtype=float))
        if s.empty:
            results.append(dict(sku=sku,article=p.article,name=p['name'],category=str(p.category),supplier=ds.supplier,stock=p.stock,recommended=np.nan,status='Нет истории',reason='Нет завершённых месяцев продаж. Нужна ручная оценка.'));continue
        full=pd.date_range(s.index.min(),cutoff-pd.offsets.MonthBegin(1),freq='MS')
        s=s.reindex(full,fill_value=0)
        regular=s.clip(lower=0).astype(float).copy(); excluded=0.0
        if cfg.remove_outliers:
            if not removed.empty and sku in removed.index.get_level_values(0):
                adjustments=removed.loc[sku].reindex(full,fill_value=0)
                regular=(regular-adjustments).clip(lower=0)
                excluded=float((s.clip(lower=0)-regular).sum())
            elif tx.empty and len(s)>=8:
                # Conservative fallback: compare same calendar month across years.
                for dt in full:
                    peers=s[(s.index.month==dt.month)&(s.index<dt)]
                    if len(peers)>=2:
                        baseline=max(float(peers.median()),1)
                        if regular.loc[dt]>5*baseline:
                            excluded+=regular.loc[dt]-baseline;regular.loc[dt]=baseline
        corrected=regular.copy(); lost=0.; stockout_mode='Нет подтверждённых периодов'
        periods=ds.stockouts[ds.stockouts.sku==sku] if not ds.stockouts.empty else ds.stockouts
        if cfg.compensate_stockout:
            known=ds.stocks[ds.stocks.sku==sku] if not ds.stocks.empty else ds.stocks
            for dt in full:
                days=dt.days_in_month; absent=set()
                for _,r in periods.iterrows():
                    start=max(pd.Timestamp(r.start),dt); end=min(pd.Timestamp(r.end),dt+pd.offsets.MonthEnd(0))
                    if start<=end:absent.update(pd.date_range(start,end).date)
                if absent:
                    available=days-len(absent)
                    peers=regular.drop(dt).tail(12)
                    daily=float((peers/peers.index.days_in_month).median()) if len(peers) else 0
                    replacement=regular.loc[dt]/available*days if available>0 else daily*days
                    corrected.loc[dt]=max(regular.loc[dt],replacement);stockout_mode='Подтверждённые периоды'
                elif cfg.approximate_stockout and not known.empty:
                    snapshot=known.loc[known.date.eq(dt),'quantity']
                    if len(snapshot) and snapshot.notna().any() and snapshot.iloc[0]==0:
                        peers=regular.drop(dt).tail(12)
                        if len(peers):corrected.loc[dt]=max(regular.loc[dt],float(peers.median()));stockout_mode='Приближение по месячным снимкам'
            lost=float((corrected-regular).sum())
        # Per-SKU seasonality from completed years; supplied factors provide a fallback/prior.
        factors={m:1. for m in range(1,13)}
        if cfg.seasonality:
            ratios={m:[] for m in range(1,13)}
            for year,g in corrected.groupby(corrected.index.year):
                if len(g)==12 and g.sum()>0:
                    daily=g/g.index.days_in_month; mean=float(daily.mean())
                    for dt,v in daily.items():ratios[dt.month].append(float(v)/mean)
            for m in factors:
                local=float(np.median(ratios[m])) if ratios[m] else None
                supplied=ds.seasonal.get(m)
                factors[m]=np.clip(.8*local+.2*supplied if local is not None and supplied else local if local is not None else supplied if supplied else 1,.25,3)
            norm=np.mean(list(factors.values()));factors={m:v/norm for m,v in factors.items()}
        deseasonal=pd.Series([v/dt.days_in_month/factors[dt.month] for dt,v in corrected.items()],index=corrected.index)
        recent=deseasonal.tail(6)
        daily=float(recent.mean()) if len(recent) else 0
        growth=0.
        if cfg.trend and len(deseasonal)>=6:
            old=float(deseasonal.iloc[-6:-3].mean());new=float(deseasonal.iloc[-3:].mean())
            if old>0: growth=float(np.clip(new/old-1,-.5,1))
        horizon=cfg.lead_days+cfg.review_days
        future=pd.date_range(asof,periods=horizon)
        future_season=float(np.mean([factors[d.month] for d in future]))
        rate=max(0,daily*(1+growth)*(1+cfg.growth_percent/100)*future_season)
        daily_rates=np.full(horizon,rate)
        model_info=dict(method='Эвристическая модель',demand_type='Не классифицирован',cv_months=0,cv_wape=np.nan,stress_daily=0.,candidates=[])
        source_growth=number(p.get('source_growth_percent',np.nan),np.nan)
        has_source_growth=cfg.use_source_growth and np.isfinite(source_growth)
        if cfg.forecast_method in ('adaptive','pooled'):
            daily_rates,model_info=adaptive_forecast(corrected,future,cfg.seasonality,cfg.trend and not has_source_growth,ds.seasonal)
            if cfg.forecast_method=='pooled' and sku in (_pooled or {}):
                daily_rates=_pooled[sku].copy()
                model_info.update(method='Общая модель HistGradientBoosting + среднее',cv_months=0,cv_wape=np.nan,stress_daily=0.,candidates=[])
            daily_rates*=max(0,1+cfg.growth_percent/100)
            growth=0.  # Selected methods carry their own level/trend; never multiply it twice.
        if has_source_growth:
            if cfg.forecast_method not in ('adaptive','pooled'):daily_rates/=1+growth
            daily_rates*=max(0,1+source_growth/100)
            growth=source_growth/100
        rate=float(daily_rates.mean())
        forecast=float(daily_rates.sum())
        cat=float(cfg.category_factors.get(str(p.category),1))
        safety=rate*cfg.safety_days*cat
        arriving=ds.transit[ds.transit.sku==sku] if not ds.transit.empty else ds.transit
        due=0.; late=0.; unknown_eta=0.
        if not arriving.empty:
            within=arriving.eta.notna()&arriving.eta.ge(asof)&arriving.eta.lt(asof+pd.Timedelta(days=horizon))
            due=float(arriving.loc[within,'quantity'].clip(lower=0).sum())
            late=float(arriving.loc[arriving.eta.ge(asof+pd.Timedelta(days=horizon)),'quantity'].clip(lower=0).sum())
            unknown_eta=float(arriving.loc[arriving.eta.isna()|arriving.eta.lt(asof),'quantity'].clip(lower=0).sum())
        stock=number(p.stock,np.nan);pack=max(number(p.pack,1),1);moq=max(number(p.moq,1),1)
        need=max(0,forecast+safety-stock-due) if np.isfinite(stock) else np.nan
        order=math.ceil(max(need,moq)/pack)*pack if np.isfinite(need) and need>1e-8 else 0 if np.isfinite(need) else np.nan
        # Date-aware deficit before the newly placed order arrives.
        balance=stock; shortage=False; first_shortage=None
        if np.isfinite(stock):
            for i,d in enumerate(pd.date_range(asof,periods=cfg.lead_days)):
                if not arriving.empty:balance+=float(arriving.loc[arriving.eta.eq(d),'quantity'].clip(lower=0).sum())
                balance-=daily_rates[i]
                if balance<0:
                    shortage=True
                    if first_shortage is None:first_shortage=str(d.date())
        status='Нужен остаток' if not np.isfinite(stock) else 'Срочно' if shortage else 'Заказать' if order>0 else 'Достаточно'
        reason=(f'Спрос {forecast:.1f} + страховой запас {safety:.1f} − свободный остаток {stock:.1f} − поступления {due:.1f}. '
                f'Минимум {moq:g}, кратность {pack:g}. Исключено разовых продаж: {excluded:.1f}; восстановлено спроса: {lost:.1f}.') if np.isfinite(stock) else f'Текущий остаток неизвестен. При остатке ниже {max(0,forecast+safety-due):.1f} ед. по этой модели нужно пополнение. Количество рассчитывается после выбора сценария или ввода остатка.'
        if unknown_eta:reason+=f' Требуют уточнения даты поступления: {unknown_eta:g} ед.'
        if cfg.forecast_method in ('adaptive','pooled'):reason+=f' Метод: {model_info["method"]}; внутренних проверочных месяцев: {model_info["cv_months"]}.'
        if has_source_growth:reason+=f' Рост из сводки {source_growth:g}% применён вместо тренда истории.'
        results.append(dict(sku=sku,article=p.article,name=p['name'],category=str(p.category),supplier=ds.supplier,unit=p.unit,stock=stock,stock_threshold=round(max(0,forecast+safety-due),2),in_transit=due,late_transit=late,forecast=round(forecast,2),safety=round(safety,2),daily=round(rate,3),growth=round(growth*100,1),season=round(future_season,3),excluded=round(excluded,2),lost=round(lost,2),pack=pack,moq=moq,recommended=order,status=status,stockout_mode=stockout_mode,reason=reason))
        histories[sku]=pd.DataFrame({'Дата':full,'Продажи':s.values,'Без всплесков':regular.values,'С учётом отсутствия':corrected.values})
        stress=max(0,model_info['stress_daily'])*(horizon+cfg.safety_days*cat)*max(0,1+cfg.growth_percent/100)
        if has_source_growth:stress*=max(0,1+source_growth/100)
        stress_need=max(0,forecast+safety+stress-stock-due) if np.isfinite(stock) else np.nan
        stress_order=math.ceil(max(stress_need,moq)/pack)*pack if np.isfinite(stress_need) and stress_need>1e-8 else 0 if np.isfinite(stress_need) else np.nan
        results[-1].update(method=model_info['method'],demand_type=model_info['demand_type'],cv_months=model_info['cv_months'],cv_wape=model_info['cv_wape'],stress_recommended=stress_order,first_shortage=first_shortage,source_growth_percent=source_growth if has_source_growth else np.nan)
        histories[sku].attrs['model_info']=model_info
        histories[sku].attrs['future']=[{'Дата':str(d.date()),'Прогноз в день':float(v)} for d,v in zip(future,daily_rates)]
    return pd.DataFrame(results),histories,anomalies
