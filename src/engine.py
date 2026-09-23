"""Deterministic, inspectable inventory policy. No network or model calls."""
from dataclasses import dataclass, field, replace
import math
import numpy as np
import pandas as pd
from .data import Dataset, number
from .forecasting import forecast_series, adaptive_forecast, demand_profile
from .reconciliation import reconcile_sales

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
    forecast_method: str = 'classic'
    anomaly_decisions: dict = field(default_factory=dict)
    use_source_growth: bool = True

from .anomalies import clean_transactions

def calculate(ds: Dataset, settings: Settings, _pooled=None):
    cfg=settings; asof=pd.Timestamp(cfg.as_of).normalize(); cutoff=asof.replace(day=1)
    for name,low,high in [('lead_days',1,365),('review_days',0,90),('safety_days',0,90),('growth_percent',-50,100)]:
        value=getattr(cfg,name)
        if not np.isfinite(value) or value<low or value>high:raise ValueError(f'Недопустимый параметр: {name}')
        if name.endswith('_days') and value!=int(value):raise ValueError(f'{name}: укажите целое число дней')
    if any(not np.isfinite(v) or v<0 or v>5 for v in cfg.category_factors.values()):raise ValueError('Множитель категории должен быть от 0 до 5')
    if cfg.forecast_method not in ('classic','legacy','auto','mean3','mean6','seasonal_naive','seasonal_level','ses','adaptive','pooled'):raise ValueError('Неизвестный метод прогноза')
    if cfg.forecast_method=='pooled' and _pooled is None:
        from .pooled import pooled_forecasts
        _,prepared,_=calculate(ds,replace(cfg,forecast_method='classic'))
        max_lead=max([cfg.lead_days]+[int(number(v,cfg.lead_days)) for v in ds.products.get('lead_days',[])])
        future=pd.date_range(asof,periods=max_lead+cfg.review_days)
        _pooled=pooled_forecasts(prepared,future,cfg.seasonality,cfg.trend,ds.seasonal)
    raw=ds.sales.copy()
    tx=ds.transactions.copy()
    if not tx.empty:
        tx=tx[tx.date<cutoff].copy() # Incomplete month must not affect historical outlier thresholds.
        cleaned,anomalies=clean_transactions(tx,cfg.anomaly_decisions)
        raw,removed,reconciliation=reconcile_sales(raw,tx,cleaned)
        if not anomalies.empty:
            anomalies['month']=pd.to_datetime(anomalies.date).dt.to_period('M').dt.to_timestamp()
            anomalies=anomalies.merge(reconciliation[['sku','date','consistent']].rename(columns={'date':'month'}),on=['sku','month'],how='left')
            anomalies['application']=np.where(anomalies.consistent.eq(True),'Сверено с месячным итогом','Не вычтено: расхождение источников')
    else:removed=pd.Series(dtype=float); anomalies=pd.DataFrame();reconciliation=pd.DataFrame()
    raw=raw[raw.date<cutoff].copy() # Do not use the incomplete current month.
    histories={}; results=[]
    sales_by_sku={k:g for k,g in raw.groupby('sku')}
    transit_by_sku={k:g for k,g in ds.transit.groupby('sku')} if not ds.transit.empty else {}
    stocks_by_sku={k:g for k,g in ds.stocks.groupby('sku')} if not ds.stocks.empty else {}
    stockouts_by_sku={k:g for k,g in ds.stockouts.groupby('sku')} if not ds.stockouts.empty else {}
    mismatch_by_sku=reconciliation.loc[reconciliation.monthly.notna()&reconciliation.detail.notna()&~reconciliation.consistent].groupby('sku').size().to_dict() if not reconciliation.empty else {}
    for _,p in ds.products.iterrows():
        sku=p.sku
        sku_lead=number(p.get('lead_days'),cfg.lead_days)
        if sku_lead<1 or sku_lead>365:raise ValueError(f'{sku}: срок поставки должен быть 1–365 дней')
        if sku_lead!=int(sku_lead):raise ValueError(f'{sku}: срок поставки должен быть целым числом дней')
        sku_lead=int(sku_lead)
        s=sales_by_sku.get(sku,raw.iloc[:0]).groupby('date').quantity.sum().sort_index()
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
        periods=stockouts_by_sku.get(sku,ds.stockouts.iloc[:0])
        if cfg.compensate_stockout and (not periods.empty or cfg.approximate_stockout):
            known=stocks_by_sku.get(sku,ds.stocks.iloc[:0])
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
        # Supplied planning growth takes precedence, rather than multiplying two trends.
        source_growth=number(p.get('source_growth'),number(p.get('source_growth_percent'),np.nan)/100)
        has_source_growth=cfg.trend and cfg.use_source_growth and np.isfinite(source_growth)
        growth_source='История продаж'
        if has_source_growth:
            growth=float(np.clip(source_growth,-.9,3));growth_source='Коэффициент поставщика'
        horizon=sku_lead+cfg.review_days
        future=pd.date_range(asof,periods=horizon)
        future_season=float(np.mean([factors[d.month] for d in future]))
        rate=max(0,daily*(1+growth)*(1+cfg.growth_percent/100)*future_season)
        forecast=rate*horizon
        model_name='Исходная модель Axioma';forecast_low=forecast;forecast_high=forecast
        calibration_months=0;stockout_floor_added=0.
        model_info=dict(method=model_name,demand_type=demand_profile(corrected)[0],cv_months=0,cv_wape=np.nan,stress_daily=0.,candidates=[])
        if cfg.forecast_method in ('adaptive','pooled'):
            if cfg.forecast_method=='pooled' and not has_source_growth and sku in (_pooled or {}):
                daily_path=_pooled[sku][:horizon].copy()
                model_info.update(method='Общая модель HistGradientBoosting + среднее')
                forecast_low=np.nan;forecast_high=np.nan
            else:
                daily_path,model_info=adaptive_forecast(corrected,future,cfg.seasonality,cfg.trend and not has_source_growth,ds.seasonal)
                calibration_months=model_info['cv_months']
                forecast_low=np.nan;forecast_high=np.nan
            scenario=(1+cfg.growth_percent/100)*(1+growth if has_source_growth else 1)
            daily_path*=scenario
            forecast=float(daily_path.sum());rate=forecast/horizon;model_name=model_info['method']
            if not has_source_growth:growth=0.;growth_source='Учитывается выбранной моделью'
            future_season=np.nan
        elif cfg.forecast_method not in ('classic','legacy'):
            fitted=forecast_series(corrected,future,method=cfg.forecast_method,seasonality=cfg.seasonality,trend=cfg.trend and not has_source_growth,supplied=ds.seasonal)
            scenario=(1+cfg.growth_percent/100)*(1+growth if has_source_growth else 1)
            daily_path=fitted['daily']*scenario
            if cfg.compensate_stockout and lost>0:
                # A seasonal analogue may otherwise ignore a recent confirmed
                # availability loss entirely. Carry a six-month average of the
                # restored daily demand as a floor above the same model without
                # the correction; do not add it twice if the model already reacts.
                restored_rate=float(((corrected-regular)/corrected.index.days_in_month).tail(6).mean())
                counterfactual=forecast_series(regular,future,method=fitted['model'],seasonality=cfg.seasonality,trend=cfg.trend and not has_source_growth,supplied=ds.seasonal)
                floor=(counterfactual['daily']+restored_rate)*scenario
                adjusted=np.maximum(daily_path,floor)
                stockout_floor_added=float((adjusted-daily_path).sum());daily_path=adjusted
            forecast=float(daily_path.sum());rate=forecast/horizon
            forecast_low=fitted['low']*scenario;forecast_high=fitted['high']*scenario
            forecast_low+=stockout_floor_added;forecast_high+=stockout_floor_added
            model_name=fitted['model_name'];calibration_months=fitted['calibration_months']
            if not has_source_growth:
                growth=0.;growth_source='Учитывается выбранной моделью'
            if fitted['model'] in ('mean3','mean6','ses'):future_season=1.
        else:daily_path=np.repeat(rate,horizon)
        cat=float(cfg.category_factors.get(str(p.category),1))
        safety=rate*cfg.safety_days*cat
        arriving=transit_by_sku.get(sku,ds.transit.iloc[:0])
        due=0.; late=0.; unknown_eta=0.
        if not arriving.empty:
            within=arriving.eta.notna()&arriving.eta.ge(asof)&arriving.eta.lt(asof+pd.Timedelta(days=horizon))
            due=float(arriving.loc[within,'quantity'].clip(lower=0).sum())
            late=float(arriving.loc[arriving.eta.ge(asof+pd.Timedelta(days=horizon)),'quantity'].clip(lower=0).sum())
            unknown_eta=float(arriving.loc[arriving.eta.isna()|arriving.eta.lt(asof),'quantity'].clip(lower=0).sum())
        stock=number(p.stock,np.nan);pack=max(number(p.pack,1),1);moq=max(number(p.moq,1),1)
        receipts=pd.Series(0.,index=future)
        if not arriving.empty:
            dated=arriving.dropna(subset=['eta']).copy()
            dated['eta']=pd.to_datetime(dated.eta).dt.normalize()
            dated['quantity']=dated.quantity.clip(lower=0)
            receipts=dated.groupby('eta').quantity.sum().reindex(future,fill_value=0.)
        net_demand=np.cumsum(daily_path-receipts.to_numpy())
        bridge_threshold=max(0.,float(net_demand[sku_lead:].max())) if sku_lead<horizon else 0.
        stock_threshold=max(0.,forecast+safety-due,bridge_threshold)
        need=max(0,stock_threshold-stock) if np.isfinite(stock) else np.nan
        order=math.ceil(max(need,moq)/pack)*pack if np.isfinite(need) and need>1e-8 else 0 if np.isfinite(need) else np.nan
        # Date-aware deficit before the newly placed order arrives.
        shortage=False;first_shortage=None
        if np.isfinite(stock):
            deficits=np.flatnonzero(stock-net_demand[:sku_lead]<-1e-8)
            if len(deficits):shortage=True;first_shortage=str(future[deficits[0]].date())
        bridge_need=max(0.,bridge_threshold-stock) if np.isfinite(stock) else np.nan
        expedite_need=max(0.,float(net_demand[:sku_lead].max())-stock) if np.isfinite(stock) else np.nan
        status='Нужен остаток' if not np.isfinite(stock) else 'Срочно' if shortage else 'Заказать' if order>0 else 'Достаточно'
        reason=(f'Спрос {forecast:.1f} + страховой запас {safety:.1f} − свободный остаток {stock:.1f} − поступления {due:.1f}. '
                f'Минимум {moq:g}, кратность {pack:g}. Исключено разовых продаж: {excluded:.1f}; восстановлено спроса: {lost:.1f}.') if np.isfinite(stock) else f'Введите актуальный свободный остаток. Порог пополнения: {stock_threshold:.1f} ед. Неизвестный остаток не подменён нулём.'
        if bridge_threshold>max(0.,forecast+safety-due)+1e-8:reason+=f' Проверка по датам повышает порог остатка до {stock_threshold:.1f}: позднее поступление не закрывает промежуточный дефицит.'
        if shortage:reason+=f' До новой поставки не хватает до {expedite_need:.1f} ед.; требуется ускорение или перемещение, обычный заказ не успеет.'
        if unknown_eta:reason+=f' Требуют уточнения даты поступления: {unknown_eta:g} ед.'
        reason+=f' Модель: {model_name}.'
        if has_source_growth:reason+=f' Рост из сводки {source_growth*100:g}% применён вместо тренда истории.'
        if has_source_growth and cfg.forecast_method=='pooled':reason+=' Для явного сценария прироста использована статистическая база без экстраполяции тренда.'
        if stockout_floor_added>0:reason+=f' Минимальная поправка на подтверждённый упущенный спрос за последние 6 месяцев: +{stockout_floor_added:.1f} ед.; уже учтённая моделью часть повторно не добавляется.'
        mismatches=int(mismatch_by_sku.get(sku,0))
        if mismatches:reason+=f' Расхождение месячного отчёта и накладных в {mismatches} мес.: автоматическое вычитание всплесков для них отключено.'
        results.append(dict(sku=sku,article=p.article,name=p['name'],category=str(p.category),supplier=ds.supplier,unit=p.unit,stock=stock,in_transit=due,late_transit=late,unknown_transit=unknown_eta,forecast=round(forecast,2),safety=round(safety,2),daily=round(rate,3),growth=round(growth*100,1),growth_source=growth_source,season=round(future_season,3),excluded=round(excluded,2),lost=round(lost,2),pack=pack,moq=moq,recommended=order,status=status,stockout_mode=stockout_mode,reason=reason))
        results[-1].update(model=model_name,forecast_low=round(forecast_low,2),forecast_high=round(forecast_high,2),calibration_months=calibration_months,lead_days=sku_lead,source_mismatch_months=mismatches)
        stress=max(0,model_info['stress_daily'])*(horizon+cfg.safety_days*cat)*(1+cfg.growth_percent/100)*(1+growth if has_source_growth else 1)
        stress_need=max(0,stock_threshold+stress-stock) if np.isfinite(stock) else np.nan
        stress_order=math.ceil(max(stress_need,moq)/pack)*pack if np.isfinite(stress_need) and stress_need>1e-8 else 0 if np.isfinite(stress_need) else np.nan
        results[-1].update(method=model_name,demand_type=model_info['demand_type'],cv_months=model_info['cv_months'],cv_wape=model_info['cv_wape'],stress_recommended=stress_order,first_shortage=first_shortage,source_growth_percent=source_growth*100 if has_source_growth else np.nan,stock_threshold=round(stock_threshold,2),bridge_need=round(bridge_need,2),expedite_need=round(expedite_need,2),order_arrival=str((asof+pd.Timedelta(days=sku_lead)).date()))
        histories[sku]=pd.DataFrame({'Дата':full,'Продажи':s.values,'Без всплесков':regular.values,'С учётом отсутствия':corrected.values})
        histories[sku].attrs['model_info']=model_info
        histories[sku].attrs['future']=[{'Дата':str(d.date()),'Прогноз в день':float(v)} for d,v in zip(future,daily_path)]
    return pd.DataFrame(results),histories,anomalies
