"""Deterministic, inspectable inventory policy. No network or model calls."""
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd
from .data import Dataset, number

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
            # Three or more distinct purchase days by a known large client are
            # treated as recurring demand, not deleted as a one-off project.
            large=g[mask & g.client_id.fillna('').ne('')]
            recurring=large.groupby('client_id').day.nunique()
            recurring=set(recurring[recurring>=3].index)
            mask &= ~g.client_id.isin(recurring)
            for _,r in g[mask].iterrows():out.append(dict(sku=sku,date=r.day,document=r.document,original=r.quantity,regular=median,excluded=r.quantity-median,reason='Крупная разовая накладная'))
            g.loc[mask,'quantity']=median
            # Catch one large client-day split into many smaller invoices.
            known=g[g.client_id.fillna('').ne('')]
            if not known.empty:
                cg=known.groupby(['client_id','day']).quantity.sum()
                positive=cg[cg>0]
                if len(positive)>=8:
                    a,b=positive.quantile([.25,.75]); med=float(positive.median()); cap=max(5*med,float(b+3*(b-a)),1)
                    counts=cg[cg>cap].reset_index().groupby('client_id').day.nunique()
                    for (client,day),qty in cg[cg>cap].items():
                        if counts.get(client,0)>=3:continue
                        ix=g.client_id.eq(client)&g.day.eq(day)&g.quantity.gt(0)
                        total=g.loc[ix,'quantity'].sum()
                        if total>0:
                            g.loc[ix,'quantity']*=med/total
                            out.append(dict(sku=sku,date=day,document='Группа клиента',original=qty,regular=med,excluded=qty-med,reason='Разовый всплеск клиента за день'))
        groups.append(g)
    result=pd.concat(groups,ignore_index=True).rename(columns={'day':'date'})
    return result,pd.DataFrame(out)

def calculate(ds: Dataset, settings: Settings):
    cfg=settings; asof=pd.Timestamp(cfg.as_of).normalize(); cutoff=asof.replace(day=1)
    for name,low,high in [('lead_days',1,365),('review_days',1,90),('safety_days',0,90),('growth_percent',-50,100)]:
        value=getattr(cfg,name)
        if not np.isfinite(value) or value<low or value>high:raise ValueError(f'Недопустимый параметр: {name}')
    if any(not np.isfinite(v) or v<0 or v>5 for v in cfg.category_factors.values()):raise ValueError('Множитель категории должен быть от 0 до 5')
    raw=ds.sales.copy()
    tx=ds.transactions.copy()
    if not tx.empty:
        tx=tx[tx.date<cutoff].copy() # Incomplete month must not affect historical outlier thresholds.
        cleaned,anomalies=clean_transactions(tx)
        raw_tx=tx.assign(date=tx.date.dt.to_period('M').dt.to_timestamp()).groupby(['sku','date']).quantity.sum()
        clean_tx=cleaned.assign(date=cleaned.date.dt.to_period('M').dt.to_timestamp()).groupby(['sku','date']).quantity.sum()
        removed=(raw_tx-clean_tx).clip(lower=0)
        # Fill only missing SKU/month keys; never add duplicate monthly representations.
        detailed=raw_tx.reset_index()
        if raw.empty:raw=detailed
        else:
            known=pd.MultiIndex.from_frame(raw[['sku','date']])
            missing=~pd.MultiIndex.from_frame(detailed[['sku','date']]).isin(known)
            raw=pd.concat([raw,detailed.loc[missing]],ignore_index=True)
    else:removed=pd.Series(dtype=float); anomalies=pd.DataFrame()
    raw=raw[raw.date<cutoff].copy() # Do not use the incomplete current month.
    histories={}; results=[]
    sales_by_sku={k:g for k,g in raw.groupby('sku')}
    transit_by_sku={k:g for k,g in ds.transit.groupby('sku')} if not ds.transit.empty else {}
    stocks_by_sku={k:g for k,g in ds.stocks.groupby('sku')} if not ds.stocks.empty else {}
    stockouts_by_sku={k:g for k,g in ds.stockouts.groupby('sku')} if not ds.stockouts.empty else {}
    for _,p in ds.products.iterrows():
        sku=p.sku
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
        source_growth=number(p.get('source_growth'),np.nan)
        growth_source='История продаж'
        if cfg.trend and np.isfinite(source_growth):
            growth=float(np.clip(source_growth,-.9,3));growth_source='Коэффициент поставщика'
        horizon=cfg.lead_days+cfg.review_days
        future=pd.date_range(asof,periods=horizon)
        future_season=float(np.mean([factors[d.month] for d in future]))
        rate=max(0,daily*(1+growth)*(1+cfg.growth_percent/100)*future_season)
        forecast=rate*horizon
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
        need=max(0,forecast+safety-stock-due) if np.isfinite(stock) else np.nan
        order=math.ceil(max(need,moq)/pack)*pack if np.isfinite(need) and need>1e-8 else 0 if np.isfinite(need) else np.nan
        # Date-aware deficit before the newly placed order arrives.
        balance=stock; shortage=False
        if np.isfinite(stock):
            for d in pd.date_range(asof,periods=cfg.lead_days):
                if not arriving.empty:balance+=float(arriving.loc[arriving.eta.eq(d),'quantity'].clip(lower=0).sum())
                balance-=rate
                if balance<0:shortage=True
        status='Нужен остаток' if not np.isfinite(stock) else 'Срочно' if shortage else 'Заказать' if order>0 else 'Достаточно'
        reason=(f'Спрос {forecast:.1f} + страховой запас {safety:.1f} − свободный остаток {stock:.1f} − поступления {due:.1f}. '
                f'Минимум {moq:g}, кратность {pack:g}. Исключено разовых продаж: {excluded:.1f}; восстановлено спроса: {lost:.1f}.') if np.isfinite(stock) else 'Введите актуальный свободный остаток. Он не подменён нулём.'
        if unknown_eta:reason+=f' Требуют уточнения даты поступления: {unknown_eta:g} ед.'
        results.append(dict(sku=sku,article=p.article,name=p['name'],category=str(p.category),supplier=ds.supplier,unit=p.unit,stock=stock,in_transit=due,late_transit=late,unknown_transit=unknown_eta,forecast=round(forecast,2),safety=round(safety,2),daily=round(rate,3),growth=round(growth*100,1),growth_source=growth_source,season=round(future_season,3),excluded=round(excluded,2),lost=round(lost,2),pack=pack,moq=moq,recommended=order,status=status,stockout_mode=stockout_mode,reason=reason))
        histories[sku]=pd.DataFrame({'Дата':full,'Продажи':s.values,'Без всплесков':regular.values,'С учётом отсутствия':corrected.values})
    return pd.DataFrame(results),histories,anomalies
