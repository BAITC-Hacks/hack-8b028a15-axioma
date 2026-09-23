"""Rolling historical evaluation. Test targets never enter training inputs."""
from copy import deepcopy
import numpy as np
import pandas as pd
from .engine import Settings, calculate


def backtest(dataset, as_of='2026-09-23', months=3, max_skus=30, include_pooled=False):
    cutoff = pd.Timestamp(as_of).replace(day=1).normalize()
    targets = pd.date_range(end=cutoff-pd.offsets.MonthBegin(1), periods=months, freq='MS')
    # Select by information available before the FIRST evaluation month.
    old = dataset.sales[dataset.sales.date.lt(targets[0])]
    counts = old.groupby('sku').date.nunique()
    eligible = sorted(counts[counts.ge(6)].index)
    skus = eligible[:max_skus]
    rows = []
    for target in targets:
        train = deepcopy(dataset)
        train.products = train.products[train.products.sku.isin(skus)].copy()
        train.sales = train.sales[train.sales.sku.isin(skus) & train.sales.date.lt(target)].copy()
        train.transactions = train.transactions[train.transactions.sku.isin(skus) & train.transactions.date.lt(target)].copy()
        train.stocks = train.stocks[train.stocks.sku.isin(skus) & train.stocks.date.lt(target)].copy()
        # Supplied seasonal factors may contain future data: recompute from history.
        train.seasonal = {}
        train.transit = train.transit.iloc[:0].copy()
        # Dates when these retrospective intervals became known are unavailable.
        train.stockouts = train.stockouts.iloc[:0].copy()
        train.products=train.products.drop(columns=['source_growth_percent','source_growth'],errors='ignore')
        cfg = Settings(as_of=str(target.date()),lead_days=target.days_in_month,review_days=0,safety_days=0,compensate_stockout=False)
        model = calculate(train,cfg)[0]
        from dataclasses import replace
        adaptive=calculate(train,replace(cfg,forecast_method='adaptive'))[0]
        pooled=calculate(train,replace(cfg,forecast_method='pooled'))[0].set_index('sku') if include_pooled else None
        baseline_cfg = Settings(as_of=str(target.date()),lead_days=target.days_in_month,review_days=0,safety_days=0,remove_outliers=False,seasonality=False,trend=False,compensate_stockout=False)
        basic = calculate(train,baseline_cfg)[0]
        if model.empty: continue
        actual = dataset.sales[dataset.sales.date.eq(target)].groupby('sku').quantity.sum().clip(lower=0)
        for sku in skus:
            if sku not in actual.index: continue # missing target is not observed zero
            m=model[model.sku.eq(sku)]; b=basic[basic.sku.eq(sku)]
            if m.empty or b.empty or 'forecast' not in m: continue
            prediction=float(m.iloc[0].forecast); naive=float(b.iloc[0].forecast)
            if not np.isfinite(prediction) or not np.isfinite(naive): continue
            a=adaptive[adaptive.sku.eq(sku)].iloc[0]
            rows.append(dict(sku=sku,month=str(target.date()),actual=float(actual[sku]),model=prediction,baseline=naive,adaptive=float(a.forecast),method=a.method,pooled=float(pooled.loc[sku,'forecast']) if pooled is not None else np.nan))
    detail=pd.DataFrame(rows,columns=['sku','month','actual','model','baseline','adaptive','method','pooled'])
    summary=[]
    if not detail.empty:
        for sku,g in detail.groupby('sku'):
            denom=float(g.actual.sum())
            summary.append(dict(sku=sku,months=len(g),actual_total=denom,
                                model_wape=float((g.model-g.actual).abs().sum()/denom*100) if denom else np.nan,
                                baseline_wape=float((g.baseline-g.actual).abs().sum()/denom*100) if denom else np.nan,
                                adaptive_wape=float((g.adaptive-g.actual).abs().sum()/denom*100) if denom else np.nan,
                                pooled_wape=float((g.pooled-g.actual).abs().sum()/denom*100) if denom and include_pooled else np.nan))
    scores=pd.DataFrame(summary,columns=['sku','months','actual_total','model_wape','baseline_wape','adaptive_wape','pooled_wape'])
    return detail,scores,{'eligible_skus':len(eligible),'selected_skus':len(skus),'evaluated_skus':len(scores),'months':list(targets.strftime('%Y-%m'))}

