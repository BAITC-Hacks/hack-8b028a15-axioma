"""Shared daily stock policy. No forecasting or model fitting here."""
import math
import numpy as np
import pandas as pd


def receipt_schedule(transit, dates, delay_days=0):
    """Shift only future dated receipts; overdue/unknown receipts stay excluded."""
    dates=pd.DatetimeIndex(dates)
    receipts=np.zeros(len(dates),dtype=float)
    if transit.empty:return receipts
    eta=pd.to_datetime(transit.eta,errors='coerce').dt.normalize()
    valid=eta.notna() & eta.ge(dates[0])
    dated=pd.DataFrame({'eta':eta[valid]+pd.Timedelta(days=delay_days),
                        'quantity':pd.to_numeric(transit.loc[valid,'quantity'],errors='coerce').fillna(0).clip(lower=0)})
    return dated.groupby('eta').quantity.sum().reindex(dates,fill_value=0.).to_numpy(dtype=float)


def inventory_plan(rates, receipts, stock, lead_days, safety, pack=1., moq=1.):
    """End-of-day demand balance, beginning-of-day receipts, unmet demand carried."""
    rates=np.asarray(rates,dtype=float);receipts=np.asarray(receipts,dtype=float)
    if len(rates)==0 or rates.shape!=receipts.shape or not 1<=lead_days<=len(rates):
        raise ValueError('Некорректный горизонт дневного расчёта')
    if not np.isfinite(rates).all() or (rates<0).any() or not np.isfinite(receipts).all() or (receipts<0).any():
        raise ValueError('Спрос и поступления должны быть конечными и неотрицательными')
    if not all(np.isfinite(x) for x in (safety,pack,moq)) or safety<0 or pack<=0 or moq<=0:
        raise ValueError('Некорректный запас, минимум или кратность')
    net=np.cumsum(rates-receipts)
    bridge=max(0.,float(net[lead_days:].max())) if lead_days<len(rates) else 0.
    threshold=max(0.,float(net[-1])+safety,bridge)
    known=np.isfinite(stock)
    need=max(0.,threshold-stock) if known else np.nan
    order=math.ceil(max(need,moq)/pack)*pack if known and need>1e-8 else 0. if known else np.nan
    deficits=np.flatnonzero(stock-net[:lead_days]<-1e-8) if known else []
    return dict(recommended=order,stock_threshold=threshold,bridge_threshold=bridge,
                bridge_need=max(0.,bridge-stock) if known else np.nan,
                expedite_need=max(0.,float(net[:lead_days].max())-stock) if known else np.nan,
                urgent=bool(len(deficits)),first_shortage_index=int(deficits[0]) if len(deficits) else None)
