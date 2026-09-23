from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
from src.data import demo_data, Dataset
from src.engine import Settings, calculate, clean_transactions

def baseline():
    d=Dataset('Test')
    d.products=pd.DataFrame([dict(sku='A',article='A',name='Test',category='1',stock=0.,pack=1,moq=1,unit='шт')])
    d.sales=pd.DataFrame([dict(sku='A',date=x,quantity=float(x.days_in_month*10)) for x in pd.date_range('2024-01-01','2026-08-01',freq='MS')])
    return d

def row(d,cfg=None):return calculate(d,cfg or Settings())[0].iloc[0]

def test_order_arithmetic():
    r=row(baseline());assert r.recommended==420

def test_transit_reduces_order():
    d=baseline();before=row(d).recommended
    d.transit=pd.DataFrame([dict(sku='A',eta=pd.Timestamp('2026-09-25'),quantity=70)])
    assert row(d).recommended==before-70

def test_late_and_past_transit_not_counted():
    d=baseline();d.transit=pd.DataFrame([dict(sku='A',eta=pd.Timestamp(dt),quantity=1000) for dt in ['2026-12-01','2026-09-01']])
    assert row(d).recommended==420

def test_unknown_stock_blocks_order():
    d=baseline();d.products.loc[0,'stock']=np.nan
    assert np.isnan(row(d).recommended)

def test_zero_need_does_not_trigger_moq():
    d=baseline();d.products.loc[0,'stock']=10000;d.products.loc[0,'moq']=100
    assert row(d).recommended==0

def test_pack_and_moq():
    d=baseline();d.products.loc[0,'pack']=12;d.products.loc[0,'moq']=500
    assert row(d).recommended==504

def test_known_stockout_increases_forecast():
    d=baseline();d.sales.loc[d.sales.date.eq(pd.Timestamp('2026-08-01')),'quantity']=110
    d.stockouts=pd.DataFrame([dict(sku='A',start=pd.Timestamp('2026-08-01'),end=pd.Timestamp('2026-08-20'))])
    assert row(d).forecast>row(d,Settings(compensate_stockout=False)).forecast

def test_growth_and_category_change_result():
    d=baseline();assert row(d,Settings(growth_percent=20)).recommended>row(d).recommended
    assert row(d,Settings(category_factors={'1':2})).recommended>row(d).recommended

def test_seasonal_pattern_changes_forecast():
    d=baseline();d.sales.loc[d.sales.date.dt.month.eq(10),'quantity']*=3
    assert row(d,Settings(as_of='2026-10-01')).forecast>row(d,Settings(as_of='2026-10-01',seasonality=False)).forecast

def test_incomplete_month_excluded():
    d=baseline();before=row(d).recommended
    d.sales=pd.concat([d.sales,pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-09-01'),quantity=100000)])],ignore_index=True)
    assert row(d).recommended==before

def test_invoice_spike_is_robust():
    d=demo_data();a=calculate(d,Settings())[0].set_index('sku').loc['DEMO-001']
    d.transactions=d.transactions[d.transactions.document!='ONE-OFF']
    d.sales.loc[(d.sales.sku=='DEMO-001')&d.sales.date.eq(pd.Timestamp('2026-08-01')),'quantity']-=1500
    b=calculate(d,Settings())[0].set_index('sku').loc['DEMO-001']
    assert abs(a.forecast-b.forecast)<5
    assert a.excluded>=1490

def test_client_split_invoices_detected():
    tx=pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-08-01')+pd.Timedelta(days=i),quantity=2,document=f'd{i}',client_id=f'c{i}') for i in range(20)])
    extra=pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-08-25'),quantity=2,document=f'x{i}',client_id='same-client') for i in range(30)])
    cleaned,anomalies=clean_transactions(pd.concat([tx,extra],ignore_index=True))
    assert cleaned[cleaned.client_id=='same-client'].quantity.sum()<10
    assert 'Разовый всплеск клиента за день' in anomalies.reason.values

def test_no_future_information():
    d=baseline();d.sales=pd.concat([d.sales,pd.DataFrame([dict(sku='A',date=pd.Timestamp('2027-01-01'),quantity=100000)])])
    assert row(d).recommended==420

