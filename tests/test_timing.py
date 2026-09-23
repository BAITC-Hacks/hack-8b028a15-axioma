import numpy as np
import pandas as pd
from src.engine import Settings, calculate
from test_engine import baseline


def late_receipt(stock=220):
    ds=baseline()
    ds.products.loc[0,'stock']=stock
    ds.transit=pd.DataFrame([dict(sku='A',eta=pd.Timestamp('2026-10-27'),quantity=500.)])
    return ds


def test_end_of_period_surplus_cannot_hide_intermediate_shortage():
    ds=late_receipt()
    r,h,_=calculate(ds,Settings())
    row=r.iloc[0]
    assert row.recommended==120  # 34 days * 10 demand - 220 stock
    assert row.bridge_need==120
    assert row.stock_threshold==340
    assert row.status=='Заказать'
    assert row.expedite_need==0
    future=pd.DataFrame(h['A'].attrs['future'])
    balance=220.
    for i,day in future.iterrows():
        if i==21:balance+=row.recommended
        if day['Дата']=='2026-10-27':balance+=500
        balance-=day['Прогноз в день']
        assert balance>=-1e-8


def test_same_quantity_earlier_receipt_removes_bridge_order():
    ds=late_receipt()
    ds.transit.loc[0,'eta']=pd.Timestamp('2026-10-14')
    assert calculate(ds,Settings())[0].iloc[0].recommended==0


def test_unknown_stock_exposes_time_aware_threshold_without_fabricated_order():
    row=calculate(late_receipt(np.nan),Settings())[0].iloc[0]
    assert row.stock_threshold==340
    assert pd.isna(row.recommended)


def test_expedite_is_separate_from_standard_order():
    row=calculate(late_receipt(20),Settings())[0].iloc[0]
    assert row.expedite_need==190
    assert row.first_shortage=='2026-09-25'
    assert row.status=='Срочно'
    assert 'обычный заказ не успеет' in row.reason


def test_bridge_order_rounds_to_pack():
    ds=late_receipt(225)
    ds.products.loc[0,'pack']=12
    row=calculate(ds,Settings())[0].iloc[0]
    assert row.bridge_need==115
    assert row.recommended==120
