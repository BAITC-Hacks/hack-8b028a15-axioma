from copy import deepcopy
import numpy as np
import pandas as pd
from src.engine import calculate,Settings
from test_engine import baseline


def panel():
    ds=baseline()
    ds.products=pd.concat([ds.products.assign(sku=f'A{i}') for i in range(12)],ignore_index=True)
    ds.sales=pd.concat([ds.sales.assign(sku=f'A{i}',quantity=ds.sales.quantity*(i+1)) for i in range(12)],ignore_index=True)
    return ds


def test_pooled_forecast_is_scale_aware_and_cannot_see_future_sales():
    ds=panel();cfg=Settings(forecast_method='pooled')
    before=calculate(ds,cfg)[0].set_index('sku')
    assert before.loc['A0','forecast']==350
    assert before.loc['A11','forecast']==4200
    assert before.method.str.contains('HistGradientBoosting').all()
    future=ds.sales[ds.sales.date.eq(pd.Timestamp('2026-08-01'))].copy()
    future['date']=pd.Timestamp('2026-10-01');future.quantity*=10000
    ds.sales=pd.concat([ds.sales,future],ignore_index=True)
    after=calculate(ds,cfg)[0].set_index('sku')
    assert np.allclose(before.forecast,after.forecast)


def test_small_panel_falls_back_to_explained_statistical_model():
    result=calculate(baseline(),Settings(forecast_method='pooled'))[0]
    assert not result.method.str.contains('HistGradientBoosting').any()
    assert result.iloc[0].forecast==350


def test_shared_model_does_not_resurrect_dormant_products():
    ds=panel()
    ds.sales.loc[ds.sales.sku.eq('A0') & ds.sales.date.ge(pd.Timestamp('2026-03-01')),'quantity']=0
    result=calculate(ds,Settings(forecast_method='pooled'))[0].set_index('sku')
    assert result.loc['A0','forecast']==0
    assert result.loc['A0','recommended']==0

