from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
from src.data import demo_data
from src.engine import calculate,Settings
from test_engine import baseline


def test_default_ui_model_handles_project_and_stockout():
    ds=demo_data();cfg=Settings(forecast_method='auto',use_source_growth=False)
    clean=calculate(ds,cfg)[0].set_index('sku')
    raw=calculate(ds,replace(cfg,remove_outliers=False,compensate_stockout=False))[0].set_index('sku')
    assert clean.loc['DEMO-001','excluded']>1400
    assert clean.loc['DEMO-001','forecast']<raw.loc['DEMO-001','forecast']*.5
    assert clean.loc['DEMO-004','lost']>0
    assert clean.loc['DEMO-004','forecast']>raw.loc['DEMO-004','forecast']


def test_auto_responds_to_season_stock_transit_category_and_growth():
    ds=baseline();cfg=Settings(forecast_method='auto')
    base=calculate(ds,cfg)[0].iloc[0]
    seasonal=deepcopy(ds);seasonal.sales.loc[seasonal.sales.date.dt.month.eq(9),'quantity']*=5
    # Horizon spans 8 September days and 27 October days: 8*50+27*10.
    assert np.isclose(calculate(seasonal,cfg)[0].iloc[0].forecast,670)
    stocked=deepcopy(ds);stocked.products['stock']=100
    assert calculate(stocked,cfg)[0].iloc[0].recommended<base.recommended
    transit=deepcopy(ds);transit.transit=pd.DataFrame([dict(sku='A',eta=pd.Timestamp('2026-09-24'),quantity=100)])
    assert calculate(transit,cfg)[0].iloc[0].recommended<base.recommended
    category=str(ds.products.iloc[0].category)
    assert calculate(ds,replace(cfg,category_factors={category:2}))[0].iloc[0].recommended>base.recommended
    growth=deepcopy(ds);growth.products['source_growth']=.2
    assert calculate(growth,cfg)[0].iloc[0].forecast>base.forecast


def test_explicit_growth_does_not_stack_on_compact_trend():
    ds=baseline();ds.sales.loc[ds.sales.date.ge('2026-06-01'),'quantity']*=2
    ds.products['source_growth']=.5
    cfg=Settings(forecast_method='seasonal_level',use_source_growth=True)
    actual=calculate(ds,cfg)[0].iloc[0].forecast
    expected=calculate(ds,replace(cfg,trend=False,use_source_growth=False))[0].iloc[0].forecast*1.5
    assert np.isclose(actual,expected,atol=.02)


def test_per_item_lead_time_changes_horizon_and_shortage_date():
    ds=baseline();cfg=Settings(forecast_method='auto')
    ds.products['lead_days']=7
    row=calculate(ds,cfg)[0].iloc[0]
    assert row.lead_days==7 and row.forecast==210
    assert row.first_shortage=='2026-09-23'
