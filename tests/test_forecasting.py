import numpy as np
import pandas as pd
from src.forecasting import adaptive_forecast,predict_rates
from src.engine import calculate,Settings
from test_engine import baseline


def test_constant_daily_rate_and_future_calendar():
    s=pd.Series([d.days_in_month*10. for d in pd.date_range('2024-01-01',periods=30,freq='MS')],index=pd.date_range('2024-01-01',periods=30,freq='MS'))
    rates,info=adaptive_forecast(s,pd.date_range('2026-07-15',periods=60))
    assert np.allclose(rates,10)
    assert info['cv_months']==6
    assert info['cv_wape']<1e-10


def test_repeating_seasonal_demand_selects_seasonal_method():
    dates=pd.date_range('2022-01-01',periods=48,freq='MS')
    s=pd.Series([d.days_in_month*(30 if d.month in (10,11,12) else 3) for d in dates],index=dates)
    rates,info=adaptive_forecast(s,pd.date_range('2026-10-01',periods=31))
    assert np.allclose(rates,30)
    assert 'Сезонный' in info['method']


def test_tsb_decays_during_zero_demand():
    dates=pd.date_range('2024-01-01',periods=24,freq='MS')
    y=np.array([0.,0.,10.]*8)
    before=predict_rates(y,dates,dates[-1:],'tsb')[0]
    after=predict_rates(np.r_[y,np.zeros(12)],pd.date_range('2024-01-01',periods=36,freq='MS'),dates[-1:],'tsb')[0]
    assert 0<after<before/10


def test_adaptive_zero_demand_never_creates_an_order():
    ds=baseline();ds.sales.quantity=0
    r=calculate(ds,Settings(forecast_method='adaptive'))[0].iloc[0]
    assert r.forecast==0 and r.recommended==0 and r.stress_recommended==0


def test_source_growth_is_explicit_and_not_multiplied_by_trend_twice():
    ds=baseline();ds.products['source_growth_percent']=50.
    cfg=Settings(forecast_method='adaptive',use_source_growth=True)
    r=calculate(ds,cfg)[0].iloc[0]
    assert r.forecast==525 and r.recommended==630
    assert 'Рост из сводки 50%' in r.reason


def test_stress_scenario_respects_surplus_stock():
    ds=baseline();ds.sales.loc[ds.sales.index[-1],'quantity']*=2
    ds.products.stock=100000
    r=calculate(ds,Settings(forecast_method='adaptive'))[0].iloc[0]
    assert r.recommended==0 and r.stress_recommended==0
