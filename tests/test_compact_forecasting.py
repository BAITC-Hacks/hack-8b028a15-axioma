import pandas as pd
import numpy as np
from src.forecasting import forecast_series
from src.backtest import evaluate_sales

def constant():
    dates=pd.date_range('2024-01-01','2026-08-01',freq='MS')
    return pd.Series([d.days_in_month*10. for d in dates],index=dates)

def test_all_models_predict_constant_daily_demand():
    s=constant();future=pd.date_range('2026-09-01',periods=30)
    for m in ['mean3','mean6','ses','seasonal_naive','seasonal_level','auto']:
        result=forecast_series(s,future,method=m)
        assert np.isclose(result['daily'].sum(),300)

def test_auto_picks_seasonal_for_repeatable_calendar_pattern():
    s=constant();s.loc[s.index.month.isin([6,7,8])]*=4
    r=forecast_series(s,pd.date_range('2026-09-01',periods=30))
    assert r['model']=='seasonal_naive'
    assert np.isclose(r['daily'].sum(),300)

def test_future_data_rejected():
    import pytest
    with pytest.raises(ValueError):forecast_series(constant(),pd.date_range('2026-08-01',periods=31))

def test_backtest_predictions_do_not_change_when_future_actuals_change():
    s=constant();sales=s.rename_axis('date').rename('quantity').reset_index().assign(sku='A')
    _,before,_=evaluate_sales(sales,periods=3)
    sales.loc[sales.date==pd.Timestamp('2026-08-01'),'quantity']=999999
    _,after,_=evaluate_sales(sales,periods=3)
    assert np.allclose(before.prediction,after.prediction)
    assert after.loc[after.month=='2026-08','actual'].eq(999999).all()

def test_backtest_uses_only_initial_training_population():
    s=constant();sales=s.rename_axis('date').rename('quantity').reset_index().assign(sku='A')
    sales=pd.concat([sales,pd.DataFrame([dict(sku='FUTURE',date=pd.Timestamp('2026-08-01'),quantity=99999)])],ignore_index=True)
    summary,detail,meta=evaluate_sales(sales,periods=3)
    assert list(detail.sku.unique())==['A']
    assert meta['observations']==3
    assert np.allclose(summary.wape,0)
