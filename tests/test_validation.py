from copy import deepcopy
import pandas as pd
from src.validation import backtest
from test_engine import baseline


def test_constant_daily_demand_predicts_heldout_months():
    detail,scores,meta=backtest(baseline())
    assert len(detail)==3
    assert scores.iloc[0].model_wape==0
    assert scores.iloc[0].baseline_wape==0
    assert meta['evaluated_skus']==1


def test_heldout_and_future_changes_cannot_change_earlier_prediction():
    d=baseline();before=backtest(d)[0]
    d.sales.loc[d.sales.date.ge(pd.Timestamp('2026-06-01')),'quantity']*=30
    d.seasonal={m:50 if m==6 else 1 for m in range(1,13)}
    after=backtest(d)[0]
    assert before.iloc[0].model==after.iloc[0].model
    assert before.iloc[0].adaptive==after.iloc[0].adaptive
    assert before.iloc[0].actual!=after.iloc[0].actual


def test_missing_target_not_scored_as_zero():
    d=baseline(); d.sales=d.sales[~d.sales.date.eq(pd.Timestamp('2026-08-01'))]
    detail,scores,_=backtest(d)
    assert len(detail)==2
    assert scores.iloc[0].months==2
