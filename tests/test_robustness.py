"""Synthetic invariants of the finite scenario grid, not measured business gains."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from src.engine import Settings, calculate
from src.robustness import (ScenarioSettings, analyse_scenarios, recount_queue,
                            ALL_ORDER, DEPENDS, NO_ORDER, INSUFFICIENT)
from src.planning import inventory_plan, receipt_schedule
from test_engine import baseline


def evaluate(ds,options=ScenarioSettings(),cfg=Settings()):
    result,histories,_=calculate(ds,cfg)
    return analyse_scenarios(ds,cfg,result,histories,options)


def test_order_needed_in_every_explicit_scenario():
    ds=baseline();ds.products.loc[0,'stock']=10
    summary,detail,_=evaluate(ds)
    assert summary.iloc[0].robustness==ALL_ORDER
    assert len(detail)==18
    assert detail.scenario_order.gt(0).all()
    assert summary.iloc[0].order_min==detail.scenario_order.min()
    assert summary.iloc[0].order_max==detail.scenario_order.max()
    assert set(detail.demand_multiplier)=={.8,1.,1.2}
    assert set(detail.delay_days)=={0,7}
    assert set(detail.scenario_stock)=={8.,10.,12.}


def test_count_changes_order_yes_no_at_fixed_demand_and_delivery():
    ds=baseline();ds.products.loc[0,'stock']=420
    summary,detail,_=evaluate(ds,ScenarioSettings(20,0,0))
    item=summary.iloc[0]
    assert item.robustness==DEPENDS
    assert item.order_min==0 and item.order_max==84
    assert item.stock_changes_order and item.check_stock
    assert 'остаток' in item.sensitivity_reason
    ds.products.loc[0,'stock']=336
    assert calculate(ds,Settings())[0].iloc[0].recommended==84
    ds.products.loc[0,'stock']=504
    assert calculate(ds,Settings())[0].iloc[0].recommended==0


def test_delay_creates_intermediate_shortage_even_when_order_unchanged():
    ds=baseline();ds.products.loc[0,'stock']=60
    ds.transit=pd.DataFrame([dict(sku='A',eta=pd.Timestamp('2026-09-28'),quantity=300.)])
    _,_,comparison=evaluate(ds)
    item=comparison.iloc[0]
    assert item.delay_new_urgent
    assert item.delay_order_before==item.delay_order_after==60
    assert item.delay_expedite_before==0
    assert item.delay_expedite_after==60
    assert item.delay_first_shortage=='2026-09-29'
    dates=pd.date_range('2026-09-23',periods=35)
    receipts=receipt_schedule(ds.transit,dates,7)
    balance=60+np.cumsum(receipts-10)+np.where(np.arange(35)>=21,item.delay_order_after,0)
    assert balance[:21].min()==-60  # Placing the ordinary order cannot repair this gap.


def test_delay_past_horizon_changes_order_and_excludes_receipt():
    ds=baseline();ds.products.loc[0,'stock']=220
    ds.transit=pd.DataFrame([dict(sku='A',eta=pd.Timestamp('2026-10-21'),quantity=300.)])
    _,_,comparison=evaluate(ds)
    item=comparison.iloc[0]
    assert item.delay_order_before==60
    assert item.delay_order_after==200
    assert item.delay_order_change==140


def test_missing_stock_without_anchor_is_not_zero_or_a_scenario():
    ds=baseline();ds.products.loc[0,'stock']=np.nan
    summary,detail,_=evaluate(ds)
    assert summary.iloc[0].robustness==INSUFFICIENT
    assert summary.iloc[0].check_stock
    assert pd.isna(summary.iloc[0].order_min)
    assert detail.empty


def test_snapshot_anchors_preserve_signed_sales_and_exclude_future_data():
    ds=baseline();ds.products.loc[0,'stock']=np.nan
    ds.stocks=pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-09-01'),quantity=400.),
                            dict(sku='A',date=pd.Timestamp('2026-10-01'),quantity=9999.)])
    ds.transactions=pd.DataFrame([dict(sku='A',date=pd.Timestamp(day),quantity=q,document=str(q),client_id='')
                                 for day,q in [('2026-09-02',100.),('2026-09-03',-20.),('2026-09-24',900.)]])
    summary,detail,_=evaluate(ds,ScenarioSettings(0,0,0))
    assert set(detail.scenario_stock)=={320.,400.}
    assert summary.iloc[0].scenario_snapshot_age_days==22
    assert '80' in summary.iloc[0].scenario_basis
    assert pd.isna(ds.products.loc[0,'stock'])


def test_missing_sales_rows_do_not_mean_zero_depletion():
    ds=baseline();ds.products.loc[0,'stock']=np.nan
    ds.stocks=pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-09-01'),quantity=400.)])
    _,detail,_=evaluate(ds,ScenarioSettings(0,0,0))
    assert len(detail)==1 and detail.iloc[0].scenario_stock==400
    assert 'минус продажи' not in detail.iloc[0].stock_assumption


def test_no_history_blocks_scenarios_even_with_stock():
    ds=baseline();ds.sales=ds.sales.iloc[:0]
    summary,detail,_=evaluate(ds)
    assert summary.iloc[0].robustness==INSUFFICIENT and detail.empty
    assert 'истори' in summary.iloc[0].sensitivity_reason


def test_sufficient_stock_in_all_scenarios_has_separate_status():
    ds=baseline();ds.products.loc[0,'stock']=10000
    summary,detail,_=evaluate(ds)
    assert summary.iloc[0].robustness==NO_ORDER
    assert not summary.iloc[0].check_stock
    assert detail.scenario_order.eq(0).all()


def test_stock_flip_not_confused_with_demand_flip():
    ds=baseline();ds.products.loc[0,'stock']=420
    summary,_,_=evaluate(ds,ScenarioSettings(0,20,0))
    item=summary.iloc[0]
    assert item.robustness==DEPENDS
    assert not item.stock_changes_order and not item.check_stock
    assert item.sensitivity_reason=='Изменяются спрос'


def test_queue_uses_age_then_relative_sensitivity_not_mixed_units():
    rows=[dict(sku=sku,check_stock=True,count_priority=priority,stock_changes_urgency=False,stock_changes_order=priority==1,
               scenario_snapshot_age_days=age,relative_order_span=span,unit=unit)
          for sku,priority,age,span,unit in [('A',1,2,.9,'м'),('B',1,20,.1,'шт'),('C',2,200,1.,'бухта')]]
    frame=pd.DataFrame(rows)
    assert recount_queue(frame).sku.tolist()==['B','A','C']
    frame['unit']='другая единица'
    assert recount_queue(frame).sku.tolist()==['B','A','C']


def test_grid_reuses_existing_forecast_and_matches_selected_calculation():
    ds=baseline();ds.products.loc[0,'stock']=12
    original=deepcopy(ds)
    cfg=Settings(category_factors={'1':1.5})
    result,histories,_=calculate(ds,cfg)
    hist_before=deepcopy(histories['A'])
    _,detail,comparison=analyse_scenarios(ds,cfg,result,histories,ScenarioSettings())
    center=detail.query('scenario_stock == 12 and demand_multiplier == 1 and delay_days == 0').iloc[0]
    assert center.scenario_order==result.iloc[0].recommended
    assert comparison.iloc[0].delay_order_before==result.iloc[0].recommended
    pd.testing.assert_frame_equal(ds.products,original.products)
    pd.testing.assert_frame_equal(histories['A'],hist_before)
    assert histories['A'].attrs==hist_before.attrs


def test_overdue_and_unknown_receipts_are_never_rehabilitated_by_delay():
    dates=pd.date_range('2026-09-23',periods=35)
    transit=pd.DataFrame({'eta':[pd.Timestamp('2026-09-22'),pd.NaT], 'quantity':[9999,9999]})
    assert receipt_schedule(transit,dates,7).sum()==0


@pytest.mark.parametrize('options',[ScenarioSettings(-1,20,7),ScenarioSettings(20,101,7),ScenarioSettings(20,20,-1)])
def test_grid_parameters_are_validated(options):
    with pytest.raises(ValueError):evaluate(baseline(),options)
