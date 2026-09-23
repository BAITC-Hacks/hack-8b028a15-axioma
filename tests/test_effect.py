from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from test_engine import baseline
from src.effect import (Experiment, compare_inventory, simulate_path, training_data,
                        aggregate_metrics, forecast_diagnostics, scenario_costs, protocol_zip, pilot_zip)


def path(mode='lost', opening=None, initial=0., pack=1, moq=1):
    exp=Experiment(lead_days=2,review_days=2,safety_days=0)
    dates=pd.date_range('2026-06-01',periods=6)
    bank={('mean',d,'A'):np.full(4,10.) for d in dates[::2]}
    return simulate_path('A',dates,np.full(6,10.),initial,pack,moq,exp,'mean',mode,bank,opening)


@pytest.mark.parametrize('mode',['lost','backorder'])
def test_daily_and_whole_period_conservation(mode):
    row,daily,orders=path(mode)
    np.testing.assert_allclose(daily.stock_start+daily.received-daily.served_now-daily.served_backlog,daily.stock_end)
    np.testing.assert_allclose(daily.backlog_start+daily.demand-daily.served_now-daily.served_backlog-daily.lost,daily.backlog_end)
    assert row['initial_stock']+daily.received.sum()==pytest.approx(daily.served_now.sum()+daily.served_backlog.sum()+row['end_stock'])
    assert row['ordered_units']==pytest.approx(row['generated_received']+row['end_generated_pipeline'])
    assert row['demand']==pytest.approx(daily.served_now.sum()+daily.served_backlog.sum()+row['lost_units']+row['end_backlog'])


def test_receipt_beginning_day_and_backlog_are_not_lost_sales():
    lost,dl,ol=path();back,db,ob=path('backorder')
    assert dl.received.iloc[:2].sum()==0 and dl.received.iloc[2]==40
    assert dl.served_now.iloc[2]==10
    assert dl.lost.sum()==20 and dl.backlog_end.sum()==0
    assert db.lost.sum()==0 and db.served_backlog.iloc[2]==20
    assert back['eventual_fill_rate']>back['immediate_fill_rate']
    assert back['ordered_units']>lost['ordered_units']
    assert any(o['arrives_after_end'] for o in ob)


def test_pack_moq_apply_to_every_generated_order():
    _,_,orders=path(pack=12,moq=50)
    assert orders and all(o['quantity']%12==0 and o['quantity']>=50 for o in orders)


def test_opening_pipeline_is_separate_and_received_once():
    opening=pd.DataFrame([dict(sku='A',eta='2026-06-02',known_on='2026-05-31',quantity=25)])
    row,daily,orders=path(opening=opening)
    assert row['initial_incoming']==25 and daily.opening_received.sum()==25
    assert daily.opening_received.iloc[1]==25
    assert all(o['source']=='policy_generated' for o in orders)
    opening.loc[0,'known_on']='2026-06-02'
    with pytest.raises(ValueError,match='известен'):path(opening=opening)


def test_future_and_unverifiable_inputs_cannot_change_first_decision():
    ds=baseline();changed=deepcopy(ds)
    changed.sales.loc[changed.sales.date.ge('2026-06-01'),'quantity']*=100
    changed.products['stock']=999999;changed.products['source_growth']=3;changed.products['lead_days']=365
    changed.transit=pd.DataFrame([dict(sku='A',eta=pd.Timestamp('2026-06-02'),quantity=999999)])
    changed.seasonal={i:9 for i in range(1,13)}
    a=compare_inventory(ds);b=compare_inventory(changed)
    pd.testing.assert_frame_equal(a[5][a[5].review_date.lt('2026-07-01')],b[5][b[5].review_date.lt('2026-07-01')])
    pd.testing.assert_frame_equal(a[2][a[2].ordered_on.eq('2026-06-01')],b[2][b[2].ordered_on.eq('2026-06-01')])
    train=training_data(changed,['A'],'2026-06-15')
    assert train.sales.date.max()<pd.Timestamp('2026-06-01') and train.transit.empty and not train.seasonal
    assert 'source_growth' not in train.products


def test_missing_target_is_not_imputed_as_zero():
    ds=baseline();ds.sales=ds.sales[ds.sales.date.ne('2026-07-01')]
    result,_,_,excluded,meta,_=compare_inventory(ds)
    assert result.empty and len(excluded)==1 and meta['selected_skus']==1 and meta['evaluated_skus']==0


def test_short_history_not_selected_using_future_months():
    ds=baseline();ds.sales=ds.sales[ds.sales.date.ge('2026-02-01')]
    assert compare_inventory(ds)[4]['eligible_skus']==0


def test_zero_demand_not_reported_as_perfect_fill():
    ds=baseline();ds.sales['quantity']=0
    r,d,o,e,m,f=compare_inventory(ds)
    assert r.immediate_fill_rate.isna().all() and r.ordered_units.eq(0).all()
    agg=aggregate_metrics(r)
    assert agg.positive_demand_skus.eq(0).all() and agg.zero_demand_skus.eq(1).all()
    assert agg.macro_immediate_fill_pct.isna().all()
    assert forecast_diagnostics(f,d,14).wape_pct.isna().all()


def test_policy_conditions_identical_and_no_cross_unit_stock_total():
    ds=baseline();ds.products.loc[0,'unit']='м'
    r,*_=compare_inventory(ds)
    assert r.initial_stock.nunique()==1 and r.initial_stock.iloc[0]==210
    assert r.unit.eq('м').all()
    assert 'average_stock' not in aggregate_metrics(r).columns
    assert r.immediate_fill_rate.eq(1).all()


def test_costs_separate_operating_cash_and_residual_and_vary_rates():
    r,*_=compare_inventory(baseline())
    costs=scenario_costs(r,100,2,7,13,3,'Тестовые ставки')
    assert len(costs)==54
    mid=costs[(costs.holding_factor==1)&(costs.shortage_factor==1)].iloc[0];row=r.iloc[0]
    assert mid.operating_cost==pytest.approx(row.stock_unit_days*2+row.orders*7+row.lost_units*13)
    assert mid.generated_purchase_commitment==row.ordered_units*100
    assert mid.residual_stock_value==row.end_stock*100
    with pytest.raises(ValueError):scenario_costs(r,1,1,1,1,1,'')
    with pytest.raises(ValueError):scenario_costs(r,float('nan'),1,1,1,1,'Тест')


def test_protocol_and_blank_counterbalanced_pilot_exports():
    ds=baseline();result=compare_inventory(ds)
    with zipfile.ZipFile(BytesIO(protocol_zip(*result))) as z:
        assert {'daily_balance.csv','generated_orders.csv','metadata.json','forecast_origins.csv'}.issubset(z.namelist())
        assert json.loads(z.read('metadata.json'))['settings']['start']=='2026-06-01'
    ds.products.loc[0,'name']='=UNSAFE()'
    with zipfile.ZipFile(BytesIO(pilot_zip(ds.products,'batch-id',{},[]))) as z:
        tasks=pd.read_csv(BytesIO(z.read('assignments.csv')))
        assert list(tasks.workflow)==['Excel','Axioma','Axioma','Excel']
        assert tasks.active_minutes.isna().all() and tasks.started_at.isna().all()
        assert tasks.status.eq('Не выполнено').all()
        assert "'=UNSAFE()" in z.read('frozen_product_rules.csv').decode('utf-8-sig')


@pytest.mark.parametrize('settings',[dict(start='2026-06-02'),dict(lead_days=0),dict(initial_days=float('nan'))])
def test_invalid_experiment_rejected(settings):
    with pytest.raises(ValueError):replace(Experiment(),**settings).validate()


@pytest.mark.parametrize('bad_value',[float('nan'),float('inf'),-1.])
def test_invalid_month_is_missing_not_zero(bad_value):
    ds=baseline();ds.sales.loc[ds.sales.date.eq('2026-07-01'),'quantity']=bad_value
    r,_,_,excluded,meta,_=compare_inventory(ds)
    assert r.empty and len(excluded)==1 and meta['evaluated_skus']==0


def test_partial_forecast_window_is_not_scored_as_complete():
    r,d,o,e,m,f=compare_inventory(baseline())
    scores=forecast_diagnostics(f,d,14)
    assert f.groupby('policy').size().eq(7).all()
    assert scores.windows.eq(6).all()
    assert scores.actual.eq(840).all() and scores.wape_pct.eq(0).all()


def test_cost_comparison_requires_matching_baseline():
    rows=compare_inventory(baseline())[0]
    with pytest.raises(ValueError,match='простая политика'):
        scenario_costs(rows[rows.policy.eq('adaptive')],1,1,1,1,1,'Тест')
