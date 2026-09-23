import pandas as pd
from src.anomalies import clean_transactions

def source():
    normal=[dict(sku='A',date=pd.Timestamp('2026-01-01')+pd.Timedelta(days=i),quantity=2.,document=f'd{i}',client_id=f'c{i}') for i in range(20)]
    project=[dict(sku='A',date=pd.Timestamp('2026-08-21')+pd.Timedelta(days=i),quantity=100.,document=f'p{i}',client_id='project') for i in range(3)]
    return pd.DataFrame(normal+project)

def test_project_split_across_three_days_still_excluded():
    cleaned,events=clean_transactions(source())
    assert cleaned.loc[cleaned.client_id.eq('project'),'quantity'].sum()==6
    assert events.excluded.sum()==294

def test_manager_can_keep_or_exclude_without_changing_source():
    tx=source();_,events=clean_transactions(tx)
    decisions={event:'regular' for event in events.event_id}
    cleaned,review=clean_transactions(tx,decisions)
    assert cleaned.quantity.sum()==tx.quantity.sum()
    assert review.excluded.sum()==0
    assert tx.loc[tx.client_id.eq('project'),'quantity'].sum()==300

def test_event_identity_survives_sorting():
    _,a=clean_transactions(source());_,b=clean_transactions(source().iloc[::-1])
    assert set(a.event_id)==set(b.event_id)

def test_anonymized_client_not_exposed_in_events():
    _,events=clean_transactions(source())
    assert 'client_id' not in events
    assert not events.astype(str).apply(lambda s:s.str.contains('project',regex=False)).any().any()

def test_wholesale_tier_without_client_is_preserved_but_isolated_spike_removed():
    normal=[dict(sku='A',date=pd.Timestamp('2025-01-01')+pd.Timedelta(days=i),quantity=2.,document=f'n{i}',client_id='') for i in range(100)]
    bulk=[dict(sku='A',date=pd.Timestamp('2025-01-10')+pd.DateOffset(months=i//3)+pd.Timedelta(days=i%3),quantity=100.,document=f'b{i}',client_id='') for i in range(18)]
    spike=[dict(sku='A',date=pd.Timestamp('2025-07-20'),quantity=10000.,document='project',client_id='')]
    cleaned,events=clean_transactions(pd.DataFrame(normal+bulk+spike))
    assert cleaned.loc[cleaned.document.str.startswith('b'),'quantity'].sum()==1800
    assert cleaned.loc[cleaned.document.eq('project'),'quantity'].sum()<100
