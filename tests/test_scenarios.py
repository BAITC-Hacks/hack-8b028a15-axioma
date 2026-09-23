import numpy as np
import pandas as pd
from src.scenarios import stock_scenario, annotate_scenario
from src.engine import calculate, Settings
from test_engine import baseline


def source():
    d = baseline()
    d.products.loc[0, 'stock'] = np.nan
    d.stocks = pd.DataFrame([dict(sku='A', date=pd.Timestamp('2026-09-01'), quantity=100),
                             dict(sku='A', date=pd.Timestamp('2026-10-01'), quantity=999)])
    d.transactions = pd.DataFrame([dict(sku='A', date=pd.Timestamp(day), quantity=q, document=str(i), client_id='')
                                  for i, (day,q) in enumerate([('2026-09-02',80),('2026-09-03',-10),('2026-09-23',40)])])
    return d


def test_snapshot_future_exclusion_and_original_unchanged():
    d = source()
    planned, evidence = stock_scenario(d, '2026-09-23')
    assert planned.products.iloc[0].stock == 100
    assert np.isnan(d.products.iloc[0].stock)
    assert evidence.iloc[0].snapshot_age_days == 22


def test_depletion_uses_actual_signed_sales_before_asof():
    d, evidence = stock_scenario(source(), '2026-09-23', 'depletion')
    assert d.products.iloc[0].stock == 30
    result = annotate_scenario(calculate(d, Settings())[0], evidence)
    assert 'не актуальный остаток' in result.iloc[0].reason
    assert result.iloc[0].recorded_sales == 70


def test_preserve_actual_stock_and_unknown_without_snapshot():
    d = source(); d.products.loc[0, 'stock'] = 42
    planned, evidence = stock_scenario(d, '2026-09-23')
    assert planned.products.iloc[0].stock == 42 and evidence.empty
    d.products.loc[0, 'stock'] = np.nan; d.stocks = d.stocks.iloc[0:0]
    planned, evidence = stock_scenario(d, '2026-09-23')
    assert np.isnan(planned.products.iloc[0].stock) and evidence.empty


def test_no_transactions_does_not_invent_depletion():
    d = source(); d.transactions = d.transactions.iloc[0:0]
    planned, evidence = stock_scenario(d, '2026-09-23', 'depletion')
    assert np.isnan(planned.products.iloc[0].stock)

