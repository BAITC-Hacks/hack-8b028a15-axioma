from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from axioma.model import Dataset
from axioma.engine import calculate, Policy, clean_transactions
from axioma.demo import make_demo


def constant_dataset(qty=10):
    dates = pd.date_range('2024-01-01', '2026-09-22')
    product = dict(supplier='Test', sku='001', article='TEST-001', name='Синтетический товар', stock=0,
                   category='C', lead_days=10, review_days=20, unit='шт', moq=0, pack=1, purchase_factor=1,
                   growth_pct=np.nan, assumptions='test')
    sales = pd.DataFrame({'supplier': 'Test', 'sku': '001', 'date': dates, 'qty': qty,
                          'document': [f'D{i}' for i in range(len(dates))], 'client_id': 'anon'})
    return Dataset(pd.DataFrame([product]), sales)


def first(ds, policy=None):
    return calculate(ds, policy)[0].iloc[0]


def test_formula_and_input_sensitivity():
    ds = constant_dataset()
    assert first(ds).recommended_qty == 370  # 30 days × 10 + 7 safety days × 10
    ds.products.loc[0, 'stock'] = 100
    assert first(ds).recommended_qty == 270
    ds.incoming = pd.DataFrame([dict(supplier='Test', sku='001', eta='2026-09-25', qty=70)])
    assert first(ds).recommended_qty == 200
    ds.products.loc[0, 'category'] = 'A'
    assert first(ds).recommended_qty == 270
    ds.products.loc[0, 'growth_pct'] = 50
    assert first(ds).recommended_qty == 490


def test_incoming_after_horizon_not_deducted_and_overdue_not_assumed_received():
    ds = constant_dataset()
    for eta in ['2026-12-01', '2026-09-20']:
        ds.incoming = pd.DataFrame([dict(supplier='Test', sku='001', eta=eta, qty=5000)])
        assert first(ds).recommended_qty == 370


def test_late_arrival_does_not_hide_early_shortage():
    ds = constant_dataset()
    ds.products.loc[0, 'stock'] = 20
    ds.incoming = pd.DataFrame([dict(supplier='Test', sku='001', eta='2026-10-15', qty=5000)])
    row = first(ds)
    assert row.recommended_qty == 0
    assert row.urgency == 'Срочно'
    assert row.shortage_date == '2026-09-25'


def test_stockout_restores_only_confirmed_intervals():
    ds = constant_dataset()
    mask = ds.sales.date.between('2026-08-15', '2026-09-05')
    ds.sales.loc[mask, 'qty'] = 0
    raw = first(ds)
    ds.stockouts = pd.DataFrame([dict(supplier='Test', sku='001', start='2026-08-15', end='2026-09-05')])
    restored = first(ds)
    assert restored.restored_qty == 220
    assert restored.recommended_qty > raw.recommended_qty
    assert first(ds, Policy(compensate_stockouts=False)).restored_qty == 0


def test_one_off_order_is_excluded_and_regular_demand_stable():
    ds = constant_dataset()
    baseline = first(ds).recommended_qty
    spike = ds.sales.iloc[-1].copy()
    spike['qty'], spike['document'], spike['client_id'] = 100000, 'PROJECT', 'anon-project'
    ds.sales = pd.concat([ds.sales, spike.to_frame().T], ignore_index=True)
    row = first(ds)
    assert row.excluded_qty == 100000
    assert abs(row.recommended_qty - baseline) / baseline < .05
    assert first(ds, Policy(remove_outliers=False)).recommended_qty > baseline * 2


def test_split_client_order_and_repeated_bulk_demand():
    dates = pd.date_range('2026-01-01', periods=50)
    frame = pd.DataFrame({'date': dates, 'qty': 10, 'document': [str(i) for i in range(50)], 'client_id': 'anon-regular'})
    split = pd.DataFrame([dict(date=dates[-1], qty=15, document=f'bulk{i}', client_id='anon-project') for i in range(20)])
    cleaned = clean_transactions(pd.concat([frame, split]))
    assert cleaned.loc[cleaned.client_id == 'anon-project', 'outlier'].all()
    repeated = frame.copy()
    repeated.loc[[10, 20, 30], 'qty'] = 1000
    assert not clean_transactions(repeated).outlier.any()


def test_seasonality_and_trend():
    ds = constant_dataset()
    ds.sales['qty'] = ds.sales.date.dt.month.map(lambda m: 30 if m in [10, 11, 12] else 10)
    _, details = calculate(ds)
    forecast = details[('Test', '001')]['forecast']
    assert forecast[forecast.index.month == 10].mean() > forecast[forecast.index.month == 9].mean() * 1.5
    ds = constant_dataset()
    ds.sales['qty'] = 5 + np.arange(len(ds.sales)) * .05
    assert first(ds).growth_factor > 1
    ds.products.loc[0, 'growth_pct'] = 0
    assert first(ds).growth_factor == 1


def test_missing_stock_no_history_and_conversion_block():
    ds = constant_dataset()
    ds.products.loc[0, 'stock'] = np.nan
    assert pd.isna(first(ds).recommended_qty)
    ds.products.loc[0, 'stock'] = 0
    ds.products['unit_conversion_required'] = True
    assert pd.isna(first(ds).recommended_qty)
    ds.products['unit_conversion_required'] = False
    ds.sales = ds.sales.iloc[:0]
    assert pd.isna(first(ds).recommended_qty)


def test_moq_pack_and_conversion_are_separate():
    ds = constant_dataset()
    ds.products.loc[0, ['purchase_factor', 'moq', 'pack']] = [100, 5, 3]
    row = first(ds)
    assert row.recommended_qty == 6
    assert row.stock_units == 600
    ds.products.loc[0, 'stock'] = 10000
    assert first(ds).recommended_qty == 0


def test_returns_not_turned_into_positive_demand():
    ds = constant_dataset()
    ds.sales.loc[len(ds.sales) - 1, 'qty'] = -10000
    row = first(ds)
    assert row.recommended_qty < 370
    assert row.excluded_qty == 0


def test_future_sales_are_not_used():
    ds = constant_dataset()
    baseline = first(ds).recommended_qty
    future = ds.sales.iloc[-1].copy()
    future['date'], future['qty'] = pd.Timestamp('2026-12-01'), 1000000
    ds.sales = pd.concat([ds.sales, future.to_frame().T], ignore_index=True)
    assert first(ds).recommended_qty == baseline


def test_demo_explanations_and_supplier_groups():
    rows, details = calculate(make_demo())
    assert rows.supplier.nunique() == 2
    assert rows.explanation.str.len().min() > 100
    assert rows.recommended_qty.notna().all()
    assert rows.loc[rows.sku == 'DEMO-005', 'restored_qty'].iloc[0] > 0
    assert rows.loc[rows.sku == 'DEMO-004', 'excluded_qty'].iloc[0] == 10000


def test_invalid_input_is_rejected():
    ds = constant_dataset()
    ds.products.loc[0, 'pack'] = 0
    with pytest.raises(ValueError):
        calculate(ds)


def test_current_day_transactions_with_time_are_included():
    ds = constant_dataset()
    ds.sales['date'] += pd.Timedelta(hours=15)
    assert first(ds).recommended_qty == 370


def test_nonfinite_and_fractional_parameters_rejected():
    for col, value in [('stock', np.inf), ('pack', np.inf), ('lead_days', 1.5), ('growth_pct', -101)]:
        ds = constant_dataset()
        ds.products[col] = ds.products[col].astype(float)
        ds.products.loc[0, col] = value
        with pytest.raises(ValueError):
            calculate(ds)


def test_old_return_does_not_create_fake_zero_history():
    ds = constant_dataset()
    ds.sales = ds.sales[ds.sales.date >= '2025-01-01'].copy()
    old_return = ds.sales.iloc[0].copy()
    old_return['date'], old_return['qty'] = pd.Timestamp('2023-01-01'), -10
    ds.sales = pd.concat([ds.sales, old_return.to_frame().T], ignore_index=True)
    _, details = calculate(ds)
    assert details[('Test', '001')]['history'].index.min() == pd.Timestamp('2025-01-01')


def test_earlier_monthly_history_informs_seasonality_without_double_count():
    ds = constant_dataset()
    ds.sales = ds.sales[ds.sales.date >= '2026-01-01'].copy()
    ds.monthly_sales = pd.DataFrame([dict(supplier='Test', sku='001', month=m,
                                        qty=(30 if m.month in [10, 11, 12] else 10) * m.days_in_month)
                                   for m in pd.date_range('2024-01-01', '2025-12-01', freq='MS')])
    row = first(ds)
    assert 'ранняя месячная история' in row.method
    _, details = calculate(ds)
    f = details[('Test', '001')]['forecast']
    assert f[f.index.month == 10].mean() > f[f.index.month == 9].mean() * 1.5
    baseline = row.recommended_qty
    ds.monthly_sales = pd.concat([ds.monthly_sales, pd.DataFrame([dict(supplier='Test', sku='001', month='2026-08-01', qty=1000000)])], ignore_index=True)
    assert first(ds).recommended_qty == baseline
