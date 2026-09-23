"""Explicit planning assumptions; never overwrite known current stock."""
from copy import deepcopy
import pandas as pd


def stock_scenario(dataset, as_of, mode='reference'):
    """Fill unknown stock from latest beginning-month snapshot before as_of.

    reference holds the snapshot unchanged; depletion subtracts signed recorded
    sales, assuming no receipts or reservations. Neither is current inventory or
    a confidence bound. Raw sales (including large orders) physically use stock.
    """
    if mode not in ('reference', 'depletion'):
        raise ValueError('Unknown stock scenario')
    ds = deepcopy(dataset)
    asof = pd.Timestamp(as_of).normalize()
    evidence = []
    snapshots = ds.stocks.dropna(subset=['date', 'quantity'])
    snapshots = snapshots[snapshots.date.le(asof)].sort_values('date')
    for idx, product in ds.products.iterrows():
        if pd.notna(product.stock):
            continue
        history = snapshots[snapshots.sku.eq(product.sku)]
        if history.empty:
            continue
        snapshot = history.iloc[-1]
        sales = 0.
        if mode == 'depletion':
            if ds.transactions.empty:
                continue
            tx = ds.transactions[ds.transactions.sku.eq(product.sku)
                                 & ds.transactions.date.ge(snapshot.date)
                                 & ds.transactions.date.lt(asof)]
            # No recorded rows cannot distinguish no sales from missing coverage.
            if tx.empty:
                continue
            sales = float(tx.quantity.sum())
        estimated = max(0., float(snapshot.quantity) - sales)
        ds.products.loc[idx, 'stock'] = estimated
        evidence.append(dict(sku=product.sku, stock_basis='Сценарий: снимок без изменений' if mode == 'reference' else 'Сценарий: без поступлений и резервов',
                             snapshot_date=str(snapshot.date.date()), snapshot_stock=float(snapshot.quantity),
                             recorded_sales=sales, scenario_stock=estimated,
                             snapshot_age_days=(asof-snapshot.date).days))
    return ds, pd.DataFrame(evidence, columns=['sku','stock_basis','snapshot_date','snapshot_stock','recorded_sales','scenario_stock','snapshot_age_days'])


def annotate_scenario(result, evidence):
    result = result.merge(evidence, on='sku', how='left', validate='one_to_one')
    mask = result.stock_basis.notna()
    result.loc[mask, 'reason'] = result.loc[mask].apply(
        lambda r: f"{r.stock_basis}; снимок {r.snapshot_date}: {r.snapshot_stock:g}, продажи после снимка: {r.recorded_sales:g}. Это допущение, не актуальный остаток. " + r.reason, axis=1)
    result['stock_basis'] = result.stock_basis.fillna('Остаток из входных данных' )
    result.loc[result.stock.isna(), 'stock_basis'] = 'Неизвестен'
    return result

