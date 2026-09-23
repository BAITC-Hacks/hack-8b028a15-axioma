"""Reproducible synthetic examples. No partner/customer data."""
import numpy as np
import pandas as pd
from .model import Dataset


def make_demo():
    rng = np.random.default_rng(42)
    dates = pd.date_range('2024-01-01', '2026-09-22')
    specs = [
        ('IEK', 'DEMO-001', 'Автоматический выключатель', 'A', 100, 12, 'steady'),
        ('IEK', 'DEMO-002', 'Светильник уличный — сезонный', 'B', 180, 8, 'seasonal'),
        ('IEK', 'DEMO-003', 'Кабель — закупка бухтами', 'A', 610, 45, 'growth'),
        ('Systeme Electric', 'DEMO-004', 'Розетка — разовая крупная продажа', 'B', 60, 9, 'outlier'),
        ('Systeme Electric', 'DEMO-005', 'Выключатель — отсутствие товара', 'A', 20, 11, 'stockout'),
        ('Systeme Electric', 'DEMO-006', 'Рамка — достаточный запас', 'C', 10000, 4, 'steady'),
    ]
    products, sales, incoming, stockouts = [], [], [], []
    for supplier, sku, name, category, stock, daily, kind in specs:
        products.append(dict(supplier=supplier, sku=sku, article=sku, name=name, category=category,
                             unit='м' if kind == 'growth' else 'шт', purchase_unit='бухта' if kind == 'growth' else 'шт',
                             stock=stock, stock_date='2026-09-22', lead_days=14, review_days=30,
                             moq=1, pack=1 if kind == 'growth' else 5, purchase_factor=305 if kind == 'growth' else 1,
                             growth_pct=np.nan, assumptions='Синтетический пример; сроки и запас заданы для демонстрации.'))
        for i, date in enumerate(dates):
            season = 1 + .65 * np.cos((date.month - 11) * 2 * np.pi / 12) if kind == 'seasonal' else 1
            trend = 1 + i / len(dates) * 1.2 if kind == 'growth' else 1
            qty = float(rng.poisson(daily * season * trend))
            if kind == 'stockout' and pd.Timestamp('2026-08-15') <= date <= pd.Timestamp('2026-09-05'):
                qty = 0
            sales.append(dict(supplier=supplier, sku=sku, date=date, qty=qty,
                              document=f'SYN-{sku}-{i}', client_id=f'anon-{i % 20:02}', warehouse='Демо-склад'))
        if kind == 'outlier':
            sales.append(dict(supplier=supplier, sku=sku, date=pd.Timestamp('2026-09-10'), qty=10000,
                              document='SYN-ONE-OFF', client_id='anon-project', warehouse='Демо-склад'))
        if kind == 'stockout':
            stockouts.append(dict(supplier=supplier, sku=sku, start='2026-08-15', end='2026-09-05'))
        incoming.extend([dict(supplier=supplier, sku=sku, eta='2026-09-28', qty=daily * 4),
                         dict(supplier=supplier, sku=sku, eta='2026-12-01', qty=daily * 20)])
    return Dataset(pd.DataFrame(products), pd.DataFrame(sales), pd.DataFrame(incoming), pd.DataFrame(stockouts))
