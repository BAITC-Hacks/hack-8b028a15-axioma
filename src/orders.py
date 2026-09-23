"""Explicitly reviewed order lines, keyed by supplier and SKU."""
import numpy as np
import pandas as pd


def add_reviewed_lines(cart, lines):
    if lines.empty:return cart.copy()
    required={'supplier','sku','quantity','pack','moq','reason'}
    if not required.issubset(lines):raise ValueError('Не хватает полей утверждённого заказа')
    for c in ('quantity','pack','moq'):
        if not np.isfinite(pd.to_numeric(lines[c],errors='coerce')).all():
            raise ValueError('Количество и ограничения должны быть конечными числами')
    if (lines.quantity<=0).any() or (lines.pack<=0).any() or (lines.moq<=0).any():
        raise ValueError('Количество, минимум и кратность должны быть положительными')
    if (lines.quantity<lines.moq).any() or not np.isclose(lines.quantity%lines.pack,0).all():
        raise ValueError('Нарушены минимальная партия или кратность')
    if lines[['supplier','sku']].isna().any().any():raise ValueError('Не указан поставщик или код товара')
    combined=pd.concat([cart,lines],ignore_index=True) if not cart.empty else lines.copy()
    return combined.drop_duplicates(['supplier','sku'],keep='last').sort_values(['supplier','sku']).reset_index(drop=True)

