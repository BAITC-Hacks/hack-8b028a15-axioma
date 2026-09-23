"""Validate inventory/planning updates before changing any dataset."""
import numpy as np
import pandas as pd

EDITABLE=['stock','category','pack','moq','lead_days','source_growth']

def apply_updates(products,updates):
    extra=updates.copy()
    if 'sku' not in extra:raise ValueError('Нужна колонка sku')
    if extra.sku.isna().any():raise ValueError('Код товара не может быть пустым')
    extra['sku']=extra.sku.astype(str).str.strip()
    if extra.sku.eq('').any() or extra.sku.duplicated().any():raise ValueError('Коды должны быть непустыми и уникальными')
    for col in ['stock','pack','moq','lead_days','source_growth']:
        if col not in extra:continue
        extra[col]=pd.to_numeric(extra[col],errors='raise')
        values=extra[col].dropna()
        if not np.isfinite(values).all():raise ValueError(f'{col}: требуется конечное число')
        lo,hi={'stock':(0,1e12),'pack':(1,1e9),'moq':(1,1e9),'lead_days':(1,365),'source_growth':(-.9,3)}[col]
        if ((values<lo)|(values>hi)).any():raise ValueError(f'{col}: допустимо от {lo} до {hi}')
        if col=='lead_days' and (values!=np.floor(values)).any():raise ValueError('lead_days: укажите целое число дней')
    base=products.set_index('sku').copy();extra=extra.set_index('sku')
    for col in EDITABLE:
        if col in extra and col not in base:base[col]=np.nan if col!='category' else ''
    base.update(extra[[c for c in EDITABLE if c in extra]])
    return base.reset_index(),len(base.index.intersection(extra.index)),list(extra.index.difference(base.index))

def stock_template(products):
    missing=products[products.stock.isna()].copy()
    if missing.empty:missing=products.copy()
    output=missing.reindex(columns=['sku','name'],fill_value='').copy()
    output['stock']='';output['lead_days']=''
    return output.to_csv(index=False).encode('utf-8-sig')
