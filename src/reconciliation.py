"""Reconcile two representations of sales before transferring corrections.

The monthly 1C report is authoritative where present. Transaction detail fills
missing keys. Outlier deductions transfer only if both totals agree (2% or one
accounting unit). A discrepancy is not evidence that either source is wrong.
"""
import numpy as np
import pandas as pd


def reconcile_sales(sales, transactions, cleaned=None):
    raw=sales.groupby(['sku','date']).quantity.sum().astype(float)
    empty=pd.DataFrame(columns=['sku','date','monthly','detail','difference','detected','applied','consistent'])
    if transactions.empty:return raw.reset_index(),pd.Series(0.,index=raw.index),empty
    def monthly(frame):
        return frame.assign(date=frame.date.dt.to_period('M').dt.to_timestamp()).groupby(['sku','date']).quantity.sum().astype(float)
    detail=monthly(transactions)
    excluded=(detail-monthly(cleaned if cleaned is not None else transactions)).clip(lower=0)
    combined=raw.combine_first(detail)
    comparison=pd.DataFrame({'monthly':raw,'detail':detail})
    comparison['difference']=comparison.detail-comparison.monthly
    tolerance=np.maximum(1.,comparison.monthly.abs()*.02)
    comparison['consistent']=comparison.monthly.isna() | comparison.difference.abs().le(tolerance)
    comparison['detected']=excluded.reindex(comparison.index,fill_value=0)
    comparison['applied']=comparison.detected.where(comparison.consistent,0).clip(lower=0)
    applied=comparison.applied.reindex(combined.index,fill_value=0)
    return combined.reset_index(),applied,comparison.reset_index()
