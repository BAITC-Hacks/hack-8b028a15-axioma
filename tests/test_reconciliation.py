import pandas as pd
from src.reconciliation import reconcile_sales


def data(monthly, detail, clean):
    s=pd.DataFrame([{'sku':'A','date':pd.Timestamp('2026-01-01'),'quantity':monthly}])
    t=pd.DataFrame([{'sku':'A','date':pd.Timestamp('2026-01-15'),'quantity':detail}])
    c=t.assign(quantity=clean)
    return s,t,c


def test_mismatched_representations_do_not_double_subtract():
    raw,removed,report=reconcile_sales(*data(100,1100,100))
    assert raw.quantity.iloc[0]==100
    assert removed.iloc[0]==0
    assert not report.consistent.iloc[0]
    assert report.detected.iloc[0]==1000


def test_matching_representation_transfers_deduction():
    raw,removed,report=reconcile_sales(*data(1100,1100,100))
    assert raw.quantity.iloc[0]-removed.iloc[0]==100
    assert report.consistent.iloc[0]


def test_missing_month_is_filled_once_and_can_be_cleaned():
    s,t,c=data(100,1100,100)
    t.date=pd.Timestamp('2026-02-15');c.date=t.date
    raw,removed,report=reconcile_sales(s,t,c)
    assert len(raw)==2
    assert raw.quantity.sum()==1200
    assert removed.sum()==1000


def test_report_only_month_has_no_fake_discrepancy_or_deduction():
    s,t,c=data(100,100,100)
    raw,removed,report=reconcile_sales(s,t.iloc[:0],c.iloc[:0])
    assert raw.quantity.sum()==100
    assert removed.sum()==0
    assert report.empty
