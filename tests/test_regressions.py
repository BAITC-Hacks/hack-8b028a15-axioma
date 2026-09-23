from dataclasses import replace
import numpy as np
import pandas as pd
from src.engine import Settings, calculate
from src.data import Dataset

def fixture():
    d=Dataset('Test')
    d.products=pd.DataFrame([dict(sku='A',article='A',name='A',category='1',stock=0.,pack=1.,moq=1.,unit='шт')])
    d.sales=pd.DataFrame([dict(sku='A',date=dt,quantity=dt.days_in_month*10.) for dt in pd.date_range('2024-01-01','2026-08-01',freq='MS')])
    return d

def test_supplied_growth_changes_recommendation():
    d=fixture();base=calculate(d,Settings())[0].iloc[0]
    d.products['source_growth']=.2
    result=calculate(d,Settings())[0].iloc[0]
    assert result.recommended>base.recommended
    assert result.growth==20
    assert calculate(d,Settings(trend=False))[0].iloc[0].recommended==base.recommended

def test_missing_monthly_sku_uses_transactions_without_double_counting():
    d=fixture()
    d.products=pd.concat([d.products,d.products.assign(sku='B')],ignore_index=True)
    d.transactions=pd.DataFrame([dict(sku=sku,date=pd.Timestamp('2026-08-01'),quantity=100.,document=sku,client_id='') for sku in ['A','B']])
    result=calculate(d,Settings())[0].set_index('sku')
    assert result.loc['A','recommended']==420
    assert result.loc['B','recommended']>0

def test_current_month_cannot_change_historical_outlier_threshold():
    d=fixture()
    d.transactions=pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-08-01')+pd.Timedelta(days=i),quantity=2.,document=str(i),client_id='') for i in range(10)])
    d.transactions=pd.concat([d.transactions,pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-08-20'),quantity=100.,document='spike',client_id='')])],ignore_index=True)
    before=calculate(d,Settings())[0].iloc[0].recommended
    extra=pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-09-01'),quantity=10000.,document=f'future-{i}',client_id='') for i in range(30)])
    d.transactions=pd.concat([d.transactions,extra],ignore_index=True)
    assert calculate(d,Settings())[0].iloc[0].recommended==before

def test_invalid_settings_rejected():
    import pytest
    with pytest.raises(ValueError):calculate(fixture(),Settings(lead_days=0))
    with pytest.raises(ValueError):calculate(fixture(),Settings(category_factors={'1':float('nan')}))

def test_late_delivery_does_not_hide_earlier_shortage():
    d=fixture();d.products.loc[0,'stock']=1
    d.transit=pd.DataFrame([dict(sku='A',eta=pd.Timestamp('2026-10-10'),quantity=1000)])
    r=calculate(d,Settings())[0].iloc[0]
    assert r.status=='Срочно'
    assert r.recommended==0

def test_zip_import_reads_members_without_extracting_paths():
    from io import BytesIO
    import zipfile
    from src.data import unpack_excel_archive
    buffer=BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:z.writestr('../../sample.xlsx',b'example')
    assert unpack_excel_archive(buffer.getvalue())==[('sample.xlsx',b'example')]

def test_recurring_large_customer_is_not_deleted_as_one_off():
    from src.engine import clean_transactions
    tx=pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-08-01')+pd.Timedelta(days=i),quantity=2.,document=f'd{i}',client_id=f'c{i}') for i in range(20)])
    repeat=pd.DataFrame([dict(sku='A',date=pd.Timestamp('2026-01-21')+pd.DateOffset(months=i*2),quantity=100.,document=f'big{i}',client_id='regular-large') for i in range(3)])
    cleaned,_=clean_transactions(pd.concat([tx,repeat],ignore_index=True))
    assert cleaned.loc[cleaned.client_id.eq('regular-large'),'quantity'].sum()==300

def test_moq_trailing_note_does_not_create_fake_sku(monkeypatch):
    from src.data import parse_files
    frame=pd.DataFrame([['№','Код 1с','Артикул поставщика','Наименование','Мин. разр. к отгр.'],[1,'A','ART','Товар',2],[None,0,'Расширение 3кв 24',None,None]])
    monkeypatch.setattr(pd,'read_excel',lambda *a,**k:frame)
    ds=parse_files([('MOQ.xlsx',b'fixture')],'IEK')
    assert list(ds.products.sku)==['A']
