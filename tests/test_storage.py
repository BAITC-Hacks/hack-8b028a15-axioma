import pandas as pd
import pytest
from src.storage import Store
from src.engine import Settings

def setup(tmp_path):
    store=Store(tmp_path/'db.sqlite3')
    df=pd.DataFrame([dict(sku='A',stock=0.,recommended=12.,pack=6.,moq=6.,name='Товар')])
    run=store.save_run('IEK','dataset',Settings(),df,[])
    return store,run,df

def test_persist_and_reopen(tmp_path):
    s,r,df=setup(tmp_path);s2=Store(s.path)
    assert s2.get_run(r)['rows'][0]['recommended']==12
    df.loc[0,'recommended']=999
    assert s2.get_run(r)['rows'][0]['recommended']==12

def test_approval_idempotency_and_reason(tmp_path):
    s,r,_=setup(tmp_path)
    with pytest.raises(ValueError):s.approve(r,[dict(sku='A',quantity=18)],'Manager')
    choices=[dict(sku='A',quantity=18,reason='Подтверждён дополнительный спрос')]
    a=s.approve(r,choices,'Manager');b=s.approve(r,choices,'Manager')
    assert a==b and len(s.list_orders())==1

def test_invalid_approval_rejected(tmp_path):
    s,r,_=setup(tmp_path)
    for q in [0,-1,float('nan'),float('inf'),7]:
        with pytest.raises(ValueError):s.approve(r,[dict(sku='A',quantity=q,reason='test')],'Manager')
    with pytest.raises(ValueError):s.approve(r,[dict(sku='MISSING',quantity=6)],'Manager')

def test_missing_stock_cannot_be_approved(tmp_path):
    s,_,df=setup(tmp_path);df['stock']=None;df['recommended']=None
    r=s.save_run('IEK','other',Settings(),df,[])
    with pytest.raises(ValueError):s.approve(r,[dict(sku='A',quantity=12)],'Manager')

def test_preferences_isolated_by_input_fingerprint(tmp_path):
    s,_,_=setup(tmp_path);s.save_preferences('one','anomalies',{'id':'regular'})
    assert Store(s.path).preferences('one','anomalies')=={'id':'regular'}
    assert s.preferences('two','anomalies',{})=={}
