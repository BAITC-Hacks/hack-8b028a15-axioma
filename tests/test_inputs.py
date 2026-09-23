import pandas as pd
import numpy as np
import pytest
from src.inputs import apply_updates,stock_template


def test_missing_inventory_remains_unknown_and_known_zero_is_applied():
    base=pd.DataFrame({'sku':['A','B'],'stock':[np.nan,5.]})
    result,matched,unknown=apply_updates(base,pd.DataFrame({'sku':['A','B','UNKNOWN'],'stock':[np.nan,0,5]}))
    assert np.isnan(result.iloc[0].stock) and result.iloc[1].stock==0
    assert matched==2 and unknown==['UNKNOWN']
    assert b'A' in stock_template(base)


@pytest.mark.parametrize('values',[{'sku':[None],'stock':[1]}, {'sku':['A'],'lead_days':[1.5]}, {'sku':['A'],'stock':[float('inf')]}, {'sku':['A','A'],'stock':[1,2]}])
def test_invalid_updates_rejected(values):
    with pytest.raises(ValueError):apply_updates(pd.DataFrame({'sku':['A'],'stock':[np.nan]}),pd.DataFrame(values))
