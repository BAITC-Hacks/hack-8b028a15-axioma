import pandas as pd
import pytest
from src.orders import add_reviewed_lines


def lines(supplier='IEK',quantity=12):
    return pd.DataFrame([dict(supplier=supplier,sku='A',quantity=quantity,pack=6,moq=10,reason='Проверено')])


def test_supplier_identity_and_reapproval_replace_not_duplicate():
    cart=add_reviewed_lines(pd.DataFrame(),lines())
    cart=add_reviewed_lines(cart,lines('Systeme Electric'))
    cart=add_reviewed_lines(cart,lines(quantity=18))
    assert len(cart)==2
    assert cart.loc[cart.supplier.eq('IEK'),'quantity'].iloc[0]==18


@pytest.mark.parametrize('quantity',[0,9,13,float('nan'),float('inf')])
def test_invalid_order_quantities_cannot_enter_cart(quantity):
    with pytest.raises(ValueError):add_reviewed_lines(pd.DataFrame(),lines(quantity=quantity))

