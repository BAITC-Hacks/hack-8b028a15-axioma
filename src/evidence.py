"""Executable case acceptance examples; synthetic, not a business ROI claim."""
from copy import deepcopy
from dataclasses import replace
import pandas as pd
from .data import Dataset, demo_data
from .engine import Settings, calculate
from .orders import add_reviewed_lines


def acceptance_examples():
    ds=Dataset('Учебный поставщик')
    ds.products=pd.DataFrame([dict(sku='CHECK-001',article='CHECK-001',name='Тестовый товар',category='1',stock=0.,pack=1.,moq=1.,unit='шт')])
    ds.sales=pd.DataFrame([dict(sku='CHECK-001',date=d,quantity=d.days_in_month*10.) for d in pd.date_range('2024-01-01','2026-08-01',freq='MS')])
    cfg=Settings(seasonality=False,trend=False)
    def order(data,settings=cfg):return calculate(data,settings)[0].iloc[0]
    rows=[]
    def record(criterion,before,after,condition,passed):
        rows.append({'Проверка':criterion,'До':before,'После':after,'Ожидаемое поведение':condition,'Результат':'Пройдено' if passed else 'Ошибка'})
    base=order(ds).recommended
    changed=deepcopy(ds);changed.transit=pd.DataFrame([dict(sku='CHECK-001',eta=pd.Timestamp('2026-09-25'),quantity=70.)])
    value=order(changed).recommended
    record('1. Поставка в пути',base,value,'70 ед. в пути уменьшают заказ на 70',value==base-70)
    changed=deepcopy(ds);changed.products.loc[0,'stock']=100
    value=order(changed).recommended
    record('1. Свободный остаток',base,value,'100 ед. остатка уменьшают заказ на 100',value==base-100)
    value=order(ds,replace(cfg,category_factors={'1':2.})).recommended
    record('1. Категория',base,value,'Множитель 2 увеличивает страховой запас',value>base)
    value=order(ds,replace(cfg,growth_percent=20)).recommended
    record('1. Прогноз прироста',base,value,'Прирост +20% увеличивает потребность',value>base)
    changed=deepcopy(ds);changed.sales.loc[changed.sales.date.dt.month.eq(10),'quantity']*=3
    off=order(changed,replace(cfg,as_of='2026-10-01')).forecast
    on=order(changed,replace(cfg,as_of='2026-10-01',seasonality=True)).forecast
    record('2. Сезонность',off,on,'Октябрьский пик повышает прогноз',on>off)
    changed=deepcopy(ds);changed.sales.loc[changed.sales.date.ge('2026-06-01'),'quantity']*=1.5
    off=order(changed).forecast;on=order(changed,replace(cfg,trend=True)).forecast
    record('2. Устойчивый рост',off,on,'Три месяца роста повышают прогноз',on>off)
    changed=deepcopy(ds);changed.sales.loc[changed.sales.date.eq('2026-08-01'),'quantity']=110
    changed.stockouts=pd.DataFrame([dict(sku='CHECK-001',start=pd.Timestamp('2026-08-01'),end=pd.Timestamp('2026-08-20'))])
    off=order(changed,replace(cfg,compensate_stockout=False)).forecast;on=order(changed).forecast
    record('3. 20 дней отсутствия',off,on,'Подтверждённое отсутствие повышает оценку спроса',on>off)
    demo=demo_data()
    clean=calculate(demo,Settings())[0].set_index('sku').loc['DEMO-001']
    demo.transactions=demo.transactions[demo.transactions.document.ne('ONE-OFF')]
    demo.sales.loc[demo.sales.sku.eq('DEMO-001')&demo.sales.date.eq('2026-08-01'),'quantity']-=1500
    no_spike=calculate(demo,Settings())[0].set_index('sku').loc['DEMO-001']
    record('4. Разовая покупка 1500 ед.',no_spike.forecast,clean.forecast,'Разница прогноза менее 5 ед. после очистки',abs(clean.forecast-no_spike.forecast)<5)
    changed=deepcopy(ds)
    changed.transactions=pd.DataFrame([dict(sku='CHECK-001',date=pd.Timestamp('2026-08-01')+pd.Timedelta(days=i),quantity=2.,document=f'd{i}',client_id=f'client{i}') for i in range(20)])
    changed.sales.loc[changed.sales.date.eq('2026-08-01'),'quantity']=40
    off=order(changed).forecast
    extra=pd.DataFrame([dict(sku='CHECK-001',date=pd.Timestamp('2026-08-25'),quantity=2.,document=f'x{i}',client_id='one-client') for i in range(30)])
    changed.transactions=pd.concat([changed.transactions,extra],ignore_index=True)
    changed.sales.loc[changed.sales.date.eq('2026-08-01'),'quantity']+=60
    on=order(changed).forecast
    record('4. Один клиент, 30 накладных',off,on,'Разница прогноза менее 1 ед. после очистки',abs(on-off)<1)
    lines=pd.DataFrame([dict(supplier=s,sku='CHECK-001',quantity=12.,pack=6.,moq=10.,reason='Проверка: MOQ 10, кратность 6') for s in ['Поставщик A','Поставщик B']])
    cart=add_reviewed_lines(pd.DataFrame(),lines)
    record('5. Заказы по поставщикам',0,len(cart),'Два поставщика, две строки с объяснениями',len(cart)==2 and cart.reason.str.len().gt(0).all())
    changed=deepcopy(ds);changed.products.loc[0,'stock']=220
    changed.transit=pd.DataFrame([dict(sku='CHECK-001',eta=pd.Timestamp('2026-10-27'),quantity=500.)])
    row=order(changed)
    record('Дополнительно: позднее поступление',0,row.recommended,'При итоговом избытке нужен заказ 120 ед. для разрыва по датам',row.recommended==120)
    return pd.DataFrame(rows)

