"""Local-only adapters for the supplied IEK and Systeme Electric Excel exports."""
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
import re
import zipfile
import hashlib
import numpy as np
import pandas as pd

MONTHS = {'янв':1,'фев':2,'мар':3,'апр':4,'май':5,'июн':6,'июл':7,'авг':8,'сен':9,'окт':10,'ноя':11,'дек':12}

def unpack_excel_archive(content):
    """Read XLSX members in memory; never extract paths onto the filesystem."""
    with zipfile.ZipFile(BytesIO(content)) as archive:
        entries=[e for e in archive.infolist() if not e.is_dir() and e.filename.lower().endswith('.xlsx')]
        if not entries:raise ValueError('В архиве нет Excel-файлов')
        if len(entries)>30 or sum(e.file_size for e in entries)>150_000_000:raise ValueError('Архив превышает лимит: 30 файлов / 150 МБ')
        files=[]
        for entry in entries:
            name=entry.filename
            if not entry.flag_bits & 0x800:
                try:name=name.encode('cp437').decode('cp866')
                except UnicodeError:pass
            files.append((name.replace('\\','/').split('/')[-1],archive.read(entry)))
        return files

def key(value):
    if pd.isna(value): return ''
    return str(value).strip().removesuffix('.0')

def number(value, default=0):
    try:
        x=float(value)
        return x if np.isfinite(x) else default
    except (ValueError, TypeError): return default

def month(value):
    s=str(value).lower()
    year=re.search(r'20\d{2}',s)
    for name, n in MONTHS.items():
        if name in s and year: return pd.Timestamp(int(year.group()),n,1)
    return None

@dataclass
class Dataset:
    supplier: str
    products: pd.DataFrame = field(default_factory=lambda:pd.DataFrame(columns=['sku','article','name','category','stock','pack','moq','unit']))
    sales: pd.DataFrame = field(default_factory=lambda:pd.DataFrame(columns=['sku','date','quantity']))
    transactions: pd.DataFrame = field(default_factory=lambda:pd.DataFrame(columns=['sku','date','quantity','document','client_id']))
    stocks: pd.DataFrame = field(default_factory=lambda:pd.DataFrame(columns=['sku','date','quantity']))
    transit: pd.DataFrame = field(default_factory=lambda:pd.DataFrame(columns=['sku','eta','quantity']))
    stockouts: pd.DataFrame = field(default_factory=lambda:pd.DataFrame(columns=['sku','start','end']))
    seasonal: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    sources: list = field(default_factory=list)

def parse_files(files, supplier):
    ds=Dataset(supplier)
    products={}; sales=[]; transactions=[]; stocks=[]; transit=[]
    def product(sku, **values):
        if not sku or sku.lower() in ('итого','nan'): return
        if sku=='0' and not values.get('name'):return
        p=products.setdefault(sku,dict(sku=sku,article='',name='',category='Не указана',stock=np.nan,pack=1,moq=1,unit='шт'))
        for k,v in values.items():
            if v is not None and not (isinstance(v,float) and np.isnan(v)): p[k]=v
    for filename, content in files:
        try:
            frame=pd.read_excel(BytesIO(content),header=None,dtype=object)
            if frame.empty: raise ValueError('Пустой лист')
            rows=frame.values.tolist(); first=[str(x).strip() for x in rows[0]]
            detected='Не распознан'
            if {'Дата','Документ','Количество'}.issubset(first):
                detected='Детальные продажи'; ix={v:i for i,v in enumerate(first)}
                dates=pd.to_datetime(frame.iloc[1:,ix['Дата']],format='mixed',dayfirst=True,errors='coerce')
                for row,dt in zip(rows[1:],dates):
                    sku=key(row[ix['Код']])
                    if not sku or pd.isna(dt): continue
                    product(sku,name=str(row[ix['Номенклатура']]),unit=key(row[ix['Ед.']]))
                    transactions.append(dict(sku=sku,date=dt,quantity=number(row[ix['Количество']]),document=key(row[ix['Номер']]),client_id=key(row[ix['client_id']]) if 'client_id' in ix else ''))
            elif any('Свободный остаток' in str(x) for row in rows[:4] for x in row):
                detected='Остаток + категории + продажи + путь'
                hi=next(i for i,row in enumerate(rows[:4]) if 'Свободный остаток' in [str(x).strip() for x in row])
                headers=[str(x).strip() for x in rows[hi]]; ix={x:i for i,x in enumerate(headers)}
                mc=[(i,month(x)) for i,x in enumerate(headers) if month(x)]
                tc=[i for i,x in enumerate(headers) if 'в пути' in x.lower()]
                for row in rows[hi+1:]:
                    sku=key(row[ix['Код 1с']]);
                    if not sku: continue
                    product(sku,article=key(row[ix['Артикул поставщика']]),name=key(row[ix['Наименование']]),category=key(row[ix['Категория 2026']]),stock=number(row[ix['Свободный остаток']],np.nan),source_growth=number(row[ix['Кэф. Роста']],np.nan) if 'Кэф. Роста' in ix else None)
                    # This summary is a fallback only; the dedicated monthly report wins.
                    for ci,dt in mc: sales.append(dict(sku=sku,date=dt,quantity=number(row[ci]),priority=0))
                    for ci in tc:
                        match=re.search(r'(\d{2})\.(\d{2})',headers[ci]); eta=pd.Timestamp(2026,int(match[2]),int(match[1])) if match else pd.NaT
                        transit.append(dict(sku=sku,eta=eta,quantity=number(row[ci])))
            elif any('поступление до' in x.lower() for x in first):
                detected='Товары в пути'; ci=first.index('Код 1с')
                for row in rows[1:]:
                    sku=key(row[ci]); product(sku,article=key(row[1]),name=key(row[2]))
                    for j,h in enumerate(first):
                        match=re.search(r'поступление до (\d{2}\.\d{2}\.\d{4})',h)
                        if match: transit.append(dict(sku=sku,eta=pd.to_datetime(match[1],dayfirst=True),quantity=number(row[j])))
            elif any(month(x) for x in first):
                is_stock='остат' in str(rows[2]).lower() or ('Ед.изм' in first)
                detected='Месячные остатки' if is_stock else 'Месячные продажи'
                code_col=next(i for i,x in enumerate(first) if x in ('Номенклатура.Код','Код 1с'))
                name_col=first.index('Номенклатура')
                month_cols=[(i,month(x)) for i,x in enumerate(first) if month(x)]
                for row in rows[1:]:
                    sku=key(row[code_col])
                    if not sku or key(row[name_col]).lower()=='итого':continue
                    product(sku,name=key(row[name_col]))
                    if 'Артикул' in first:product(sku,article=key(row[first.index('Артикул')]))
                    for j,dt in month_cols:
                        if is_stock:
                            # Blank stock is unknown, never evidence of an empty warehouse.
                            stocks.append(dict(sku=sku,date=dt,quantity=number(row[j],np.nan)))
                        else:sales.append(dict(sku=sku,date=dt,quantity=number(row[j]),priority=1))
            elif 'Мин. разр. к отгр.' in first or 'Кратность' in first:
                detected='Минимальная партия / кратность'
                code_col=next(i for i,x in enumerate(first) if x in ('Код 1с','Номенклатура.Код'))
                article_col=next(i for i,x in enumerate(first) if x in ('Артикул','Артикул поставщика'))
                value_col=first.index('Кратность') if 'Кратность' in first else first.index('Мин. разр. к отгр.')
                name_col=first.index('Номенклатура') if 'Номенклатура' in first else first.index('Наименование')
                for row in rows[1:]:
                    sku=key(row[code_col]); n=number(row[value_col],1)
                    if not key(row[name_col]):continue # Ignore trailing spreadsheet notes, not products.
                    product(sku,article=key(row[article_col]),name=key(row[name_col]),**({'pack':max(1,n)} if 'Кратность' in first else {'moq':max(1,n)}))
            elif any('СЕЗОННОСТЬ' in str(x).upper() for row in rows for x in row):
                detected='Сезонность поставщика'
                for row in rows:
                    if len(row)>11:
                        name=str(row[1]).lower().strip()
                        if name[:3] in MONTHS:
                            value=number(row[11],np.nan)
                            if np.isfinite(value) and value>0: ds.seasonal[MONTHS[name[:3]]]=value
                if not ds.seasonal:ds.warnings.append('В файле сезонности не найден готовый положительный коэффициент: сезонность будет оценена по продажам.')
            ds.sources.append(dict(Файл=filename,Тип=detected,Строк=len(frame),SHA256=hashlib.sha256(content).hexdigest()))
            if detected=='Не распознан':ds.warnings.append(f'Не распознан файл: {filename}')
        except Exception as exc:ds.warnings.append(f'{filename}: {exc}')
    ds.products=pd.DataFrame(products.values()) if products else ds.products
    if 'source_growth' not in ds.products:ds.products['source_growth']=np.nan
    if sales:
        ds.sales=pd.DataFrame(sales).sort_values('priority').drop_duplicates(['sku','date'],keep='last').drop(columns='priority')
    if transactions:ds.transactions=pd.DataFrame(transactions)
    if stocks:ds.stocks=pd.DataFrame(stocks)
    if transit:ds.transit=pd.DataFrame(transit)
    if not ds.transactions.empty and not ds.transactions.client_id.astype(bool).any():ds.warnings.append('ID клиента отсутствует. Выбросы определяются по накладным; объединение покупок одного клиента недоступно.')
    if not ds.products.empty:
        missing=int(ds.products.stock.isna().sum())
        if missing:ds.warnings.append(f'Для {missing} позиций нет актуального свободного остатка. Расчёт заказа для них заблокирован до ввода остатка.')
    ds.warnings.append('Месячные снимки остатков не определяют точные дни stockout. Приближение применяется только по явному выбору пользователя.')
    return ds

def demo_data():
    ds=Dataset('Демо · синтетические данные')
    names=['Розетка ATLAS','Выключатель BRITE','Кабель силовой','Автомат защиты','Светильник LED','Коробка монтажная']
    ds.products=pd.DataFrame([dict(sku=f'DEMO-{i+1:03}',article=f'ART-{i+1:03}',name=n,category=str(i%3+1),stock=[10,np.nan,12,0,np.nan,15][i],pack=[10,5,1,12,1,10][i],moq=1,unit='м' if i==2 else 'шт') for i,n in enumerate(names)])
    rng=np.random.default_rng(42); tx=[]
    for i,p in ds.products.iterrows():
        for day in pd.date_range('2024-01-01','2026-09-22'):
            if i==3 and pd.Timestamp('2026-07-01')<=day<=pd.Timestamp('2026-07-20'):continue
            season=1+.35*np.sin((day.month-3)/12*2*np.pi)
            growth=1+.12*(day.year-2024)
            qty=rng.poisson((i+2)*.55*season*growth)
            if qty:tx.append(dict(sku=p.sku,date=day,quantity=qty,document=f'{i}-{day.date()}',client_id=f'client-{day.day%9}'))
    tx.append(dict(sku='DEMO-001',date=pd.Timestamp('2026-08-15'),quantity=1500,document='ONE-OFF',client_id='project-client'))
    ds.transactions=pd.DataFrame(tx)
    ds.sales=ds.transactions.assign(date=ds.transactions.date.dt.to_period('M').dt.to_timestamp()).groupby(['sku','date'],as_index=False).quantity.sum()
    ds.transit=pd.DataFrame([dict(sku='DEMO-001',eta=pd.Timestamp('2026-09-28'),quantity=30),dict(sku='DEMO-003',eta=pd.Timestamp('2026-12-01'),quantity=500)])
    ds.stocks=pd.DataFrame([dict(sku='DEMO-002',date=pd.Timestamp('2026-09-01'),quantity=120.)])
    ds.stockouts=pd.DataFrame([dict(sku='DEMO-004',start=pd.Timestamp('2026-07-01'),end=pd.Timestamp('2026-07-20'))])
    ds.products.loc[ds.products.sku.eq('DEMO-003'),['unit','pack','moq']]=['м',50,50]
    ds.sources=[dict(Файл='Встроенный пример',Тип='Синтетический набор, не данные компании',Строк=len(tx))]
    return ds
