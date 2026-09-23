from io import BytesIO
import zipfile
import pandas as pd
from openpyxl import Workbook
from axioma.importers import import_supplier_zips


def workbook(rows):
    stream = BytesIO()
    wb = Workbook()
    for row in rows:
        wb.active.append(row)
    wb.save(stream)
    return stream.getvalue()


def test_original_iek_layout_preserves_sku_returns_eta_and_missing_stock():
    data = BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('IEK/Динамика продаж_2025-2026.xlsx', workbook([
            ['Дата', 'Номер', 'Документ', 'Код', 'Номенклатура', 'Ед.', 'Склад', 'Количество'],
            ['21.09.2026 12:00:00', 'D1', 'Продажа', '0001_', 'Тест', 'шт', 'Склад', 10],
            ['22.09.2026 12:00:00', 'D2', 'Возврат', '0001_', 'Тест', 'шт', 'Склад', -2],
        ]))
        archive.writestr('IEK/MOQ ИЭК.xlsx', workbook([
            ['№', 'Код 1с', 'Артикул поставщика', 'Наименование', 'Мин. разр. к отгр.'],
            [1, '0001_', 'ARTICLE', 'Тест', 5],
        ]))
        archive.writestr('IEK/Путь ИЭК.xlsx', workbook([
            ['Код 1с', 'Артикул ИЭК', 'Наименование', 'ПП (поступление до 01.10.2026)'],
            ['0001_', 'ARTICLE', 'Тест', 20],
        ]))
        archive.writestr('IEK/Ежемесячные продажи.xlsx', workbook([
            ['Номенклатура', 'Номенклатура.Код', 'сент. 2026'],
            [None, None, 'Количество'], ['Тест', '0001_', 8],
        ]))
    ds = import_supplier_zips([('IEK.zip', data.getvalue())])
    assert ds.products.iloc[0].sku == '0001_'
    assert ds.products.iloc[0].moq == 5
    assert ds.products.iloc[0].pack == 1
    assert pd.isna(ds.products.iloc[0].stock)
    assert ds.sales.qty.tolist() == [10, -2]
    assert ds.sales.client_id.eq('').all()
    assert ds.incoming.iloc[0].eta == pd.Timestamp('2026-10-01')
    assert 'Расхождения >0.1%: 0' in ds.source_checks.iloc[-1]['check']


def test_systeme_layout_stock_category_and_pack_are_imported():
    data = BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('Systeme electric/Динамика продаж.xlsx', workbook([
            ['Дата', 'Номер', 'Документ', 'Код', 'Номенклатура', 'Ед.', 'Склад', 'Количество'],
            ['21.09.2026 12:00:00', 'D1', 'Продажа', '001', 'Тест', 'шт', 'Склад', 10],
        ]))
        archive.writestr('Systeme electric/MOQ.xlsx', workbook([
            ['№', 'Номенклатура', 'Номенклатура.Код', 'Артикул', 'Кратность'],
            [1, 'Тест', '001', 'ART', 6],
        ]))
        archive.writestr('Systeme electric/Товар в пути.xlsx', workbook([
            ['СКЛАДЫ', None, None, None, None, None, None],
            ['Код 1с', 'Наименование', 'Артикул поставщика', 'Свободный остаток', 'Категория 2026', 'Кэф. Роста', 'СЭ в пути 24.09'],
            ['001', 'Тест', 'ART', 55, '2', .15, 40],
        ]))
    ds = import_supplier_zips([('Systeme electric.zip', data.getvalue())])
    product = ds.products.iloc[0]
    assert product.stock == 55
    assert product.category == '2'
    assert product.pack == 6
    assert product.growth_pct == 15
    assert ds.incoming.iloc[0].qty == 40

