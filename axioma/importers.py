"""Read original HackAlem supplier ZIPs or a documented canonical CSV bundle."""
from io import BytesIO
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
from .model import Dataset


def _text(value):
    return '' if pd.isna(value) else str(value).strip()


def _number(value, default=np.nan):
    if value is None or pd.isna(value) or str(value).strip() == '':
        return default
    return pd.to_numeric(str(value).replace('\xa0', '').replace(' ', '').replace(',', '.'), errors='coerce')


def _zip_name(entry):
    if entry.flag_bits & 0x800:
        return entry.filename
    try:
        return entry.filename.encode('cp437').decode('cp866')
    except (UnicodeError, LookupError):
        return entry.filename


def _rows(raw):
    # Reading only sheet 0 avoids duplicated seasonality summaries.
    return pd.read_excel(BytesIO(raw), sheet_name=0, header=None, dtype=object)


def _month(value):
    names = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
    text = _text(value).lower()
    match = re.search(r'20\d{2}', text)
    if not match:
        return None
    for index, name in enumerate(names, start=1):
        if text.startswith(name):
            return f'{match.group(0)}-{index:02}'
    return None


def import_supplier_zips(files, as_of='2026-09-22'):
    products, sales_parts, arrivals, warnings = {}, [], [], []
    checks, monthly_totals, monthly_records = [], [], []
    as_of = pd.Timestamp(as_of)
    for filename, content in files:
        supplier = 'Systeme Electric' if 'system' in filename.lower() else 'IEK'
        with zipfile.ZipFile(BytesIO(content)) as archive:
            entries = [e for e in archive.infolist() if e.filename.lower().endswith('.xlsx')]
            if sum(e.file_size for e in entries) > 300_000_000:
                raise ValueError('Архив слишком большой: лимит распакованных Excel 300 МБ.')
            def ensure(sku, name='', unit='шт'):
                key = supplier, _text(sku)
                if not key[1] or key[1].lower() in ['итого', 'nan']:
                    return None
                if key not in products:
                    products[key] = dict(supplier=supplier, sku=key[1], article=key[1], name=_text(name), unit=_text(unit) or 'шт',
                                         purchase_unit=_text(unit) or 'шт', category='unknown', stock=np.nan, stock_date='',
                                         lead_days=14, review_days=30, moq=0, pack=1, purchase_factor=1,
                                         growth_pct=np.nan, assumptions='Допущения: срок 14 дн., пересмотр 30 дн.; проверьте единицу закупки и условия поставщика.')
                return products[key]
            # Metadata before sales: explicit unit from transaction data updates it later.
            for entry in entries:
                name = _zip_name(entry).lower()
                frame = _rows(archive.read(entry))
                if 'динамика продаж' in name:
                    frame.columns = [_text(x) for x in frame.iloc[0]]
                    body = frame.iloc[1:].copy()
                    required = ['Дата', 'Код', 'Количество']
                    if not all(c in body for c in required):
                        raise ValueError(f'{name}: нет обязательных столбцов {required}')
                    normalized = pd.DataFrame({'supplier': supplier, 'sku': body['Код'].map(_text),
                                               'date': pd.to_datetime(body['Дата'], dayfirst=True, errors='coerce'),
                                               'qty': body['Количество'].map(_number),
                                               'document': body['Номер'].map(_text), 'client_id': '',
                                               'warehouse': body['Склад'].map(_text)})
                    invalid = normalized.date.isna() | normalized.qty.isna() | normalized.sku.eq('')
                    if invalid.any():
                        warnings.append(f'{supplier}: пропущено некорректных строк продаж: {int(invalid.sum())}.')
                    normalized = normalized.loc[~invalid]
                    sales_parts.append(normalized)
                    for row in body.drop_duplicates('Код').to_dict('records'):
                        p = ensure(row['Код'], row.get('Номенклатура', ''), row.get('Ед.', 'шт'))
                        if p:
                            p['unit'] = _text(row.get('Ед.', p['unit'])) or p['unit']
                            p['purchase_unit'] = p['unit']
                            if not p['name']:
                                p['name'] = _text(row.get('Номенклатура', ''))
                elif 'moq' in name:
                    headers = [_text(v) for v in frame.iloc[0]]
                    frame.columns = headers
                    code_col = 'Код 1с' if 'Код 1с' in headers else 'Номенклатура.Код'
                    article_col = 'Артикул поставщика' if 'Артикул поставщика' in headers else 'Артикул'
                    for row in frame.iloc[1:].to_dict('records'):
                        p = ensure(row.get(code_col), row.get('Наименование', row.get('Номенклатура', '')))
                        if not p:
                            continue
                        p['article'] = _text(row.get(article_col)) or p['article']
                        if supplier == 'IEK':
                            p['moq'] = max(0, _number(row.get('Мин. разр. к отгр.'), 0))
                        else:
                            p['pack'] = max(1, _number(row.get('Кратность'), 1))
                elif supplier == 'IEK' and 'путь' in name:
                    headers = list(frame.iloc[0])
                    for values in frame.iloc[1:].itertuples(index=False, name=None):
                        p = ensure(values[0], values[2])
                        if not p:
                            continue
                        p['article'] = _text(values[1]) or p['article']
                        if 'БУХТАМИ' in _text(values[2]).upper():
                            p['assumptions'] += ' Кабель: требуется подтверждение перевода метров в бухты.'
                            p['unit_conversion_required'] = True
                        for idx, header in enumerate(headers[3:], start=3):
                            qty = _number(values[idx], 0)
                            if not qty:
                                continue
                            match = re.search(r'поступление до\s*(\d{2}\.\d{2}\.\d{4})', str(header))
                            if match:
                                arrivals.append(dict(supplier=supplier, sku=p['sku'], eta=pd.to_datetime(match.group(1), dayfirst=True), qty=qty))
                            else:
                                warnings.append(f'IEK {p["sku"]}: поступление без даты не учтено.')
                elif supplier == 'Systeme Electric' and 'товар в пути' in name:
                    headers = [_text(v) for v in frame.iloc[1]]
                    # Select by position because this export contains many unnamed columns.
                    columns = {h: i for i, h in enumerate(headers) if h}
                    for values in frame.iloc[2:].itertuples(index=False, name=None):
                        def get(col, default=None):
                            return values[columns[col]] if col in columns else default
                        p = ensure(get('Код 1с'), get('Наименование'))
                        if not p:
                            continue
                        p['article'] = _text(get('Артикул поставщика')) or p['article']
                        p['stock'] = _number(get('Свободный остаток'))
                        p['stock_date'] = '2026-09-22'
                        p['category'] = _text(get('Категория 2026')) or 'unknown'
                        g = _number(get('Кэф. Роста'))
                        if pd.notna(g):
                            if np.isfinite(g) and g >= -1:
                                p['growth_pct'] = g * 100
                                p['assumptions'] += ' Кэф. Роста принят как доля прироста; требуется подтверждение методики.'
                            else:
                                p['assumptions'] += ' Некорректный Кэф. Роста (<−100% или бесконечность); вместо него используется автоматический тренд.'
                                warnings.append(f'{supplier} {p["sku"]}: недопустимый коэффициент роста; включён автоматический тренд, требуется проверка.')
                        for header, idx in columns.items():
                            if 'в пути' in header.lower():
                                qty = _number(values[idx], 0)
                                match = re.search(r'(\d{2})\.(\d{2})', header)
                                if qty and match:
                                    arrivals.append(dict(supplier=supplier, sku=p['sku'], qty=qty,
                                                         eta=pd.Timestamp(2026, int(match.group(2)), int(match.group(1)))))
                elif 'ежемесячные остатки' in name:
                    valid_rows = frame.iloc[1:][frame.iloc[1:, 2].notna()]
                    month_cols = [i for i, value in enumerate(frame.iloc[0]) if _month(value)]
                    observed = int(valid_rows.iloc[:, month_cols].notna().sum().sum())
                    checks.append(dict(supplier=supplier, source='Месячные остатки', rows=len(valid_rows),
                                       check=f'{len(month_cols)} месяцев, {observed} непустых значений; контроль покрытия, не текущий остаток.'))
                    warnings.append(f'{supplier}: месячные остатки прочитаны как исторический источник; они не заменяют текущий остаток и точные stockout.')
                elif 'ежемесячные продажи' in name:
                    valid_rows = frame.iloc[1:][frame.iloc[1:, 1].notna()]
                    for idx, header in enumerate(frame.iloc[0]):
                        month = _month(header)
                        if month:
                            total = pd.to_numeric(valid_rows.iloc[:, idx], errors='coerce').sum()
                            monthly_totals.append((supplier, month, float(total)))
                            for code, amount in zip(valid_rows.iloc[:, 1], valid_rows.iloc[:, idx]):
                                number = _number(amount)
                                if pd.notna(number) and _text(code):
                                    monthly_records.append(dict(supplier=supplier, sku=_text(code), month=month + '-01', qty=float(number)))
                    checks.append(dict(supplier=supplier, source='Месячные продажи', rows=len(valid_rows),
                                       check='Сверка помесячных сумм с детализацией; повторное сложение исключено.'))
                    warnings.append(f'{supplier}: ранняя месячная история дополняет сезонность до начала детализации; совпадающие периоды не суммируются.')
                elif 'сезонность' in name:
                    years = frame.iloc[:, 0].map(lambda x: str(x).strip()).isin(['2024', '2025', '2026'])
                    checks.append(dict(supplier=supplier, source='Сезонность (денежная)', rows=int(years.sum()),
                                       check='Проверено покрытие годовых строк. Денежные суммы не переводятся в штуки.'))
                    warnings.append(f'{supplier}: сводная сезонность в денежных суммах не подменяет сезонность количества; коэффициенты оцениваются по транзакциям.')
    if not sales_parts or not products:
        raise ValueError('Не найдены поддерживаемые данные поставщиков.')
    product_frame = pd.DataFrame(products.values())
    product_frame['unit_conversion_required'] = product_frame.get('unit_conversion_required', pd.Series(False, index=product_frame.index)).fillna(False)
    sales_frame = pd.concat(sales_parts, ignore_index=True)
    grouped = sales_frame.groupby([sales_frame.supplier, sales_frame.date.dt.strftime('%Y-%m')]).qty.sum()
    for supplier_name in sales_frame.supplier.unique():
        compared = [(m, q, float(grouped.get((s, m), 0))) for s, m, q in monthly_totals if s == supplier_name]
        mismatches = [(m, q, actual) for m, q, actual in compared if abs(q - actual) > max(.01, abs(q) * .001)]
        checks.append(dict(supplier=supplier_name, source='Сверка продаж', rows=len(compared),
                           check=f'Расхождения >0.1%: {len(mismatches)} из {len(compared)} месяцев. Детализация остаётся расчётным источником.'))
        if mismatches:
            warnings.append(f'{supplier_name}: месячная сводка и детализация расходятся в {len(mismatches)} месяцах; проверьте полноту/даты выгрузок.')
    sale_keys = set(zip(sales_frame.supplier, sales_frame.sku))
    warnings += [
        'ID клиента отсутствует: выбросы определяются по документам. Клиентский сценарий проверяется на синтетике.',
        'Возвраты (отрицательные количества) сохраняются в истории, но не считаются положительным спросом.',
        'Расчёт агрегирован по складам. Для поскладового заказа нужны остатки и поступления по каждому складу.',
        'Нет точных периодов stockout. Загрузите подтверждённые интервалы отдельно.',
        'Категории и сроки без исходных значений являются настраиваемыми допущениями.',
    ]
    inactive = sum((r.supplier, r.sku) not in sale_keys for r in product_frame.itertuples())
    warnings.append(f'Товаров без транзакционной истории: {inactive}; рекомендации для них блокируются.')
    return Dataset(product_frame, sales_frame, pd.DataFrame(arrivals, columns=['supplier', 'sku', 'eta', 'qty']),
                   warnings=warnings, source='Реальные выгрузки партнёра', source_checks=pd.DataFrame(checks),
                   monthly_sales=pd.DataFrame(monthly_records, columns=['supplier', 'sku', 'month', 'qty'])).validate()


def read_canonical(files):
    """files maps products.csv/sales.csv/[incoming.csv,stockouts.csv] to bytes."""
    tables = {}
    for name, content in files.items():
        name = Path(name).name.lower()
        if name in ['products.csv', 'sales.csv', 'incoming.csv', 'stockouts.csv', 'monthly_sales.csv']:
            tables[name] = pd.read_csv(BytesIO(content), dtype={'sku': str, 'supplier': str, 'article': str, 'category': str})
    if not {'products.csv', 'sales.csv'} <= set(tables):
        raise ValueError('Требуются products.csv и sales.csv.')
    kwargs = {key: tables[key + '.csv'] for key in ['incoming', 'stockouts', 'monthly_sales'] if key + '.csv' in tables}
    return Dataset(tables['products.csv'], tables['sales.csv'], **kwargs, source='Загруженные CSV').validate()


def csv_bundle(dataset):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['products', 'sales', 'incoming', 'stockouts', 'monthly_sales']:
            archive.writestr(name + '.csv', getattr(dataset, name).to_csv(index=False).encode('utf-8-sig'))
    return buffer.getvalue()
