from copy import deepcopy
from datetime import date
from io import BytesIO
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from axioma.demo import make_demo
from axioma.engine import calculate, Policy
from axioma.importers import import_supplier_zips, read_canonical, csv_bundle
from axioma.export import approve_orders, export_csv, export_xlsx, state_fingerprint

CACHE_VERSION = 5

st.set_page_config(page_title='AXIOMA · Закупки', page_icon='◈', layout='wide')
st.markdown('''<style>
.block-container{max-width:1500px;padding-top:2rem} h1{letter-spacing:-1.6px}
[data-testid="stMetric"]{background:white;border:1px solid #e0e9e4;border-radius:14px;padding:18px}
[data-testid="stSidebar"]{border-right:1px solid #e0e9e4}
.eyebrow{font-size:12px;letter-spacing:3px;font-weight:700;color:#138c77}
.lead{color:#647970;font-size:17px;max-width:850px;margin-bottom:22px}
</style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def demo_data(schema_version=CACHE_VERSION):
    return make_demo()


@st.cache_data(show_spinner='Читаем Excel и проверяем структуру…')
def import_data(payload, kind, as_of, schema_version=CACHE_VERSION):
    return import_supplier_zips(payload, as_of) if kind == 'Архивы поставщиков' else read_canonical(dict(payload))


@st.cache_data(show_spinner='Оцениваем спрос и формируем рекомендации…', max_entries=4)
def run_calculation(dataset, policy, schema_version=CACHE_VERSION):
    return calculate(dataset, policy)


with st.sidebar:
    st.markdown('## ◈ AXIOMA')
    st.caption('ПОПОЛНЕНИЕ СКЛАДА')
    mode = st.radio('Источник данных', ['Демонстрация', 'Архивы поставщиков', 'CSV по шаблону'])
    as_of = st.date_input('Дата расчёта', date(2026, 9, 22))
    uploads = []
    if mode == 'Архивы поставщиков':
        uploads = st.file_uploader('IEK.zip и Systeme electric.zip', type=['zip'], accept_multiple_files=True)
    elif mode == 'CSV по шаблону':
        uploads = st.file_uploader('products.csv, sales.csv и дополнительные таблицы', type=['csv'], accept_multiple_files=True)
    st.divider()
    st.caption('Все вычисления выполняются локально. Заказ поставщику автоматически не отправляется.')
    st.download_button('Скачать синтетический пример', csv_bundle(demo_data(CACHE_VERSION)), 'axioma-demo.zip', mime='application/zip')

st.markdown('<div class="eyebrow">AXIOMA / SUPPLY INTELLIGENCE</div>', unsafe_allow_html=True)
st.title('Закупки с понятным основанием')
st.markdown('<div class="lead">От истории продаж к решению: сколько заказать, когда возникнет дефицит и какие данные повлияли на расчёт.</div>', unsafe_allow_html=True)
if mode != 'Демонстрация' and not uploads:
    st.info('Загрузите исходные файлы слева. Для знакомства с приложением доступен демонстрационный режим.')
    st.stop()
try:
    dataset = deepcopy(demo_data(CACHE_VERSION) if mode == 'Демонстрация' else import_data(tuple((u.name, u.getvalue()) for u in uploads), mode, str(as_of), CACHE_VERSION))
except Exception as exc:
    st.error(f'Не удалось прочитать данные: {exc}')
    st.stop()

st.caption(f'{dataset.source} · {len(dataset.products):,} товаров · {len(dataset.sales):,} строк продаж · расчёт на {as_of:%d.%m.%Y}')
data_key = state_fingerprint(dataset.products, {'mode': mode, 'date': str(as_of)})[:12]
if mode == 'Демонстрация':
    st.info('Синтетический набор: сезонный товар, устойчивый рост, крупная разовая покупка, stockout, кабель в бухтах и избыточный запас.')
else:
    st.warning('Реальные данные агрегированы по складам. Перед утверждением проверьте актуальность остатков, сроков и единиц закупки.')

with st.expander('Данные и параметры расчёта', expanded=False):
    st.write('Пустой остаток блокирует рекомендацию. Сроки и страховой запас — открытые допущения, которые можно изменить.')
    editable = ['supplier', 'sku', 'name', 'category', 'stock', 'lead_days', 'review_days', 'growth_pct', 'moq', 'pack', 'purchase_factor', 'purchase_unit']
    if 'unit_conversion_required' in dataset.products:
        editable.append('unit_conversion_required')
        st.caption('Для кабеля заполните коэффициент перевода и единицу закупки, затем снимите «требуется перевод единиц».')
    edited_products = st.data_editor(dataset.products[editable], disabled=['supplier', 'sku', 'name'], hide_index=True,
                                     key=f'products-{data_key}', width='stretch', num_rows='fixed',
                                     column_config={'stock': st.column_config.NumberColumn('Доступный остаток', min_value=0),
                                     'growth_pct': st.column_config.NumberColumn('Прирост, % (заменяет тренд)', min_value=-100),
                                     'lead_days': st.column_config.NumberColumn('Срок поставки, дней', min_value=0, max_value=365, step=1),
                                     'review_days': st.column_config.NumberColumn('Период пополнения, дней', min_value=1, max_value=365, step=1),
                                     'purchase_factor': st.column_config.NumberColumn('Складских единиц в закупочной', min_value=.001),
                                     'unit_conversion_required': st.column_config.CheckboxColumn('Требуется перевод единиц')})
    for col in editable:
        dataset.products[col] = edited_products[col].values
    defaults = {'1': 14, '2': 10, '3': 7, 'A': 14, 'B': 10, 'C': 7, 'unknown': 7}
    cat_frame = pd.DataFrame([{'category': c, 'safety_days': defaults.get(c, 7)} for c in sorted(set(dataset.products.category.astype(str)))])
    cat_key = state_fingerprint(cat_frame, {'data': data_key})[:12]
    cat_edited = st.data_editor(cat_frame, disabled=['category'], hide_index=True, key=f'cats-{cat_key}',
                               column_config={'safety_days': st.column_config.NumberColumn('Страховой запас, дней', min_value=0, max_value=180)})
    c1, c2 = st.columns(2)
    remove = c1.checkbox('Исключать разовые крупные продажи', value=True)
    restore = c2.checkbox('Компенсировать подтверждённый stockout', value=True)
    extra = st.file_uploader('Дополнить stockouts.csv или incoming.csv', type=['csv'], accept_multiple_files=True)
    try:
        for upload in extra:
            if upload.name not in ['stockouts.csv', 'incoming.csv']:
                raise ValueError('Допустимы только stockouts.csv и incoming.csv')
            frame = pd.read_csv(BytesIO(upload.getvalue()), dtype={'supplier': str, 'sku': str})
            setattr(dataset, upload.name.split('.')[0], frame)
    except Exception as exc:
        st.error(f'Дополнительная таблица: {exc}')
        st.stop()
    st.caption('Дополнительная таблица полностью заменяет соответствующий источник, не суммируется с ним.')

policy = Policy(str(as_of), dict(zip(cat_edited.category.astype(str), cat_edited.safety_days)), remove, restore)
try:
    results, details = run_calculation(dataset, policy, CACHE_VERSION)
except Exception as exc:
    st.error(f'Проверьте данные: {exc}')
    st.stop()

metrics = st.columns(4)
metrics[0].metric('Товаров к заказу', int((results.recommended_qty > 0).sum()))
metrics[1].metric('Риск до поставки', int((results.urgency == 'Срочно').sum()))
metrics[2].metric('Требуют уточнения', int(results.recommended_qty.isna().sum()))
metrics[3].metric('Поставщиков', results.supplier.nunique())

orders_tab, analytics_tab, quality_tab = st.tabs(['Рекомендации и заказ', 'Почему столько', 'Качество данных'])
with orders_tab:
    left, right = st.columns([2, 1])
    selected = left.multiselect('Поставщик', results.supplier.unique().tolist(), default=results.supplier.unique().tolist())
    urgency = right.multiselect('Статус', results.urgency.unique().tolist())
    view = results[results.supplier.isin(selected)]
    if urgency:
        view = view[view.urgency.isin(urgency)]
    st.caption('Количество заказа указано в единицах закупки. Строки с неполными данными нельзя утвердить.')
    edit_cols = ['supplier', 'sku', 'name', 'urgency', 'purchase_unit', 'recommended_qty', 'order_qty', 'explanation']
    order_frame = view.assign(order_qty=view.recommended_qty.fillna(0))[edit_cols]
    fingerprint = state_fingerprint(results, {'date': str(as_of), 'selected': selected, 'urgency': urgency})
    edited = st.data_editor(order_frame, hide_index=True, width='stretch', key=f'orders-{fingerprint[:12]}',
                           disabled=[c for c in edit_cols if c != 'order_qty'],
                           column_config={'order_qty': st.column_config.NumberColumn('Заказать', min_value=0),
                                          'recommended_qty': st.column_config.NumberColumn('Рекомендация'),
                                          'explanation': st.column_config.TextColumn('Обоснование', width='large'),
                                          'purchase_unit': 'Ед. закупки', 'urgency': 'Статус', 'supplier': 'Поставщик', 'name': 'Товар'})
    reviewer = st.text_input('Ответственный — имя или инициалы', placeholder='Например, менеджер А.К.')
    current_signature = state_fingerprint(edited, {'calculation': fingerprint, 'reviewer': reviewer})
    if st.button('Утвердить выбранные позиции', type='primary', disabled=edited.empty):
        try:
            approved = approve_orders(view, edited, reviewer, as_of)
            if approved.empty:
                st.warning('Нет положительных количеств для заказа.')
            else:
                st.session_state['approved'] = (current_signature, approved)
                st.success('Заказ утверждён. Скачайте файл для передачи в учётную систему.')
        except ValueError as exc:
            st.error(str(exc))
    approval = st.session_state.get('approved')
    if approval and approval[0] == current_signature:
        a, b = st.columns(2)
        a.download_button('Скачать утверждённый CSV', export_csv(approval[1]), 'axioma-approved.csv', 'text/csv')
        b.download_button('Скачать утверждённый Excel', export_xlsx(approval[1]), 'axioma-approved.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.caption('CSV/XLSX — универсальный формат. Прямой импорт в конкретную конфигурацию 1С требует согласования схемы.')
    elif approval:
        st.caption('После изменения данных, количества, фильтра или ответственного требуется повторное утверждение.')

with analytics_tab:
    options = [(r.supplier, r.sku) for r in results.itertuples()]
    chosen = st.selectbox('Товар для разбора', options, format_func=lambda key: f'{key[0]} · {key[1]} · {results.loc[(results.supplier == key[0]) & (results.sku == key[1]), "name"].iloc[0]}')
    row = results[(results.supplier == chosen[0]) & (results.sku == chosen[1])].iloc[0]
    st.write(row.explanation)
    st.caption(row.method)
    detail = details[chosen]
    history = detail['history']
    if not history.empty:
        chart = history[['raw', 'restored']].resample('W').sum().rename(columns={'raw': 'Фактические продажи', 'restored': 'Регулярный + восстановленный спрос'})
        chart['Прогноз'] = detail['forecast'].resample('W').sum()
        chart = chart.combine_first(pd.DataFrame({'Прогноз': detail['forecast'].resample('W').sum()}))
        st.line_chart(chart, color=['#b5c5bf', '#138c77', '#d38b25'])
        st.caption('Недельные суммы. Неполные крайние недели содержат меньше дней.')
        with st.expander('Исключённые разовые операции'):
            st.dataframe(detail['events'].loc[detail['events'].outlier], hide_index=True)
        with st.expander('Дневная история и восстановление'):
            st.dataframe(history.tail(120))
        if not detail['monthly_prior'].empty:
            with st.expander('Дополнительная месячная история'):
                st.caption('Для сезонности используются только месяцы до начала дневной детализации. На этих агрегатах невозможно проверить клиента или отдельную разовую операцию.')
                st.dataframe(detail['monthly_prior'], hide_index=True)
    else:
        st.info('Для этого товара нет истории продаж.')

with quality_tab:
    if not dataset.source_checks.empty:
        st.markdown('**Проверка дополнительных источников**')
        st.dataframe(dataset.source_checks, hide_index=True, width='stretch')
    for warning in dataset.warnings:
        st.warning(warning)
    st.markdown('**Что известно и что является допущением**')
    st.dataframe(dataset.products[['supplier', 'sku', 'stock', 'category', 'assumptions']], hide_index=True)
    st.caption('Нулевые продажи не доказывают отсутствие товара. Без подтверждённых интервалов потерянный спрос не добавляется.')
    st.caption('Система не измеряет фактическую экономию и не обещает её: эффект требует проверки на реальных закупках.')
