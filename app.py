from copy import deepcopy
from dataclasses import replace
from datetime import date
from io import BytesIO
from pathlib import Path
import os
import hashlib
import json
import numpy as np
import pandas as pd
import streamlit as st
from src.data import demo_data, parse_files, unpack_excel_archive
from src.engine import Settings, calculate
from src.scenarios import stock_scenario, annotate_scenario
from src.orders import add_reviewed_lines
from src.robustness import ScenarioSettings, analyse_scenarios, recount_queue, INSUFFICIENT
from src.planning import receipt_schedule

st.set_page_config(page_title='Axioma · Закупки',page_icon='◈',layout='wide')
st.markdown('''<style>
.block-container{padding-top:4rem;max-width:1450px} h1{letter-spacing:-1.5px}
[data-testid="stMetric"]{background:white;border:1px solid #e3e8ef;border-radius:14px;padding:18px}
[data-testid="stSidebar"]{border-right:1px solid #e3e8ef}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:3px;color:#0d9488}
.intro{font-size:18px;color:#61718a;max-width:850px;margin-bottom:24px}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_demo(version):return demo_data()
@st.cache_data(show_spinner=False)
def load_files(files,supplier):return parse_files(files,supplier)
CALCULATION_VERSION=hashlib.sha256(b''.join((Path(__file__).parent/'src'/name).read_bytes() for name in ['engine.py','forecasting.py','pooled.py','planning.py','robustness.py','data.py','effect.py'])).hexdigest()
@st.cache_data(show_spinner=False)
def cached_run(ds,cfg,version):return calculate(ds,cfg)
def run(ds,cfg):return cached_run(ds,cfg,CALCULATION_VERSION)
@st.cache_data(show_spinner=False)
def cached_scenarios(ds,cfg,result,histories,options,version):
    return analyse_scenarios(ds,cfg,result,histories,options)

LABELS={'stock_threshold':'Порог пополнения','stock_basis':'Основание остатка','snapshot_date':'Дата снимка','snapshot_stock':'Остаток в снимке','recorded_sales':'Продажи после снимка','scenario_stock':'Сценарный остаток','snapshot_age_days':'Возраст снимка, дней','sku':'Код 1С','article':'Артикул','name':'Товар','category':'Категория','supplier':'Поставщик','unit':'Ед.','stock':'Свободный остаток','in_transit':'Приедет в период','late_transit':'Приедет позже','forecast':'Прогноз спроса','safety':'Страховой запас','daily':'Спрос в день','growth':'Тренд, %','season':'Сезонный коэффициент','excluded':'Исключено всплесков','lost':'Восстановлено спроса','pack':'Кратность','moq':'Минимальная партия','recommended':'Рекомендация','status':'Статус','reason':'Обоснование','stockout_mode':'Оценка отсутствия'}
LABELS.update(method='Выбранный метод',demand_type='Характер спроса',cv_months='Месяцев внутренней проверки',cv_wape='Внутренняя ошибка, %',stress_recommended='Заказ в стресс-сценарии',first_shortage='Первый риск дефицита',source_growth_percent='Рост из сводки, %',quantity='К заказу',calculated_at='Дата расчёта',planning_method='Метод расчёта')
LABELS.update(bridge_need='Потребность до позднего поступления',expedite_need='Не хватает до новой поставки',order_arrival='Приход нового заказа',reviewer='Проверил',calculation_id='Версия расчёта',planning_parameters='Параметры расчёта')
LABELS.update(source_growth='Прирост поставщика, доля',growth_source='Источник прироста',unknown_transit='Путь с неясной датой')

LABELS.update(robustness='Устойчивость решения',order_min='Минимум заказа в сценариях',order_max='Максимум заказа в сценариях',scenario_count='Проверено сценариев',scenario_assumptions='Границы проверенной сетки',scenario_basis='Основания сценарных остатков',scenario_snapshot_date='Дата опорного снимка',scenario_snapshot_age_days='Давность снимка, дней',sensitivity_reason='Причины изменения решения',count_reason='Почему пересчитать',count_priority='Приоритет пересчёта',relative_order_span='Относительный разброс заказа',stock_min='Минимум проверенного остатка',stock_max='Максимум проверенного остатка',stock_origin='Источник уточнения',input_batch='Версия входных файлов',source_files='Источники данных',scenario_order='Заказ в сценарии',scenario_expedite='Дефицит до нового заказа',scenario_shortage='Начало дефицита',demand_multiplier='Множитель спроса',delay_days='Задержка поступлений, дней',stock_assumption='Допущение об остатке',delay_order_before='Заказ: вовремя',delay_order_after='Заказ: задержка 7 дней',delay_order_change='Изменение заказа',delay_expedite_before='Дефицит до нового заказа: вовремя',delay_expedite_after='Дефицит до нового заказа: задержка',delay_new_urgent='Стал срочным',delay_first_shortage='Начало дефицита при задержке')

def export_excel(frame, cfg):
    buffer=BytesIO()
    safe=frame.rename(columns=LABELS).copy()
    # Prevent spreadsheet formula injection through imported product names.
    for col in safe.select_dtypes(include=['object','str']).columns:
        safe[col]=safe[col].map(lambda x:"'"+x if isinstance(x,str) and x.startswith(('=','+','-','@')) else x)
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer:
        safe.to_excel(writer,index=False,sheet_name='Рекомендации')
        parameters=cfg if isinstance(cfg,dict) else vars(cfg)
        pd.DataFrame([{'Параметр':k,'Значение':str(v)} for k,v in parameters.items()]).to_excel(writer,index=False,sheet_name='Параметры')
        ws=writer.sheets['Рекомендации'];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        from openpyxl.styles import Font, PatternFill
        for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='0D9488')
        for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=min(60,max(14,len(str(col[0].value))+3))
    return buffer.getvalue()

with st.sidebar:
    st.markdown('## ◈ AXIOMA')
    st.caption('Помощник менеджера закупок')
    mode=st.radio('Источник данных',['Показать пример','Файлы поставщика'])
    supplier=st.selectbox('Поставщик',['Systeme Electric','IEK']) if mode=='Файлы поставщика' else 'Демо'
    st.divider()
    st.markdown('**Период планирования**')
    as_of=st.date_input('Дата расчёта',date(2026,9,23))
    lead=st.number_input('Срок новой поставки, дней',min_value=1,max_value=365,value=21,help='Допущение. Уточните срок у поставщика.')
    review=st.number_input('До следующего заказа, дней',min_value=1,max_value=90,value=14)
    safety=st.number_input('Страховой запас, дней',min_value=0,max_value=90,value=7)
    growth=st.slider('Дополнительный прогноз прироста, %',-50,100,0,help='Ручной сценарий поверх тренда, оценённого по истории.')
    st.divider()
    method=st.selectbox('Метод расчёта',['Общая ML-модель','Автовыбор по истории','Axioma: настроенная модель','Обычное среднее за 6 месяцев'],help='Общая модель обучается на прошлых месяцах всех загруженных товаров. При менее 200 обучающих примерах применяется статистический автовыбор. Автовыбор сравнивает методы на шести доступных месяцах внутри истории. Для независимой проверки нужен новый период.')
    st.caption('ML не гарантирует лучшего заказа. В «Проверке эффекта» сравните сервис и запас с простым средним; проигрыши показаны отдельно.')
    source_growth=st.checkbox('Применить рост из сводки вместо тренда',False,help='Поле «Кэф. Роста» импортируется как доля изменения: 0,2 означает +20%. Это явный сценарий менеджера; подтвердите смысл коэффициента у владельца данных.')
    remove=st.toggle('Исключать разовые всплески',True)
    seasonal=st.toggle('Учитывать сезонность',True)
    trend=st.toggle('Учитывать устойчивый рост',True)
    compensate=st.toggle('Восстанавливать упущенный спрос',True)
    approx=st.checkbox('Приближение по месячным остаткам',False,help='Только явный нулевой снимок. Это не точные даты отсутствия товара.')
    st.caption('Расчёт выполняется локально. API-ключи и платные подписки не нужны.')

st.markdown('<div class="eyebrow">AXIOMA / INVENTORY INTELLIGENCE</div>',unsafe_allow_html=True)
st.title('Заказы поставщикам')
st.markdown('<div class="intro">Что заказать, что ускорить и что сначала проверить. Неполные данные превращаются в объяснимое решение с явными допущениями.</div>',unsafe_allow_html=True)

if mode=='Показать пример':
    ds=deepcopy(load_demo(CALCULATION_VERSION))
    st.info('Демонстрационный набор: 6 товаров, сезонность, рост, разовая покупка на 1 500 единиц и известный период отсутствия товара. Это синтетические данные.')
else:
    uploaded=st.file_uploader('Загрузите ZIP-архив или все 6 Excel-файлов выбранного поставщика',type=['xlsx','zip'],accept_multiple_files=True)
    local_root=os.environ.get('AXIOMA_DATA_DIR')
    local_files=[]
    if local_root:
        folder=Path(local_root)/('Systeme electric' if supplier=='Systeme Electric' else 'IEK')
        if folder.is_dir():
            if st.button('Открыть предоставленные локальные файлы',type='primary'):st.session_state['local_supplier']=supplier
            if st.session_state.get('local_supplier')==supplier:local_files=[(p.name,p.read_bytes()) for p in sorted(folder.glob('*.xlsx'))]
    inputs=[]
    try:
        for f in uploaded:
            inputs.extend(unpack_excel_archive(f.getvalue()) if f.name.lower().endswith('.zip') else [(f.name,f.getvalue())])
        if len({name for name,_ in inputs})!=len(inputs):raise ValueError('Один файл загружен несколько раз; оставьте архив или отдельные XLSX')
    except Exception as exc:st.error(f'Архив не прочитан: {exc}');st.stop()
    inputs=inputs or local_files
    if not inputs:
        st.markdown('**Начните с примера слева или выберите файлы выше.** После загрузки здесь появятся рекомендации.')
        st.stop()
    with st.spinner('Читаю файлы и связываю товары по коду 1С…'):ds=deepcopy(load_files(inputs,supplier))
    if ds.products.empty:st.error('Не удалось найти товары. Проверьте отчёт об импорте ниже.');st.write(ds.warnings);st.stop()

# Scope manual corrections to the actual input batch, not just supplier name.
batch_payload=ds.products.to_json(date_format='iso')+ds.stocks.to_json(date_format='iso')+ds.sales.to_json(date_format='iso')+ds.transit.to_json(date_format='iso')
if mode=='Файлы поставщика':batch_payload+=''.join(name+hashlib.sha256(raw).hexdigest() for name,raw in inputs)
dataset_id=hashlib.sha256((supplier+batch_payload).encode()).hexdigest()[:16]
count_key=f'counts-{dataset_id}-{as_of}'
ds.products['stock_origin']='Остаток из входных файлов' if mode=='Файлы поставщика' else 'Синтетический остаток демо'

with st.expander('Данные и допущения · проверить перед заказом',expanded=False):
    st.dataframe(pd.DataFrame(ds.sources),hide_index=True,width='stretch')
    for warning in ds.warnings:st.warning(warning)
    st.caption('Для расчёта используются завершённые месяцы. Отрицательные продажи сохраняются как корректировки; отрицательный спрос не прогнозируется. Срок поставки и страховой запас задаются пользователем.')
    missing=ds.products.stock.isna().sum()
    st.markdown(f'**Актуальный остаток не указан: {missing} позиций.** Можно исправить таблицу ниже или загрузить CSV.')
    st.caption('CSV остатков: sku,stock,category,pack,moq. Обязательны sku и stock; коды храните текстом. pack — кратность в единицах учёта. Для бухт переведите кратность в метры.')
    overrides=st.file_uploader('Актуальные остатки и параметры товаров',type=['csv'])
    if overrides:
        try:
            extra=pd.read_csv(overrides,dtype={'sku':str})
            if not {'sku','stock'}.issubset(extra.columns):raise ValueError('Нужны колонки sku и stock')
            if extra.sku.duplicated().any():raise ValueError('Повторяются коды SKU')
            extra=extra.set_index('sku');base=ds.products.set_index('sku')
            for c in ['stock','pack','moq']:
                if c in extra:extra[c]=pd.to_numeric(extra[c],errors='raise')
            if 'stock' in extra and (extra.stock<0).any():raise ValueError('Свободный остаток должен быть неотрицательным')
            if any(c in extra and (extra[c]<=0).any() for c in ['pack','moq']):raise ValueError('Кратность и минимум должны быть положительными')
            base.update(extra[[c for c in ['stock','category','pack','moq'] if c in extra]])
            changed_codes=base.index.intersection(extra.index[extra.stock.notna()])
            base.loc[changed_codes,'stock_origin']=f'CSV остатков: {overrides.name}; дата снимка не задана'
            ds.sources.append({'file':overrides.name,'type':'Ручной CSV остатков','rows':len(changed_codes)})
            ds.products=base.reset_index();st.caption(f'Обновлено {len(base.index.intersection(extra.index))} товаров; не найдено {len(extra.index.difference(base.index))}.')
        except Exception as exc:st.error(f'CSV не применён: {exc}')
    changed=st.data_editor(ds.products.rename(columns=LABELS),disabled=['Код 1С','Артикул','Товар','Ед.','Источник уточнения'],hide_index=True,width='stretch',key=f'products-{dataset_id}',column_config={'Свободный остаток':st.column_config.NumberColumn(min_value=0),'Кратность':st.column_config.NumberColumn(min_value=1),'Минимальная партия':st.column_config.NumberColumn(min_value=1),'Прирост поставщика, доля':st.column_config.NumberColumn(min_value=-.9,max_value=3.,help='0.2 означает +20%. Если задан, заменяет тренд из истории.')})
    changed=changed.rename(columns={v:k for k,v in LABELS.items()})
    edited_stock=~(changed.stock.eq(ds.products.stock)|(changed.stock.isna()&ds.products.stock.isna()))
    changed.loc[edited_stock,'stock_origin']='Ручное изменение остатка в таблице; дата снимка не задана'
    ds.products=changed
    st.caption('CSV отсутствия товара: sku,start,end, даты YYYY-MM-DD. Оба конца периода включены.')
    stockout_file=st.file_uploader('Подтверждённые периоды отсутствия',type=['csv'])
    if stockout_file:
        try:
            so=pd.read_csv(stockout_file,dtype={'sku':str})
            if not {'sku','start','end'}.issubset(so.columns):raise ValueError('Нужны sku,start,end')
            for c in ['start','end']:so[c]=pd.to_datetime(so[c],errors='raise')
            if (so.end<so.start).any() or so[['start','end']].isna().any().any():raise ValueError('Некорректные границы периода')
            ds.stockouts=so
        except Exception as exc:st.error(f'Периоды не применены: {exc}')
    st.markdown('**Правила категорий**')
    st.caption('Множитель страхового запаса. По умолчанию 1 для всех: бизнес-значение кодов категорий не предоставлено. Установите согласованные с менеджером правила.')
    categories=sorted(ds.products.category.astype(str).unique())
    cat_frame=st.data_editor(pd.DataFrame({'Категория':categories,'Множитель':[1.]*len(categories)}),disabled=['Категория'],hide_index=True,key=f'cat-{mode}-{supplier}',column_config={'Множитель':st.column_config.NumberColumn(min_value=0.,max_value=5.)})
    category_factors=dict(zip(cat_frame['Категория'],cat_frame['Множитель']))

# A confirmed physical count is a separate, dated override of imported data.
for code,amount in st.session_state.get(count_key,{}).items():
    ds.products.loc[ds.products.sku.eq(code),['stock','stock_origin']]=[amount,'Ручной пересчёт']
source_ds=deepcopy(ds)
with st.expander('Проверенные допущения · устойчивость решения'):
    st.caption('Конечная сетка, а не доверительный интервал. За её пределами устойчивость не проверена. Для неизвестного остатка опора — датированный снимок и, если есть продажи, снимок минус продажи без учёта других движений.')
    sc1,sc2,sc3=st.columns(3)
    stock_pct=sc1.number_input('Отклонение остатка, ±%',min_value=0.,max_value=100.,value=20.,step=5.)
    demand_pct=sc2.number_input('Отклонение спроса, ±%',min_value=0.,max_value=100.,value=20.,step=5.)
    delay_days=sc3.number_input('Задержка в сетке, дней',min_value=0,max_value=90,value=7)
scenario_options=ScenarioSettings(stock_pct,demand_pct,int(delay_days))
stock_evidence=pd.DataFrame()
if ds.products.stock.isna().any() and not ds.stocks.empty:
    st.markdown('**Расчёт при неполных данных об остатках**')
    scenario=st.radio('Как рассчитать неизвестные остатки?', ['Только известные остатки','Сценарий: остаток на начало месяца','Сценарий: вычесть продажи без поступлений'], horizontal=True)
    if scenario!='Только известные остатки':
        ds,stock_evidence=stock_scenario(ds,str(as_of),'reference' if scenario=='Сценарий: остаток на начало месяца' else 'depletion')
        st.warning(f'Сценарный расчёт для {len(stock_evidence)} товаров. Поступления, резервы и другие движения после снимка неизвестны. Это не фактические остатки и не границы возможного запаса. Позиции без достаточных данных остаются неизвестными.')
        with st.expander('Из чего получились сценарные остатки'):
            st.dataframe(stock_evidence.rename(columns=LABELS),hide_index=True,width='stretch')

cfg=Settings(str(as_of),int(lead),int(review),int(safety),float(growth),remove,seasonal,trend,compensate,approx,category_factors)
cfg=replace(cfg,forecast_method={'Автовыбор по истории':'adaptive','Общая ML-модель':'pooled'}.get(method,'legacy'),use_source_growth=source_growth)
if method=='Обычное среднее за 6 месяцев':
    cfg=replace(cfg,remove_outliers=False,seasonality=False,trend=False,compensate_stockout=False)
    st.info('Выбрано обычное среднее. Переключатели очистки, сезонности, тренда и восстановления спроса не применяются. Ручной прирост и параметры запаса сохранены.')
with st.spinner('Считаю спрос и заказы…'):result,histories,anomalies=run(ds,cfg)
if not stock_evidence.empty:result=annotate_scenario(result,stock_evidence)
if result.empty:st.warning('Нет позиций для расчёта');st.stop()
with st.spinner('Проверяю устойчивость решения…'):
    scenario_summary,scenario_details,delay_comparison=cached_scenarios(source_ds,cfg,result,histories,scenario_options,CALCULATION_VERSION)
result=result.merge(scenario_summary,on='sku',validate='one_to_one')
if 'stock_basis' not in result:result['stock_basis']=np.nan
result['stock_basis']=result.stock_basis.astype(object).where(result.stock_basis.notna(),result.sku.map(source_ds.products.set_index('sku').stock_origin))
result.loc[result.stock.isna(),'stock_basis']='Остаток неизвестен'
confirmed=set(st.session_state.get(count_key,{}))
known_codes=source_ds.products.loc[source_ds.products.stock.notna(),'sku']
result.loc[result.sku.isin(known_codes),'stock_basis']=result.loc[result.sku.isin(known_codes),'sku'].map(source_ds.products.set_index('sku').stock_origin)
result.loc[result.sku.isin(confirmed),'stock_basis']=f'Ручной пересчёт на {as_of}'
result['input_batch']=dataset_id
result['source_files']=json.dumps(source_ds.sources,ensure_ascii=False,default=str)
queue=recount_queue(result)
result=result.assign(_priority=result.status.map({'Срочно':0,'Заказать':1,'Нужен остаток':2,'Нет истории':3,'Достаточно':4})).sort_values(['_priority','sku']).drop(columns='_priority').reset_index(drop=True)
cards=st.columns(4)
cards[0].metric('Товаров в анализе',len(result))
cards[1].metric('Подготовить заказ',int(result.recommended.gt(0).sum()))
cards[2].metric('Ускорить · включая сценарии',int((result.urgency_any|result.expedite_need.gt(0)).sum()))
cards[3].metric('Проверить остаток',len(queue))
calculation_id=hashlib.sha256((result.to_json(date_format='iso')+json.dumps(vars(cfg),sort_keys=True,ensure_ascii=False)+ds.transit.to_json(date_format='iso')).encode()).hexdigest()[:16]
orders_tab,detail_tab,quality_tab,comparison_tab,evidence_tab,effect_tab=st.tabs(['Рабочее место','Почему столько','Качество данных','Сравнение методов','Проверка кейса','Проверка эффекта'])
with orders_tab:
    urgent=result[result.status.eq('Срочно')]
    if not urgent.empty:
        st.warning(f'{len(urgent)} позиций требуют ускорения поставки или перемещения: обычный заказ не успеет до дефицита. Количества и даты — во вкладке «Почему столько».')
    st.caption('1. Проверьте данные → 2. Откройте обоснование → 3. Утвердите позиции → 4. Выгрузите заказ поставщику.')
    a,b=st.columns([2,1]);search=a.text_input('Найти товар',placeholder='Название, артикул или код')
    statuses=b.multiselect('Статус',list(result.status.unique()),default=list(result.status.unique()))
    selected=result[result.status.isin(statuses)].copy()
    if search:selected=selected[selected[['sku','article','name']].astype(str).apply(lambda s:s.str.contains(search,case=False,regex=False)).any(axis=1)]
    chosen_categories=st.multiselect('Категории',categories)
    if chosen_categories:selected=selected[selected.category.astype(str).isin(chosen_categories)]
    action=st.radio('Выберите действие',['Подготовить заказ','Ускорить поставку','Проверить остаток'],horizontal=True)
    st.caption('Группы могут пересекаться: товару одновременно нужны заказ и ускорение. Количества разных товаров не складываются.')
    if action=='Подготовить заказ':selected=selected[selected.recommended.gt(0)]
    elif action=='Ускорить поставку':
        selected=selected[selected.urgency_any|selected.expedite_need.gt(0)]
        st.caption('Здесь показан дефицит в выбранном расчёте или хотя бы одном проверенном сценарии. Откройте товар, чтобы отличить эти основания.')
    else:
        selected=queue[queue.sku.isin(selected.sku)]
        st.subheader('Что пересчитать на складе в первую очередь')
        st.caption('Сначала остаток, меняющий заказ да/нет или срочность; затем неизвестный остаток. Внутри приоритета — срочность, давность снимка и относительная чувствительность. Денежная выгода не оценивалась.')
    columns=['sku','name','count_priority','count_reason','scenario_snapshot_age_days','robustness','order_min','order_max','recommended','unit','first_shortage','expedite_need','stock','stock_basis'] if action=='Проверить остаток' else ['sku','name','recommended','unit','order_min','order_max','robustness','first_shortage','expedite_need','stock','stock_basis']
    st.dataframe(selected[[c for c in columns if c in selected]].rename(columns=LABELS),hide_index=True,width='stretch',column_config={'Код 1С':st.column_config.TextColumn(width='small'),'Товар':st.column_config.TextColumn(width='medium'),'Минимум заказа в сценариях':st.column_config.NumberColumn('Заказ от',width='small'),'Максимум заказа в сценариях':st.column_config.NumberColumn('Заказ до',width='small'),'Приоритет пересчёта':st.column_config.NumberColumn('Приоритет',width='small'),'Почему пересчитать':st.column_config.TextColumn(width='large')})
    if st.checkbox('Показать задержку на 7 дней'):
        st.markdown('**Что изменится, если ожидаемые поставки опоздают**')
        st.caption('Все известные будущие поступления сдвинуты на 7 дней. Выбранные остатки, спрос и срок нового заказа неизменны. Просроченные поступления и путь без даты не считаются прибывшими. Это отдельный эксперимент, независимо от задержки в сетке.')
        evaluated=delay_comparison[delay_comparison.delay_evaluable]
        st.write(f'Стали срочными: {int(evaluated.delay_new_urgent.sum())} товаров. Изменился заказ: {int(evaluated.delay_order_change.abs().gt(1e-8).sum())}. Рассчитано {len(evaluated)} из {len(result)} товаров; остальным нужны остатки или история.')
        show_delay=evaluated.merge(result[['sku','name','unit']],on='sku')
        st.dataframe(show_delay.drop(columns=['delay_evaluable','delay_stock']).rename(columns=LABELS),hide_index=True,width='stretch')
        st.warning('Неизменный размер заказа не означает отсутствие риска. Дефицит до прихода нового заказа устраняется ускорением или перемещением, а не обычным заказом.')
    if not selected.empty:
        focused=st.selectbox('Открыть решение по товару',selected.sku.tolist(),format_func=lambda code:f"{code} · {result.set_index('sku').loc[code,'name']}")
        item=result.set_index('sku').loc[focused]
        original=source_ds.products.set_index('sku').loc[focused]
        st.markdown('#### Исходные данные → Допущения → Расчёт → Действие')
        st.markdown('**1. Исходные данные**')
        st.write(f'Остаток: {original.stock:g} {item.unit}' if pd.notna(original.stock) else 'Актуальный остаток неизвестен — он не подменён нулём.')
        st.write(f'Основание выбранного расчёта: {item.stock_basis}. Поставщик: {item.supplier}. Кратность: {item.pack:g}; минимум: {item.moq:g} {item.unit}.')
        incoming=source_ds.transit[source_ds.transit.sku.eq(focused)]
        if not incoming.empty:st.dataframe(incoming.rename(columns={'eta':'Ожидаемая дата','quantity':'Количество','sku':'Код 1С'}),hide_index=True)
        st.markdown('**2. Допущения**')
        st.write(item.scenario_assumptions)
        if item.scenario_count:
            st.write(f'Проверенные остатки: {item.stock_min:g}–{item.stock_max:g} {item.unit}. Все основания и множители доступны в таблице сочетаний ниже.')
        if pd.isna(original.stock):
            snapshots=source_ds.stocks[source_ds.stocks.sku.eq(focused)&source_ds.stocks.date.le(pd.Timestamp(as_of))].sort_values('date')
            if not snapshots.empty:
                latest=snapshots.iloc[-1];st.write(f'Опорный снимок {latest.date.date()}: {latest.quantity:g} {item.unit}. Это не подтверждённый текущий остаток.')
        st.markdown('**3. Расчёт**')
        st.write(item.robustness)
        st.write(item.sensitivity_reason)
        if item.scenario_count:
            st.write(f'Проверено {item.scenario_count:g} сочетаний. Заказ от {item.order_min:g} до {item.order_max:g} {item.unit}. {item.sensitivity_reason}.')
            with st.expander('Каждое проверенное сочетание'):
                trace=scenario_details[scenario_details.sku.eq(focused)]
                st.dataframe(trace.rename(columns=LABELS),hide_index=True,width='stretch')
                st.download_button('Скачать сценарии этого товара',export_excel(trace,{**vars(cfg),**vars(scenario_options)}),file_name=f'axioma-scenarios-{focused}.xlsx')
        st.write(item.reason)
        st.markdown('**4. Действие**')
        if item.expedite_need>0:st.error(f'Ускорить или переместить: до {item.order_arrival} не хватает до {item.expedite_need:g} {item.unit}; первый риск {item.first_shortage}. Обычный заказ этот дефицит не устраняет.')
        elif item.urgency_any:st.warning('Срочность возникает в части проверенных сценариев. Уточните остаток и даты до решения об ускорении.')
        if pd.isna(item.recommended):st.warning('Утверждение заказа недоступно до уточнения данных или явного выбора расчётного сценария остатка.')
        elif item.recommended>0:st.write(f'Подготовить {item.recommended:g} {item.unit} по выбранному расчёту; проверить и утвердить ниже. Диапазон сценариев не заменяет выбранную рекомендацию.')
        else:st.write('По выбранному расчёту обычный заказ не требуется.')
        if item.check_stock:st.info(item.count_reason)
        with st.expander('Внести подтверждённый остаток после пересчёта'):
            counted=st.number_input(f'Подтверждённый свободный остаток · {focused}',min_value=0.,value=float(original.stock) if pd.notna(original.stock) else 0.,key=f'count-value-{dataset_id}-{focused}-{as_of}')
            st.caption(f'Указывайте фактический свободный остаток на {as_of}, в единицах {item.unit}. Исправление сохраняется в этой сессии для этих файлов и даты; оно попадёт в происхождение заказа.')
            if st.button('Применить уточнённый остаток'):
                saved=dict(st.session_state.get(count_key,{}));saved[focused]=counted;st.session_state[count_key]=saved;st.rerun()
    else:st.info('В этой группе нет товаров по выбранным фильтрам.')
    insufficient=result[result.robustness.eq(INSUFFICIENT)]
    if not insufficient.empty:
        with st.expander(f'Недостаточно данных даже для сценариев · {len(insufficient)}'):
            st.dataframe(insufficient[['sku','name','sensitivity_reason']].rename(columns=LABELS),hide_index=True,width='stretch')
    st.download_button('Скачать все рекомендации · Excel',export_excel(result,cfg),file_name=f'axioma-{as_of}.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    with st.expander('Проверить и утвердить заказ'):
        draft=result[result.recommended.gt(0)].copy()
        if draft.empty:st.info('Нет положительных рекомендаций для утверждения.')
        else:
            draft=draft[[c for c in ['sku','article','name','supplier','unit','recommended','pack','moq','stock_basis','snapshot_date','reason'] if c in draft]].copy();draft['Утвердить']=False;draft['К заказу']=draft.recommended
            draft=draft[['Утвердить','К заказу']+[c for c in draft.columns if c not in ['Утвердить','К заказу']]]
            reviewed=st.data_editor(draft.rename(columns=LABELS),hide_index=True,disabled=['Код 1С','Артикул','Товар','Поставщик','Ед.','Рекомендация','Кратность','Минимальная партия','Основание остатка','Дата снимка','Обоснование'],column_config={'К заказу':st.column_config.NumberColumn(min_value=0)},key=f'approval-{mode}-{supplier}-{calculation_id}')
            approved=reviewed[reviewed['Утвердить']&reviewed['К заказу'].gt(0)]
            bad=approved[(approved['К заказу']<approved['Минимальная партия'])|~np.isclose(approved['К заказу']%approved['Кратность'],0)]
            if len(bad):st.error('Исправьте количество: оно должно соответствовать минимуму и кратности.')
            reviewer=st.text_input('Кто проверил заказ',key=f'reviewer-{mode}-{supplier}',placeholder='Имя или внутренний идентификатор менеджера')
            approved=approved.copy()
            lineage=['stock','stock_basis','snapshot_date','snapshot_stock','recorded_sales','scenario_stock','snapshot_age_days','robustness','order_min','order_max','scenario_count','scenario_assumptions','scenario_basis','scenario_snapshot_date','scenario_snapshot_age_days','sensitivity_reason','expedite_need','first_shortage','input_batch','source_files']
            extra=[c for c in lineage if c in result and LABELS.get(c,c) not in approved.columns]
            approved=approved.merge(result[['sku']+extra].rename(columns=LABELS),on='Код 1С',how='left',validate='one_to_one')
            approved['Проверил']=reviewer.strip()
            approved['Версия расчёта']=calculation_id
            approved['Параметры расчёта']=json.dumps({**vars(cfg),'scenario_grid':vars(scenario_options)},ensure_ascii=False,sort_keys=True)
            st.caption('При изменении входных данных или настроек отметки утверждения сбрасываются. В корзине остаются отдельно сохранённые версии.')
            acknowledge=True if stock_evidence.empty else st.checkbox('Понимаю: это сценарный заказ с предполагаемыми остатками',key=f'scenario-ack-{calculation_id}')
            st.download_button('Скачать утверждённые позиции',export_excel(approved,cfg),file_name='axioma-approved.xlsx',disabled=approved.empty or not bad.empty or not acknowledge or not reviewer.strip())
            if st.button('Добавить утверждённые позиции в общую корзину',disabled=approved.empty or not bad.empty or not acknowledge or not reviewer.strip()):
                lines=approved.rename(columns={v:k for k,v in LABELS.items()}).copy()
                lines=lines.drop(columns=['Утвердить'],errors='ignore')
                lines['calculated_at']=str(as_of);lines['planning_method']=method
                st.session_state['order_cart']=add_reviewed_lines(st.session_state.get('order_cart',pd.DataFrame()),lines)
                st.success(f'Добавлено {len(lines)} позиций. Теперь можно рассчитать другого поставщика.')
            st.caption('Скачивание не отправляет заказ поставщику. Утверждение действует в текущей сессии; сохраните файл.')
    cart=st.session_state.get('order_cart',pd.DataFrame())
    if not cart.empty:
        st.subheader('Общая корзина поставщиков')
        st.caption('Сохранённые утверждения этой сессии. Новые расчёты не меняют их автоматически: повторное добавление заменяет строку того же поставщика и товара. Корзина хранит дату и метод каждого расчёта.')
        for supplier_name,group in cart.groupby('supplier'):
            with st.expander(f'{supplier_name} · {len(group)} позиций',expanded=True):
                st.dataframe(group.rename(columns=LABELS),hide_index=True,width='stretch')
        st.download_button('Скачать общую корзину · Excel',export_excel(cart,{}),file_name='axioma-supplier-orders.xlsx')
        if st.button('Очистить корзину этой сессии'):
            st.session_state['order_cart']=pd.DataFrame();st.rerun()
with detail_tab:
    sku=st.selectbox('Выберите товар',list(histories),format_func=lambda k:f"{k} · {result.set_index('sku').loc[k,'name']}") if histories else None
    if sku:
        row=result.set_index('sku').loc[sku];st.subheader(row['name']);st.write(row.reason)
        if row.expedite_need>0:st.error(f'До {row.order_arrival} не хватает до {row.expedite_need:g} {row.unit}. Действие: ускорить существующую поставку или согласовать перемещение. Этот объём не нужно автоматически прибавлять к заказу.')
        elif row.bridge_need>max(0,row.forecast+row.safety-row.stock-row.in_transit)+0.1:st.info(f'Проверка по датам: нужно перекрыть {row.bridge_need:g} {row.unit} до позднего поступления. Его общий объём достаточен, но срок не закрывает промежуточный спрос.')
        c1,c2,c3=st.columns(3);c1.metric('Рекомендовано',f'{row.recommended:g}' if pd.notna(row.recommended) else 'Нужен остаток');c2.metric('Порог пополнения',f'{row.stock_threshold:g}');c3.metric('Восстановлено спроса',row.lost)
        st.write('Метод:',row.method,'· Характер спроса:',row.demand_type)
        st.line_chart(histories[sku].set_index('Дата'),color=['#94A3B8','#0D9488','#F59E0B'])
        st.caption('Серый — фактические продажи; зелёный — без разовых всплесков; оранжевый — с компенсацией отсутствия товара.')
        st.write('Оценка отсутствия:',row.stockout_mode)
        model_info=histories[sku].attrs.get('model_info',{})
        if model_info.get('candidates'):
            with st.expander('Почему выбран этот метод'):
                st.write(f'Сравнение на {row.cv_months:g} последних доступных месяцах внутри истории. Это выбор метода, а не независимая оценка качества. При близкой ошибке усредняются до трёх методов.')
                st.dataframe(pd.DataFrame(model_info['candidates']).rename(columns={'method':'Метод','mae_daily':'Средняя абсолютная ошибка, ед./день'}),hide_index=True,width='stretch')
        future_frame=pd.DataFrame(histories[sku].attrs.get('future',[]))
        if not future_frame.empty and pd.notna(row.stock):
            future_frame['Дата']=pd.to_datetime(future_frame['Дата'])
            transit=ds.transit[ds.transit.sku.eq(sku)]
            arrivals=transit.groupby('eta').quantity.sum() if not transit.empty else pd.Series(dtype=float)
            future_frame['Ожидаемые поступления']=future_frame['Дата'].map(arrivals).fillna(0).clip(lower=0)
            future_frame['Без нового заказа']=row.stock+(future_frame['Ожидаемые поступления']-future_frame['Прогноз в день']).cumsum()
            future_frame['С новым заказом']=future_frame['Без нового заказа']+np.where(future_frame['Дата']>=pd.Timestamp(as_of)+pd.Timedelta(days=lead),row.recommended,0)
            st.markdown('**Что будет с запасом по дням**')
            st.line_chart(future_frame.set_index('Дата')[['Без нового заказа','С новым заказом']],color=['#EF4444','#0D9488'])
            st.caption('Отрицательный баланс показывает неудовлетворённую потребность. Поступления учитываются в начале дня; новый заказ — через заданный срок. Это сценарий по прогнозу, а не фактическое движение склада.')
            if pd.notna(row.first_shortage):st.warning(f'Риск дефицита с {row.first_shortage}, до прихода нового заказа. Нужна ускоренная поставка или перемещение со склада.')
            if cfg.forecast_method=='adaptive' and row.cv_months>=3:
                st.info(f'Стресс-сценарий: {row.stress_recommended:g} ед. к заказу, если спрос превысит прогноз на величину 90-го процентиля прошлых положительных ошибок. Это ориентир для проверки, не гарантированный уровень сервиса.')
with quality_tab:
    st.subheader('Прозрачный расчёт')
    st.write('Заказ = прогноз на срок поставки и интервал пересмотра + страховой запас − свободный остаток − поступления в этот период. Дополнительно проверяется баланс каждого дня после прихода нового заказа: позднее поступление не должно скрыть промежуточный дефицит. Берётся большая из двух потребностей, затем учитываются минимум и кратность.')
    st.write('Крупные накладные и всплески клиента за день проверяются устойчивым порогом по распределению объёмов. Подозрительный объём заменяется типичным; исходные файлы не изменяются.')
    st.dataframe(pd.DataFrame(ds.sources),hide_index=True,width='stretch')
    if not anomalies.empty:st.markdown('**Обнаруженные всплески**');st.dataframe(anomalies,hide_index=True,width='stretch')
    else:st.info('Всплески не обнаружены либо нет детальных продаж для их проверки.')
    st.warning('Без выбранного сценария неизвестный остаток блокирует заказ. Сценарные остатки всегда помечены в результатах. Неизвестные или просроченные даты поступления требуют уточнения. Без client_id невозможно проверить покупки одного клиента между разными накладными.')
    st.caption('Прототип HackAlem AI · AI-агент Codex использован при разработке. Расчёт детерминированный; внешние AI API не вызываются.')

with comparison_tab:
    import json
    published=[]
    for report_name in ['benchmark-systeme.json','benchmark-iek.json']:
        report_path=Path(__file__).parent/'docs'/report_name
        if report_path.exists():
            report=json.loads(report_path.read_text(encoding='utf-8'))
            published.append({'Поставщик':report['supplier'],'Проверено товаров':report['evaluated_skus'],'С положительным фактом':report['scored_skus'],'Ошибка исходной модели, %':round(report['macro_wape_model'],2),'Ошибка среднего, %':round(report['macro_wape_baseline'],2),'Ошибка автовыбора, %':round(report['macro_wape_adaptive'],2),'Ошибка общей ML-модели, %':round(report['macro_wape_pooled'],2)})
    if published:
        st.subheader('Сохранённая проверка полного подходящего каталога')
        st.dataframe(pd.DataFrame(published),hide_index=True,width='stretch')
        st.caption('Июнь–август 2026. Каждый прогноз обучен на предшествующей истории. Это сохранённый протокол, а не результат текущих настроек. Средняя WAPE считается по товарам с положительным фактом; ошибки могут превышать 100%. Ограниченные выборки использовались при разработке, поэтому для независимого подтверждения нужен новый период. Учёт товаров с нулевыми продажами и точные команды — в docs/validation.md.')
    st.subheader('Что меняет обработка спроса')
    st.write('Сравните три расчёта на одинаковых товарах, остатках и сроках: обычное среднее за 6 завершённых месяцев; текущая модель без удаления всплесков; выбранная вами модель.')
    st.caption('Это сравнение рекомендаций, а не проверка точности на будущих продажах. Меньший заказ сам по себе не доказывает экономию. Количества разных товаров не складываются.')
    if st.button('Рассчитать сравнение'):
        with st.spinner('Сравниваю методы на одинаковых входных данных…'):
            simple=run(ds,replace(cfg,forecast_method='legacy',remove_outliers=False,seasonality=False,trend=False,compensate_stockout=False))[0]
            unclean=run(ds,replace(cfg,remove_outliers=False))[0]
        comparison=result[[c for c in ['sku','name','recommended','stock_basis','snapshot_date'] if c in result]].rename(columns={'recommended':'Выбранная модель'})
        comparison=comparison.merge(simple[['sku','recommended']].rename(columns={'recommended':'Обычное среднее'}),on='sku',validate='one_to_one')
        comparison=comparison.merge(unclean[['sku','recommended']].rename(columns={'recommended':'Без удаления всплесков'}),on='sku',validate='one_to_one')
        comparison['Разница со средним']=comparison['Выбранная модель']-comparison['Обычное среднее']
        comparison['Влияние фильтра всплесков']=comparison['Выбранная модель']-comparison['Без удаления всплесков']
        comparison=comparison.sort_values('Влияние фильтра всплесков',key=lambda x:x.abs(),ascending=False)
        st.dataframe(comparison.rename(columns=LABELS),hide_index=True,width='stretch')
        st.download_button('Скачать сравнение методов',export_excel(comparison,cfg),file_name='axioma-method-comparison.xlsx')

    st.divider()
    st.subheader('Проверка на прошлых месяцах')
    st.write('Предсказываем 3 завершённых месяца по очереди: каждый прогноз видит только более раннюю историю. Затем сравниваем его с фактическими продажами и обычным средним.')
    st.caption('Для быстрого запуска: первые 30 кодов с минимум 6 месяцами истории до проверки. Это выборка, не весь каталог. Готовые сезонные коэффициенты и периоды отсутствия исключены, поскольку неизвестно, когда они стали доступны. Факт продаж не равен скрытому спросу при дефиците и включает разовые покупки.')
    if st.button('Проверить прогноз на истории'):
        from src.validation import backtest
        with st.spinner('Проверяю прогноз без доступа к будущим продажам…'):
            detail,scores,coverage=backtest(ds,str(as_of),include_pooled=True)
        st.write(f"Проверены {coverage['evaluated_skus']} товаров из {coverage['eligible_skus']} подходящих. Месяцы: {', '.join(coverage['months'])}.")
        valid=scores.dropna(subset=['model_wape','baseline_wape'])
        if valid.empty:st.info('Недостаточно наблюдаемых продаж для оценки ошибки.')
        else:
            c1,c2,c3,c4=st.columns(4)
            c1.metric('Средняя ошибка Axioma по товарам',f'{valid.model_wape.mean():.1f}%')
            c2.metric('Средняя ошибка обычного среднего',f'{valid.baseline_wape.mean():.1f}%')
            c3.metric('Средняя ошибка автовыбора',f'{valid.adaptive_wape.mean():.1f}%')
            c4.metric('Средняя ошибка общей ML-модели',f'{valid.pooled_wape.mean():.1f}%')
            st.caption('Для каждого товара WAPE = сумма абсолютных ошибок / сумма факта; затем берём среднее по товарам с положительным фактом. Меньше — лучше, ошибка может быть выше 100%. Это не «процент точности» и не денежная экономия.')
            st.dataframe(scores.rename(columns={'sku':'Код 1С','months':'Месяцев','actual_total':'Факт за период','model_wape':'Ошибка исходной модели, %','baseline_wape':'Ошибка среднего, %','adaptive_wape':'Ошибка автовыбора, %','pooled_wape':'Ошибка общей ML-модели, %'}),hide_index=True,width='stretch')
            st.download_button('Скачать результаты проверки',export_excel(detail.rename(columns={'month':'Месяц','actual':'Факт','model':'Прогноз Axioma','baseline':'Прогноз среднего'}),cfg),file_name='axioma-validation.xlsx')

with evidence_tab:
    st.subheader('Пять обязательных требований — проверка в один клик')
    st.write('Система сама меняет по одному входному условию и показывает результат до и после. Это воспроизводимые синтетические примеры, а не оценка экономии или точности на данных компании.')
    st.caption('Для изоляции арифметики примеры используют исходную модель с явно заданными переключателями. Прогноз ML на реальных данных проверяется отдельно во вкладке «Сравнение методов».')
    if st.button('Запустить проверку требований',type='primary'):
        from src.evidence import acceptance_examples
        st.session_state['acceptance_evidence']=acceptance_examples()
    if 'acceptance_evidence' in st.session_state:
        checks=st.session_state['acceptance_evidence']
        passed=int(checks['Результат'].eq('Пройдено').sum())
        if passed==len(checks):st.success(f'{passed} из {len(checks)} проверок пройдено: все пять обязательных требований представлены.')
        else:st.error(f'Пройдено {passed} из {len(checks)}: проверьте строки с ошибкой.')
        st.dataframe(checks,hide_index=True,width='stretch')
        st.download_button('Скачать протокол проверки · CSV',checks.to_csv(index=False).encode('utf-8-sig'),file_name='axioma-case-checks.csv',mime='text/csv')
    st.markdown('**Что требуется для пилота у Электрокомплект**')
    st.write('Ежедневный свободный остаток и резервы; обезличенный ID клиента; точные периоды отсутствия; согласованные сроки, MOQ и единицы закупки. Подключение этих источников позволит проверять решения на новом периоде.')
    st.caption('План пилота: 2 недели параллельного расчёта с менеджером без автоматической отправки. Измерять время подготовки заказа, долю ручных правок и дни дефицита; денежный эффект считать только после получения цен и фактических затрат.')


with effect_tab:
    from src.effect import (Experiment, compare_inventory, aggregate_metrics, forecast_diagnostics,
                            scenario_costs, protocol_zip, pilot_zip, POLICIES, MODES)
    st.subheader('Проверка эффекта: обслуживание, запас и цена допущений')
    st.write('Сравниваем решения по запасам на одной истории и при одинаковых правилах. Простое среднее — явно заданная политика сравнения; реальный процесс закупщика нам неизвестен.')
    with st.expander('Что уже доказано, а что предстоит измерить'):
        st.dataframe(pd.DataFrame([
            {'Тип доказательства':'Воспроизводимые тесты','Что подтверждено':'Арифметика, сроки поступлений, сценарии, MOQ, кратность, отсутствие доступа к будущим данным'},
            {'Тип доказательства':'Измерение на файлах компании','Что подтверждено':'Время вычислений и чувствительность рекомендаций; это не время работы менеджера'},
            {'Тип доказательства':'Ретроспективная симуляция','Что подтверждено':'Сравнение политик при выбранных начальном запасе, сроках и распределении продаж по дням'},
            {'Тип доказательства':'Реальный пилот — ещё не проведён','Что подтверждено':'Экономия денег, активное время закупщика и внедрение требуют подтверждения заказчика'},
        ]),hide_index=True,width='stretch')
    st.caption(f'Источник: {source_ds.supplier} · версия входных файлов {dataset_id}. '
               + ('Синтетические данные демо.' if mode=='Показать пример' else 'Загруженные отчёты; коммерческие исходники не публикуются.'))
    last_complete=pd.Timestamp(as_of).replace(day=1)-pd.Timedelta(days=1)
    first_complete=(last_complete-pd.DateOffset(months=2)).replace(day=1)
    c1,c2,c3=st.columns(3)
    sim_start=c1.date_input('Начало симуляции · первое число',value=first_complete.date(),key='effect-start')
    sim_end=c2.date_input('Конец симуляции · последнее число',value=last_complete.date(),key='effect-end')
    sim_limit=c3.number_input('Максимум товаров в проверке',min_value=1,max_value=100,value=30,key='effect-limit')
    c1,c2,c3,c4=st.columns(4)
    sim_lead=c1.number_input('Срок поставки в симуляции, дней',1,365,21,key='effect-lead')
    sim_review=c2.number_input('Пересмотр в симуляции, дней',1,90,14,key='effect-review')
    sim_safety=c3.number_input('Страховой запас в симуляции, дней',0,90,7,key='effect-safety')
    sim_initial=c4.number_input('Начальный запас, дней прошлого спроса',0.,365.,21.,key='effect-initial')
    experiment=Experiment(str(sim_start),str(sim_end),sim_lead,sim_review,sim_safety,sim_initial,sim_limit)
    st.write('**Три политики:** среднее за 6 завершённых месяцев без очистки и сезонности; текущий автовыбор; общая ML-модель (на малой истории — статистический резервный метод). Модели сохранены без подгонки по результату. Общая модель здесь обучается на выбранных товарах, а не на всём каталоге.')
    st.caption('Начальный запас = средний дневной спрос до старта × выбранное число дней. MOQ и кратность из текущих входов заморожены одинаково. Текущие остатки, текущий путь и готовые сезонные коэффициенты исключены: их историческая доступность неизвестна. Начальный путь здесь нулевой; новые заказы моделируются отдельно.')
    st.warning('Это симуляция, не предотвращённые потери. Месячные продажи равномерно распределены по дням; реальный спрос мог быть ограничен дефицитом. Показываем и потерянную продажу, и отложенный заказ. Июнь–август 2026 уже использовались при разработке и не являются независимым финальным тестом.')
    signature=hashlib.sha256((dataset_id+CALCULATION_VERSION+str(vars(experiment))+
                            source_ds.sales.to_json(date_format='iso')+source_ds.transactions.to_json(date_format='iso')+
                            source_ds.products.to_json()).encode()).hexdigest()
    if st.button('Сравнить политики запасов',type='primary'):
        try:
            with st.spinner('Пересчитываю решения только по доступной на каждую дату истории…'):
                effect=compare_inventory(source_ds,experiment)
            effect[4].update(input_batch=dataset_id,calculation_version=CALCULATION_VERSION,
                             data_type='Синтетические данные' if mode=='Показать пример' else 'Отчёты компании',
                             calculated_at=pd.Timestamp.now(tz='UTC').isoformat())
            st.session_state['effect_result']=(signature,effect)
        except ValueError as error:
            st.error(str(error));st.session_state.pop('effect_result',None)
    saved=st.session_state.get('effect_result')
    if saved is not None and saved[0]!=signature:
        st.info('Входы или условия изменились. Пересчитайте сравнение: прежние результаты скрыты.')
    if saved is not None and saved[0]==signature:
        er,ed,eo,ee,em,ef=saved[1]
        st.markdown('**Ретроспективная симуляция · '+('синтетический пример' if mode=='Показать пример' else 'на загруженной истории продаж')+'**')
        st.write(f"Период: {experiment.start} — {experiment.end}. Проверено {em['evaluated_skus']} из {em['selected_skus']} выбранных товаров; подходят по прошлой истории {em['eligible_skus']}. Исключено из-за пропусков целевых месяцев: {em['excluded_skus']}.")
        if er.empty:
            st.info('Недостаточно данных: нужны минимум 6 наблюдаемых месяцев до старта и все месяцы оцениваемого периода. Пропуск не считается нулевой продажей.')
            if not ee.empty:st.dataframe(ee,hide_index=True)
        else:
            agg=aggregate_metrics(er)
            effect_labels={'policy':'Политика','mode':'Неудовлетворённый спрос','evaluated_skus':'Проверено SKU',
                'positive_demand_skus':'SKU в знаменателе обслуживания','zero_demand_skus':'SKU с нулевым спросом',
                'macro_immediate_fill_pct':'Обслужено сразу, % · среднее по SKU','macro_eventual_fill_pct':'Обслужено к концу, % · среднее по SKU',
                'mean_shortage_days':'Средние дни дефицита','worse_service_skus':'SKU с худшим обслуживанием',
                'better_service_skus':'SKU с лучшим обслуживанием','less_stock_skus':'SKU с меньшим запасом','lower_stock_worse_service_skus':'SKU: меньше запас, хуже сервис','more_stock_skus':'SKU с большим запасом','service_stock_tradeoff_skus':'SKU: лучше сервис, больше запас',
                'sku':'Код 1С','unit':'Ед.','name':'Товар','shortage_days':'Дни дефицита','backlog_days':'Дни с отложенным спросом',
                'immediate_fill_rate':'Доля обслуживания сразу','eventual_fill_rate':'Доля обслуживания к концу',
                'average_stock':'Средний запас','orders':'Число заказов','end_stock':'Конечный остаток','end_backlog':'Не выполнено к концу',
                'end_pipeline':'Путь после конца периода','fill_change_pp':'Изменение обслуживания, п.п.','stock_change':'Изменение среднего запаса',
                'worse_service':'Хуже обслуживание','more_stock':'Больше запас','tradeoff':'Лучше сервис ценой запаса',
                'history_zero_fraction':'Доля нулевых месяцев в прошлых 6','history_months':'Месяцев прошлой истории'}
            def effect_view(frame):
                return frame.replace({'policy':POLICIES,'mode':MODES}).rename(columns=effect_labels)
            summary_mode=st.radio('Режим дефицита в итогах',list(MODES),format_func=MODES.get,horizontal=True,key='effect-summary-mode')
            group=agg[agg['mode'].eq(summary_mode)].set_index('policy')
            reference=group.loc['mean','macro_immediate_fill_pct']
            for col,policy in zip(st.columns(3),POLICIES):
                item=group.loc[policy];fill=item.macro_immediate_fill_pct
                delta=f'{fill-reference:+.2f} п.п. к среднему' if policy!='mean' and pd.notna(fill) else None
                col.metric(POLICIES[policy],f'{fill:.2f}%' if pd.notna(fill) else 'Нет спроса',delta=delta)
                col.caption(f"Обслужено сразу · {int(item.positive_demand_skus)} SKU в знаменателе. Хуже среднего: {int(item.worse_service_skus)} SKU; больший запас: {int(item.more_stock_skus)}.")
            with st.expander('Все агрегаты: оба режима, нулевой спрос, запас и компромиссы'):
                st.dataframe(effect_view(agg).round(2),hide_index=True,width='stretch')
            st.caption('Обслуживание: среднее долей по SKU с положительными продажами. Товары с нулём учитываются отдельно. Дни дефицита — дни неудовлетворённого нового спроса в симуляции. Запасы разных товаров и единиц не складываются.')
            columns=['sku','name','unit','policy','mode','shortage_days','backlog_days','immediate_fill_rate','eventual_fill_rate','average_stock','orders','end_stock','end_backlog','end_pipeline','fill_change_pp','stock_change','worse_service','more_stock','history_zero_fraction']
            with st.expander('Все товары и натуральные показатели'):
                st.dataframe(effect_view(er[columns]).round(3),hide_index=True,width='stretch')
            worse=er[er.policy.ne('mean')&(er.worse_service|er.more_stock)]
            st.markdown('**Где обслуживание хуже или запас больше среднего**')
            st.caption('Отрицательное изменение обслуживания означает, что выбранная модель проиграла простому среднему. Для текущего заказа простой метод доступен в настройке «Метод расчёта». Больший запас — обратная сторона решения, а не автоматически денежный убыток. Строки включают оба режима неудовлетворённого спроса.')
            if worse.empty:st.info('В выбранных условиях таких случаев нет. Это не гарантия для других условий.')
            else:st.dataframe(effect_view(worse[columns]).round(3),hide_index=True,width='stretch')
            with st.expander('Ошибка и систематическое смещение прогноза'):
                diagnostics=forecast_diagnostics(ef,ed,experiment.review_days)
                st.caption('Непересекающиеся полные окна пересмотра; факт распределён по дням искусственно. WAPE не точность. Положительное смещение — завышение, отрицательное — занижение. При нулевом факте проценты не определены. Это диагностика сохранённых моделей, не независимый подбор.')
                st.dataframe(effect_view(diagnostics).rename(columns={'wape_pct':'WAPE, %','signed_bias_pct':'Смещение, %','windows':'Полных окон','actual':'Продажи за окна','zero_actual_positive_forecast':'Нулевой факт, положительный прогноз'}).round(2),hide_index=True,width='stretch')
                st.dataframe(ef[['policy','method']].drop_duplicates().replace({'policy':POLICIES}),hide_index=True)
            sku_options=list(er.sku.drop_duplicates())
            effect_sku=st.selectbox('Товар для дневного баланса и затрат',sku_options,key='effect-sku-'+signature[:12])
            selected_policy=st.selectbox('Политика для дневного баланса',list(POLICIES),format_func=POLICIES.get,key='effect-policy')
            selected_mode=st.selectbox('Модель дефицита для дневного баланса',list(MODES),format_func=MODES.get,key='effect-mode')
            trace=ed[ed.sku.eq(effect_sku)&ed.policy.eq(selected_policy)&ed['mode'].eq(selected_mode)]
            st.line_chart(trace.set_index('date')[['stock_end','unfilled_today','backlog_end']].rename(columns={'stock_end':'Остаток','unfilled_today':'Не обслужено сегодня','backlog_end':'Отложено к концу дня'}))
            st.caption(f"Один товар {effect_sku}; единица: {er[er.sku.eq(effect_sku)].unit.iloc[0]}. Поступление — в начале дня, затем заказ, выполнение отложенного спроса и продажи дня. Заказы после конца периода сохраняются в пути, а не исчезают.")
            costs=None
            if st.checkbox('Рассчитать сценарные затраты по моим ставкам',key='effect-cost-enable'):
                unit=str(er[er.sku.eq(effect_sku)].unit.iloc[0])
                st.info('Цен в исходных данных нет. Введите собственные ставки и источник. Эти затраты сценарные; закупочные обязательства и остаточная стоимость показаны отдельно от операционных затрат.')
                source=st.text_input('Источник ставок и дата согласования',key='effect-cost-source-'+effect_sku)
                currency=st.text_input('Валюта ставок',value='KZT',key='effect-cost-currency')
                cc1,cc2=st.columns(2)
                price=cc1.number_input(f'Цена закупки, {currency}/{unit}',min_value=0.,key='effect-price-'+effect_sku)
                holding=cc2.number_input(f'Хранение, {currency}/{unit}/день',min_value=0.,key='effect-holding-'+effect_sku)
                line_cost=cc1.number_input(f'Оформление одной строки заказа, {currency}',min_value=0.,key='effect-line-'+effect_sku)
                lost_cost=cc2.number_input(f'Потерянная продажа, {currency}/{unit}',min_value=0.,key='effect-lost-'+effect_sku)
                backlog_cost=cc1.number_input(f'Ожидание отложенного спроса, {currency}/{unit}/день',min_value=0.,key='effect-backlog-'+effect_sku)
                st.caption(f'Период {experiment.start} — {experiment.end}. Затраты = сумма дневных запасов × хранение + число строк заказов × оформление + потерянные единицы × ставка потери (либо сумма дневного отложенного спроса × ставка ожидания). Ставки хранения и дефицита варьируются независимо ×0,5 / ×1 / ×1,5. Цена закупки не прибавляется к этим затратам; полная прибыль не рассчитывается.')
                if source.strip() and currency.strip():
                    costs=scenario_costs(er[er.sku.eq(effect_sku)],price,holding,line_cost,lost_cost,backlog_cost,source,currency)
                    costs['period_start']=experiment.start;costs['period_end']=experiment.end
                    st.dataframe(effect_view(costs).rename(columns={'operating_cost':'Сценарные операционные затраты','generated_purchase_commitment':'Закупочные обязательства','residual_stock_value':'Стоимость конечного запаса','pending_generated_value':'Стоимость созданного пути','holding_factor':'Множитель хранения','shortage_factor':'Множитель дефицита','cost_change_vs_mean':'Разница затрат со средним','baseline_operating_cost':'Затраты простого среднего','storage_cost':'Хранение','ordering_cost':'Оформление строк','shortage_cost':'Дефицит','received_generated_value':'Стоимость полученных новых заказов','source':'Источник ставок','currency':'Валюта','unit_price':'Цена единицы','holding_per_unit_day':'Хранение единицы в день','order_line_cost':'Оформление строки','lost_per_unit':'Ставка потери единицы','backlog_per_unit_day':'Ожидание единицы в день','period_start':'Начало периода','period_end':'Конец периода'}).round(2),hide_index=True,width='stretch')
                else:st.caption('До указания источника ставки не считаются подтверждёнными входами; денежный результат не выводится.')
            st.download_button('Скачать протокол симуляции · ZIP',protocol_zip(er,ed,eo,ee,em,ef,costs),file_name='axioma-inventory-experiment.zip',mime='application/zip')
    st.divider()
    st.subheader('Подготовить контрольный пилот с закупщиком')
    st.write('Пилот пока не проведён. Экспорт содержит пустые задания Excel/Axioma с чередованием порядка, версию входов и зафиксированные правила. Отдельно записываются активное время, ожидание, правки, причины и ошибки. Участника и проверяющего нужно назначить с заказчиком.')
    st.download_button('Скачать задания пилота · ZIP',pilot_zip(source_ds.products,dataset_id,vars(cfg),source_ds.sources),file_name='axioma-pilot-assignments.zip',mime='application/zip')
    st.caption('Критерии до начала: отсутствие критических ошибок, меньшее активное время при не худшем качестве. Порог и число заданий согласуются до измерения. Для складского результата дополнительно нужен новый период с ежедневными остатками, резервами, спросом и фактическими поступлениями.')
