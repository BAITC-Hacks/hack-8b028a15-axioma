from copy import deepcopy
from datetime import date
from io import BytesIO
from pathlib import Path
import os
import hashlib
import json
from dataclasses import asdict,replace
import numpy as np
import pandas as pd
import streamlit as st
from src.data import demo_data, parse_files, unpack_excel_archive
from src.engine import Settings, calculate
from src.forecasting import MODEL_NAMES
from src.backtest import evaluate_dataset,summarize
from src.storage import Store, fingerprint, records, canonical
from src.inputs import apply_updates, stock_template
from src.scenarios import stock_scenario,annotate_scenario
from src.orders import add_reviewed_lines

st.set_page_config(page_title='Axioma · Закупки',page_icon='◈',layout='wide')
st.markdown('''<style>
.block-container{padding-top:4rem;max-width:1450px} h1{letter-spacing:-1.5px}
[data-testid="stMetric"]{background:white;border:1px solid #e3e8ef;border-radius:14px;padding:18px}
[data-testid="stSidebar"]{border-right:1px solid #e3e8ef}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:3px;color:#0d9488}
.intro{font-size:18px;color:#61718a;max-width:850px;margin-bottom:24px}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_demo():return demo_data()
@st.cache_data(show_spinner=False)
def load_files(files,supplier):return parse_files(files,supplier)
@st.cache_data(show_spinner=False)
def run(ds,cfg,version):return calculate(ds,cfg)
@st.cache_data(show_spinner=False)
def benchmark(ds,end,periods,max_skus,version):return evaluate_dataset(ds,end,periods,max_skus)

ENGINE_VERSION=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((Path(__file__).parent/'src').glob('*.py')))).hexdigest()

store=Store(os.environ.get('AXIOMA_DB','data/axioma.sqlite3'))

LABELS={'sku':'Код 1С','article':'Артикул','name':'Товар','category':'Категория','supplier':'Поставщик','unit':'Ед.','stock':'Свободный остаток','in_transit':'Приедет в период','late_transit':'Приедет позже','forecast':'Прогноз спроса','safety':'Страховой запас','daily':'Спрос в день','growth':'Тренд, %','season':'Сезонный коэффициент','excluded':'Исключено всплесков','lost':'Восстановлено спроса','pack':'Кратность','moq':'Минимальная партия','recommended':'Рекомендация','status':'Статус','reason':'Обоснование','stockout_mode':'Оценка отсутствия'}
LABELS.update({'source_growth':'Прирост поставщика, доля','growth_source':'Источник прироста','unknown_transit':'Путь с неясной датой'})
LABELS.update({'model':'Модель прогноза','forecast_low':'Нижний ориентир спроса','forecast_high':'Верхний ориентир спроса','calibration_months':'Месяцев проверки','lead_days':'Срок поставки, дней','approved_quantity':'Утверждено','adjustment_reason':'Причина корректировки'})
LABELS['source_mismatch_months']='Месяцев с расхождением источников'
LABELS.update(bridge_need='Потребность до позднего поступления',expedite_need='Не хватает до новой поставки',order_arrival='Приход нового заказа',reviewer='Проверил',scenario_acknowledged='Сценарий подтверждён')
LABELS.update(stock_threshold='Порог пополнения',stock_basis='Основание остатка',snapshot_date='Дата снимка',snapshot_stock='Остаток в снимке',recorded_sales='Продажи после снимка',scenario_stock='Сценарный остаток',snapshot_age_days='Возраст снимка, дней',first_shortage='Первый риск дефицита',demand_type='Характер спроса',quantity='К заказу',calculated_at='Дата расчёта',planning_method='Метод расчёта')

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
    method=st.selectbox('Метод прогноза',['auto','pooled','adaptive','classic','mean6','seasonal_naive','seasonal_level','ses'],format_func=lambda m:MODEL_NAMES[m])
    use_source_growth=st.checkbox('Применить рост из сводки',False,help='Явный сценарий: коэффициент поставщика заменяет исторический тренд. Уточните смысл коэффициента у владельца данных.')
    st.markdown('**Период планирования**')
    as_of=st.date_input('Дата расчёта',date(2026,9,23))
    lead=st.number_input('Срок новой поставки, дней',min_value=1,max_value=365,value=21,help='Допущение. Уточните срок у поставщика.')
    review=st.number_input('До следующего заказа, дней',min_value=1,max_value=90,value=14)
    safety=st.number_input('Страховой запас, дней',min_value=0,max_value=90,value=7)
    growth=st.slider('Дополнительный прогноз прироста, %',-50,100,0,help='Ручной сценарий поверх тренда, оценённого по истории.')
    st.divider()
    remove=st.toggle('Исключать разовые всплески',True)
    seasonal=st.toggle('Учитывать сезонность',True)
    trend=st.toggle('Учитывать устойчивый рост',True)
    compensate=st.toggle('Восстанавливать упущенный спрос',True)
    approx=st.checkbox('Приближение по месячным остаткам',False,help='Только явный нулевой снимок. Это не точные даты отсутствия товара.')
    st.caption('Расчёт выполняется локально. API-ключи и платные подписки не нужны.')

st.markdown('<div class="eyebrow">AXIOMA / INVENTORY INTELLIGENCE</div>',unsafe_allow_html=True)
st.title('Заказывайте то, что нужно.')
st.markdown('<div class="intro">Продажи, склад и ожидаемые поставки — в одном расчёте. Каждая рекомендация сопровождается понятным объяснением.</div>',unsafe_allow_html=True)

if mode=='Показать пример':
    ds=deepcopy(load_demo())
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

dataset_key=fingerprint(ds.supplier,ds.sources)
saved_products=store.preferences(dataset_key,'products',[])
if saved_products:
    ds.products,_,_=apply_updates(ds.products,pd.DataFrame(saved_products))
anomaly_decisions=store.preferences(dataset_key,'anomalies',{})
if 'lead_days' not in ds.products:ds.products['lead_days']=np.nan
saved_stockouts=store.preferences(dataset_key,'stockouts',[])
if saved_stockouts:
    ds.stockouts=pd.DataFrame(saved_stockouts)
    for c in ['start','end']:ds.stockouts[c]=pd.to_datetime(ds.stockouts[c])

with st.expander('Данные и допущения · проверить перед заказом',expanded=False):
    st.dataframe(pd.DataFrame(ds.sources),hide_index=True,width='stretch')
    for warning in ds.warnings:st.warning(warning)
    st.caption('Для расчёта используются завершённые месяцы. Отрицательные продажи сохраняются как корректировки; отрицательный спрос не прогнозируется. Срок поставки и страховой запас задаются пользователем.')
    missing=ds.products.stock.isna().sum()
    st.markdown(f'**Актуальный остаток не указан: {missing} позиций.** Можно исправить таблицу ниже или загрузить CSV.')
    st.download_button('Скачать шаблон недостающих остатков · CSV',stock_template(ds.products),file_name='axioma-stock-template.csv',mime='text/csv')
    st.caption('В шаблоне уже стоят коды товаров. Заполните stock — свободный остаток, lead_days — согласованный срок поставки. Пустое поле не превращается в ноль.')
    st.caption('CSV остатков: sku,stock,category,pack,moq. Обязательны sku и stock; коды храните текстом. pack — кратность в единицах учёта. Для бухт переведите кратность в метры.')
    overrides=st.file_uploader('Актуальные остатки и параметры товаров',type=['csv'])
    if overrides:
        try:
            extra=pd.read_csv(overrides,dtype={'sku':str})
            ds.products,matched,unknown=apply_updates(ds.products,extra)
            st.caption(f'Обновлено {matched} товаров; не найдено {len(unknown)}.')
        except Exception as exc:st.error(f'CSV не применён: {exc}')
    changed=st.data_editor(ds.products.rename(columns=LABELS),disabled=['Код 1С','Артикул','Товар','Ед.'],hide_index=True,width='stretch',key=f'products-{mode}-{supplier}',column_config={'Свободный остаток':st.column_config.NumberColumn(min_value=0),'Кратность':st.column_config.NumberColumn(min_value=1),'Минимальная партия':st.column_config.NumberColumn(min_value=1),'Прирост поставщика, доля':st.column_config.NumberColumn(min_value=-.9,max_value=3.,help='0.2 означает +20%. Если задан, заменяет тренд из истории.')})
    ds.products=changed.rename(columns={v:k for k,v in LABELS.items()})
    if st.button('Сохранить остатки и параметры на этом ноутбуке'):
        try:
            ds.products,_,_=apply_updates(ds.products,ds.products)
            store.save_preferences(dataset_key,'products',records(ds.products[['sku']+[c for c in ['stock','category','pack','moq','lead_days','source_growth'] if c in ds.products]]))
            st.success('Сохранено. Эти значения восстановятся при повторном открытии тех же исходных файлов.')
        except Exception as exc:st.error(str(exc))
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
    if not ds.stockouts.empty and st.button('Сохранить периоды отсутствия на этом ноутбуке'):
        store.save_preferences(dataset_key,'stockouts',records(ds.stockouts));st.success('Периоды сохранены для текущих исходных данных.')
    st.markdown('**Правила категорий**')
    st.caption('Множитель страхового запаса. По умолчанию 1 для всех: бизнес-значение кодов категорий не предоставлено. Установите согласованные с менеджером правила.')
    categories=sorted(ds.products.category.astype(str).unique())
    cat_frame=st.data_editor(pd.DataFrame({'Категория':categories,'Множитель':[1.]*len(categories)}),disabled=['Категория'],hide_index=True,key=f'cat-{mode}-{supplier}',column_config={'Множитель':st.column_config.NumberColumn(min_value=0.,max_value=5.)})
    category_factors=dict(zip(cat_frame['Категория'],cat_frame['Множитель']))

stock_evidence=pd.DataFrame()
if ds.products.stock.isna().any() and not ds.stocks.empty:
    scenario=st.radio('Расчёт при неизвестных остатках',['Только известные остатки','Сценарий: остаток на начало месяца','Сценарий: вычесть продажи без поступлений'],horizontal=True)
    if scenario!='Только известные остатки':
        ds,stock_evidence=stock_scenario(ds,str(as_of),'reference' if scenario=='Сценарий: остаток на начало месяца' else 'depletion')
        st.warning(f'Сценарий для {len(stock_evidence)} товаров: это предполагаемые остатки, а не текущий свободный склад.')
        with st.expander('Основания сценарных остатков'):st.dataframe(stock_evidence.rename(columns=LABELS),hide_index=True,width='stretch')
cfg=Settings(str(as_of),int(lead),int(review),int(safety),float(growth),remove,seasonal,trend,compensate,approx,category_factors)
cfg.forecast_method=method;cfg.anomaly_decisions=anomaly_decisions;cfg.use_source_growth=use_source_growth
with st.spinner('Считаю спрос и заказы…'):result,histories,anomalies=run(ds,cfg,ENGINE_VERSION)
if not stock_evidence.empty:result=annotate_scenario(result,stock_evidence)
if result.empty:st.warning('Нет позиций для расчёта');st.stop()
result=result.assign(_priority=result.status.map({'Срочно':0,'Заказать':1,'Нужен остаток':2,'Нет истории':3,'Достаточно':4})).sort_values(['_priority','sku']).drop(columns='_priority').reset_index(drop=True)
cards=st.columns(4)
cards[0].metric('Товаров в анализе',len(result))
cards[1].metric('Нужно заказать',int(result.recommended.gt(0).sum()))
cards[2].metric('Риск дефицита',int(result.status.eq('Срочно').sum()))
cards[3].metric('Нужны данные',int(result.recommended.isna().sum()))
run_signature=hashlib.sha256(canonical({'dataset':dataset_key,'config':asdict(cfg),'rows':records(result)}).encode()).hexdigest()
orders_tab,detail_tab,anomaly_tab,backtest_tab,history_tab,quality_tab,evidence_tab=st.tabs(['Рекомендации','Почему столько','Разовые продажи','Проверка прогноза','История','Качество данных','Проверка кейса'])
with orders_tab:
    urgent=result[result.status.eq('Срочно')]
    if len(urgent):st.warning(f'{len(urgent)} позиций требуют ускорения поставки или перемещения: обычный заказ не успеет до дефицита. Даты и объёмы — в «Почему столько».')
    a,b=st.columns([2,1]);search=a.text_input('Найти товар',placeholder='Название, артикул или код')
    statuses=b.multiselect('Статус',list(result.status.unique()),default=list(result.status.unique()))
    selected=result[result.status.isin(statuses)].copy()
    if search:selected=selected[selected[['sku','article','name']].astype(str).apply(lambda s:s.str.contains(search,case=False,regex=False)).any(axis=1)]
    chosen_categories=st.multiselect('Категории',categories)
    if chosen_categories:selected=selected[selected.category.astype(str).isin(chosen_categories)]
    columns=['status','sku','name','recommended','unit','first_shortage','stock','in_transit','forecast','stock_threshold','stock_basis','article']
    table=selected[[c for c in columns if c in selected]].copy()
    if 'first_shortage' in table:table['first_shortage']=table.first_shortage.fillna('—')
    st.dataframe(table.rename(columns=LABELS),hide_index=True,width='stretch')
    st.download_button('Скачать все рекомендации · Excel',export_excel(result,cfg),file_name=f'axioma-{as_of}.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    if st.button('Сохранить этот расчёт в историю'):
        if st.session_state.get('saved_signature')!=run_signature:
            st.session_state['saved_run']=store.save_run(ds.supplier,dataset_key,cfg,result,ds.sources)
            st.session_state['saved_signature']=run_signature
        st.success('Сохранён расчёт '+st.session_state['saved_run'])
    with st.expander('Проверить и утвердить заказ'):
        draft=result[result.recommended.gt(0)].copy()
        if draft.empty:st.info('Нет положительных рекомендаций для утверждения.')
        else:
            draft=draft[['sku','article','name','supplier','recommended','pack','moq']].copy();draft['Утвердить']=False;draft['К заказу']=draft.recommended;draft['Причина корректировки']=''
            draft=draft[['Утвердить','sku','К заказу','recommended','name','Причина корректировки','pack','moq','article','supplier']]
            st.caption('Отметьте товары слева. «К заказу» можно изменить; для отличия от рекомендации укажите причину.')
            reviewed=st.data_editor(draft.rename(columns=LABELS),hide_index=True,width='stretch',disabled=['Код 1С','Артикул','Товар','Поставщик','Рекомендация','Кратность','Минимальная партия'],column_config={'Утвердить':st.column_config.CheckboxColumn(width='small'),'К заказу':st.column_config.NumberColumn(min_value=0,width='small'),'Товар':st.column_config.TextColumn(width='medium')},key=f'approval-{run_signature[:16]}')
            approved=reviewed[reviewed['Утвердить']&reviewed['К заказу'].gt(0)]
            bad=approved[(approved['К заказу']<approved['Минимальная партия'])|~np.isclose(approved['К заказу']%approved['Кратность'],0)]
            if len(bad):st.error('Исправьте количество: оно должно соответствовать минимуму и кратности.')
            reviewer=st.text_input('Ответственный за утверждение',placeholder='Имя или рабочий идентификатор')
            order_note=st.text_input('Комментарий к заказу',placeholder='Например, пополнение на две недели')
            acknowledge=True if stock_evidence.empty else st.checkbox('Понимаю: это сценарный заказ с предполагаемыми остатками')
            if st.button('Утвердить и сохранить заказ',type='primary',disabled=approved.empty or not bad.empty or not acknowledge or not reviewer.strip()):
                try:
                    if st.session_state.get('saved_signature')!=run_signature:
                        st.session_state['saved_run']=store.save_run(ds.supplier,dataset_key,cfg,result,ds.sources)
                        st.session_state['saved_signature']=run_signature
                    selections=[{'sku':str(r['Код 1С']),'quantity':float(r['К заказу']),'reason':r['Причина корректировки']} for _,r in approved.iterrows()]
                    order_id=store.approve(st.session_state['saved_run'],selections,reviewer,order_note,acknowledge_scenario=acknowledge)
                    st.success('Заказ '+order_id+' сохранён в истории. Поставщику не отправлен.')
                except Exception as exc:st.error(str(exc))
            st.download_button('Скачать выбранные позиции · черновик',export_excel(approved,cfg),file_name='axioma-draft.xlsx',disabled=approved.empty or not bad.empty or not acknowledge)
            st.caption('Сохранение фиксирует исходный расчёт, количество, ответственного и причину изменения в локальной базе. Изменение расчёта сбрасывает выбор строк для утверждения.')
with detail_tab:
    sku=st.selectbox('Выберите товар',list(histories),format_func=lambda k:f"{k} · {result.set_index('sku').loc[k,'name']}") if histories else None
    if sku:
        row=result.set_index('sku').loc[sku];st.subheader(row['name']);st.write(row.reason)
        if row.expedite_need>0:st.error(f'До {row.order_arrival} не хватает до {row.expedite_need:g} {row.unit}. Ускорьте существующую поставку или согласуйте перемещение. Этот объём не нужно автоматически прибавлять к обычному заказу.')
        c1,c2,c3=st.columns(3);c1.metric('Рекомендовано',f'{row.recommended:g}' if pd.notna(row.recommended) else 'Нужен остаток');c2.metric('Порог пополнения',f'{row.stock_threshold:g}');c3.metric('Восстановлено спроса',row.lost)
        st.write('Выбранная модель:',row.model)
        if pd.notna(row.forecast_low) and row.calibration_months:
            st.caption(f'Ориентир спроса: {row.forecast_low:g}–{row.forecast_high:g} ед. Разброс построен по ошибкам {int(row.calibration_months)} прошлых месяцев; это не гарантия покрытия.')
        st.line_chart(histories[sku].set_index('Дата'),color=['#94A3B8','#0D9488','#F59E0B'])
        st.caption('Серый — фактические продажи; зелёный — без разовых всплесков; оранжевый — с компенсацией отсутствия товара.')
        st.write('Оценка отсутствия:',row.stockout_mode)
        st.caption(f'Источник прироста: {row.growth_source}. Сценарий по коэффициенту поставщика применяется только при включённом переключателе.')
        model_info=histories[sku].attrs.get('model_info',{})
        if model_info.get('candidates'):
            with st.expander('Почему выбран этот метод'):
                st.caption('Сравнение кандидатов внутри прошлой истории. Это подбор метода, не независимая проверка.')
                st.dataframe(pd.DataFrame(model_info['candidates']).rename(columns={'method':'Метод','mae_daily':'Средняя ошибка, ед./день'}),hide_index=True,width='stretch')
        future_frame=pd.DataFrame(histories[sku].attrs.get('future',[]))
        if not future_frame.empty and pd.notna(row.stock):
            future_frame['Дата']=pd.to_datetime(future_frame['Дата'])
            transit=ds.transit[ds.transit.sku.eq(sku)]
            arrivals=transit.groupby('eta').quantity.sum() if not transit.empty else pd.Series(dtype=float)
            future_frame['Ожидаемые поступления']=future_frame['Дата'].map(arrivals).fillna(0).clip(lower=0)
            future_frame['Без нового заказа']=row.stock+(future_frame['Ожидаемые поступления']-future_frame['Прогноз в день']).cumsum()
            future_frame['С новым заказом']=future_frame['Без нового заказа']+np.where(future_frame['Дата']>=pd.Timestamp(as_of)+pd.Timedelta(days=int(row.lead_days)),row.recommended,0)
            st.markdown('**Что будет с запасом по дням**')
            st.line_chart(future_frame.set_index('Дата')[['Без нового заказа','С новым заказом']],color=['#EF4444','#0D9488'])
            st.caption('Отрицательный баланс показывает неудовлетворённую потребность. Поступления учитываются в начале дня; новый заказ — через срок поставки этого товара.')
            if pd.notna(row.first_shortage):st.warning(f'Риск дефицита с {row.first_shortage}, до прихода нового заказа. Нужна ускоренная поставка или перемещение.')
            if cfg.forecast_method=='adaptive' and row.cv_months>=3:
                st.info(f'Стресс-сценарий: {row.stress_recommended:g} ед. к заказу, если спрос превысит прогноз на 90-й процентиль прошлых положительных ошибок. Это ориентир, не гарантированный уровень сервиса.')
with anomaly_tab:
    st.subheader('Отделить разовый проект от регулярного спроса')
    st.write('Проверяется повторяемость крупных покупок клиента и устойчивый поток крупных накладных по товару. Три поставки одного проекта за несколько дней не считаются регулярным спросом. Решение по каждому событию можно изменить.')
    if anomalies.empty:st.info('Нет событий для проверки. При отсутствии ID клиента анализ ограничен накладными.')
    else:
        anomaly_sku=st.selectbox('Товар для проверки разовых продаж',sorted(anomalies.sku.unique()),key='anomaly-sku')
        subset=anomalies[anomalies.sku.eq(anomaly_sku)].copy()
        decision_labels={'auto':'Автоматически','regular':'Регулярная продажа','one_off':'Разовая продажа'}
        display=subset[['event_id','date','document','original','regular','excluded','reason','application','decision']].rename(columns={'event_id':'Событие','date':'Дата','document':'Документ','original':'Факт','regular':'Регулярная часть накладной','excluded':'Кандидат на исключение','reason':'Обоснование','decision':'Решение','application':'Сверка источников'})
        display['Решение']=display['Решение'].map(decision_labels)
        reviewed_events=st.data_editor(display,hide_index=True,width='stretch',disabled=[c for c in display if c!='Решение'],column_config={'Решение':st.column_config.SelectboxColumn(options=list(decision_labels.values()),required=True)},key=f'events-{dataset_key[:8]}-{anomaly_sku}')
        if st.button('Применить решения и пересчитать'):
            decisions=dict(anomaly_decisions);reverse={v:k for k,v in decision_labels.items()}
            for _,event in reviewed_events.iterrows():
                value=reverse[event['Решение']]
                if value=='auto':decisions.pop(event['Событие'],None)
                else:decisions[event['Событие']]=value
            store.save_preferences(dataset_key,'anomalies',decisions)
            st.rerun()
        st.caption('Решения сохраняются локально и привязаны к исходным файлам и событию. Исходные продажи не изменяются.')
        st.caption('Если месячный итог отличается от суммы накладных больше чем на 2% или 1 единицу, вычитание кандидата отключено: требуется сверка выгрузок. Решение менеджера не обходит эту проверку.')

with backtest_tab:
    st.subheader('Проверка на прошлых месяцах')
    st.write('На каждом шаге модель прогнозирует следующий месяц, используя только более раннюю историю. Сравнение проводится с неизменёнными фактическими продажами, включая разовые проекты.')
    b1,b2=st.columns(2)
    backtest_periods=b1.selectbox('Сколько месяцев проверить',[3,6],index=1)
    backtest_size=b2.selectbox('Объём проверки',['100 SKU по прошлому объёму','Все товары'],index=0)
    end_month=(pd.Timestamp(as_of).replace(day=1)-pd.DateOffset(months=1)).strftime('%Y-%m-01')
    test_key=f'{dataset_key}-{end_month}-{backtest_periods}-{backtest_size}'
    if st.button('Запустить проверку прогноза',type='primary'):
        with st.spinner('Сравниваю прогнозы с последующими продажами. Полный каталог может потребовать несколько минут…'):
            stats,detail,meta=benchmark(ds,end_month,backtest_periods,100 if backtest_size.startswith('100') else None,ENGINE_VERSION)
            st.session_state['benchmark_result']=(test_key,stats,detail,meta)
    saved_test=st.session_state.get('benchmark_result')
    if saved_test and saved_test[0]==test_key:
        _,stats,detail,meta=saved_test
        st.caption(f"Период {meta['start']}–{meta['end']}; {meta['sku_count']} SKU, {meta['observations']} наблюдений. Исключено из-за короткой истории: {meta['skipped']}.")
        if stats.empty:st.warning('Недостаточно завершённых месяцев для проверки.')
        else:
            shown=stats[['model','macro_wape','median_wape','scored_skus','zero_actual_skus','zero_actual_with_forecast']].copy();shown[['macro_wape','median_wape']]*=100
            st.dataframe(shown.rename(columns={'model':'Модель','macro_wape':'Средняя WAPE по SKU, %','median_wape':'Медианная WAPE по SKU, %','scored_skus':'SKU с положительным фактом','zero_actual_skus':'SKU с нулевым фактом','zero_actual_with_forecast':'Нулевой факт, есть прогноз'}),hide_index=True,width='stretch')
            st.bar_chart(stats.set_index('model')[['macro_wape']],color='#0D9488')
            st.caption('Для каждого товара: WAPE = сумма абсолютных ошибок / сумма продаж. Затем среднее по товарам с положительным фактом — каждый имеет равный вес. Нулевой факт показан отдельно. Ошибка может превышать 100%; меньше — лучше.')
            if 'unit' in detail:
                with st.expander('Ошибка по объёму внутри одной единицы учёта'):
                    unit=st.selectbox('Единица учёта для оценки',sorted(detail.unit.unique()))
                    by_unit=summarize(detail[detail.unit.eq(unit)])
                    unit_table=by_unit[['model','wape','bias']].copy();unit_table[['wape','bias']]*=100
                    st.dataframe(unit_table.rename(columns={'model':'Модель','wape':'WAPE по объёму, %','bias':'Смещение, %'}),hide_index=True,width='stretch')
                    st.caption('Здесь больший объём даёт больший вес. Штуки, метры и другие единицы не складываются. Это не денежная экономия.')
            per_sku=detail.groupby(['sku','method']).abs_error.mean().unstack('method')
            if 'axioma' in per_sku and 'mean6' in per_sku:
                win=int((per_sku.axioma<per_sku.mean6).sum());loss=int((per_sku.axioma>per_sku.mean6).sum())
                st.write(f'Axioma против среднего за 6 месяцев: лучше на {win} SKU, хуже на {loss}, равенство на {len(per_sku)-win-loss}.')
            st.download_button('Скачать подробную проверку · CSV',detail.to_csv(index=False).encode('utf-8-sig'),file_name='axioma-backtest.csv',mime='text/csv')
            st.info(meta['limitation'])
            st.caption('В проверке не применяются текущие коэффициенты поставщика, остатки, ретроспективные stockout-периоды и ручные решения. Эти месяцы используются для диагностики разработки; для независимой проверки нужен новый период.')
    else:st.caption('Проверка запускается по кнопке и не замедляет обычный пересчёт заказа.')
    with st.expander('Сравнить текущие рекомендации при разных допущениях'):
        st.caption('Это влияние настроек на заказ, а не доказательство точности или экономии.')
        if st.button('Рассчитать сравнение'):
            with st.spinner('Сравниваю расчёты при одинаковых остатках и сроках…'):
                simple=run(ds,replace(cfg,forecast_method='classic',remove_outliers=False,seasonality=False,trend=False,compensate_stockout=False),ENGINE_VERSION)[0]
                unclean=run(ds,replace(cfg,remove_outliers=False),ENGINE_VERSION)[0]
            comparison=result[['sku','name','recommended']].rename(columns={'recommended':'Выбранная модель'})
            comparison=comparison.merge(simple[['sku','recommended']].rename(columns={'recommended':'Обычное среднее'}),on='sku',validate='one_to_one')
            comparison=comparison.merge(unclean[['sku','recommended']].rename(columns={'recommended':'Без удаления всплесков'}),on='sku',validate='one_to_one')
            comparison['Влияние фильтра всплесков']=comparison['Выбранная модель']-comparison['Без удаления всплесков']
            st.dataframe(comparison.rename(columns=LABELS),hide_index=True,width='stretch')
            st.download_button('Скачать сравнение методов',export_excel(comparison,cfg),file_name='axioma-method-comparison.xlsx')

with history_tab:
    st.subheader('Сохранённые расчёты и заказы')
    runs=store.list_runs(ds.supplier)
    if runs:
        st.dataframe(pd.DataFrame(runs)[['id','created_at','supplier']].rename(columns={'id':'Расчёт','created_at':'Создан (UTC)','supplier':'Поставщик'}),hide_index=True,width='stretch')
        chosen_run=st.selectbox('Открыть сохранённый расчёт',[r['id'] for r in runs])
        snapshot=store.get_run(chosen_run)
        st.download_button('Скачать сохранённый расчёт',export_excel(pd.DataFrame(snapshot['rows']),Settings(**snapshot['settings'])),file_name=f'{chosen_run}.xlsx')
    else:st.info('Сохраните текущий расчёт на вкладке «Рекомендации».')
    orders=store.list_orders(ds.supplier)
    if orders:
        st.dataframe(pd.DataFrame([{k:o[k] for k in ['id','run_id','created_at','reviewer','note']} for o in orders]).rename(columns={'id':'Заказ','run_id':'Расчёт','created_at':'Утверждён (UTC)','reviewer':'Ответственный','note':'Комментарий'}),hide_index=True,width='stretch')
        order_id=st.selectbox('Открыть утверждённый заказ',[o['id'] for o in orders])
        chosen_order=next(o for o in orders if o['id']==order_id)
        order_frame=pd.DataFrame(chosen_order['rows'])
        order_frame['order_id']=order_id;order_frame['reviewer']=chosen_order['reviewer'];order_frame['approved_at']=chosen_order['created_at'];order_frame['run_id']=chosen_order['run_id']
        st.dataframe(order_frame[['sku','name','recommended','approved_quantity','adjustment_reason']].rename(columns=LABELS),hide_index=True,width='stretch')
        order_run=store.get_run(chosen_order['run_id'])
        st.download_button('Скачать сохранённый утверждённый заказ',export_excel(order_frame,Settings(**order_run['settings'])),file_name=f'{order_id}.xlsx')
    else:st.caption('Утверждённых заказов пока нет.')
    st.caption('История хранится в локальной SQLite-базе этого ноутбука. Сохранённые расчёты не меняются после правок исходных данных. Автоматическая отправка поставщикам отсутствует.')
    all_orders=store.list_orders()
    if all_orders:
        st.subheader('Общая корзина поставщиков')
        order_map={o['id']:o for o in all_orders}
        cart_ids=st.multiselect('Включить сохранённые заказы в корзину',list(order_map),format_func=lambda k:f"{order_map[k]['supplier']} · {k} · {order_map[k]['created_at'][:16]}")
        if cart_ids:
            cart=pd.DataFrame()
            for oid in sorted(cart_ids,key=lambda k:order_map[k]['created_at']):
                o=order_map[oid];lines=pd.DataFrame(o['rows']).copy()
                lines['quantity']=lines.approved_quantity;lines['order_id']=oid;lines['reviewer']=o['reviewer'];lines['approved_at']=o['created_at']
                cart=add_reviewed_lines(cart,lines)
            st.caption('Если один товар утверждён повторно у того же поставщика, в корзине используется наиболее позднее выбранное утверждение. История всех заказов сохранена.')
            for supplier_name,group in cart.groupby('supplier'):
                with st.expander(f'{supplier_name} · {len(group)} позиций',expanded=True):
                    st.dataframe(group[['sku','name','quantity','unit','order_id','reviewer']].rename(columns=LABELS),hide_index=True,width='stretch')
            st.download_button('Скачать общую корзину · Excel',export_excel(cart,{'Заказы':', '.join(cart_ids)}),file_name='axioma-supplier-orders.xlsx')

with quality_tab:
    st.subheader('Прозрачный расчёт')
    st.write('Потребность = прогноз на срок поставки и интервал пересмотра + страховой запас − свободный остаток − поступления в этот период. Дополнительно проверяется баланс по дням: поздняя поставка не должна скрывать дефицит после прихода нового заказа. Положительный результат округляется с учётом минимума и кратности.')
    st.write('Крупные накладные и всплески клиента за день проверяются устойчивым порогом по распределению объёмов. Подозрительный объём заменяется типичным; исходные файлы не изменяются.')
    st.dataframe(pd.DataFrame(ds.sources),hide_index=True,width='stretch')
    mismatched=result[result.get('source_mismatch_months',pd.Series(0,index=result.index)).fillna(0).gt(0)]
    if len(mismatched):
        st.warning(f'У {len(mismatched)} товаров расходятся месячная ведомость и сумма накладных. Для несовпадающих месяцев сохранён месячный итог; автоматическое вычитание всплесков заблокировано.')
        st.dataframe(mismatched[['sku','name','source_mismatch_months']].rename(columns=LABELS),hide_index=True,width='stretch')
    if not anomalies.empty:st.markdown('**Обнаруженные всплески**');st.dataframe(anomalies,hide_index=True,width='stretch')
    else:st.info('Всплески не обнаружены либо нет детальных продаж для их проверки.')
    st.warning('Неизвестный остаток блокирует заказ. Неизвестные или просроченные даты поступления требуют уточнения. Без client_id невозможно проверить покупки одного клиента между разными накладными.')
    st.caption('Прототип HackAlem AI · AI-агент Codex использован при разработке. Расчёт детерминированный; внешние AI API не вызываются.')

with evidence_tab:
    st.subheader('Пять обязательных требований — проверка в один клик')
    st.write('Изменяем по одному входному условию и показываем результат до и после. Это воспроизводимые синтетические примеры, не оценка экономии или точности на данных компании.')
    st.caption('Для изоляции арифметики эти примеры используют исходную модель с указанными переключателями. Современные модели также покрыты тестами; реальный прогноз проверяется отдельно.')
    if st.button('Запустить проверку требований',type='primary'):
        from src.evidence import acceptance_examples
        st.session_state['acceptance_evidence']=acceptance_examples()
    if 'acceptance_evidence' in st.session_state:
        checks=st.session_state['acceptance_evidence'];passed=int(checks['Результат'].eq('Пройдено').sum())
        if passed==len(checks):st.success(f'{passed} из {len(checks)} проверок пройдено: все пять обязательных требований представлены.')
        else:st.error(f'Пройдено {passed} из {len(checks)}: проверьте строки с ошибкой.')
        st.dataframe(checks,hide_index=True,width='stretch')
        st.download_button('Скачать протокол проверки · CSV',checks.to_csv(index=False).encode('utf-8-sig'),file_name='axioma-case-checks.csv',mime='text/csv')
    st.markdown('**Пилот у Электрокомплект**')
    st.write('Две недели параллельного расчёта с менеджером: измерять время подготовки заказа, долю ручных правок и дни дефицита. Для этого нужны ежедневные свободные остатки, обезличенный ID клиента, периоды отсутствия и согласованные сроки. Денежный эффект — после получения цен и фактических затрат.')
