from test_engine import baseline
from test_app import open_app
from src.engine import Settings,calculate
from src.robustness import ScenarioSettings,analyse_scenarios


def test_individual_lead_time_matches_scenario_order():
    ds=baseline();ds.products['lead_days']=7;ds.products['stock']=80
    cfg=Settings(forecast_method='auto')
    result,histories,_=calculate(ds,cfg)
    summary,detail,comparison=analyse_scenarios(ds,cfg,result,histories,ScenarioSettings(0,0,0))
    assert detail.iloc[0].scenario_order==result.iloc[0].recommended
    assert comparison.iloc[0].delay_order_before==result.iloc[0].recommended
    assert '7 дн. неизменен' in summary.iloc[0].scenario_assumptions


def test_scenario_recount_persists_for_same_date_and_changes_order(tmp_path,monkeypatch):
    from src.storage import Store
    app=open_app(tmp_path,monkeypatch)
    next(c for c in app.checkbox if c.label=='Показать задержку на 7 дней').check().run()
    assert not app.exception
    next(r for r in app.radio if r.label=='Выберите действие').set_value('Проверить остаток').run()
    next(s for s in app.selectbox if s.label=='Открыть решение по товару').set_value('DEMO-002').run()
    next(n for n in app.number_input if n.label=='Подтверждённый свободный остаток · DEMO-002').set_value(20).run()
    next(b for b in app.button if b.label=='Применить уточнённый остаток').click().run()
    assert not app.exception
    next(b for b in app.button if b.label=='Сохранить этот расчёт в историю').click().run()
    saved=Store(tmp_path/'ui.sqlite3');run=saved.get_run(saved.list_runs()[0]['id'])
    row=next(r for r in run['rows'] if r['sku']=='DEMO-002')
    assert row['stock']==20 and row['recommended']>0
    assert row['stock_basis']=='Ручной пересчёт на 2026-09-23'
    assert 'scenario_assumptions' in row and row['scenario_count']>0
    app=open_app(tmp_path,monkeypatch)
    assert not app.exception
    assert saved.preferences(run['dataset_key'],'counts:2026-09-23')['DEMO-002']==20
    assert saved.preferences(run['dataset_key'],'counts:2026-09-24',{})=={}


def test_no_history_products_are_safe_in_scenario_workspace(tmp_path,monkeypatch):
    from src.data import demo_data
    import src.data
    import streamlit as st
    ds=demo_data();ds.sales=ds.sales.iloc[:0];ds.transactions=ds.transactions.iloc[:0]
    monkeypatch.setattr(src.data,'demo_data',lambda:ds)
    st.cache_data.clear()
    try:
        app=open_app(tmp_path,monkeypatch)
        next(r for r in app.radio if r.label=='Выберите действие').set_value('Проверить остаток').run()
        assert not app.exception
        assert any('Нет завершённой истории продаж' in m.value for m in app.markdown)
    finally:st.cache_data.clear()


def test_xlsx_import_count_approval_and_audited_export(tmp_path,monkeypatch):
    from io import BytesIO
    import pandas as pd
    import streamlit as st
    from test_importers import workbook
    uploaded=BytesIO(workbook([
        ['Дата','Номер','Документ','Код','Номенклатура','Ед.','Склад','Количество'],
        ['15.07.2026','D1','Продажа','UPLOAD-001','Тест загрузки','шт','Склад',310],
        ['15.08.2026','D2','Продажа','UPLOAD-001','Тест загрузки','шт','Склад',310],
    ]));uploaded.name='synthetic-transactions.xlsx'
    real_upload=st.file_uploader;real_editor=st.data_editor;real_download=st.download_button
    downloads={}
    def upload(label,*args,**kwargs):
        value=real_upload(label,*args,**kwargs)
        return [uploaded] if label.startswith('Загрузите ZIP') else value
    def review(data,*args,**kwargs):
        value=real_editor(data,*args,**kwargs)
        if str(kwargs.get('key','')).startswith('approval-'):
            value=value.copy();value.loc[value.index[0],'Утвердить']=True
        return value
    def capture(label,data,*args,**kwargs):
        downloads[label]=(data,kwargs)
        return real_download(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'file_uploader',upload)
    monkeypatch.setattr(st,'data_editor',review)
    monkeypatch.setattr(st,'download_button',capture)
    app=open_app(tmp_path,monkeypatch)
    next(r for r in app.radio if r.label=='Источник данных').set_value('Файлы поставщика').run()
    assert not app.exception and app.metric[0].value=='1'
    next(r for r in app.radio if r.label=='Выберите действие').set_value('Проверить остаток').run()
    next(n for n in app.number_input if n.label=='Подтверждённый свободный остаток · UPLOAD-001').set_value(20).run()
    next(b for b in app.button if b.label=='Применить уточнённый остаток').click().run()
    next(i for i in app.text_input if i.label=='Ответственный за утверждение').set_value('Проверка загрузки').run()
    next(b for b in app.button if b.label=='Утвердить и сохранить заказ').click().run()
    assert not app.exception
    payload,_=downloads['Скачать сохранённый утверждённый заказ']
    frame=pd.read_excel(BytesIO(payload));row=frame.iloc[0]
    assert len(frame)==1 and row['Код 1С']=='UPLOAD-001'
    assert row['Утверждено']>0 and row['Утверждено']%row['Кратность']==0
    assert row['Проверил']=='Проверка загрузки'
    assert row['Основание остатка']=='Ручной пересчёт на 2026-09-23'
    assert 'synthetic-transactions.xlsx' in row['Источники данных']
    for col in ['Устойчивость решения','Границы проверенной сетки','Заказ','Расчёт']:
        assert col in frame and pd.notna(row[col])
