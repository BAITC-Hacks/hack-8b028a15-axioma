from streamlit.testing.v1 import AppTest
from pathlib import Path

def test_demo_and_toggles():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    assert not app.exception
    assert app.metric[0].value=='6'
    app.toggle[0].set_value(False).run()
    assert not app.exception
    app.text_input[0].set_value('DEMO-001').run()
    assert not app.exception


def test_comparison_runs_without_error():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    next(b for b in app.button if b.label=='Рассчитать сравнение').click().run()
    assert not app.exception
    tables=[d.value for d in app.dataframe]
    assert any('Влияние фильтра всплесков' in t.columns for t in tables)


def test_historical_validation_ui():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    next(b for b in app.button if b.label=='Проверить прогноз на истории').click().run()
    assert not app.exception
    assert any(m.label=='Средняя ошибка Axioma по товарам' for m in app.metric)


def test_baseline_method_can_be_selected():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    next(s for s in app.selectbox if s.label=='Метод расчёта').set_value('Обычное среднее за 6 месяцев').run()
    assert not app.exception
    assert any('Выбрано обычное среднее' in i.value for i in app.info)


def test_existing_cart_renders_and_can_be_exported():
    import pandas as pd
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90)
    app.session_state['order_cart']=pd.DataFrame([dict(supplier='IEK',sku='A',quantity=12,pack=6,moq=10,reason='Проверенный пример',calculated_at='2026-09-23',planning_method='Test')])
    app.run()
    assert not app.exception
    assert any(s.value=='Общая корзина поставщиков' for s in app.subheader)
    next(b for b in app.button if b.label=='Очистить корзину этой сессии').click().run()
    assert not app.exception
    assert app.session_state['order_cart'].empty


def test_case_acceptance_is_visible_and_all_examples_pass():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    next(b for b in app.button if b.label=='Запустить проверку требований').click().run()
    assert not app.exception
    checks=app.session_state['acceptance_evidence']
    assert len(checks)==11
    assert checks['Результат'].eq('Пройдено').all()


def test_approval_requires_reviewer_and_resets_after_input_change(monkeypatch):
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    editor=next(d for d in app.dataframe if 'Утвердить' in d.value.columns)
    old_id=editor.proto.id
    # AppTest has no data_editor editing API; emulate its reviewed return value.
    import streamlit as st
    original_editor=st.data_editor
    old_key=old_id.split('-approval-',1)[1]
    def review_first(data,*args,**kwargs):
        value=original_editor(data,*args,**kwargs)
        if kwargs.get('key')=='approval-'+old_key:
            value=value.copy();value.loc[value.index[0],'Утвердить']=True
        return value
    monkeypatch.setattr(st,'data_editor',review_first)
    app.run()
    add=lambda:next(b for b in app.button if b.label=='Добавить утверждённые позиции в общую корзину')
    assert add().disabled
    next(i for i in app.text_input if i.label=='Кто проверил заказ').set_value('Тестовый менеджер').run()
    assert not add().disabled
    add().click().run()
    assert not app.exception
    cart=app.session_state['order_cart']
    assert len(cart)==1 and cart.iloc[0].reviewer=='Тестовый менеджер'
    assert 'lead_days' in cart.iloc[0].planning_parameters
    app.toggle[0].set_value(False).run()
    assert not app.exception
    new_id=next(d for d in app.dataframe if 'Утвердить' in d.value.columns).proto.id
    assert old_id!=new_id
    assert add().disabled
    assert app.session_state['order_cart'].equals(cart)


def test_workbench_count_and_delay_flow():
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    next(c for c in app.checkbox if c.label=='Показать задержку на 7 дней').check().run()
    assert not app.exception
    assert any('Стали срочными: 1 товаров' in m.value for m in app.markdown)
    next(r for r in app.radio if r.label=='Выберите действие').set_value('Проверить остаток').run()
    next(s for s in app.selectbox if s.label=='Открыть решение по товару').set_value('DEMO-002').run()
    assert not app.exception
    assert any('Решение зависит от допущений' in m.value for m in app.markdown)
    next(n for n in app.number_input if n.label=='Подтверждённый свободный остаток · DEMO-002').set_value(20).run()
    next(b for b in app.button if b.label=='Применить уточнённый остаток').click().run()
    assert not app.exception
    next(r for r in app.radio if r.label=='Выберите действие').set_value('Подготовить заказ').run()
    draft=next(d.value for d in app.dataframe if 'Утвердить' in d.value.columns)
    row=draft.set_index('Код 1С').loc['DEMO-002']
    assert row['Рекомендация']>0
    assert row['Основание остатка']=='Ручной пересчёт на 2026-09-23'


def test_approved_excel_contains_scenario_lineage_and_pack(monkeypatch):
    from io import BytesIO
    import pandas as pd
    import streamlit as st
    captured={}
    real_download=st.download_button
    real_editor=st.data_editor
    def reviewed(data,*args,**kwargs):
        value=real_editor(data,*args,**kwargs)
        if str(kwargs.get('key','')).startswith('approval-'):
            value=value.copy();value.loc[value.index[0],'Утвердить']=True
        return value
    def capture(label,data,*args,**kwargs):
        captured[label]=(data,kwargs)
        return real_download(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'data_editor',reviewed)
    monkeypatch.setattr(st,'download_button',capture)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    next(i for i in app.text_input if i.label=='Кто проверил заказ').set_value('Демо менеджер').run()
    assert not app.exception
    payload,options=captured['Скачать утверждённые позиции']
    assert not options['disabled']
    frame=pd.read_excel(BytesIO(payload))
    assert len(frame)==1
    row=frame.iloc[0]
    assert row['К заказу']%row['Кратность']==0 and row['К заказу']>=row['Минимальная партия']
    assert row['Проверил']=='Демо менеджер'
    for col in ['Устойчивость решения','Минимум заказа в сценариях','Максимум заказа в сценариях','Границы проверенной сетки','Основания сценарных остатков','Основание остатка','Версия входных файлов','Источники данных','Версия расчёта']:
        assert col in frame and pd.notna(row[col])
    assert 'scenario_grid' in row['Параметры расчёта']


def test_no_history_product_can_be_opened_in_count_queue(monkeypatch):
    from src.data import demo_data
    import src.data
    ds=demo_data();ds.sales=ds.sales.iloc[:0];ds.transactions=ds.transactions.iloc[:0]
    monkeypatch.setattr(src.data,'demo_data',lambda:ds)
    import streamlit as st
    st.cache_data.clear()
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    next(r for r in app.radio if r.label=='Выберите действие').set_value('Проверить остаток').run()
    assert not app.exception
    assert any('Нет завершённой истории продаж' in m.value for m in app.markdown)
    st.cache_data.clear()


def test_xlsx_upload_to_count_to_reviewed_export(monkeypatch):
    """The real uploader return shape is emulated; XLSX parsing/export is real."""
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
        real=real_upload(label,*args,**kwargs)
        return [uploaded] if label.startswith('Загрузите ZIP') else real
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
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    next(r for r in app.radio if r.label=='Источник данных').set_value('Файлы поставщика').run()
    assert not app.exception and app.metric[0].value=='1'
    next(r for r in app.radio if r.label=='Выберите действие').set_value('Проверить остаток').run()
    assert any('Недостаточно данных для сценариев' in m.value for m in app.markdown)
    next(n for n in app.number_input if n.label=='Подтверждённый свободный остаток · UPLOAD-001').set_value(20).run()
    next(b for b in app.button if b.label=='Применить уточнённый остаток').click().run()
    next(i for i in app.text_input if i.label=='Кто проверил заказ').set_value('Проверка загрузки').run()
    assert not app.exception
    payload,options=downloads['Скачать утверждённые позиции']
    assert not options['disabled']
    frame=pd.read_excel(BytesIO(payload))
    assert frame.iloc[0]['Код 1С']=='UPLOAD-001'
    assert frame.iloc[0]['К заказу']>0
    assert 'synthetic-transactions.xlsx' in frame.iloc[0]['Источники данных']
    assert frame.iloc[0]['Основание остатка']=='Ручной пересчёт на 2026-09-23'


def test_inventory_effect_ui_export_and_stale_result_invalidation(monkeypatch):
    import streamlit as st
    import zipfile
    import json
    from io import BytesIO
    exports={};download=st.download_button
    def capture(label,data,*args,**kwargs):
        exports[label]=data
        return download(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'download_button',capture)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    next(b for b in app.button if b.label=='Сравнить политики запасов').click().run()
    assert not app.exception
    assert any('Обслужено сразу, % · среднее по SKU' in d.value.columns for d in app.dataframe)
    with zipfile.ZipFile(BytesIO(exports['Скачать протокол симуляции · ZIP'])) as z:
        metadata=json.loads(z.read('metadata.json'))
        assert metadata['data_type']=='Синтетические данные' and metadata['evaluated_skus']==6
        assert metadata['input_batch'] and 'forecast_diagnostics.csv' in z.namelist()
    next(c for c in app.checkbox if c.label=='Рассчитать сценарные затраты по моим ставкам').check().run()
    assert not app.exception
    next(i for i in app.text_input if i.label=='Источник ставок и дата согласования').set_value('Синтетические ставки теста').run()
    assert not app.exception
    with zipfile.ZipFile(BytesIO(exports['Скачать протокол симуляции · ZIP'])) as z:
        assert 'scenario_costs.csv' in z.namelist()
    next(n for n in app.number_input if n.label=='Срок поставки в симуляции, дней').set_value(28).run()
    assert not app.exception
    assert any('прежние результаты скрыты' in i.value for i in app.info)
    assert not any('Обслужено сразу, % · среднее по SKU' in d.value.columns for d in app.dataframe)
