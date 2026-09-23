from streamlit.testing.v1 import AppTest
from pathlib import Path

def test_demo_and_toggles(tmp_path,monkeypatch):
    monkeypatch.setenv('AXIOMA_DB',str(tmp_path/'ui.sqlite3'))
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
    assert not app.exception
    assert app.metric[0].value=='6'
    app.toggle[0].set_value(False).run()
    assert not app.exception
    app.text_input[0].set_value('DEMO-001').run()
    assert not app.exception


def open_app(tmp_path,monkeypatch):
    monkeypatch.setenv('AXIOMA_DB',str(tmp_path/'ui.sqlite3'))
    return AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()


def test_comparison_runs_without_error(tmp_path,monkeypatch):
    app=open_app(tmp_path,monkeypatch)
    next(b for b in app.button if b.label=='Рассчитать сравнение').click().run()
    assert not app.exception
    assert any('Влияние фильтра всплесков' in d.value.columns for d in app.dataframe)


def test_historical_validation_ui(tmp_path,monkeypatch):
    app=open_app(tmp_path,monkeypatch)
    next(b for b in app.button if b.label=='Запустить проверку прогноза').click().run()
    assert not app.exception
    assert any('Средняя WAPE по SKU, %' in d.value.columns for d in app.dataframe)


def test_all_forecast_modes_render(tmp_path,monkeypatch):
    app=open_app(tmp_path,monkeypatch)
    for method in ['mean6','pooled','adaptive','classic','auto']:
        next(s for s in app.selectbox if s.label=='Метод прогноза').set_value(method).run()
        assert not app.exception


def test_saved_calculation_survives_new_session(tmp_path,monkeypatch):
    from src.storage import Store
    app=open_app(tmp_path,monkeypatch)
    next(b for b in app.button if b.label=='Сохранить этот расчёт в историю').click().run()
    assert not app.exception
    saved=Store(tmp_path/'ui.sqlite3').list_runs()
    assert len(saved)==1
    app=open_app(tmp_path,monkeypatch)
    assert not app.exception
    assert any(saved[0]['id'] in d.value.to_string() for d in app.dataframe)


def test_case_evidence_all_requirements_pass(tmp_path,monkeypatch):
    app=open_app(tmp_path,monkeypatch)
    next(b for b in app.button if b.label=='Запустить проверку требований').click().run()
    assert not app.exception
    table=next(d.value for d in app.dataframe if 'Ожидаемое поведение' in d.value.columns)
    assert len(table)>=11 and table['Результат'].eq('Пройдено').all()


def test_reviewed_order_persists_and_input_change_resets_selection(tmp_path,monkeypatch):
    import streamlit as st
    from src.storage import Store
    app=open_app(tmp_path,monkeypatch)
    original=st.data_editor
    first_key=None
    def select_first(data,*args,**kwargs):
        nonlocal first_key
        value=original(data,*args,**kwargs)
        key=kwargs.get('key','')
        if key.startswith('approval-'):
            if first_key is None:first_key=key
            if key==first_key:
                value=value.copy();value.loc[value.index[0],'Утвердить']=True
        return value
    monkeypatch.setattr(st,'data_editor',select_first)
    app.run()
    submit=lambda:next(b for b in app.button if b.label=='Утвердить и сохранить заказ')
    assert submit().disabled
    next(i for i in app.text_input if i.label=='Ответственный за утверждение').set_value('Проверка интерфейса').run()
    assert not submit().disabled
    submit().click().run()
    assert not app.exception
    saved=Store(tmp_path/'ui.sqlite3').list_orders()
    assert len(saved)==1 and saved[0]['reviewer']=='Проверка интерфейса'
    assert len(saved[0]['rows'])==1
    app.toggle[0].set_value(False).run()
    assert not app.exception and submit().disabled
    app=open_app(tmp_path,monkeypatch)
    assert not app.exception
    assert any(saved[0]['id'] in d.value.to_string() for d in app.dataframe)
    cart=next(m for m in app.multiselect if m.label=='Включить сохранённые заказы в корзину')
    cart.set_value([saved[0]['id']]).run()
    assert not app.exception
    assert any('К заказу' in d.value.columns and 'Проверил' in d.value.columns for d in app.dataframe)


def test_inventory_effect_ui_export_and_stale_result_invalidation(tmp_path,monkeypatch):
    monkeypatch.setenv('AXIOMA_DB',str(tmp_path/'effect-ui.sqlite3'))
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
