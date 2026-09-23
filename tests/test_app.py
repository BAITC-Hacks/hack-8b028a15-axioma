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
