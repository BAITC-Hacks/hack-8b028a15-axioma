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
