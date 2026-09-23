from streamlit.testing.v1 import AppTest
from pathlib import Path

def test_demo_and_toggles(tmp_path,monkeypatch):
    monkeypatch.setenv('AXIOMA_DB',str(tmp_path/'ui.sqlite3'))
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py',default_timeout=90).run()
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
    assert app.metric[0].value=='6'
    app.toggle[0].set_value(False).run()
    assert not app.exception
    app.text_input[0].set_value('DEMO-001').run()
    assert not app.exception
