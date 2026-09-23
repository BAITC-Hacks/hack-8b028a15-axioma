from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_demo_loads_and_approval_requires_reviewer():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app_extended.py', default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == 'Закупки с понятным основанием'
    assert len(app.metric) == 4
    app.button[0].click().run()
    assert not app.exception
    assert any('ответственного' in e.value for e in app.error)
    app.text_input[0].set_value('Демо-менеджер').run()
    app.button[0].click().run()
    assert not app.exception
    assert any('утверждён' in e.value for e in app.success)
