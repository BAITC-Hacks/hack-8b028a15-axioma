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
