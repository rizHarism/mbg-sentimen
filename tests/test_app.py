from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_berjalan_tanpa_error():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert len(at.sidebar.date_input) >= 1
    assert len(at.sidebar.checkbox) >= 1


def test_contoh_komentar_per_kelas_tersedia():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert "Cara memilih contoh" in [box.label for box in at.selectbox]
    assert len(at.slider) >= 2          # slider contoh + slider maks komentar/video
    assert len(at.button) >= 1
