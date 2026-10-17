"""Підказки «?» і сторінка «Посібник»."""
import pytest
from streamlit.testing.v1 import AppTest

from ui.locales import uk

WIDGET_KINDS = ["selectbox", "slider", "number_input", "toggle", "multiselect", "text_input"]


# ---- словник -------------------------------------------------------------------------------
def test_every_help_entry_has_label_and_text():
    for key, e in uk.HELP.items():
        assert e["label"].strip() and e["what"].strip(), key
        assert set(e) <= {"label", "what", "effect", "default", "tip"}, key


def test_guide_groups_cover_exactly_the_help_keys():
    grouped = [k for _, keys in uk.FORM_GROUPS for k in keys]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(uk.HELP)


def test_locale_structure_is_complete():
    for name in ("UI", "HELP", "FORM_GROUPS", "TERMS", "GUIDE"):
        assert getattr(uk, name)
    assert {"intro", "steps", "results", "cautions", "faq", "formulas"} <= set(uk.GUIDE)


def test_unknown_language_falls_back_to_ukrainian(monkeypatch):
    from ui import i18n
    monkeypatch.setattr(i18n, "LANG", "xx")
    assert i18n.texts() is uk and i18n.entry("alpha")["label"] == uk.HELP["alpha"]["label"]


def test_tip_builds_text_and_respects_switch(monkeypatch):
    from ui import help as h
    monkeypatch.setattr(h, "hints_enabled", lambda: True)
    t = h.tip("alpha")
    assert "Що це." in t and "Типове значення." in t
    monkeypatch.setattr(h, "hints_enabled", lambda: False)
    assert h.tip("alpha") is None
    monkeypatch.setattr(h, "hints_enabled", lambda: True)
    with pytest.raises(KeyError):
        h.tip("немає такого")


def test_env_switch_default(monkeypatch):
    import importlib

    from ui import settings
    monkeypatch.setenv("CLEANTEXT_HINTS", "0")
    assert importlib.reload(settings).HINTS_DEFAULT is False
    monkeypatch.delenv("CLEANTEXT_HINTS")
    assert importlib.reload(settings).HINTS_DEFAULT is True


# ---- віджети ------------------------------------------------------------------------------------
def _page(code: str) -> AppTest:
    return AppTest.from_string(code, default_timeout=180).run()


@pytest.mark.slow
@pytest.mark.parametrize("page", ["experiment", "grid"])
def test_every_input_widget_has_a_hint(page):
    at = _page(f"from ui.pages import {page}\n{page}()")
    assert not at.exception
    missing = []
    for kind in WIDGET_KINDS + ["segmented_control", "pills"]:
        for w in getattr(at, kind, []):
            if not getattr(w, "help", None):
                missing.append((kind, w.label))
    assert not missing, missing


@pytest.mark.slow
def test_hints_can_be_switched_off():
    at = _page("import streamlit as st\nst.session_state['ui_show_hints'] = False\n"
               "from ui.pages import experiment\nexperiment()")
    assert not at.exception
    assert all(not getattr(w, "help", None) for w in list(at.slider) + list(at.number_input) + list(at.toggle))


@pytest.mark.slow
def test_hint_text_of_widget_is_from_glossary():
    at = _page("from ui.pages import experiment\nexperiment()")
    alpha = [s for s in at.slider if s.label.startswith("α")][0]
    assert "невпевненість" in alpha.help.lower() or "ознаки C" in alpha.help


# ---- сторінка «Посібник» ----------------------------------------------------------------------------
@pytest.mark.slow
def test_guide_page_opens_and_has_all_sections():
    at = _page("from ui.pages import guide\nguide()")
    assert not at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    for title, _ in uk.GUIDE.values():
        assert title in text
    for term, _ in uk.TERMS:
        assert term in text
    for _, keys in uk.FORM_GROUPS:
        for k in keys:
            assert uk.HELP[k]["label"] in text, k


@pytest.mark.slow
def test_app_registers_guide_page():
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert re.search(r'"guide": st\.Page\(pages\.guide', src) and "hints_toggle()" in src


@pytest.mark.slow
def test_toggling_hints_keeps_form_values():
    code = ("import streamlit as st\nfrom ui.settings import hints_toggle\nhints_toggle()\n"
            "from ui.pages import experiment\nexperiment()")
    at = AppTest.from_string(code, default_timeout=180).run()
    rate = [s for s in at.slider if s.label.startswith("Рівень шуму")][0]
    rate.set_value(0.3).run()
    at.sidebar.toggle(key="ui_show_hints").set_value(False).run()
    assert not at.exception
    rate = [s for s in at.slider if s.label.startswith("Рівень шуму")][0]
    assert rate.value == pytest.approx(0.3) and not rate.help
