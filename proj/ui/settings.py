"""Налаштування інтерфейсу. Підказки «?» вимикаються змінною CLEANTEXT_HINTS=0 або перемикачем на бічній панелі."""
from __future__ import annotations

import os

import streamlit as st

from .i18n import ui

HINTS_DEFAULT = os.environ.get("CLEANTEXT_HINTS", "1").strip().lower() not in ("0", "false", "no", "off")
HINTS_KEY = "ui_show_hints"


def hints_enabled() -> bool:
    try:
        return bool(st.session_state.get(HINTS_KEY, HINTS_DEFAULT))
    except Exception:
        return HINTS_DEFAULT


def hints_toggle() -> None:
    with st.sidebar:
        st.toggle(ui("hints_toggle"), HINTS_DEFAULT, key=HINTS_KEY, help=ui("hints_toggle_help"))
