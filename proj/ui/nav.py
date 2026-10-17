"""Реєстр сторінок для переходів між ними (заповнюється в app.py)."""
import streamlit as st

PAGES: dict = {}


def go(name: str, run_path: str | None = None) -> None:
    if run_path:
        st.session_state["run_path"] = run_path
    st.switch_page(PAGES[name])


def link(name: str, label: str, icon: str | None = None) -> None:
    st.page_link(PAGES[name], label=label, icon=icon)
