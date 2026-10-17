"""Мова інтерфейсних текстів (підказки, посібник). Нова мова = новий файл ui/locales/<код>.py з тими самими іменами.

Мову задає змінна оточення CLEANTEXT_LANG (за замовчуванням uk); якщо в обраній мові бракує ключа,
береться українська.
"""
from __future__ import annotations

import importlib
import os

DEFAULT_LANG = "uk"
LANG = os.environ.get("CLEANTEXT_LANG", DEFAULT_LANG)


def _load(lang: str):
    try:
        return importlib.import_module(f"ui.locales.{lang}")
    except ModuleNotFoundError:
        return importlib.import_module(f"ui.locales.{DEFAULT_LANG}")


def texts():
    """Модуль з текстами поточної мови."""
    return _load(LANG)


def entry(key: str) -> dict | None:
    """Запис довідки за ключем (з поверненням до української)."""
    return texts().HELP.get(key) or _load(DEFAULT_LANG).HELP.get(key)


def ui(key: str) -> str:
    """Короткий інтерфейсний підпис (назви розділів і кнопок довідки)."""
    return texts().UI.get(key) or _load(DEFAULT_LANG).UI[key]
