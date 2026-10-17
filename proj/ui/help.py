"""Підказки «?» до віджетів: tip("ключ") передається в параметр help= віджета Streamlit."""
from __future__ import annotations

from .i18n import entry, ui
from .settings import hints_enabled


def tip(key: str) -> str | None:
    """Markdown-текст підказки або None, якщо підказки вимкнено."""
    if not hints_enabled():
        return None
    e = entry(key)
    if e is None:
        raise KeyError(f"немає підказки для «{key}»")
    parts = [f"**{ui('what')}** {e['what']}"]
    if e.get("effect"):
        parts.append(f"**{ui('effect')}** {e['effect']}")
    if e.get("default"):
        parts.append(f"**{ui('default')}** {e['default']}")
    if e.get("tip"):
        parts.append(f"**{ui('advice')}** {e['tip']}")
    return "\n\n".join(parts)
