"""Виконання довгих обчислень із живим прогресом у Streamlit."""
from __future__ import annotations

import time
import traceback

import streamlit as st


def run_with_progress(label: str, fn, **kwargs):
    """fn(log=..., on_progress=..., **kwargs) → результат; показує статус, прогрес і журнал."""
    t0 = time.time()
    lines: list[str] = []
    with st.status(label, expanded=True) as status:
        bar = st.progress(0.0, text="Підготовка…")
        log_box = st.empty()

        def log(msg):
            lines.append(f"{time.time() - t0:6.1f} с  {msg}")
            log_box.code("\n".join(lines[-14:]), language=None)

        def on_progress(frac, text):
            el = time.time() - t0
            eta = el / frac - el if frac > 0.03 else None
            bar.progress(min(max(frac, 0.0), 1.0),
                         text=f"{text} · {frac:.0%} · {el:.0f} с" + (f" · залишилось ≈{eta:.0f} с" if eta else ""))

        try:
            result = fn(log=log, on_progress=on_progress, **kwargs)
        except Exception as e:  # показуємо помилку, не обриваючи застосунок
            status.update(label=f"Помилка: {e}", state="error", expanded=True)
            st.code(traceback.format_exc(), language=None)
            return None
        status.update(label=f"{label} — завершено за {time.time() - t0:.0f} с", state="complete", expanded=False)
    return result
