import streamlit as st

from cleantext.pipeline import run_experiment
from ui.form import config_form
from ui.nav import PAGES, go
from ui.runner import run_with_progress
from ui.help import tip
from ui.theme import note, page_head


def experiment():
    page_head("Етапи E0–E5", "Новий експеримент",
              "Один сценарій: корпус × тип шуму × рівень η × seed. Усі методи отримують однаковий бюджет K, "
              "а класифікатор після очищення навчається з ідентичними гіперпараметрами.")
    if "guide" in PAGES:
        st.page_link(PAGES["guide"], label="Не знайомі з термінами? Відкрийте «Посібник»", icon=":material/help:")
    cfg, csv_df, errors = config_form("exp")
    for e in errors:
        note(e, "warn")

    st.write("")
    c1, c2 = st.columns([1, 3])
    start = c1.button("Запустити експеримент", type="primary", icon=":material/play_arrow:",
                      disabled=cfg is None, width="stretch", help=tip("btn_run"))
    c2.caption("Перший запуск завантажує корпус і модель MiniLM з Hugging Face (потрібен Інтернет) і кешує їх "
               "у теці data/cache — наступні запуски працюють швидше та офлайн.")
    if start and cfg is not None:
        path = run_with_progress("Виконується експеримент", run_experiment, cfg=cfg, csv_df=csv_df)
        if path is not None:
            st.session_state["run_path"] = str(path)
            st.session_state["last_finished"] = str(path)
    last = st.session_state.get("last_finished")
    if last:
        note(f"Запуск збережено: <span class='mono'>{last}</span>", "ok")
        if st.button("Відкрити результати", icon=":material/monitoring:", type="primary", help=tip("btn_open")):
            go("results", last)
