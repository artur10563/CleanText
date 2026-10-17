"""CleanText — інтерфейс дослідницької програми. Запуск: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="CleanText — шум у мітках", page_icon=":material/fact_check:", layout="wide",
                   initial_sidebar_state="expanded")

from cleantext.models import DEVICE  # noqa: E402
from ui import pages  # noqa: E402
from ui.nav import PAGES  # noqa: E402
from ui.settings import hints_toggle  # noqa: E402
from ui.theme import inject_css, sidebar_brand, sidebar_footer  # noqa: E402

inject_css()
sidebar_brand()
hints_toggle()

PAGES.update({
    "home": st.Page(pages.home, title="Огляд", icon=":material/space_dashboard:", url_path="home", default=True),
    "experiment": st.Page(pages.experiment, title="Новий експеримент", icon=":material/science:", url_path="experiment"),
    "results": st.Page(pages.results, title="Результати запуску", icon=":material/monitoring:", url_path="results"),
    "grid": st.Page(pages.grid, title="Серія експериментів", icon=":material/grid_view:", url_path="grid"),
    "history": st.Page(pages.history, title="Історія та порівняння", icon=":material/history:", url_path="history"),
    "guide": st.Page(pages.guide, title="Посібник", icon=":material/help:", url_path="guide"),
    "method": st.Page(pages.method, title="Про метод", icon=":material/menu_book:", url_path="method"),
})
nav = st.navigation({
    "Дослідження": [PAGES[k] for k in ("home", "experiment", "results", "grid")],
    "Довідка": [PAGES[k] for k in ("guide", "history", "method")],
})
sidebar_footer(DEVICE.type)
nav.run()
