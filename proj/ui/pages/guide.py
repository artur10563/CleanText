"""Сторінка «Посібник»: пояснення без технічних термінів для людини, не знайомої з областю."""
import streamlit as st

from ui.i18n import texts, ui
from ui.nav import PAGES
from ui.theme import note, page_head, section


def _block(key: str) -> None:
    title, paragraphs = texts().GUIDE[key]
    section(title)
    for p in paragraphs:
        st.markdown(p)


def guide():
    t = texts()
    page_head(ui("guide_eyebrow"), ui("guide_title"), ui("guide_lead"))
    note("Біля кожного поля форми є маленька кнопка «?» з коротким поясненням. Тут зібрано те саме, а також загальну картину. "
         "Підказки можна вимкнути на бічній панелі.", "info")

    tabs = st.tabs(["З чого почати", "Поля форми", "Словник термінів", "Формули", "Результати та застереження", "Питання"])
    with tabs[0]:
        _block("intro")
        _block("steps")
        if "experiment" in PAGES:
            st.page_link(PAGES["experiment"], label="Перейти до нового експерименту", icon=":material/science:")
    with tabs[1]:
        section("Що означає кожне поле", "Порядок збігається з порядком у формі та на сторінках результатів")
        for group, keys in t.FORM_GROUPS:
            with st.expander(group):
                for k in keys:
                    e = t.HELP[k]
                    lines = [f"**{e['label']}** — {e['what']}"]
                    for field, name in (("effect", ui("effect")), ("default", ui("default")), ("tip", ui("advice"))):
                        if e.get(field):
                            lines.append(f"&nbsp;&nbsp;&nbsp;*{name}* {e[field]}")
                    st.markdown("\n\n".join(lines))
    with tabs[2]:
        section("Словник термінів", "Слова, які трапляються в програмі, простими словами")
        for term, text in t.TERMS:
            st.markdown(f"**{term}** — {text}")
    with tabs[3]:
        _block("formulas")
    with tabs[4]:
        _block("results")
        _block("cautions")
    with tabs[5]:
        _block("faq")
