import pandas as pd
import streamlit as st

from cleantext.config import NOISE_TYPES
from cleantext.runs import list_runs
from ui.nav import go
from ui.theme import section


def home():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Магістерське дослідження · 2026</div>
      <h1>Виявлення некоректно розмічених україномовних текстів<br>у процесі навчання нейромереж</h1>
      <p>Програма ранжує навчальні зразки за ризиком помилкової мітки, поєднуючи динаміку навчання нейромережі
      (впевненість і незгоду за епохами) з локальною семантичною узгодженістю, і порівнює метод
      із Loss, margin-показником, Cleanlab OOF та випадковим відбором.</p>
      <div class="chips"><span class="chip">UA-News · UAReviews · власний CSV</span>
      <span class="chip">Симетричний · асиметричний · семантичний шум</span>
      <span class="chip">AP · PR-AUC · Precision@K · Macro-F1</span>
      <span class="chip">Бутстреп 95 % CI</span></div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    feats = [
        (c1, "#1baf7a", "C", "Невпевненість", "Cᵢ = 1 − (1/T)·Σₜ pₜ(ỹᵢ|xᵢ) — середня ймовірність наявної мітки за епохами."),
        (c2, "#eda100", "D", "Незгода", "Dᵢ = 1 − (1/T)·Σₜ 1[argmax pₜ = ỹᵢ] — як часто передбачення суперечить мітці."),
        (c3, "#008300", "S", "Семантика", "Sᵢ = 1 − частка k найближчих за змістом сусідів із тією самою міткою."),
        (c4, "#2a78d6", "R", "Інтегральний ризик", "Rᵢ = α·Cᵢ + β·Dᵢ + γ·Sᵢ; відбір top-K = ⌊qN⌋ для перевірки або вилучення."),
    ]
    for col, color, letter, title, text in feats:
        col.markdown(f"""<div class="feature"><div class="ico" style="background:{color}">{letter}</div>
        <h4>{title}</h4><p>{text}</p></div>""", unsafe_allow_html=True)

    section("Експериментальний конвеєр", "Кожен запуск проходить усі етапи E0–E5 і зберігає машинозчитувані артефакти.")
    steps = [("E0", "Контроль даних", "Нормалізація, дублікати між train/val/test, маніфест"),
             ("E1", "Інʼєкція шуму", "Лише train; η = 0–40 %; три типи шуму"),
             ("E2", "Навчання", "Нейромережа; pₜ(c|xᵢ) після кожної епохи"),
             ("E3", "Ранжування", "R, Loss, margin, Cleanlab OOF, random"),
             ("E4", "Абляція", "C+D проти S проти повного R; динаміка за епохами"),
             ("E5", "Очищення", "Однаковий K; перенавчання; Macro-F1 на чистому тесті")]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="step"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
        for n, t, d in steps) + "</div>", unsafe_allow_html=True)

    st.write("")
    b1, b2, b3, b4, _ = st.columns([1.2, 1.2, 1.2, 1.2, 2])
    if b1.button("Новий експеримент", type="primary", icon=":material/play_arrow:", width="stretch"):
        go("experiment")
    if b2.button("Серія експериментів", icon=":material/grid_view:", width="stretch"):
        go("grid")
    if b3.button("Про метод", icon=":material/menu_book:", width="stretch"):
        go("method")
    if b4.button("Посібник", icon=":material/help:", width="stretch"):
        go("guide")

    runs = list_runs()
    section("Останні запуски", "Відкрийте запуск, щоб переглянути таблиці, графіки та кандидатів для експертної перевірки.")
    if not runs:
        st.info("Ще немає жодного запуску. Почніть із пробного режиму: UA-News, семантичний шум, η = 0,20.")
        return
    df = pd.DataFrame(runs[:8])
    df["noise"] = df.noise.map(NOISE_TYPES)
    sel = st.dataframe(
        df[["created", "dataset", "noise", "rate", "seed", "classifier", "n_train", "AP_R", "F1_noisy", "F1_R"]],
        hide_index=True, on_select="rerun", selection_mode="single-row", key="home_runs",
        column_config={
            "created": "Створено", "dataset": "Корпус", "noise": "Шум", "rate": st.column_config.NumberColumn("η", format="%.2f"),
            "seed": "Seed", "classifier": "Мережа", "n_train": "N train",
            "AP_R": st.column_config.ProgressColumn("AP методу R", min_value=0, max_value=1, format="%.3f"),
            "F1_noisy": st.column_config.NumberColumn("Macro-F1 без очищення", format="%.3f"),
            "F1_R": st.column_config.NumberColumn("Macro-F1 після R", format="%.3f"),
        })
    rows = sel.selection.rows if sel else []
    if rows:
        go("results", df.iloc[rows[0]].path)
