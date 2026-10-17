import pandas as pd
import streamlit as st

from cleantext.config import NOISE_TYPES
from cleantext.runs import list_runs, load_run
from cleantext.scoring import SHORT_NAMES
from ui.nav import go
from ui.help import tip
from ui.theme import note, page_head, section

NUM3 = lambda label: st.column_config.NumberColumn(label, format="%.3f")


def history():
    page_head("Архів", "Історія та порівняння запусків",
              "Усі одиночні запуски з теки runs/. Оберіть кілька рядків, щоб порівняти методи між сценаріями.")
    runs = list_runs()
    if not runs:
        st.info("Немає збережених запусків.")
        return
    df = pd.DataFrame(runs)
    df["noise_ua"] = df.noise.map(NOISE_TYPES)
    sel = st.dataframe(
        df[["created", "dataset", "noise_ua", "rate", "seed", "classifier", "n_train", "AP_R", "F1_noisy", "F1_R", "name"]],
        hide_index=True, on_select="rerun", selection_mode="multi-row", key="hist_tbl",
        column_config={"created": "Створено", "dataset": "Корпус", "noise_ua": "Шум",
                       "rate": st.column_config.NumberColumn("η", format="%.2f"), "seed": "Seed", "classifier": "Мережа",
                       "n_train": "N train", "AP_R": NUM3("AP R"), "F1_noisy": NUM3("F1 шум"), "F1_R": NUM3("F1 після R"),
                       "name": "Тека"})
    rows = sel.selection.rows if sel else []
    if not rows:
        note("Позначте один або кілька запусків у таблиці.", "info")
        return
    if len(rows) == 1 and st.button("Відкрити результати", type="primary", icon=":material/monitoring:", help=tip("btn_open")):
        go("results", df.iloc[rows[0]].path)

    section("Порівняння AP методів між обраними запусками")
    comp = []
    for i in rows:
        r = load_run(df.iloc[i].path)
        label = f"{r.cfg.dataset} · {r.cfg.noise_type} {r.cfg.noise_rate:.2f} · s{r.cfg.seed}"
        for _, d in r.detection.iterrows():
            comp.append({"Запуск": label, "Метод": SHORT_NAMES.get(d.key, d.key), "AP": d.AP, "P@K": d["Precision@K"]})
    c = pd.DataFrame(comp)
    for metric in ("AP", "P@K"):
        st.markdown(f"**{metric}**")
        st.dataframe(c.pivot_table(index="Запуск", columns="Метод", values=metric).reset_index(), hide_index=True,
                     column_config={m: NUM3(m) for m in c["Метод"].unique()})
