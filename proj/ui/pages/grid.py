import json

import numpy as np
import pandas as pd
import streamlit as st

from cleantext.config import NOISE_TYPES
from cleantext.grid import run_grid
from cleantext.runs import grid_tables, list_grids
from cleantext.scoring import SHORT_NAMES
from ui import charts
from ui.form import config_form
from ui.help import tip
from ui.nav import go
from ui.runner import run_with_progress
from ui.theme import note, page_head, section, show

NUM3 = lambda label: st.column_config.NumberColumn(label, format="%.3f")


def _agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.groupby(["kind", "rate", "key"])[metric].agg(["mean", "std", "count"]).reset_index()


def grid():
    page_head("Контрольна матриця (розділ 6.1)", "Серія експериментів",
              "Типи шуму × рівні η × seeds на одному корпусі. Корпус і семантичні подання обчислюються один раз, тож "
              "порівняння методів у межах одного seed є парними. Завершені сценарії пропускаються при повторному запуску.")
    t_run, t_view = st.tabs(["Запуск серії", "Результати серій"])

    with t_run:
        cfg, csv_df, errors = config_form("grid", with_noise=False)
        with st.container(border=True, key="card_grid_matrix"):
            st.markdown('<div class="card-title"><span class="num">5</span>Матриця сценаріїв</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            types = c1.pills("Типи шуму", list(NOISE_TYPES), selection_mode="multi", default=list(NOISE_TYPES),
                             format_func=NOISE_TYPES.get, key="g_types", help=tip("grid_types"))
            rates = c2.pills("Рівні η", [0.0, 0.1, 0.2, 0.3, 0.4], selection_mode="multi", default=[0.1, 0.2, 0.3, 0.4],
                             format_func=lambda r: f"{r:.0%}", key="g_rates", help=tip("grid_rates"))
            seeds_s = c3.text_input("Seeds (через кому)", "13, 42, 77", key="g_seeds", help=tip("grid_seeds"))
            try:
                seeds = [int(s) for s in seeds_s.replace(";", ",").split(",") if s.strip()]
            except ValueError:
                seeds = []
                errors.append("Seeds мають бути цілими числами")
            n_cases = len(types or []) * len(rates or []) * len(seeds)
            st.caption(f"Сценаріїв: **{n_cases}**. Мінімально достатній перший цикл: 3 типи × η = 0,20 × 3 seeds.")
        for e in errors:
            note(e, "warn")
        if st.button("Запустити серію", type="primary", icon=":material/play_arrow:",
                     disabled=cfg is None or n_cases == 0, help=tip("btn_run_grid")):
            out = run_with_progress(f"Серія з {n_cases} сценаріїв", run_grid, cfg=cfg, types=types,
                                    rates=sorted(rates), seeds=seeds, csv_df=csv_df)
            if out is not None:
                st.session_state["grid_path"] = str(out)
                note(f"Серію збережено: <span class='mono'>{out}</span>. Перейдіть на вкладку «Результати серій».", "ok")

    with t_view:
        grids = list_grids()
        if not grids:
            st.info("Ще немає жодної серії.")
            return
        cur = st.session_state.get("grid_path")
        paths = [str(g) for g in grids]
        gsel = st.selectbox("Серія", paths, index=paths.index(cur) if cur in paths else 0,
                            format_func=lambda p: p.split("\\")[-1].split("/")[-1], help=tip("grid_select"))
        meta = json.loads((grids[paths.index(gsel)] / "grid.json").read_text(encoding="utf-8"))
        det, cls = grid_tables(grids[paths.index(gsel)])
        if det.empty:
            st.info("У серії ще немає завершених сценаріїв.")
            return
        total = len(meta["types"]) * len(meta["rates"]) * len(meta["seeds"])
        st.caption(f"Завершено сценаріїв: {det.case.nunique()} з {total} · корпус: {meta['config']['dataset']} · "
                   f"N train = {meta['config']['n_train']} · мережа: {meta['config']['classifier']}")
        kinds = [k for k in NOISE_TYPES if k in set(det.kind)]
        all_methods = [k for k in ["proposed", "adaptive", "cleanlab", "loss", "margin", "dynamics", "semantic", "random"]
                       if k in set(det.key)]
        methods = st.pills("Методи на графіках", all_methods, selection_mode="multi",
                           default=[k for k in ["proposed", "cleanlab", "loss", "random"] if k in all_methods],
                           format_func=lambda k: SHORT_NAMES.get(k, k), key="g_methods", help=tip("grid_methods")) or all_methods
        d_pos = det[det.rate > 0]
        if not d_pos.empty:
            show(charts.grid_lines(_agg(d_pos, "AP"), "AP", methods, kinds, NOISE_TYPES), key="g_ap")
            show(charts.grid_lines(_agg(d_pos, "Precision@K"), "Precision@K", methods, kinds, NOISE_TYPES), key="g_pk")
        if not cls.empty:
            cm = [k for k in ["noisy", "random", "loss", "cleanlab", "proposed", "adaptive", "weighted", "oracle"] if k in set(cls.key)]
            show(charts.grid_lines(_agg(cls, "macro_f1"), "Macro-F1", cm, kinds, NOISE_TYPES), key="g_f1")

        section("Підсумкова таблиця (шаблон розділу 11)", "Середнє ± std за seeds")
        rows = []
        for (kind, rate), g in det.groupby(["kind", "rate"]):
            def ms(key, metric, frame=g):
                v = frame.loc[frame.key == key, metric]
                return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else (f"{v.mean():.3f}" if len(v) else "—")
            row = {"Шум": NOISE_TYPES[kind], "η": rate, "Loss AP": ms("loss", "AP"), "Cleanlab AP": ms("cleanlab", "AP"),
                   "R AP": ms("proposed", "AP"), "R Precision@K": ms("proposed", "Precision@K")}
            if not cls.empty:
                c = cls[(cls.kind == kind) & (cls.rate == rate)]
                p = c[c.key == "proposed"].set_index("seed").macro_f1
                n = c[c.key == "noisy"].set_index("seed").macro_f1
                diff = (p - n).dropna()
                row["Δ Macro-F1 R − Noisy"] = (f"{diff.mean():+.3f} ± {diff.std(ddof=1):.3f}" if len(diff) > 1
                                               else f"{diff.mean():+.3f}" if len(diff) else "—")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"η": st.column_config.NumberColumn("η", format="%.2f")})

        section("Парні різниці AP за seeds", "ΔAP = AP(R) − AP(базовий) для кожного seed; середнє та 95 % t-інтервал")
        prow = []
        for (kind, rate), g in d_pos.groupby(["kind", "rate"]):
            pv = g[g.key == "proposed"].set_index("seed").AP
            for b in [k for k in ("loss", "cleanlab", "dynamics", "semantic") if k in set(g.key)]:
                diff = (pv - g[g.key == b].set_index("seed").AP).dropna()
                n = len(diff)
                half = _t975(n - 1) * diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
                prow.append({"Шум": NOISE_TYPES[kind], "η": rate, "Проти": SHORT_NAMES[b], "n seeds": n,
                             "ΔAP": diff.mean(), "CI low": diff.mean() - half if n > 1 else np.nan,
                             "CI high": diff.mean() + half if n > 1 else np.nan})
        if prow:
            st.dataframe(pd.DataFrame(prow), hide_index=True, column_config={
                "η": st.column_config.NumberColumn("η", format="%.2f"), "ΔAP": NUM3("ΔAP"),
                "CI low": NUM3("CI low"), "CI high": NUM3("CI high")})
            note("За малої кількості повторів (3 seeds) інтервали широкі — категоричних висновків про статистичну "
                 "значущість робити не слід (розділ 6.3).", "info")

        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button("Усі метрики виявлення (CSV)", det.to_csv(index=False).encode("utf-8-sig"),
                           "all_detection.csv", "text/csv", icon=":material/download:",
                           help=tip("btn_csv_det"))
        if not cls.empty:
            c2.download_button("Усі метрики класифікації (CSV)", cls.to_csv(index=False).encode("utf-8-sig"),
                               "all_classification.csv", "text/csv", icon=":material/download:",
                               help=tip("btn_csv_cls"))
        section("Окремі сценарії")
        case = st.selectbox("Відкрити сценарій у «Результатах запуску»", sorted(det.case.unique()), key="g_case", help=tip("grid_case"))
        if st.button("Відкрити сценарій", icon=":material/open_in_new:", help=tip("btn_open")):
            go("results", str(grids[paths.index(gsel)] / case))


def _t975(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(df, 1.96)
