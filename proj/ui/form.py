"""Спільна форма параметрів експерименту (одиночний запуск і серія)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from cleantext.config import (BERT_PRESETS, CALIB_MODES, CLASSIFIERS, DATASETS, EMBEDDINGS, NOISE_TYPES, PRESETS,
                              ExperimentConfig)
from .help import tip
from .theme import card_title, note

PRESET_NAMES = {"quick": "Пробний (швидко)", "main": "Основний"}


def config_form(prefix: str, with_noise: bool = True) -> tuple[ExperimentConfig | None, pd.DataFrame | None, list[str]]:
    errors: list[str] = []
    csv_df = None
    left, right = st.columns(2, gap="medium")

    with left:
        with st.container(border=True, key=f"card_{prefix}_data"):
            card_title("1", "Корпус і вибірки", "Основний корпус — UA-News; незалежна перевірка — UAReviews.")
            dataset = st.selectbox("Набір даних", list(DATASETS), format_func=DATASETS.get, key=f"{prefix}_ds", help=tip("dataset"))
            task = "intent"
            if dataset == "uareviews":
                task = st.segmented_control("Завдання UAReviews", ["intent", "emotion"], default="intent",
                                            format_func={"intent": "Наміри (5 класів)", "emotion": "Емоції (7 класів)"}.get,
                                            key=f"{prefix}_task", help=tip("task")) or "intent"
            if dataset == "csv":
                up = st.file_uploader("CSV зі стовпцями text, label (необов'язково split = train/test)", type=["csv"],
                                      key=f"{prefix}_csv", help=tip("csv"))
                if up is not None:
                    csv_df = pd.read_csv(up)
                    missing = {"text", "label"} - set(csv_df.columns)
                    if missing:
                        errors.append(f"У CSV бракує стовпців: {', '.join(missing)}")
                    else:
                        st.caption(f"Завантажено {len(csv_df)} рядків, класів: {csv_df.label.nunique()}")
                else:
                    errors.append("Завантажте CSV-файл")
            preset = st.segmented_control("Режим", list(PRESET_NAMES), default="quick", format_func=PRESET_NAMES.get,
                                          key=f"{prefix}_preset", help=tip("preset")) or "quick"
            p = PRESETS[preset]
            c1, c2, c3 = st.columns(3)
            n_train = c1.number_input("Train", 100, 20000, p["n_train"], 100, key=f"{prefix}_ntr_{preset}", help=tip("n_train"))
            n_val = c2.number_input("Validation", 50, 5000, p["n_val"], 50, key=f"{prefix}_nva_{preset}", help=tip("n_val"))
            n_test = c3.number_input("Test", 50, 5000, p["n_test"], 50, key=f"{prefix}_nte_{preset}", help=tip("n_test"))
            if dataset == "uareviews" and n_test > 1700:
                errors.append("Внутрішній test UAReviews містить ≈1 700 текстів — зменшіть Test")

        if with_noise:
            with st.container(border=True, key=f"card_{prefix}_noise"):
                card_title("2", "Модель шуму", "Шум вноситься ЛИШЕ в train; validation і test залишаються незмінними.")
                noise_type = st.segmented_control("Тип шуму", list(NOISE_TYPES), default="semantic",
                                                  format_func=NOISE_TYPES.get, key=f"{prefix}_nt", help=tip("noise_type")) or "semantic"
                c1, c2 = st.columns([2, 1])
                rate = c1.slider("Рівень шуму η", 0.0, 0.4, 0.20, 0.05, format="%.2f", key=f"{prefix}_rate", help=tip("rate"))
                seed = c2.number_input("Random seed", 0, 10_000, 42, key=f"{prefix}_seed", help=tip("seed"))
                st.caption({"symmetric": "Обрані зразки рівноймовірно отримують одну з інших K−1 міток.",
                            "asymmetric": "Фіксована матриця переходів: клас c → клас (c+1) mod K.",
                            "semantic": "Мітка найближчого за змістом тексту іншого класу (найскладніший шум)."}[noise_type])
        else:
            noise_type, rate, seed = "semantic", 0.2, 42

        with st.container(border=True, key=f"card_{prefix}_base"):
            card_title("5" if with_noise else "4", "Базові методи та очищення")
            c1, c2 = st.columns(2)
            run_cleanlab = c1.toggle("Cleanlab OOF", True, key=f"{prefix}_cl", help=tip("run_cleanlab"))
            run_retrain = c2.toggle("Перенавчання після очищення", True, key=f"{prefix}_rt", help=tip("run_retrain"))
            run_oracle = c1.toggle("Oracle (верхня межа)", True, key=f"{prefix}_or",
                                   help=tip("run_oracle"))
            bootstrap = c2.number_input("Бутстреп-повторів", 0, 5000, 500, 100, key=f"{prefix}_bs", help=tip("bootstrap"))
            run_weighted = c1.toggle("М'яке зважування", True, key=f"{prefix}_wt",
                                     help=tip("run_weighted"))
            run_aum = c2.toggle("Повний AUM (Pleiss et al.)", True, key=f"{prefix}_aum",
                                help=tip("run_aum"))
            weight_min, weight_power, weight_scope = 0.20, 2.0, "top_k"
            if run_weighted:
                with st.expander("Параметри зважування"):
                    w1, w2 = st.columns(2)
                    weight_min = w1.number_input("w_min", 0.01, 1.0, 0.20, 0.05, key=f"{prefix}_wmin", help=tip("weight_min"))
                    weight_power = w2.number_input("Степінь p", 0.5, 6.0, 2.0, 0.5, key=f"{prefix}_wp", help=tip("weight_power"))
                    weight_scope = st.segmented_control(
                        "Кого зважувати", ["top_k", "all"], default="top_k", key=f"{prefix}_wsc",
                        format_func={"top_k": "Лише K найризикованіших", "all": "Усі за R"}.get,
                        help=tip("weight_scope")) or "top_k"
            options = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
            retrain_budgets = st.multiselect(
                "Бюджети вилучення при перенавчанні", options, default=p["retrain_budgets"],
                format_func=lambda b: f"{b:.0%}", key=f"{prefix}_rb_{preset}",
                help=tip("budgets"))

    with right:
        with st.container(border=True, key=f"card_{prefix}_method"):
            card_title("3" if with_noise else "2", "Інтегральна оцінка ризику",
                       "Rᵢ = α·Cᵢ + β·Dᵢ + γ·Sᵢ,  α + β + γ = 1. Стартові ваги задані апріорно.")
            c1, c2 = st.columns(2)
            alpha = c1.slider("α — невпевненість C", 0.0, 1.0, 0.45, 0.05, key=f"{prefix}_a", help=tip("alpha"))
            beta = c2.slider("β — незгода D", 0.0, 1.0, 0.35, 0.05, key=f"{prefix}_b", help=tip("beta"))
            gamma = round(1 - alpha - beta, 4)
            if gamma < -1e-9:
                errors.append("α + β не може перевищувати 1")
                note("α + β &gt; 1 — зменшіть одну з ваг.", "warn")
            else:
                st.markdown(f'<div class="formula">R = {alpha:.2f}·C + {beta:.2f}·D + {max(gamma, 0):.2f}·S</div>',
                            unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            k = c1.slider("k семантичних сусідів", 3, 30, 10, key=f"{prefix}_k", help=tip("k"))
            budget = c2.slider("Бюджет перевірки q", 0.05, 0.40, 0.20, 0.05, format="%.2f", key=f"{prefix}_q",
                               help=tip("budget"))
            calibrate = st.toggle("Калібрувати ваги на окремій підвибірці", True, key=f"{prefix}_cal",
                                  help=tip("calibrate"))
            calib_mode, n_calib, calib_step = "known_type", int(p["n_calib"]), 0.25
            if calibrate:
                c1, c2 = st.columns(2)
                calib_mode = c1.selectbox("Режим шуму калібрування", list(CALIB_MODES), format_func=CALIB_MODES.get,
                                          key=f"{prefix}_cm", help=tip("calib_mode"))
                n_calib = int(c2.number_input("Розмір calib", 100, 5000, int(p["n_calib"]), 50,
                                              key=f"{prefix}_nc_{preset}", help=tip("n_calib")))
                calib_step = float(c1.selectbox("Крок сітки ваг", [0.05, 0.1, 0.25], index=2, key=f"{prefix}_cs", help=tip("calib_step")))
            embedding = st.selectbox("Семантична модель", list(EMBEDDINGS), format_func=EMBEDDINGS.get,
                                     key=f"{prefix}_emb", help=tip("embedding"))

        with st.container(border=True, key=f"card_{prefix}_model"):
            card_title("4" if with_noise else "3", "Нейромережевий класифікатор",
                       "Після кожної епохи фіксуються pₜ(c|xᵢ) для всіх навчальних прикладів.")
            classifier = st.selectbox("Архітектура", list(CLASSIFIERS), format_func=CLASSIFIERS.get, key=f"{prefix}_clf", help=tip("classifier"))
            is_bert = classifier == "distilbert"
            bp = BERT_PRESETS[preset] if is_bert else p
            c1, c2, c3 = st.columns(3)
            epochs = c1.number_input("Епохи T", 1, 50, bp["epochs"], key=f"{prefix}_ep_{preset}_{classifier}", help=tip("epochs"))
            folds = c2.number_input("OOF фолди", 2, 10, bp["oof_folds"], key=f"{prefix}_f_{preset}_{classifier}", help=tip("folds"))
            oof_epochs = c3.number_input("OOF епохи", 1, 50, bp["oof_epochs"], key=f"{prefix}_oe_{preset}_{classifier}", help=tip("oof_epochs"))
            with st.expander("Додаткові гіперпараметри"):
                c1, c2, c3 = st.columns(3)
                if is_bert:
                    lr = c1.number_input("Learning rate", 1e-6, 1e-3, 3e-5, format="%.0e", key=f"{prefix}_blr", help=tip("lr"))
                    batch = c2.number_input("Batch", 4, 64, 16, key=f"{prefix}_bb", help=tip("batch"))
                    max_len = c3.number_input("Max length", 32, 512, 128, 32, key=f"{prefix}_ml", help=tip("max_len"))
                    hidden, dropout, max_feat = 256, 0.3, 30000
                else:
                    lr = c1.number_input("Learning rate", 1e-5, 1e-1, 1e-3, format="%.0e", key=f"{prefix}_lr", help=tip("lr"))
                    batch = c2.number_input("Batch", 8, 256, 32, key=f"{prefix}_bt", help=tip("batch"))
                    hidden = c3.number_input("Нейронів прихованого шару", 32, 2048, 256, 32, key=f"{prefix}_hd", help=tip("hidden"))
                    dropout = c1.slider("Dropout", 0.0, 0.7, 0.3, 0.05, key=f"{prefix}_do", help=tip("dropout"))
                    max_feat = c2.number_input("Ознак TF-IDF", 2000, 200000, 30000, 1000, key=f"{prefix}_mf", help=tip("max_feat"))
                    max_len = 128
            if is_bert:
                note("DistilBERT на CPU навчається дуже повільно (десятки хвилин на одну епоху для 4 000 текстів). "
                     "Рекомендовано GPU або режим «Пробний».", "warn")

    if errors:
        return None, csv_df, errors
    cfg = ExperimentConfig(
        dataset=dataset, review_task=task, n_train=int(n_train), n_val=int(n_val), n_test=int(n_test),
        noise_type=noise_type, noise_rate=float(rate), seed=int(seed), alpha=float(alpha), beta=float(beta),
        gamma=float(max(gamma, 0)), k_neighbors=int(k), budget_frac=float(budget),
        retrain_budgets=[float(b) for b in retrain_budgets], embedding=embedding,
        calibrate_weights=bool(calibrate), calib_mode=calib_mode, n_calib=n_calib, coeff_grid_step=calib_step,
        classifier=classifier, epochs=int(epochs), oof_folds=int(folds), oof_epochs=int(oof_epochs),
        run_cleanlab=run_cleanlab, run_retrain=run_retrain, run_oracle=run_oracle, bootstrap=int(bootstrap),
        run_aum=run_aum, run_weighted=run_weighted, weight_min=float(weight_min), weight_power=float(weight_power),
        weight_scope=weight_scope,
        hidden=int(hidden), dropout=float(dropout), max_features=int(max_feat), max_length=int(max_len),
        **({"bert_lr": float(lr), "bert_batch_size": int(batch)} if is_bert else {"lr": float(lr), "batch_size": int(batch)}),
    )
    return cfg, csv_df, errors
