import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, f1_score

from cleantext.config import CALIB_MODES, CLASSIFIERS, DATASETS, EMBEDDINGS, NOISE_TYPES
from cleantext.metrics import detection_metrics, pr_curve, top_k
from cleantext.runs import budget_view, list_runs, load_run
from cleantext.scoring import SHORT_NAMES, combine
from ui import charts
from ui.help import tip
from ui.nav import go
from ui.theme import METHOD_COLORS, esc, kpis, note, page_head, section, show

NUM3 = lambda label: st.column_config.NumberColumn(label, format="%.3f")


@st.cache_resource(show_spinner="Завантаження запуску…", max_entries=8)
def _load(path: str, mtime: float):
    return load_run(path)


@st.cache_data(show_spinner=False, max_entries=8)
def _weight_grid(path: str, c, d, s, z, step: float = 0.05):
    rows = []
    vals = np.round(np.arange(0, 1 + 1e-9, step), 2)
    for a in vals:
        for b in vals:
            if a + b <= 1 + 1e-9:
                r = combine(c, d, s, a, b, 1 - a - b)
                rows.append({"alpha": a, "beta": b, "AP": average_precision_score(z, r)})
    return pd.DataFrame(rows)


def _pick_run() -> str | None:
    runs = list_runs()
    paths = [r["path"] for r in runs]
    current = st.session_state.get("run_path")
    if current and current not in paths and Path(current).exists():
        paths.insert(0, current)
    if not paths:
        return None
    names = {r["path"]: f"{r['created'].replace('T', ' ')} · {r['dataset']} · {NOISE_TYPES.get(r['noise'], r['noise'])} "
                        f"η={r['rate']:.2f} · seed {r['seed']}" for r in runs}
    idx = paths.index(current) if current in paths else 0
    sel = st.selectbox("Запуск", paths, index=idx, format_func=lambda p: names.get(p, Path(p).parent.name + " / " + Path(p).name),
                       label_visibility="collapsed", help=tip("run_select"))
    st.session_state["run_path"] = sel
    return sel


def results():
    page_head("Аналіз запуску", "Результати експерименту",
              "Таблиці та графіки можна зберегти як PNG (іконка камери над графіком) для звіту.")
    path = _pick_run()
    if path is None:
        st.info("Немає збережених запусків.")
        if st.button("Створити експеримент", type="primary"):
            go("experiment")
        return
    run = _load(path, (Path(path) / "run_manifest.json").stat().st_mtime)
    cfg, m, A = run.cfg, run.manifest, run.arrays
    z = A["z"]
    det = run.detection.set_index("key")
    cls = run.classification_main.set_index("key") if len(run.classification) else pd.DataFrame()
    K = m["budget_K"]

    chips = [DATASETS.get(cfg.dataset, cfg.dataset) + (f" · {cfg.review_task}" if cfg.dataset == "uareviews" else ""),
             f"{NOISE_TYPES[cfg.noise_type]} шум η = {cfg.noise_rate:.2f}", f"seed {cfg.seed}",
             CLASSIFIERS[cfg.classifier].split(" (")[0], f"T = {cfg.epochs} епох",
             f"R = {cfg.alpha:.2f}C + {cfg.beta:.2f}D + {cfg.gamma:.2f}S", f"k = {cfg.k_neighbors}", f"K = {K} ({cfg.budget_frac:.0%})"]
    st.markdown('<div class="chips" style="margin:0 0 .8rem">' + "".join(
        f'<span class="chip" style="background:#eef2ff;border-color:#c7d2fe;color:#3730a3">{esc(c)}</span>' for c in chips)
        + "</div>", unsafe_allow_html=True)

    has_noise = z.sum() > 0
    rap = det.AP.get("proposed", np.nan)
    baselines = [k for k in ("loss", "margin", "cleanlab") if k in det.index]
    best_b = max(baselines, key=lambda k: det.AP[k]) if baselines and has_noise else None
    items = [
        {"label": "AP методу R", "value": f"{rap:.3f}" if has_noise else "—", "color": METHOD_COLORS["proposed"],
         "delta": (rap - det.AP[best_b]) if best_b else None,
         "sub": f"проти {SHORT_NAMES[best_b]} ({det.AP[best_b]:.3f})" if best_b else "AP не визначена за η = 0"},
        {"label": "Precision@K (R)", "value": f"{det['Precision@K'].get('proposed', np.nan):.3f}",
         "color": METHOD_COLORS["proposed"], "sub": f"TP = {int(det.TP.get('proposed', 0))} із K = {K}"},
        {"label": "Recall@K (R)", "value": f"{det['Recall@K'].get('proposed', np.nan):.3f}" if has_noise else "—",
         "color": METHOD_COLORS["proposed"], "sub": f"знайдено з {int(z.sum())} змінених"},
    ]
    if "proposed" in cls.index:
        items.append({"label": "Macro-F1 після R", "value": f"{cls.macro_f1['proposed']:.3f}",
                      "color": METHOD_COLORS["oracle"], "delta": cls.macro_f1["proposed"] - cls.macro_f1["noisy"],
                      "sub": "відносно навчання на шумі"})
    elif "noisy" in cls.index:
        items.append({"label": "Macro-F1 (шум)", "value": f"{cls.macro_f1['noisy']:.3f}", "color": METHOD_COLORS["oracle"]})
    items.append({"label": "Внесений шум", "value": f"{z.mean():.1%}", "color": "#e34948",
                  "sub": f"{int(z.sum())} з {len(z)} навчальних міток"})
    kpis(items)

    tabs = st.tabs(["Дані та шум", "Навчання", "Виявлення", "Динаміка за епохами", "Абляція та ваги",
                    "Очищення", "Аналіз помилок", "Кандидати", "Маніфест і експорт"])
    with tabs[0]:
        _tab_data(run)
    with tabs[1]:
        _tab_training(run)
    with tabs[2]:
        _tab_detection(run, has_noise)
    with tabs[3]:
        _tab_epochs(run, has_noise)
    with tabs[4]:
        _tab_weights(run, has_noise)
    with tabs[5]:
        _tab_cleaning(run)
    with tabs[6]:
        _tab_errors(run, has_noise)
    with tabs[7]:
        _tab_candidates(run)
    with tabs[8]:
        _tab_manifest(run)


# ------------------------------------------------------------------ вкладки
def _tab_data(run):
    m, A = run.manifest, run.arrays
    stats = m["data"]
    c1, c2 = st.columns([1.15, 1])
    with c1:
        show(charts.class_distribution(stats), key="cls_dist", height=400)
    with c2:
        show(charts.transition_heatmap(A["transition"], run.labels), key="trans", height=400)
    s = stats["sizes"]
    rem = stats["removed_duplicates"]
    info = pd.DataFrame({
        "Вибірка": ["train", "val", "test"],
        "Текстів": [s["train"], s["val"], s["test"]],
        "Вилучено дублікатів": [rem.get("train", 0), rem.get("val", 0), rem.get("test", 0)],
        "Середня довжина, симв.": [round(stats["length"][k]["mean"]) for k in ("train", "val", "test")],
        "Медіана довжини": [round(stats["length"][k]["median"]) for k in ("train", "val", "test")],
        "Хеш розбиття": [stats["split_hashes"][k] for k in ("train", "val", "test")],
    })
    section("Контроль даних (E0)")
    st.dataframe(info, hide_index=True)
    src = stats["source"]
    note(f"Джерело: <b>{esc(src.get('repo'))}</b>; пул train = {src.get('pool_train')}, пул test = {src.get('pool_test')}. "
         "Для UA-News пул сформовано буферним перемішуванням потоку Hugging Face (не тотожно глобальній випадковій "
         "вибірці). Нормалізовані повні дублікати вилучено з пріоритетом test → val → train; майже-дублікати не контролюються.", "info")
    if run.train is not None and A["z"].sum() > 0:
        section("Приклади навмисно змінених міток")
        tr = run.train.set_index("id")
        idx = np.where(A["z"] == 1)[0][:12]
        ex = pd.DataFrame({"Текст": tr.text.iloc[idx].str.slice(0, 220).values,
                           "Вихідна мітка y⁰": [run.labels[v] for v in A["clean_y"][idx]],
                           "Спостережувана ỹ": [run.labels[v] for v in A["noisy_y"][idx]]})
        st.dataframe(ex, hide_index=True)


def _tab_training(run):
    log = run.learning_log
    show(charts.learning_curves(log), key="lc")
    if "train_acc_observed" in log and len(log):
        acc = log.train_acc_observed.iloc[-1]
        rate = run.arrays["z"].mean()
        if acc > 1 - rate / 2 and rate > 0:
            note(f"Наприкінці навчання точність щодо зашумлених міток становить {acc:.1%} — мережа запам'ятала значну "
                 "частину навмисно змінених міток. Саме тому накопичені за епохами C і D інформативніші, ніж "
                 "loss лише останньої епохи.", "warn")
    section("Розподіли ознак ризику", "Сірим — зразки з коректною міткою, червоним — навмисно змінені", hint="sec_features")
    A = run.arrays
    comps = {"C": A["comp_C"], "D": A["comp_D"], "S": A["comp_S"], "R": A["score_proposed"]}
    show(charts.score_histograms(comps, A["z"]), key="hist")
    section("Карта навчальної динаміки (Dataset Cartography)",
            "Swayamdipta et al., 2020: низька confidence і низька variability — «hard-to-learn» (кандидати на хибну мітку); "
            "висока variability — «ambiguous»; висока confidence — «easy-to-learn»", hint="sec_cartography")
    if "cart_confidence" in A:
        has_z = bool(A["z"].sum() > 0)
        show(charts.cartography_map(A["cart_confidence"], A["cart_variability"], A["z"] if has_z else None,
                                    A["score_proposed"]), key="cart")
        note("Confidence = 1 − C, correctness = 1 − D; окремим методом ранжування Cartography не є, бо ранжування за низькою "
             "confidence збігається з ознакою C (див. абляцію).")
    section("Журнал навчання")
    st.dataframe(log.rename(columns={"epoch": "Епоха", "train_loss": "Train loss", "train_acc_observed": "Train acc (ỹ)",
                                     "val_loss": "Val loss", "val_macro_f1": "Val Macro-F1"}), hide_index=True,
                 column_config={c: NUM3(c) for c in ["Train loss", "Train acc (ỹ)", "Val loss", "Val Macro-F1"]})


def _tab_detection(run, has_noise):
    if not has_noise:
        note("За η = 0 немає навмисно змінених міток: AP, PR-AUC і Recall не визначені. Цей запуск — нульовий контроль "
             "впливу самого вилучення на Macro-F1 (див. «Очищення»).", "warn")
    det = run.detection
    section("Метрики виявлення за фіксованого бюджету K", "Відсортовано за AP. TP — знайдені навмисно змінені мітки, "
            "FP — коректні зразки, позначені підозрілими.", hint="sec_detection")
    st.dataframe(det[["method", "AP", "PR-AUC", "ROC-AUC", "Precision@K", "Recall@K", "F1@K", "TP", "FP", "FN", "K"]],
                 hide_index=True, column_config={"method": "Метод", **{c: NUM3(c) for c in
                                                                     ["AP", "PR-AUC", "ROC-AUC", "Precision@K", "Recall@K", "F1@K"]}})
    if not has_noise:
        return
    c1, c2 = st.columns(2)
    with c1:
        show(charts.metric_bars(det, "AP", "Average Precision"), key="ap_bars")
    with c2:
        show(charts.metric_bars(det, "Precision@K", f"Precision@K, K = {run.manifest['budget_K']}"), key="pk_bars")
    A = run.arrays
    curves = {k: pr_curve(A["z"], A[f"score_{k}"]) for k in det.key}
    c1, c2 = st.columns(2)
    with c1:
        show(charts.pr_curves(curves), key="pr")
    with c2:
        show(charts.budget_curves(run.budget_curve, run.cfg.noise_rate), key="budget")
    if len(run.bootstrap):
        section("Парні порівняння з R (стратифікований бутстреп)",
                f"ΔAP = AP(R) − AP(метод), {run.cfg.bootstrap} повторів. Інтервал, що не перетинає 0, вказує на стійку різницю "
                "в межах цього сценарію (одного seed недостатньо для загального висновку).", hint="sec_pairs")
        show(charts.delta_ci(run.bootstrap, "delta_AP", "ΔAP запропонованого методу відносно інших"), key="boot")
        b = run.bootstrap.set_index("versus")
        if {"dynamics", "semantic"} <= set(b.index):
            ok = all(b.loc[k, "delta_AP"] > 0 for k in ("dynamics", "semantic"))
            strict = all(b.loc[k, "ci_low"] > 0 for k in ("dynamics", "semantic"))
            text = ("<b>H1 у цьому сценарії:</b> AP(R) " + ("перевищує" if ok else "не перевищує") +
                    " AP обох складових окремо" + ("; обидва 95 % CI вище нуля." if strict else
                                                  "; принаймні один інтервал перетинає 0 — різниця не є стійкою." if ok else "."))
            note(text, "ok" if strict else "warn" if ok else "info")


def _tab_epochs(run, has_noise):
    if not has_noise:
        note("Аналіз динаміки виявлення потребує η &gt; 0.", "info")
        return
    em = run.epoch_metrics
    c1, c2 = st.columns(2)
    with c1:
        show(charts.epoch_ap(em, "AP"), key="ep_ap")
    with c2:
        show(charts.epoch_ap(em, "Precision@K"), key="ep_pk")
    piv = em.pivot(index="epoch", columns="key", values="AP")
    loss_best = int(piv["loss"].idxmax())
    r_best = int(piv["proposed"].idxmax())
    drop = piv["loss"].max() - piv["loss"].iloc[-1]
    msg = (f"Найкраща AP для Loss — після епохи {loss_best} ({piv['loss'].max():.3f}); на останній епосі — "
           f"{piv['loss'].iloc[-1]:.3f}. Найкраща AP для R — після епохи {r_best} ({piv['proposed'].max():.3f}). ")
    msg += ("Падіння якості Loss на пізніх епохах свідчить про запам'ятовування помилкових міток: діагностика за "
            "раннім припиненням або накопиченою історією доцільніша." if drop > 0.01 else
            "Суттєвого погіршення Loss на пізніх епохах не виявлено.")
    note(msg, "info")
    section("AP за епохами", hint="sec_epoch_ap")
    st.dataframe(piv.rename(columns={"C": "C", "D": "D", "loss": "Loss", "proposed": "R"}).reset_index()
                 .rename(columns={"epoch": "Епоха"}), hide_index=True,
                 column_config={c: NUM3(c) for c in ["C", "D", "Loss", "R"]})


def _calibration_section(run, det):
    cal = run.calibration
    section("Калібрування ваг на окремій підвибірці",
            "Ваги підібрано за AP на прапорцях калібрувальної підвибірки з власним шумом; цільовий train і test не використовувались.", hint="sec_calibration")
    mode = CALIB_MODES.get(cal["mode"], cal["mode"])
    kpis([{"label": "Адаптивні ваги α / β / γ", "value": f"{cal['alpha']:.2f} / {cal['beta']:.2f} / {cal['gamma']:.2f}",
           "sub": f"апріорні: {run.cfg.alpha:.2f} / {run.cfg.beta:.2f} / {run.cfg.gamma:.2f}", "color": METHOD_COLORS["adaptive"]},
          {"label": "AP на calib (адаптивні)", "value": f"{cal['ap_calib']:.3f}", "delta": cal["ap_calib"] - cal["ap_calib_fixed"],
           "sub": f"апріорні ваги: {cal['ap_calib_fixed']:.3f}", "color": METHOD_COLORS["adaptive"]},
          {"label": "Калібрувальна вибірка", "value": f"{cal['n_calib']}",
           "sub": f"{mode}; η = {cal['noise_rate']:.0%}, seed {cal['seed']}", "color": METHOD_COLORS["noisy"]}])
    if run.calibration_table is not None and len(run.calibration_table):
        show(charts.weight_heatmap(run.calibration_table, (run.cfg.alpha, run.cfg.beta),
                                   best=(cal["alpha"], cal["beta"]),
                                   title="AP на калібрувальній підвибірці (не на цільовому train)"), key="cal_heat")
    if {"proposed", "adaptive"} <= set(det.index) and not np.isnan(det.AP["proposed"]):
        d = det.AP["adaptive"] - det.AP["proposed"]
        note(f"На цільовому train: AP(адаптивні) = {det.AP['adaptive']:.3f}, AP(фіксовані) = {det.AP['proposed']:.3f} "
             f"(Δ = {d:+.3f}). Один сценарій не дає підстав для висновку — порівнюйте за серіями.",
             "ok" if d > 0 else "info")
    if cal["mode"] == "known_type":
        note("Режим «тип шуму відомий» припускає, що тип шуму реальних даних відомий. Для чесної оцінки в роботі "
             "використовуйте також тип-агностичний режим.", "warn")


def _tab_weights(run, has_noise):
    det = run.detection.set_index("key")
    section("Абляція компонентів (E4)", hint="sec_ablation")
    ab = pd.DataFrame([{"Варіант": n, "AP": det.AP.get(k), "Precision@K": det["Precision@K"].get(k),
                        "ROC-AUC": det["ROC-AUC"].get(k)} for k, n in
                       [("dynamics", "R(C, D, 0) — лише динаміка"), ("semantic", "R(0, 0, S) — лише семантика"),
                        ("proposed", "R(C, D, S) — повна модель, апріорні ваги"),
                        ("adaptive", "R(C, D, S) — ваги з калібрування")] if k in det.index])
    st.dataframe(ab, hide_index=True, column_config={c: NUM3(c) for c in ["AP", "Precision@K", "ROC-AUC"]})
    if run.calibration:
        _calibration_section(run, det)
    if not has_noise:
        return
    section("Аналіз чутливості до ваг", "Перерахунок R зі збережених C, D, S без повторного навчання.", hint="sec_sensitivity")
    note("Цей аналіз використовує zᵢ і призначений лише для дослідження чутливості. Підбирати ваги за ним для "
         "цього ж сценарію некоректно — це був би витік цільової інформації (розділ 7 плану).", "warn")
    A = run.arrays
    c, d, s, z = A["comp_C"], A["comp_D"], A["comp_S"], A["z"]
    col1, col2 = st.columns([1, 1.6])
    with col1:
        a = st.slider("α", 0.0, 1.0, float(run.cfg.alpha), 0.05, key="w_a", help=tip("w_a"))
        b = st.slider("β", 0.0, 1.0, float(run.cfg.beta), 0.05, key="w_b", help=tip("w_b"))
        if a + b > 1:
            note("α + β &gt; 1", "warn")
        else:
            g = 1 - a - b
            r = combine(c, d, s, a, b, g)
            mm = detection_metrics(z, r, run.manifest["budget_K"])
            kpis([{"label": f"AP за γ = {g:.2f}", "value": f"{mm['AP']:.3f}", "delta": mm["AP"] - det.AP["proposed"],
                   "sub": "відносно апріорних ваг", "color": METHOD_COLORS["proposed"]},
                  {"label": "Precision@K", "value": f"{mm['Precision@K']:.3f}",
                   "delta": mm["Precision@K"] - det["Precision@K"]["proposed"], "sub": "", "color": METHOD_COLORS["proposed"]}])
    with col2:
        grid = _weight_grid(str(run.path), c, d, s, z)
        show(charts.weight_heatmap(grid, (run.cfg.alpha, run.cfg.beta)), key="wh")
        best = grid.loc[grid.AP.idxmax()]
        st.caption(f"Максимум AP на сітці: α = {best.alpha:.2f}, β = {best.beta:.2f}, γ = {1 - best.alpha - best.beta:.2f} "
                   f"→ AP = {best.AP:.3f} (оптимістична оцінка, отримана на цільових мітках).")


def _tab_cleaning(run):
    full = run.classification
    if len(full) <= 1:
        note("Перенавчання після очищення в цьому запуску не виконувалось.", "info")
        if len(full):
            st.dataframe(full[["method", "macro_f1", "accuracy", "weighted_f1", "n_train"]], hide_index=True,
                         column_config={"method": "Метод", "macro_f1": NUM3("Macro-F1"), "accuracy": NUM3("Accuracy"),
                                        "weighted_f1": NUM3("Weighted-F1"), "n_train": "N train"})
        return
    budgets = run.budgets
    budget = run.cfg.budget_frac
    if len(budgets) > 1:
        budget = st.pills("Бюджет вилучення K", budgets, default=run.cfg.budget_frac if run.cfg.budget_frac in budgets
                          else budgets[-1], format_func=lambda b: f"{b:.0%}", key="clean_budget", help=tip("view_budget")) or budget
        section("Macro-F1 залежно від бюджету",
                "Пунктир — навчання без очищення. Для oracle вилучається не більше K змінених зразків.", hint="sec_f1_budget")
        show(charts.budget_f1_lines(full), key="clf_budget")
    cls = budget_view(full, budget)
    show(charts.classification_bars(cls), key="clf")
    st.caption("Вусики — 95 % парний бутстреп-інтервал різниці Macro-F1 відносно навчання без очищення, "
               "відкладений від значення методу.")
    cols = [c for c in ["method", "macro_f1", "val_macro_f1", "accuracy", "weighted_f1", "n_train", "removed",
                        "removed_corrupted", "downweighted", "weight_mean", "delta_macro_f1", "ci_low", "ci_high", "delta_vs_random",
                        "ci_low_vs_random", "ci_high_vs_random"] if c in cls]
    t = cls[cols].copy()
    st.dataframe(t, hide_index=True, column_config={
        "method": "Метод", "macro_f1": NUM3("Macro-F1 (test)"), "val_macro_f1": NUM3("Macro-F1 (val)"),
        "accuracy": NUM3("Accuracy"), "weighted_f1": NUM3("Weighted-F1"),
        "n_train": "N train", "removed": "Вилучено", "removed_corrupted": "з них змінених", "downweighted": "Знижено вагу",
        "weight_mean": NUM3("Середня вага"),
        "delta_macro_f1": NUM3("Δ vs шум"), "ci_low": NUM3("CI low"), "ci_high": NUM3("CI high"),
        "delta_vs_random": NUM3("Δ vs random"), "ci_low_vs_random": NUM3("CI low (rnd)"),
        "ci_high_vs_random": NUM3("CI high (rnd)")})
    c = cls.set_index("key")
    if "weighted" in c.index and "downweighted" in c:
        note(f"<b>М'яке зважування:</b> усі {int(c.n_train['weighted'])} текстів лишаються в навчанні, "
             f"{int(c.downweighted['weighted'])} найризикованіших мають вагу max(w_min; (1 − R)^p) = "
             f"max({run.cfg.weight_min:g}; (1 − R)^{run.cfg.weight_power:g}); Macro-F1 {c.macro_f1['weighted']:.3f}. "
             "Втрати нормуються як Σ wᵢℓᵢ / Σ wᵢ.", "info")
    if {"proposed", "random", "noisy"} <= set(c.index):
        better_noisy = c.macro_f1["proposed"] > c.macro_f1["noisy"]
        better_rand = c.macro_f1["proposed"] > c.macro_f1["random"]
        sig = c.ci_low["proposed"] > 0
        note(f"<b>H2 у цьому сценарії (K = {budget:.0%}):</b> Macro-F1 після очищення за R = {c.macro_f1['proposed']:.3f}; без очищення — "
             f"{c.macro_f1['noisy']:.3f}; випадкове вилучення — {c.macro_f1['random']:.3f}. "
             + ("Очищення за R краще за обидва контролі" if better_noisy and better_rand else
                "Очищення за R не перевершило обидва контролі") +
             ("; інтервал ΔF1 вище нуля." if sig else "; інтервал ΔF1 перетинає 0 або від'ємний.") +
             " Негативний або нульовий ефект — теж результат: вилучення корисних неоднозначних прикладів і зміна "
             "балансу класів можуть погіршити класифікатор.", "ok" if better_noisy and better_rand and sig else "warn")

    # склад вилучених за класами
    rep = run.report.set_index("id")
    rem = run.removed_ids(budget)
    if rem:
        comp = pd.DataFrame({k: rep.observed_label.reindex(ids).value_counts() for k, ids in rem.items()}).T.fillna(0)
        base = rep.observed_label.value_counts(normalize=True) * max(1, round(len(rep) * budget))
        comp.loc["train_expected"] = base
        comp = comp[sorted(comp.columns)]
        section("Контроль балансу класів серед вилучених",
                "Рядок «Пропорційно train» — скільки текстів кожного класу було б вилучено пропорційно до їхньої частки.", hint="sec_balance")
        fig = charts.removed_composition(comp.rename(index={"train_expected": "Пропорційно train"}))
        show(fig, key="comp")

    A = run.arrays
    rows = []
    for k in [k for k in ("noisy", "random", "loss", "cleanlab", "aum", "proposed", "adaptive", "weighted", "oracle") if f"pred_{k}" in A]:
        f1s = f1_score(A["test_y"], A[f"pred_{k}"], average=None, labels=range(len(run.labels)), zero_division=0)
        rows += [{"key": k, "class": run.labels[i], "f1": v} for i, v in enumerate(f1s)]
    show(charts.per_class_f1(pd.DataFrame(rows)), key="pcf1")


def _tab_errors(run, has_noise):
    if not has_noise:
        note("Аналіз хибних спрацювань потребує η &gt; 0.", "info")
        return
    A = run.arrays
    K = run.manifest["budget_K"]
    rep = run.report.set_index("id").sort_index()
    sel = np.zeros(len(rep), dtype=bool)
    sel[top_k(A["score_proposed"], K)] = True
    z = A["z"].astype(bool)
    grp = np.where(sel & z, "TP у top-K", np.where(sel & ~z, "FP у top-K", "Поза top-K"))
    df = pd.DataFrame({"class": rep.observed_label.values, "length": rep.length.values, "group": grp})
    per = df[df.group != "Поза top-K"].groupby("class").group.value_counts().unstack(fill_value=0)
    per = per.reindex(columns=["TP у top-K", "FP у top-K"], fill_value=0).rename(
        columns={"TP у top-K": "TP", "FP у top-K": "FP"}).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        show(charts.class_flag_share(per), key="cfs")
    with c2:
        show(charts.length_box(df), key="lbox")
    short_q = np.percentile(rep.length, 25)
    fp_short = ((rep.length.values <= short_q) & sel & ~z).sum() / max((sel & ~z).sum(), 1)
    note(f"Частка коротких текстів (≤ {short_q:.0f} символів, нижній квартиль) серед хибних спрацювань: {fp_short:.0%} "
         "(очікувано ≈25 % за відсутності залежності від довжини).", "info")

    cols = ["text", "original_dataset_label", "observed_label", "proposed_risk", "C_confidence_risk",
            "D_disagreement_risk", "S_semantic_risk"]
    cc = {"text": st.column_config.TextColumn("Текст", width="large"), "original_dataset_label": "Мітка датасету",
          "observed_label": "Спостережувана", "proposed_risk": NUM3("R"), "C_confidence_risk": NUM3("C"),
          "D_disagreement_risk": NUM3("D"), "S_semantic_risk": NUM3("S")}
    section("Хибні спрацювання з найвищим ризиком",
            "Мітка не змінювалась, але метод вважає її підозрілою — кандидати на реальні помилки датасету або неоднозначні тексти.", hint="sec_fp")
    fp = rep[sel & ~z].sort_values("proposed_risk", ascending=False).head(15)
    st.dataframe(fp[cols], hide_index=True, column_config=cc)
    section("Пропущені змінені мітки з найнижчим ризиком", "Змінені мітки, які метод вважає надійними.", hint="sec_fn")
    fn = rep[~sel & z].sort_values("proposed_risk").head(15)
    st.dataframe(fn[cols], hide_index=True, column_config=cc)


def _tab_candidates(run):
    rep = run.report.copy()
    note("Сліпий режим (за замовчуванням) приховує вихідні мітки та індикатор штучного шуму — у такому вигляді "
         "список передається експертам. Дослідницький режим — лише для оцінювання.", "info")
    c1, c2, c3, c4 = st.columns([2, 2, 1.2, 1.2])
    q = c1.text_input("Пошук у тексті", key="cand_q", placeholder="наприклад: матч, бюджет, смартфон…", help=tip("cand_q"))
    classes = c2.multiselect("Спостережувані класи", run.labels, key="cand_cls", placeholder="Усі класи", help=tip("cand_cls"))
    top = c3.number_input("Показати top-N", 10, len(rep), min(run.manifest["budget_K"], len(rep)), 10, key="cand_n", help=tip("cand_n"))
    research = c4.toggle("Дослідницький режим", False, key="cand_res", help=tip("cand_research"))
    v = rep.sort_values("proposed_risk", ascending=False).head(int(top))
    if q:
        v = v[v.text.str.contains(q, case=False, na=False, regex=False)]
    if classes:
        v = v[v.observed_label.isin(classes)]
    cols = ["rank", "text", "observed_label", "proposed_risk", "C_confidence_risk", "D_disagreement_risk", "S_semantic_risk"]
    if research:
        cols += ["original_dataset_label", "synthetic_corrupted"] + (["cleanlab_risk"] if "cleanlab_risk" in v else [])
    prog = lambda label: st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.3f")
    st.dataframe(v[cols], hide_index=True, height=520, column_config={
        "rank": st.column_config.NumberColumn("Ранг", width="small"),
        "text": st.column_config.TextColumn("Текст", width="large"), "observed_label": "Мітка",
        "proposed_risk": prog("Ризик R"), "C_confidence_risk": prog("C"), "D_disagreement_risk": prog("D"),
        "S_semantic_risk": prog("S"), "original_dataset_label": "Мітка датасету",
        "synthetic_corrupted": st.column_config.CheckboxColumn("Змінена"), "cleanlab_risk": NUM3("Cleanlab")})
    st.caption(f"Показано {len(v)} рядків.")
    d1, d2 = st.columns(2)
    d1.download_button("Сліпий реєстр для експертів (CSV)",
                       (run.path / "review_candidates_for_experts.csv").read_bytes(),
                       "review_candidates_for_experts.csv", "text/csv", icon=":material/download:", type="primary",
                       help=tip("btn_csv_blind"))
    d2.download_button("Повний дослідницький реєстр (CSV)",
                       (run.path / "review_candidates_research_only.csv").read_bytes(),
                       "review_candidates_research_only.csv", "text/csv", icon=":material/download:",
                       help=tip("btn_csv_full"))


def _zip_bytes(path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in path.iterdir():
            if p.is_file():
                zf.write(p, p.name)
    return buf.getvalue()


def _tab_manifest(run):
    m = run.manifest
    c1, c2 = st.columns([1.1, 1])
    with c1:
        show(charts.timings_bar(m["timings"]), key="tim")
        note("Метод R використовує одне навчання з реєстрацією динаміки (+ заморожена семантична модель), тоді як "
             "Cleanlab OOF потребує K додаткових навчань на фолдах.", "info")
        st.download_button("Завантажити всі артефакти запуску (ZIP)", _zip_bytes(run.path), f"{run.path.name}.zip",
                           "application/zip", icon=":material/folder_zip:", type="primary", help=tip("btn_zip"))
        files = sorted(p.name for p in run.path.iterdir() if p.is_file())
        st.caption("Файли: " + ", ".join(files))
    with c2:
        section("run_manifest.json")
        st.json({k: v for k, v in m.items() if k != "removed_ids"}, expanded=1)
        st.caption(f"Пристрій: {m['device']} · семантична модель: {EMBEDDINGS.get(run.cfg.embedding)}")
