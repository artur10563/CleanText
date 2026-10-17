"""Повний експериментальний конвеєр: дані → шум → навчання → ризик → виявлення → очищення."""
from __future__ import annotations

import json
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import __version__
from .aum import compute_aum
from .calibration import calibrate_weights
from .config import RUNS_DIR, ExperimentConfig
from .data import load_corpus
from .embeddings import compute_embeddings, nearest_neighbours
from .metrics import (bootstrap_ap_diff, bootstrap_f1_diff, clf_metrics, detection_metrics,
                      precision_at_curve, top_k)
from .models import DEVICE, fit_model
from .noise import add_noise
from .scoring import METHOD_NAMES, all_scores, combine, soft_weights, cartography, cleanlab_flags, per_epoch_scores

BUDGET_FRACS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


class Progress:
    """Зважений прогрес етапів; callback(frac, text)."""

    def __init__(self, callback=None, log=print):
        self.cb = callback or (lambda f, t: None)
        self.log = log
        self.stages: list[tuple[str, float]] = []
        self.done = 0.0
        self.total = 1.0

    def plan(self, stages):
        self.stages = stages
        self.total = sum(w for _, w in stages)
        self.done = 0.0

    def start(self, name):
        self.cur = name
        self.cur_w = dict(self.stages).get(name, 0.0)
        self.cb(self.done / self.total, name)
        self.log(f"▶ {name}")

    def sub(self, frac, text=None):
        self.cb(min(1.0, (self.done + self.cur_w * frac) / self.total), text or self.cur)

    def finish(self):
        self.done += self.cur_w
        self.cb(min(1.0, self.done / self.total), self.cur)


def _versions():
    import sklearn
    import torch
    v = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__,
         "numpy": np.__version__, "pandas": pd.__version__, "cleantext": __version__}
    for mod in ("transformers", "sentence_transformers", "cleanlab"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    return v


def prepare_data(cfg: ExperimentConfig, log=print, csv_df=None):
    corpus = load_corpus(cfg, log=log, csv_df=csv_df)
    emb = compute_embeddings(corpus.train.text.tolist(), cfg.embedding, log=log)
    neighbours = nearest_neighbours(emb, max(cfg.k_neighbors + 1, 35))
    return corpus, emb, neighbours


def run_experiment(cfg: ExperimentConfig, log=print, on_progress=None, csv_df=None,
                   prepared=None, out_dir: Path | None = None, save_texts: bool = True,
                   calib_cache: Path | None = None) -> Path:
    cfg.validate()
    pr = Progress(on_progress, log)
    retrain_budgets = cfg.all_budgets()
    n_strat = 3 + cfg.run_oracle + cfg.run_cleanlab + cfg.calibrate_weights + cfg.run_weighted + cfg.run_aum
    stages = [("Підготовка корпусу та семантичних подань", 1.0), ("Внесення шуму", 0.1),
              ("Навчання базової нейромережі", 3.0)]
    if cfg.calibrate_weights and cfg.n_calib > 0:
        stages.append(("Калібрування ваг α, β, γ", 0.8))
    if cfg.run_cleanlab:
        stages.append(("Cleanlab: out-of-fold передбачення", cfg.oof_folds * cfg.oof_epochs / max(cfg.epochs, 1) * 2.5))
    if cfg.run_aum:
        stages.append(("AUM: два прогони з пороговими зразками", 4.5))
    stages.append(("Обчислення ризику та метрик виявлення", 0.4))
    if cfg.run_retrain:
        stages.append(("Повторне навчання після очищення",
                       2.7 * len(retrain_budgets) * n_strat))
    stages.append(("Збереження результатів", 0.2))
    pr.plan(stages)
    timings = {}

    # E0. Дані ---------------------------------------------------------------
    pr.start(stages[0][0]); t0 = time.time()
    corpus, emb, neighbours = prepared if prepared is not None else prepare_data(cfg, log, csv_df)
    timings["data_and_embeddings_s"] = time.time() - t0
    K = corpus.num_labels
    train, val, test = corpus.train, corpus.val, corpus.test
    pr.finish()

    # E1. Шум ---------------------------------------------------------------
    pr.start("Внесення шуму")
    clean_y = train.y.to_numpy()
    noisy_y, z, transition = add_noise(clean_y, cfg.noise_type, cfg.noise_rate, cfg.seed, K, neighbours)
    log(f"Навмисно змінено {z.sum()} з {len(z)} міток ({z.mean():.1%})")
    pr.finish()

    # Базове навчання з історією ---------------------------------------------
    pr.start("Навчання базової нейромережі"); t0 = time.time()
    texts = train.text.tolist()
    base = fit_model(cfg, texts, noisy_y, K, cfg.seed, cfg.epochs, track=True,
                     val=(val.text.tolist(), val.y.to_numpy()),
                     progress=lambda e, n: pr.sub(e / n, f"Навчання базової нейромережі: епоха {int(np.ceil(e))}/{n}"))
    history = base.history.astype(np.float32)
    test_probs = base.predict_proba(test.text.tolist())
    learning_log = base.log
    base.release()
    timings["base_training_s"] = time.time() - t0
    pr.finish()

    # Калібрування ваг на окремій підвибірці (не бачить train, z і test) -------------
    calib_res = None
    if cfg.calibrate_weights and cfg.n_calib > 0:
        pr.start("Калібрування ваг α, β, γ"); t0 = time.time()
        if corpus.calib is None:
            log("Калібрування пропущено: калібрувальну вибірку не сформовано (замалий пул)")
        else:
            calib_res = calibrate_weights(
                cfg, corpus.calib, K, log=log, cache_dir=calib_cache,
                progress=lambda e, n: pr.sub(e / n, f"Калібрування: епоха {int(np.ceil(e))}/{n}"))
        timings["calibration_s"] = time.time() - t0
        pr.finish()

    # Cleanlab OOF ----------------------------------------------------------
    oof = None
    if cfg.run_cleanlab:
        pr.start("Cleanlab: out-of-fold передбачення"); t0 = time.time()
        folds = min(cfg.oof_folds, int(pd.Series(noisy_y).value_counts().min()))
        if folds < 2:
            log("Замало зразків у класі для K-fold — Cleanlab пропущено")
        else:
            oof = np.zeros((len(noisy_y), K), dtype=np.float32)
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)
            for f, (tr, ho) in enumerate(skf.split(np.zeros(len(noisy_y)), noisy_y)):
                m = fit_model(cfg, [texts[i] for i in tr], noisy_y[tr], K, cfg.seed + 1000 + f, cfg.oof_epochs,
                              progress=lambda e, n, f=f: pr.sub((f + e / n) / folds, f"Cleanlab OOF: фолд {f + 1}/{folds}"))
                oof[ho] = m.predict_proba([texts[i] for i in ho])
                m.release()
        timings["cleanlab_oof_s"] = time.time() - t0
        pr.finish()

    # AUM (Pleiss et al., 2020): фіктивний клас K+1, два прогони --------------------
    aum_res = None
    if cfg.run_aum:
        pr.start("AUM: два прогони з пороговими зразками"); t0 = time.time()
        aum_res = compute_aum(cfg, texts, noisy_y, K, cfg.aum_percentile, log=log,
                              progress=lambda f: pr.sub(f, f"AUM: прогін {int(f * 2) + 1}/2"))
        timings["aum_s"] = time.time() - t0
        pr.finish()

    # E2/E4. Ризик і виявлення ----------------------------------------------
    pr.start("Обчислення ризику та метрик виявлення"); t0 = time.time()
    scores, comps = all_scores(history, noisy_y, neighbours, cfg.k_neighbors, cfg.alpha, cfg.beta, cfg.gamma,
                               cfg.seed, oof)
    if calib_res is not None:
        scores["adaptive"] = combine(comps["C"], comps["D"], comps["S"], *calib_res.weights)
    if aum_res is not None:
        scores["aum"] = aum_res.score
    conf, var, corr = cartography(history, noisy_y)
    timings["risk_scoring_s"] = time.time() - t0
    cl_flags = cleanlab_flags(noisy_y, oof) if oof is not None else None
    budget = max(1, int(round(len(noisy_y) * cfg.budget_frac)))
    det_rows = [{"key": k, "method": METHOD_NAMES[k], **detection_metrics(z, s, budget)} for k, s in scores.items()]
    detection = pd.DataFrame(det_rows).sort_values("AP", ascending=False, na_position="last")

    budget_rows = []
    for k, s in scores.items():
        for r in precision_at_curve(z, s, BUDGET_FRACS):
            budget_rows.append({"key": k, "method": METHOD_NAMES[k], **r})
    budget_curve = pd.DataFrame(budget_rows)

    epoch_rows = []
    for rec in per_epoch_scores(history, noisy_y, neighbours, cfg.k_neighbors, cfg.alpha, cfg.beta, cfg.gamma):
        for k in ("C", "D", "loss", "proposed"):
            epoch_rows.append({"epoch": rec["epoch"], "key": k, **detection_metrics(z, rec[k], budget)})
    epoch_metrics = pd.DataFrame(epoch_rows)

    boot_rows = []
    if z.sum() > 0:
        for ref in [r for r in ("proposed", "adaptive") if r in scores]:
            for k in scores:
                if k == ref:
                    continue
                d, lo, hi = bootstrap_ap_diff(z, scores[ref], scores[k], cfg.bootstrap, cfg.seed)
                boot_rows.append({"reference": ref, "versus": k, "method": METHOD_NAMES[k], "delta_AP": d,
                                  "ci_low": lo, "ci_high": hi})
    bootstrap = pd.DataFrame(boot_rows)
    pr.finish()

    # E3. Очищення та перенавчання --------------------------------------------
    # Протокол: усі моделі (базова й перенавчені) — один seed; однаковий K для всіх стратегій у межах бюджету.
    val_texts, val_y = val.text.tolist(), val.y.to_numpy()
    test_y = test.y.to_numpy()
    base_val_f1 = float(learning_log["val_macro_f1"].iloc[-1]) if "val_macro_f1" in learning_log else np.nan
    preds = {"noisy": test_probs.argmax(1)}
    cls_rows = [{"budget_frac": np.nan, "K": np.nan, "key": "noisy", "method": "Без очищення (зашумлені дані)",
                 **clf_metrics(test.y, test_probs), "val_macro_f1": base_val_f1, "n_train": len(noisy_y),
                 "removed": 0, "removed_corrupted": 0, "downweighted": 0, "weight_mean": 1.0,
                 "budget_mismatch": False}]
    removed_ids: dict = {}
    downweighted_ids: dict = {}
    removed_rows = []
    retrain_preds = {}
    if cfg.run_retrain:
        pr.start(stages[-2][0]); t0 = time.time()
        strategies = ["random", "loss", "proposed"] + (["adaptive"] if "adaptive" in scores else []) + (["weighted"] if cfg.run_weighted else []) + (["cleanlab"] if "cleanlab" in scores else []) + (["aum"] if "aum" in scores else [])
        if cfg.run_oracle and z.sum() > 0:
            strategies.append("oracle")
        oracle_order = np.random.default_rng(cfg.seed + 989).permutation(np.where(z == 1)[0])
        total_fits = len(retrain_budgets) * len(strategies)
        done_fits = 0
        for b in retrain_budgets:
            k_b = max(1, int(round(len(noisy_y) * b)))
            removed_ids[f"{b:.2f}"] = {}
            downweighted_ids[f"{b:.2f}"] = {}
            for key in strategies:
                keep = np.ones(len(noisy_y), dtype=bool)
                weights, picks = None, np.array([], dtype=int)
                if key == "oracle":
                    keep[oracle_order[:min(k_b, len(oracle_order))]] = False
                elif key == "weighted":
                    # м'яке зважування (v2 adaptive_weighted): усі тексти лишаються, ризикові мають меншу вагу
                    weights, picks = soft_weights(scores["adaptive" if "adaptive" in scores else "proposed"], k_b,
                                                  cfg.weight_min, cfg.weight_power, cfg.weight_scope)
                    downweighted_ids[f"{b:.2f}"][key] = train.id.to_numpy()[picks].tolist()
                else:
                    keep[top_k(scores[key], k_b)] = False
                if key != "weighted":
                    removed_ids[f"{b:.2f}"][key] = train.id.to_numpy()[~keep].tolist()
                fit_kw = {} if weights is None else {"sample_weight": weights}
                m = fit_model(cfg, [texts[i] for i in np.where(keep)[0]], noisy_y[keep], K, cfg.seed, cfg.epochs,
                              val=(val_texts, val_y), **fit_kw,
                              progress=lambda e, n, j=done_fits, key=key, b=b: pr.sub(
                                  (j + e / n) / total_fits, f"Перенавчання: {key}, K = {b:.0%}"))
                probs = m.predict_proba(test.text.tolist())
                vf1 = float(m.log["val_macro_f1"].iloc[-1])
                m.release()
                done_fits += 1
                retrain_preds[(b, key)] = probs.argmax(1)
                name = "Oracle (виключено змінені, ≤ K)" if key == "oracle" else METHOD_NAMES[key]
                cls_rows.append({"budget_frac": b, "K": k_b, "key": key, "method": name,
                                 **clf_metrics(test.y, probs), "val_macro_f1": vf1, "n_train": int(keep.sum()),
                                 "removed": int((~keep).sum()), "removed_corrupted": int(z[~keep].sum()),
                                 "downweighted": int(len(picks)) if weights is not None else 0,
                                 "weight_mean": float(weights.mean()) if weights is not None else 1.0,
                                 "budget_mismatch": bool(key == "oracle" and k_b > z.sum())})
                for cls_id, cnt in pd.Series(noisy_y[~keep]).value_counts().items():
                    removed_rows.append({"budget_frac": b, "key": key, "class": corpus.labels[int(cls_id)],
                                         "count": int(cnt)})
        timings["retraining_s"] = time.time() - t0
        pr.finish()
    classification = pd.DataFrame(cls_rows)
    f1_boot = []
    for (b, key), p in retrain_preds.items():
        d, lo, hi = bootstrap_f1_diff(test_y, p, preds["noisy"], cfg.bootstrap, cfg.seed)
        row = {"budget_frac": b, "key": key, "delta_macro_f1": d, "ci_low": lo, "ci_high": hi,
               "delta_vs_random": np.nan, "ci_low_vs_random": np.nan, "ci_high_vs_random": np.nan}
        if key != "random" and (b, "random") in retrain_preds:
            row["delta_vs_random"], row["ci_low_vs_random"], row["ci_high_vs_random"] = bootstrap_f1_diff(
                test_y, p, retrain_preds[(b, "random")], cfg.bootstrap, cfg.seed)
        f1_boot.append(row)
    boot_cols = ["budget_frac", "key", "delta_macro_f1", "ci_low", "ci_high",
                 "delta_vs_random", "ci_low_vs_random", "ci_high_vs_random"]
    classification = classification.merge(pd.DataFrame(f1_boot, columns=boot_cols), on=["budget_frac", "key"], how="left")
    preds.update({f"{key}": p for (b, key), p in retrain_preds.items() if abs(b - cfg.budget_frac) < 1e-9})
    removed_by_class = pd.DataFrame(removed_rows, columns=["budget_frac", "key", "class", "count"])

    # Збереження ---------------------------------------------------------------
    pr.start("Збереження результатів")
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = RUNS_DIR / f"{stamp}_{cfg.dataset}_{cfg.noise_type}_{cfg.noise_rate:.2f}_s{cfg.seed}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = train[["id", "text", "label", "length"]].rename(columns={"label": "original_dataset_label"}).copy()
    report["observed_label"] = [corpus.labels[v] for v in noisy_y]
    report["synthetic_corrupted"] = z
    report["C_confidence_risk"] = comps["C"]
    report["D_disagreement_risk"] = comps["D"]
    report["S_semantic_risk"] = comps["S"]
    report["proposed_risk"] = scores["proposed"]
    report["rank"] = pd.Series(scores["proposed"]).rank(ascending=False, method="min").astype(int).to_numpy()
    report["confidence"], report["variability"], report["correctness"] = conf, var, corr      # Dataset Cartography
    if aum_res is not None:
        report["aum"] = aum_res.aum
        report["aum_flag"] = aum_res.flags
        report["aum_run"] = aum_res.run_of
    if "cleanlab" in scores:
        report["cleanlab_risk"] = scores["cleanlab"]
        if cl_flags is not None:
            report["cleanlab_flag"] = cl_flags
    report = report.sort_values("proposed_risk", ascending=False)
    report.to_csv(out_dir / "review_candidates_research_only.csv", index=False, encoding="utf-8-sig")
    report[["id", "text", "observed_label", "proposed_risk", "rank"]].to_csv(
        out_dir / "review_candidates_for_experts.csv", index=False, encoding="utf-8-sig")

    learning_log.to_csv(out_dir / "learning_log.csv", index=False)
    detection.to_csv(out_dir / "detection_metrics.csv", index=False)
    classification.to_csv(out_dir / "classification_metrics.csv", index=False)
    removed_by_class.to_csv(out_dir / "removed_by_class.csv", index=False)
    if calib_res is not None:
        calib_res.table.to_csv(out_dir / "calibration_table.csv", index=False)
        (out_dir / "calibration.json").write_text(json.dumps(calib_res.to_json(), ensure_ascii=False, indent=2,
                                                             default=float), encoding="utf-8")
    budget_curve.to_csv(out_dir / "budget_curve.csv", index=False)
    epoch_metrics.to_csv(out_dir / "epoch_detection_metrics.csv", index=False)
    bootstrap.to_csv(out_dir / "bootstrap_delta_ap.csv", index=False)
    arrays = {"history": history, "clean_y": clean_y, "noisy_y": noisy_y, "z": z, "transition": transition,
              "test_y": test.y.to_numpy(), "test_probs": test_probs, "neighbours": neighbours[:, :cfg.k_neighbors],
              **{f"score_{k}": v for k, v in scores.items()}, **{f"comp_{k}": v for k, v in comps.items()},
              "cart_confidence": conf, "cart_variability": var, "cart_correctness": corr,
              **{f"pred_{k}": v for k, v in preds.items()}}
    if aum_res is not None:
        arrays.update({"aum": aum_res.aum, "aum_flags": aum_res.flags, "aum_run": aum_res.run_of})
    if oof is not None:
        arrays["oof"] = oof
        np.save(out_dir / "oof_probabilities.npy", oof)
    np.savez_compressed(out_dir / "arrays.npz", **arrays)
    if save_texts:
        for name, df in (("train", train), ("val", val), ("test", test)):
            df.to_parquet(out_dir / f"{name}.parquet", index=False)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "labels": corpus.labels,
        "data": corpus.stats,
        "budget_K": budget,
        "retrain_budgets": retrain_budgets,
        "coeff_fixed": [cfg.alpha, cfg.beta, cfg.gamma],
        "coeff_adaptive": list(calib_res.weights) if calib_res is not None else None,
        "calibration": calib_res.to_json() if calib_res is not None else None,
        "aum": ({**aum_res.summary(), "flag_precision": float(z[aum_res.flags].mean()) if aum_res.flags.any() else None,
                 "flag_recall": float(aum_res.flags[z == 1].mean()) if z.sum() else None}
                if aum_res is not None else None),
        "actual_injected_noise_rate": float(z.mean()),
        "n_corrupted": int(z.sum()),
        "device": str(DEVICE),
        "timings": timings,
        "removed_ids": removed_ids,
        "downweighted_ids": downweighted_ids,
        "versions": _versions(),
        "texts_dir": "." if save_texts else "..",
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=float)
    pr.finish()
    log(f"✔ Результати збережено: {out_dir}")
    return out_dir


def zip_run(run_dir: Path) -> Path:
    return Path(shutil.make_archive(str(run_dir), "zip", run_dir))
