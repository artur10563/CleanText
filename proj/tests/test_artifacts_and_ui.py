"""Сумісність артефактів (схеми CSV, старі запуски) і відкриття сторінок Streamlit через AppTest."""
import json
import shutil

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from cleantext.runs import budget_view, grid_tables, list_runs, load_run

DETECTION_COLS = {"key", "method", "AP", "PR-AUC", "ROC-AUC", "Precision@K", "Recall@K", "F1@K", "TP", "FP", "FN", "K"}
CLASSIFICATION_COLS = {"budget_frac", "K", "key", "method", "accuracy", "macro_f1", "weighted_f1", "val_macro_f1",
                       "n_train", "removed", "removed_corrupted", "budget_mismatch", "delta_macro_f1", "ci_low",
                       "ci_high", "delta_vs_random", "ci_low_vs_random", "ci_high_vs_random"}
BOOTSTRAP_COLS = {"reference", "versus", "method", "delta_AP", "ci_low", "ci_high"}
ALWAYS_FILES = ["learning_log.csv", "detection_metrics.csv", "classification_metrics.csv", "removed_by_class.csv",
                "budget_curve.csv", "epoch_detection_metrics.csv", "bootstrap_delta_ap.csv", "arrays.npz",
                "run_manifest.json", "review_candidates_research_only.csv", "review_candidates_for_experts.csv"]


# ---- схеми артефактів -------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("name", ["full", "no_calibration", "detection_only", "zero_noise"])
def test_artifact_schemas_are_stable(run_variants, name):
    out = run_variants[name]
    for f in ALWAYS_FILES:
        assert (out / f).exists(), f
    assert DETECTION_COLS <= set(pd.read_csv(out / "detection_metrics.csv").columns)
    assert CLASSIFICATION_COLS <= set(pd.read_csv(out / "classification_metrics.csv").columns)
    if name == "zero_noise":
        return                                      # без шуму бутстреп ΔAP порожній
    assert BOOTSTRAP_COLS <= set(pd.read_csv(out / "bootstrap_delta_ap.csv").columns)


@pytest.mark.slow
def test_calibration_artifacts_only_when_calibrated(run_variants):
    assert (run_variants["full"] / "calibration.json").exists()
    assert (run_variants["full"] / "calibration_table.csv").exists()
    for name in ("no_calibration",):
        assert not (run_variants[name] / "calibration.json").exists()


@pytest.mark.slow
@pytest.mark.parametrize("name", ["full", "no_calibration", "detection_only", "zero_noise"])
def test_every_variant_loads(run_variants, name):
    run = load_run(run_variants[name])
    assert len(run.detection) and len(run.classification)
    assert run.cfg.noise_rate == (0.0 if name == "zero_noise" else 0.2)


# ---- старий формат (до кроків 2 і 3) ----------------------------------------------------
@pytest.fixture
def legacy_run(run_variants, tmp_path):
    """Емуляція запуску v1: плоска класифікація без budget_frac, плаский removed_ids, без калібрування."""
    dst = tmp_path / "legacy"
    shutil.copytree(run_variants["no_calibration"], dst)
    cls = pd.read_csv(dst / "classification_metrics.csv")
    cls = budget_view(cls, 0.10)
    cls = cls[cls.key != "weighted"][["key", "method", "accuracy", "macro_f1", "weighted_f1", "n_train", "removed", "removed_corrupted",
                                       "delta_macro_f1", "ci_low", "ci_high"]]
    cls.to_csv(dst / "classification_metrics.csv", index=False)
    mf = json.loads((dst / "run_manifest.json").read_text(encoding="utf-8"))
    mf["removed_ids"] = {"proposed": [1, 2, 3]}
    mf["config"] = {k: v for k, v in mf["config"].items()
                    if k not in ("retrain_budgets", "n_calib", "calibrate_weights", "calib_mode", "calib_noise_rate",
                                 "calib_seed", "coeff_grid_step")}
    for k in ("retrain_budgets", "coeff_fixed", "coeff_adaptive", "calibration"):
        mf.pop(k, None)
    (dst / "run_manifest.json").write_text(json.dumps(mf, ensure_ascii=False), encoding="utf-8")
    (dst / "removed_by_class.csv").unlink()
    boot = pd.read_csv(dst / "bootstrap_delta_ap.csv").drop(columns=["reference"], errors="ignore")
    boot.to_csv(dst / "bootstrap_delta_ap.csv", index=False)
    return dst


@pytest.mark.slow
def test_legacy_run_loads_with_old_semantics(legacy_run):
    run = load_run(legacy_run)
    assert run.calibration is None and run.calibration_table is None and len(run.bootstrap_adaptive) == 0
    assert run.budgets == [run.cfg.budget_frac]
    assert run.removed_ids() == {"proposed": [1, 2, 3]}
    assert budget_view(run.classification, 0.2).equals(run.classification)


@pytest.mark.slow
def test_legacy_run_results_page_has_no_exceptions(legacy_run):
    at = _results(legacy_run).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.slow
def test_list_runs_and_grid_tables_with_new_runs(run_variants, tmp_path):
    root = tmp_path / "runs"
    for name in ("full", "no_calibration"):
        shutil.copytree(run_variants[name], root / name)
    rows = list_runs(root)
    assert {r["name"] for r in rows} == {"full", "no_calibration"}
    assert all(r["AP_R"] == r["AP_R"] for r in rows)                      # не NaN
    grid = tmp_path / "grid"
    shutil.copytree(run_variants["full"], grid / "semantic_0.20_seed7")
    (grid / "grid.json").write_text("{}", encoding="utf-8")
    det, cls = grid_tables(grid)
    assert len(det) and len(cls) and {"kind", "rate", "seed"} <= set(det.columns)


# ---- сторінки UI ------------------------------------------------------------------------
def _results(run_dir):
    code = f"""
import streamlit as st
st.session_state['run_path'] = {str(run_dir.resolve())!r}
from ui.pages import results
results()
"""
    return AppTest.from_string(code, default_timeout=180)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["full", "no_calibration", "detection_only", "zero_noise"])
def test_results_page_for_each_variant(run_variants, name):
    at = _results(run_variants[name]).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.slow
def test_results_page_shows_calibration_only_when_present(run_variants):
    full = _results(run_variants["full"]).run()
    off = _results(run_variants["no_calibration"]).run()
    text = lambda at: " ".join(str(m.value) for m in at.markdown)
    assert "Калібрування ваг на окремій підвибірці" in text(full)
    assert "Калібрування ваг на окремій підвибірці" not in text(off)


@pytest.mark.slow
@pytest.mark.parametrize("page", ["home", "experiment", "grid", "history", "method"])
def test_other_pages_open(page):
    at = AppTest.from_string(f"from ui.pages import {page}\n{page}()", default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.slow
def test_experiment_form_calibration_toggle():
    at = AppTest.from_string("from ui.pages import experiment\nexperiment()", default_timeout=180).run()
    toggles = [t for t in at.toggle if "Калібрувати" in t.label]
    assert toggles and toggles[0].value is True
    toggles[0].set_value(False).run()
    assert not at.exception
