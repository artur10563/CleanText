"""Читання збережених запусків і серій експериментів."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RUNS_DIR, ExperimentConfig


@dataclass
class Run:
    path: Path
    manifest: dict
    cfg: ExperimentConfig
    labels: list[str]
    arrays: dict
    learning_log: pd.DataFrame
    detection: pd.DataFrame
    classification: pd.DataFrame
    budget_curve: pd.DataFrame
    epoch_metrics: pd.DataFrame
    bootstrap: pd.DataFrame
    report: pd.DataFrame
    train: pd.DataFrame | None
    test: pd.DataFrame | None
    calibration: dict | None = None
    calibration_table: pd.DataFrame | None = None
    bootstrap_adaptive: pd.DataFrame | None = None

    @property
    def title(self) -> str:
        c = self.cfg
        return f"{c.dataset} · {c.noise_type} η={c.noise_rate:.2f} · seed {c.seed}"

    @property
    def classification_main(self) -> pd.DataFrame:
        """Таблиця класифікації для основного бюджету (сумісно зі старими запусками)."""
        return budget_view(self.classification, self.cfg.budget_frac)

    @property
    def budgets(self) -> list[float]:
        c = self.classification
        if "budget_frac" not in c or c.budget_frac.dropna().empty:
            return [self.cfg.budget_frac]
        return sorted(c.budget_frac.dropna().unique().tolist())

    def removed_ids(self, budget: float | None = None) -> dict:
        """{метод: [id]} для бюджету; старий плаский формат манифесту теж підтримується."""
        rem = self.manifest.get("removed_ids", {})
        if rem and all(isinstance(v, dict) for v in rem.values()):
            b = self.cfg.budget_frac if budget is None else budget
            return rem.get(f"{b:.2f}", {})
        return rem

    def score(self, key):
        return self.arrays.get(f"score_{key}")


def budget_view(cls: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Рядок «noisy» + рядки заданого бюджету. Без колонки budget_frac (старі запуски) — таблиця як є."""
    if cls.empty or "budget_frac" not in cls:
        return cls
    mask = cls.key.eq("noisy") | np.isclose(cls.budget_frac.fillna(-1.0), budget)
    return cls[mask].reset_index(drop=True)


def _csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 2 else pd.DataFrame()


def _split_bootstrap(b: pd.DataFrame):
    """bootstrap_delta_ap.csv: опорний метод — у колонці reference (старі запуски: завжди proposed)."""
    if b.empty or "reference" not in b:
        return b, pd.DataFrame()
    return (b[b.reference.eq("proposed")].reset_index(drop=True),
            b[b.reference.eq("adaptive")].reset_index(drop=True))


def load_run(path: str | Path) -> Run:
    path = Path(path)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    with np.load(path / "arrays.npz") as a:
        arrays = {k: a[k] for k in a.files}
    tdir = (path / manifest.get("texts_dir", ".")).resolve()
    read = lambda n: pd.read_parquet(tdir / f"{n}.parquet") if (tdir / f"{n}.parquet").exists() else None
    boot_fixed, boot_adaptive = _split_bootstrap(_csv(path / "bootstrap_delta_ap.csv"))
    cal_path = path / "calibration.json"
    return Run(
        path=path, manifest=manifest, cfg=ExperimentConfig.from_dict(manifest["config"]),
        labels=manifest["labels"], arrays=arrays,
        learning_log=_csv(path / "learning_log.csv"),
        detection=_csv(path / "detection_metrics.csv"),
        classification=_csv(path / "classification_metrics.csv"),
        budget_curve=_csv(path / "budget_curve.csv"),
        epoch_metrics=_csv(path / "epoch_detection_metrics.csv"),
        bootstrap=boot_fixed, bootstrap_adaptive=boot_adaptive,
        calibration=json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else None,
        calibration_table=_csv(path / "calibration_table.csv") if cal_path.exists() else None,
        report=pd.read_csv(path / "review_candidates_research_only.csv", encoding="utf-8-sig"),
        train=read("train"), test=read("test"),
    )


def list_runs(root: Path = RUNS_DIR) -> list[dict]:
    out = []
    if not root.exists():
        return out
    for p in sorted(root.iterdir(), reverse=True):
        mf = p / "run_manifest.json"
        if p.is_dir() and mf.exists():
            m = json.loads(mf.read_text(encoding="utf-8"))
            c = m["config"]
            det = _csv(p / "detection_metrics.csv")
            cls = budget_view(_csv(p / "classification_metrics.csv"), c.get("budget_frac", 0.2))
            ap = det.set_index("key")["AP"].get("proposed", np.nan) if len(det) else np.nan
            f1 = cls.set_index("key")["macro_f1"] if len(cls) else pd.Series(dtype=float)
            out.append({"path": str(p), "name": p.name, "created": m.get("created", ""),
                        "dataset": c["dataset"], "noise": c["noise_type"], "rate": c["noise_rate"],
                        "seed": c["seed"], "classifier": c["classifier"], "n_train": m["data"]["sizes"]["train"],
                        "AP_R": ap, "F1_noisy": f1.get("noisy", np.nan), "F1_R": f1.get("proposed", np.nan)})
    return out


def list_grids(root: Path = RUNS_DIR) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in sorted(root.iterdir(), reverse=True) if p.is_dir() and (p / "grid.json").exists()]


def grid_tables(grid_dir: Path, all_budgets: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """all_budgets=False — у таблиці класифікації лише основний бюджет кожного сценарію."""
    det, cls = [], []
    for case in sorted(grid_dir.iterdir()):
        mf = case / "run_manifest.json"
        if not mf.exists():
            continue
        c = json.loads(mf.read_text(encoding="utf-8"))["config"]
        tag = {"case": case.name, "kind": c["noise_type"], "rate": c["noise_rate"], "seed": c["seed"]}
        d = _csv(case / "detection_metrics.csv")
        k = _csv(case / "classification_metrics.csv")
        if not all_budgets:
            k = budget_view(k, c.get("budget_frac", 0.2))
        if len(d):
            det.append(d.assign(**tag))
        if len(k):
            cls.append(k.assign(**tag))
    return (pd.concat(det, ignore_index=True) if det else pd.DataFrame(),
            pd.concat(cls, ignore_index=True) if cls else pd.DataFrame())
