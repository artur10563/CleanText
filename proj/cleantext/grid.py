"""Серія експериментів: типи шуму × рівні η × випадкові зерна на одному корпусі."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import RUNS_DIR, ExperimentConfig
from .pipeline import prepare_data, run_experiment


def run_grid(cfg: ExperimentConfig, types, rates, seeds, log=print, on_progress=None,
             csv_df=None, grid_dir: Path | None = None) -> Path:
    cases = [(t, r, s) for t in types for r in rates for s in seeds]
    if grid_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        grid_dir = RUNS_DIR / f"grid_{stamp}_{cfg.dataset}"
    grid_dir = Path(grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)
    (grid_dir / "grid.json").write_text(json.dumps(
        {"config": cfg.to_dict(), "types": list(types), "rates": list(rates), "seeds": list(seeds)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    # Корпус і семантичні подання — один раз, щоб порівняння були парними.
    log("Підготовка корпусу для всієї серії…")
    prepared = prepare_data(cfg, log, csv_df)
    corpus = prepared[0]
    for name, df in (("train", corpus.train), ("val", corpus.val), ("test", corpus.test), ("calib", corpus.calib)):
        if df is not None:
            df.to_parquet(grid_dir / f"{name}.parquet", index=False)

    for i, (t, r, s) in enumerate(cases):
        case_dir = grid_dir / f"{t}_{r:.2f}_seed{s}"
        if (case_dir / "run_manifest.json").exists():
            log(f"Пропуск завершеного сценарію {case_dir.name}")
            continue
        log(f"Сценарій {i + 1}/{len(cases)}: {case_dir.name}")
        case_cfg = replace(cfg, noise_type=t, noise_rate=r, seed=s)
        run_experiment(case_cfg, log=lambda *_: None, prepared=prepared, out_dir=case_dir, save_texts=False,
                       calib_cache=grid_dir / "calibration",
                       on_progress=(lambda f, txt, i=i: on_progress((i + f) / len(cases),
                                                                    f"[{i + 1}/{len(cases)}] {txt}"))
                       if on_progress else None)
    if on_progress:
        on_progress(1.0, "Серію завершено")
    return grid_dir
