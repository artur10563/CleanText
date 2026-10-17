"""Запуск експерименту без інтерфейсу.

Приклади:
    py cli.py --dataset ua_news --noise semantic --rate 0.2 --preset quick
    py cli.py --dataset uareviews --grid --types symmetric semantic --rates 0.1 0.2 --seeds 13 42 77
"""
import argparse
import sys
from dataclasses import replace

from cleantext.config import PRESETS, BERT_PRESETS, ExperimentConfig
from cleantext.grid import run_grid
from cleantext.pipeline import run_experiment


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ua_news", choices=["ua_news", "uareviews", "csv"])
    ap.add_argument("--csv")
    ap.add_argument("--task", default="intent", choices=["intent", "emotion"])
    ap.add_argument("--noise", default="semantic", choices=["symmetric", "asymmetric", "semantic"])
    ap.add_argument("--rate", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preset", default="quick", choices=list(PRESETS))
    ap.add_argument("--classifier", default="mlp", choices=["mlp", "distilbert"])
    ap.add_argument("--embedding", default="minilm", choices=["minilm", "tfidf_svd"])
    ap.add_argument("--no-cleanlab", action="store_true")
    ap.add_argument("--no-retrain", action="store_true")
    ap.add_argument("--budgets", nargs="+", type=float, help="бюджети перенавчання, напр. 0.05 0.10 0.20")
    ap.add_argument("--no-calibration", action="store_true", help="не калібрувати ваги (без адаптивного R)")
    ap.add_argument("--calib-mode", choices=["known_type", "mixed"], default="known_type")
    ap.add_argument("--no-weighted", action="store_true", help="не виконувати м'яке зважування")
    ap.add_argument("--no-aum", action="store_true", help="не виконувати повний AUM")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--types", nargs="+", default=["symmetric", "asymmetric", "semantic"])
    ap.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 77])
    a = ap.parse_args()

    cfg = ExperimentConfig(dataset=a.dataset, csv_path=a.csv, review_task=a.task, noise_type=a.noise,
                           noise_rate=a.rate, seed=a.seed, classifier=a.classifier, embedding=a.embedding,
                           run_cleanlab=not a.no_cleanlab, run_retrain=not a.no_retrain, calibrate_weights=not a.no_calibration,
                           calib_mode=a.calib_mode, run_weighted=not a.no_weighted, run_aum=not a.no_aum)
    cfg = replace(cfg, **PRESETS[a.preset])
    if a.classifier == "distilbert":
        cfg = replace(cfg, **BERT_PRESETS[a.preset])

    if a.budgets:
        cfg = replace(cfg, retrain_budgets=a.budgets)

    def bar(frac, text):
        print(f"\r[{'█' * int(frac * 30):<30}] {frac:5.1%} {text[:60]:<60}", end="", flush=True)

    if a.grid:
        out = run_grid(cfg, a.types, a.rates, a.seeds, log=lambda m: print("\n" + m), on_progress=bar)
    else:
        out = run_experiment(cfg, log=lambda m: print("\n" + m), on_progress=bar)
    print("\nГотово:", out)


if __name__ == "__main__":
    main()
