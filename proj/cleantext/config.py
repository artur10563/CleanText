"""Конфігурація експерименту та готові пресети."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
RUNS_DIR = ROOT / "runs"

DATASETS = {
    "ua_news": "UA-News (FIdo-AI)",
    "uareviews": "UAReviews (KSE)",
    "csv": "Власний CSV-файл",
}
NOISE_TYPES = {
    "symmetric": "Симетричний",
    "asymmetric": "Асиметричний",
    "semantic": "Семантично залежний",
}
CALIB_MODES = {
    "known_type": "Тип шуму відомий (як у ноутбуці v2)",
    "mixed": "Тип-агностичний (суміш типів)",
}
CLASSIFIERS = {
    "mlp": "MLP-нейромережа на TF-IDF (швидко, CPU)",
    "distilbert": "DistilBERT multilingual (повільно без GPU)",
}
EMBEDDINGS = {
    "minilm": "Multilingual MiniLM-L12 (sentence-transformers)",
    "tfidf_svd": "TF-IDF + SVD (LSA, без завантаження моделей)",
}

MINILM_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DISTILBERT_ID = "distilbert/distilbert-base-multilingual-cased"


@dataclass
class ExperimentConfig:
    # дані
    dataset: str = "ua_news"
    review_task: str = "intent"          # для UAReviews: intent / emotion
    csv_path: str | None = None
    n_train: int = 800
    n_val: int = 200
    n_test: int = 300
    n_calib: int = 350                   # калібрувальна підвибірка (окрема від train/val/test)
    # шум
    noise_type: str = "semantic"
    noise_rate: float = 0.20
    seed: int = 42
    # метод
    alpha: float = 0.45
    beta: float = 0.35
    gamma: float = 0.20
    k_neighbors: int = 10
    budget_frac: float = 0.20
    retrain_budgets: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])
    embedding: str = "minilm"
    # калібрування ваг на окремій підвибірці з власним шумом (без доступу до z основного train)
    calibrate_weights: bool = True
    calib_mode: str = "known_type"       # known_type — шум того самого типу; mixed — по третині кожного типу
    calib_noise_rate: float = 0.20
    calib_seed: int = 2026
    coeff_grid_step: float = 0.25
    # класифікатор
    classifier: str = "mlp"
    epochs: int = 8
    batch_size: int = 32
    lr: float = 1e-3
    hidden: int = 256
    dropout: float = 0.3
    max_features: int = 30000
    max_length: int = 128
    bert_lr: float = 3e-5
    bert_batch_size: int = 16
    # базові методи та етапи
    run_cleanlab: bool = True
    oof_folds: int = 3
    oof_epochs: int = 8              # = epochs: Cleanlab навчається за тим самим протоколом, що й базова модель
    run_retrain: bool = True
    run_oracle: bool = True
    run_aum: bool = True                 # повний AUM (Pleiss et al., 2020): +2 навчання з порогами
    aum_percentile: float = 99.0         # поріг = цей перцентиль AUM порогових зразків
    run_weighted: bool = True            # м'яке зважування підозрілих зразків замість вилучення
    weight_min: float = 0.20             # v2 MIN_SAMPLE_WEIGHT
    weight_power: float = 2.0            # v2 WEIGHT_POWER
    weight_scope: str = "top_k"          # top_k — лише K найризикованіших (як у v2); all — усі за R
    bootstrap: int = 500

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def all_budgets(self) -> list[float]:
        """Бюджети перенавчання (основний budget_frac завжди включено), за зростанням."""
        return sorted({round(float(b), 4) for b in [*self.retrain_budgets, self.budget_frac]})

    def validate(self) -> None:
        assert self.dataset in DATASETS, self.dataset
        assert self.noise_type in NOISE_TYPES, self.noise_type
        assert 0 <= self.noise_rate < 1
        assert 0 < self.budget_frac < 1
        assert self.alpha >= 0 and self.beta >= 0 and self.gamma >= 0
        assert abs(self.alpha + self.beta + self.gamma - 1) < 1e-6, "α + β + γ має дорівнювати 1"
        assert all(0 < b <= 0.5 for b in self.retrain_budgets), "бюджети перенавчання мають бути в (0; 0,5]"
        assert self.calib_mode in CALIB_MODES, self.calib_mode
        assert 0 < self.calib_noise_rate < 1 and 0 < self.coeff_grid_step <= 0.5 and self.n_calib >= 0
        assert 0 < self.weight_min <= 1 and self.weight_power > 0 and self.weight_scope in ("top_k", "all")
        assert 50 <= self.aum_percentile < 100, "перцентиль AUM має бути в [50; 100)"
        assert self.oof_folds >= 2 and self.epochs >= 1


PRESETS = {
    "quick": dict(n_train=800, n_val=200, n_test=300, n_calib=350, epochs=8, oof_folds=3, oof_epochs=8,
                  retrain_budgets=[0.10, 0.20]),
    "main": dict(n_train=4000, n_val=800, n_test=1000, n_calib=700, epochs=10, oof_folds=3, oof_epochs=10,
                 retrain_budgets=[0.05, 0.10, 0.15, 0.20]),
}
BERT_PRESETS = {
    "quick": dict(epochs=2, oof_folds=2, oof_epochs=2, retrain_budgets=[0.20]),
    "main": dict(epochs=4, oof_folds=3, oof_epochs=4, retrain_budgets=[0.05, 0.10, 0.15, 0.20]),
}
