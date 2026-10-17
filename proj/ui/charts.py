"""Побудова графіків Plotly для сторінок результатів."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cleantext.scoring import SHORT_NAMES
from .theme import CLEAN_COLOR, CORRUPT_COLOR, INK_2, METHOD_COLORS, SERIES

BAR = dict(cornerradius=4, line=dict(width=0))


def _c(key):
    return METHOD_COLORS.get(key, "#64748b")


def class_distribution(stats: dict) -> go.Figure:
    counts = stats["class_counts"]
    labels = sorted({l for d in counts.values() for l in d})
    fig = go.Figure()
    for i, split in enumerate(["train", "val", "test"]):
        vals = [counts[split].get(l, 0) for l in labels]
        fig.add_bar(name=split, x=labels, y=vals, marker=dict(color=SERIES[i], **BAR),
                    hovertemplate=f"{split}<br>%{{x}}: %{{y}}<extra></extra>")
    fig.update_layout(title="Розподіл класів за вибірками", barmode="group", yaxis_title="Кількість текстів")
    return fig


def transition_heatmap(transition: np.ndarray, labels: list[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=transition, x=labels, y=labels, colorscale=[[0, "#f8fafc"], [0.15, "#cde2fb"],
                                                                              [0.5, "#5598e7"], [1, "#104281"]],
                               text=transition, texttemplate="%{text}", showscale=False,
                               hovertemplate="вихідна: %{y}<br>спостережувана: %{x}<br>%{z}<extra></extra>",
                               xgap=2, ygap=2))
    fig.update_layout(title="Матриця переходів міток y⁰ → ỹ", xaxis_title="Спостережувана мітка ỹ",
                      yaxis_title="Вихідна мітка y⁰", yaxis_autorange="reversed")
    return fig


def learning_curves(log: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Функція втрат", "Якість на валідації та запам'ятовування"),
                        horizontal_spacing=0.09)
    fig.add_scatter(x=log.epoch, y=log.train_loss, name="Train loss (ỹ)", mode="lines+markers",
                    line=dict(color=SERIES[0], width=2), marker=dict(size=8), row=1, col=1)
    if "val_loss" in log:
        fig.add_scatter(x=log.epoch, y=log.val_loss, name="Val loss", mode="lines+markers",
                        line=dict(color=SERIES[1], width=2), marker=dict(size=8), row=1, col=1)
    if "val_macro_f1" in log:
        fig.add_scatter(x=log.epoch, y=log.val_macro_f1, name="Val Macro-F1", mode="lines+markers",
                        line=dict(color=SERIES[2], width=2), marker=dict(size=8), row=1, col=2)
    if "train_acc_observed" in log:
        fig.add_scatter(x=log.epoch, y=log.train_acc_observed, name="Train accuracy щодо ỹ", mode="lines+markers",
                        line=dict(color=SERIES[6], width=2, dash="dot"), marker=dict(size=8), row=1, col=2)
    fig.update_xaxes(title_text="Епоха", dtick=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    fig.update_layout(height=400, margin=dict(t=40, b=10), legend=dict(y=-0.28, yanchor="top"))
    fig.update_annotations(font=dict(size=13, color=INK_2))
    return fig


def score_histograms(comps: dict, z: np.ndarray) -> go.Figure:
    names = [("C", "C — невпевненість"), ("D", "D — незгода"), ("S", "S — семантика"), ("R", "R — інтегральна")]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[n for _, n in names], horizontal_spacing=0.05)
    for j, (k, _) in enumerate(names, 1):
        v = comps[k]
        for flag, color, label in [(0, CLEAN_COLOR, "Коректні мітки"), (1, CORRUPT_COLOR, "Навмисно змінені")]:
            fig.add_histogram(x=v[z == flag], nbinsx=20, histnorm="probability", name=label, marker_color=color,
                              opacity=0.75, showlegend=j == 1, legendgroup=label, row=1, col=j,
                              hovertemplate=f"{label}<br>%{{x}}: %{{y:.1%}}<extra></extra>")
    fig.update_layout(barmode="overlay", height=340, margin=dict(t=40), legend=dict(y=-0.2, yanchor="top"))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(tickformat=".0%")
    fig.update_annotations(font=dict(size=12, color=INK_2))
    return fig


def metric_bars(det: pd.DataFrame, metric: str, title: str) -> go.Figure:
    d = det.dropna(subset=[metric]).sort_values(metric)
    fig = go.Figure(go.Bar(x=d[metric], y=[SHORT_NAMES.get(k, k) for k in d.key], orientation="h",
                           marker=dict(color=[_c(k) for k in d.key], **BAR), text=d[metric].map(lambda v: f"{v:.3f}"),
                           textposition="outside", textfont=dict(color=INK_2), cliponaxis=False,
                           customdata=d.method, hovertemplate="%{customdata}<br>" + metric + ": %{x:.4f}<extra></extra>"))
    fig.update_layout(title=title, xaxis=dict(range=[0, min(1.08, d[metric].max() * 1.18 + 0.02)]),
                      yaxis=dict(ticksuffix="  "), height=60 + 42 * len(d), margin=dict(r=30))
    return fig


def pr_curves(curves: dict) -> go.Figure:
    fig = go.Figure()
    for key, (r, p) in curves.items():
        fig.add_scatter(x=r, y=p, mode="lines", name=SHORT_NAMES.get(key, key),
                        line=dict(color=_c(key), width=3 if key == "proposed" else 2,
                                  dash="dot" if key == "random" else "solid"),
                        hovertemplate=SHORT_NAMES.get(key, key) + "<br>Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>")
    fig.update_layout(title="PR-криві виявлення навмисно змінених міток", xaxis_title="Recall",
                      yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1.02], height=420)
    return fig


def budget_curves(bc: pd.DataFrame, noise_rate: float) -> go.Figure:
    fig = go.Figure()
    for key, g in bc.groupby("key", sort=False):
        fig.add_scatter(x=g.frac, y=g["Precision@K"], mode="lines+markers", name=SHORT_NAMES.get(key, key),
                        line=dict(color=_c(key), width=3 if key == "proposed" else 2), marker=dict(size=8),
                        hovertemplate=SHORT_NAMES.get(key, key) + "<br>q=%{x:.0%}<br>Precision@K %{y:.3f}<extra></extra>")
    fig.add_vline(x=noise_rate, line=dict(color="#94a3b8", dash="dash", width=1),
                  annotation_text=f"η = {noise_rate:.0%}", annotation_font_color=INK_2)
    fig.update_layout(title="Precision@K залежно від бюджету перевірки q", xaxis_title="Частка вибірки q",
                      yaxis_title="Precision@K", xaxis_tickformat=".0%", yaxis_range=[0, 1.02], height=400)
    return fig


def delta_ci(df: pd.DataFrame, value: str, title: str, label_col: str = "versus") -> go.Figure:
    d = df.dropna(subset=[value])
    labels = [SHORT_NAMES.get(k, k) for k in d[label_col]]
    fig = go.Figure(go.Scatter(
        x=d[value], y=labels, mode="markers",
        marker=dict(size=12, color=[_c(k) for k in d[label_col]], line=dict(color="#fff", width=2)),
        error_x=dict(type="data", symmetric=False, array=d.ci_high - d[value], arrayminus=d[value] - d.ci_low,
                     color="#64748b", thickness=2, width=6),
        hovertemplate="%{y}<br>Δ = %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#94a3b8", width=1, dash="dash"))
    fig.update_layout(title=title, height=80 + 46 * len(d), xaxis_title="Різниця та 95 % довірчий інтервал")
    return fig


def epoch_ap(em: pd.DataFrame, metric: str = "AP") -> go.Figure:
    names = {"C": "C (невпевненість)", "D": "D (незгода)", "loss": "Loss на епосі t", "proposed": "R (C+D+S)"}
    fig = go.Figure()
    for key, g in em.groupby("key", sort=False):
        fig.add_scatter(x=g.epoch, y=g[metric], name=names.get(key, key), mode="lines+markers",
                        line=dict(color=_c(key), width=3 if key == "proposed" else 2), marker=dict(size=8),
                        hovertemplate=names.get(key, key) + "<br>епоха %{x}<br>" + metric + " %{y:.3f}<extra></extra>")
    fig.update_layout(title=f"{metric} ранжування, обчисленого за першими t епохами", xaxis_title="Епоха t",
                      yaxis_title=metric, xaxis_dtick=1, height=400)
    return fig


def cartography_map(conf, var, z, risk=None) -> go.Figure:
    """Dataset Cartography: confidence × variability; червоним — навмисно змінені мітки (якщо z відомий)."""
    conf, var = np.asarray(conf), np.asarray(var)
    fig = go.Figure()
    if z is not None and np.asarray(z).sum() > 0:
        z = np.asarray(z).astype(bool)
        for mask, name, color, op in ((~z, "Коректна мітка", CLEAN_COLOR, 0.5), (z, "Змінена мітка", CORRUPT_COLOR, 0.8)):
            fig.add_trace(go.Scattergl(x=var[mask], y=conf[mask], mode="markers", name=name,
                                       marker=dict(size=6, color=color, opacity=op)))
    else:
        fig.add_trace(go.Scattergl(x=var, y=conf, mode="markers", name="Зразки",
                                   marker=dict(size=6, color=risk if risk is not None else CLEAN_COLOR,
                                               colorscale="Viridis", showscale=risk is not None,
                                               colorbar=dict(title="R"), opacity=0.7)))
    fig.update_layout(title="Dataset Cartography: confidence × variability", height=430,
                      xaxis_title="variability = std p(ỹ) за епохами", yaxis_title="confidence = mean p(ỹ)",
                      margin=dict(t=50), legend=dict(y=-0.2, yanchor="top"))
    return fig


def weight_heatmap(grid: pd.DataFrame, current: tuple[float, float], best: tuple[float, float] | None = None,
                   title: str = "AP на симплексі ваг (γ = 1 − α − β)") -> go.Figure:
    piv = grid.pivot(index="beta", columns="alpha", values="AP")
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                               colorscale=[[0, "#f8fafc"], [0.3, "#b7d3f6"], [0.65, "#3987e5"], [1, "#0d366b"]],
                               colorbar=dict(title="AP", thickness=12), xgap=2, ygap=2,
                               hovertemplate="α=%{x:.2f}, β=%{y:.2f}, γ=1−α−β<br>AP=%{z:.3f}<extra></extra>"))
    fig.add_scatter(x=[current[0]], y=[current[1]], mode="markers", name="Поточні ваги",
                    marker=dict(symbol="x", size=13, color="#eb6834", line=dict(width=2, color="#fff")))
    if best is not None:
        fig.add_scatter(x=[best[0]], y=[best[1]], mode="markers", name="Адаптивні ваги",
                        marker=dict(symbol="star", size=14, color="#0d9488", line=dict(width=2, color="#fff")))
    fig.update_layout(title=title, xaxis_title="α (вага C)", yaxis_title="β (вага D)",
                      height=440, showlegend=best is not None)
    return fig


def classification_bars(cls: pd.DataFrame) -> go.Figure:
    d = cls.copy()
    err = None
    if "ci_low" in d:
        # довірчі інтервали ΔF1 переносимо на рівень Macro-F1 відносно базового запуску
        err = dict(type="data", symmetric=False, array=(d.ci_high - d.delta_macro_f1).fillna(0),
                   arrayminus=(d.delta_macro_f1 - d.ci_low).fillna(0), color="#64748b", thickness=1.5, width=5)
    fig = go.Figure(go.Bar(x=[SHORT_NAMES.get(k, k) for k in d.key], y=d.macro_f1,
                           marker=dict(color=[_c(k) for k in d.key], **BAR), error_y=err,
                           text=d.macro_f1.map(lambda v: f"{v:.3f}"), textposition="inside",
                           insidetextanchor="start", textfont=dict(color="#fff"),
                           customdata=d.method, hovertemplate="%{customdata}<br>Macro-F1 %{y:.4f}<extra></extra>"))
    base = d.loc[d.key == "noisy", "macro_f1"]
    if len(base):
        fig.add_hline(y=float(base.iloc[0]), line=dict(color="#64748b", dash="dash", width=1),
                      annotation_text="без очищення", annotation_position="top left",
                      annotation_font_color=INK_2)
    fig.update_layout(title="Macro-F1 на незмінному тесті після вилучення K зразків",
                      yaxis_title="Macro-F1", yaxis_range=[0, 1], height=420)
    return fig


def budget_f1_lines(cls: pd.DataFrame) -> go.Figure:
    """Macro-F1 залежно від бюджету K за методами; пунктир — без очищення."""
    d = cls[cls.key != "noisy"].dropna(subset=["budget_frac"])
    fig = go.Figure()
    for key, g in d.groupby("key", sort=False):
        g = g.sort_values("budget_frac")
        fig.add_scatter(x=g.budget_frac, y=g.macro_f1, mode="lines+markers", name=SHORT_NAMES.get(key, key),
                        line=dict(color=_c(key), width=2.5), marker=dict(size=8),
                        hovertemplate="K = %{x:.0%}<br>Macro-F1 %{y:.4f}<extra>" + SHORT_NAMES.get(key, key) + "</extra>")
    base = cls.loc[cls.key == "noisy", "macro_f1"]
    if len(base):
        fig.add_hline(y=float(base.iloc[0]), line=dict(color="#64748b", dash="dash", width=1),
                      annotation_text="без очищення", annotation_position="top left", annotation_font_color=INK_2)
    fig.update_layout(title="Macro-F1 після очищення залежно від бюджету K", xaxis_title="Бюджет K, частка train",
                      yaxis_title="Macro-F1", xaxis=dict(tickformat=".0%", tickvals=sorted(d.budget_frac.unique())),
                      height=380)
    return fig


def removed_composition(comp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, cls in enumerate(comp.columns):
        fig.add_bar(name=cls, y=[SHORT_NAMES.get(k, k) for k in comp.index], x=comp[cls], orientation="h",
                    marker=dict(color=SERIES[i % len(SERIES)], line=dict(color="#fff", width=2)),
                    hovertemplate=f"{cls}<br>%{{y}}: %{{x}}<extra></extra>")
    fig.update_layout(barmode="stack", legend_traceorder="normal", title="Склад вилучених зразків за спостережуваними класами",
                      xaxis_title="Кількість вилучених текстів", height=110 + 44 * len(comp))
    return fig


def per_class_f1(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for key, g in df.groupby("key", sort=False):
        fig.add_bar(name=SHORT_NAMES.get(key, key), x=g["class"], y=g.f1, marker=dict(color=_c(key), **BAR),
                    hovertemplate=SHORT_NAMES.get(key, key) + "<br>%{x}: F1 %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", title="F1 за класами на тестовій вибірці", yaxis_range=[0, 1],
                      yaxis_title="F1", height=400)
    return fig


def class_flag_share(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="Правильно знайдені (TP)", x=df["class"], y=df.TP, marker=dict(color="#2a78d6", line=dict(color="#fff", width=2)))
    fig.add_bar(name="Хибні спрацювання (FP)", x=df["class"], y=df.FP, marker=dict(color="#eb6834", line=dict(color="#fff", width=2)))
    fig.update_layout(barmode="stack", legend_traceorder="normal", title="Кандидати top-K методу R за спостережуваним класом",
                      yaxis_title="Кількість у top-K", height=380)
    return fig


def length_box(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for grp, color in [("TP у top-K", "#2a78d6"), ("FP у top-K", "#eb6834"), ("Поза top-K", "#94a3b8")]:
        v = df.loc[df.group == grp, "length"]
        fig.add_box(y=v, name=grp, marker_color=color, boxmean=True, line=dict(width=1.5))
    fig.update_layout(title="Довжина текстів (символи) за групами", yaxis_title="Символів", showlegend=False, height=380)
    return fig


def timings_bar(t: dict) -> go.Figure:
    names = {"data_and_embeddings_s": "Дані + семантичні подання", "base_training_s": "Навчання з реєстрацією динаміки",
             "cleanlab_oof_s": "Cleanlab OOF (K фолдів)", "risk_scoring_s": "Обчислення C, D, S, R",
             "retraining_s": "Повторне навчання"}
    keys = [k for k in names if k in t]
    fig = go.Figure(go.Bar(y=[names[k] for k in keys], x=[t[k] for k in keys], orientation="h",
                           marker=dict(color=SERIES[0], **BAR), text=[f"{t[k]:.1f} с" for k in keys],
                           textposition="outside", cliponaxis=False))
    fig.update_layout(title="Обчислювальний час етапів", xaxis_title="Секунди", height=90 + 44 * len(keys),
                      margin=dict(r=50))
    return fig


def grid_lines(agg: pd.DataFrame, metric: str, methods: list[str], kinds: list[str], kind_names: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(kinds), subplot_titles=[kind_names.get(k, k) for k in kinds],
                        horizontal_spacing=0.05)
    for j, kind in enumerate(kinds, 1):
        for key in methods:
            g = agg[(agg.kind == kind) & (agg.key == key)].sort_values("rate")
            if g.empty:
                continue
            fig.add_scatter(x=g.rate, y=g["mean"], name=SHORT_NAMES.get(key, key), legendgroup=key, showlegend=j == 1,
                            mode="lines+markers", line=dict(color=_c(key), width=3 if key == "proposed" else 2),
                            marker=dict(size=8),
                            error_y=dict(type="data", array=g["std"].fillna(0), color=_c(key), thickness=1, width=4),
                            hovertemplate=SHORT_NAMES.get(key, key) + "<br>η=%{x:.2f}<br>" + metric + " %{y:.3f}<extra></extra>",
                            row=1, col=j)
    rates = sorted(agg.rate.unique())
    fig.update_xaxes(title_text="Рівень шуму η", tickformat=".0%", tickvals=rates,
                     range=[min(rates) - 0.03, max(rates) + 0.03])
    lo = max(0.0, float((agg["mean"] - agg["std"].fillna(0)).min()) - 0.05)
    fig.update_yaxes(range=[lo, min(1.02, float((agg["mean"] + agg["std"].fillna(0)).max()) + 0.05)])
    fig.update_yaxes(title_text=metric, col=1)
    fig.update_layout(height=440, margin=dict(t=40), legend=dict(y=-0.22, yanchor="top"))
    fig.update_annotations(font=dict(size=13, color=INK_2))
    return fig
