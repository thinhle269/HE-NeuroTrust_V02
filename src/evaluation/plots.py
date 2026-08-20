"""Publication-quality plots.

All plots share a consistent style (set in :func:`_setup_style`) and are
saved as PNG (raster, 300 dpi) and PDF (vector) so they can be embedded in
LaTeX without quality loss.

The non-interactive ``Agg`` backend is selected at import time.  Without
this, matplotlib defaults to ``TkAgg`` on Windows, which spins up Tcl/Tk
objects that hold references back to the main thread.  When we later run
joblib workers for Paillier encryption (``prefer="threads"``), Python's
garbage collector may try to destroy those Tk handles from a worker thread,
producing the noisy ``RuntimeError: main thread is not in main loop`` traces
on every figure cleanup.  ``Agg`` is purely in-memory rasterisation, so
nothing holds onto Tk state and the errors disappear.
"""
from __future__ import annotations

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # noqa: E402

from pathlib import Path                                       # noqa: E402
from typing import Dict, Iterable, List, Optional, Sequence    # noqa: E402

import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
import seaborn as sns                                          # noqa: E402


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_class_distribution(csv_path: Path, out_path: Path, title: str) -> None:
    _setup_style()
    df = pd.read_csv(csv_path).sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="Label", y="count", ax=ax, color="#4C72B0")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontsize=8, rotation=0)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
    _save(fig, out_path)


def _aggregate_by_round(per_round: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean and std deviation over ``seed`` for each (scenario, round_idx)."""
    if "seed" not in per_round.columns:
        agg = per_round.groupby(["scenario", "round_idx"])[value_col].agg(["mean"])
        agg["std"] = 0.0
        return agg.reset_index()
    g = per_round.groupby(["scenario", "round_idx"])[value_col]
    return g.agg(["mean", "std"]).fillna(0.0).reset_index()


def plot_round_curves(per_round: pd.DataFrame, out_path: Path,
                      scenarios: Optional[Iterable[str]] = None) -> None:
    """Convergence curves: mean line +/- std band across seeds per scenario."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if scenarios is None:
        scenarios = list(per_round["scenario"].unique())
    else:
        scenarios = list(scenarios)
    palette = sns.color_palette("tab10", n_colors=max(1, len(scenarios)))
    for i, scen in enumerate(scenarios):
        sub_loss = _aggregate_by_round(per_round[per_round["scenario"] == scen], "val_loss")
        sub_acc = _aggregate_by_round(per_round[per_round["scenario"] == scen], "val_accuracy")
        if sub_loss.empty:
            continue
        x = sub_loss["round_idx"] + 1
        axes[0].plot(x, sub_loss["mean"], label=scen, marker="o", linewidth=1.6,
                     color=palette[i])
        axes[0].fill_between(x,
                             sub_loss["mean"] - sub_loss["std"],
                             sub_loss["mean"] + sub_loss["std"],
                             alpha=0.18, color=palette[i], linewidth=0)
        axes[1].plot(x, sub_acc["mean"], label=scen, marker="o", linewidth=1.6,
                     color=palette[i])
        axes[1].fill_between(x,
                             sub_acc["mean"] - sub_acc["std"],
                             sub_acc["mean"] + sub_acc["std"],
                             alpha=0.18, color=palette[i], linewidth=0)
    axes[0].set_title("Validation loss vs. round (mean +/- std over seeds)")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Validation accuracy vs. round (mean +/- std over seeds)")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy")
    axes[0].legend(loc="best", fontsize=9)
    axes[1].legend(loc="best", fontsize=9)
    _save(fig, out_path)


def plot_confusion_matrix(cm: np.ndarray, label_names: Sequence[str],
                          out_path: Path, title: str,
                          normalize: bool = True) -> None:
    _setup_style()
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        plot_data = cm_norm
        fmt = ".2f"
    else:
        plot_data = cm.astype(np.int64)
        fmt = "d"
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(label_names)),
                                    max(5, 0.5 * len(label_names))))
    sns.heatmap(plot_data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                cbar=True, ax=ax, square=True, linewidths=0.4)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    _save(fig, out_path)


def plot_trust_heatmap(trust_matrix: pd.DataFrame, out_path: Path,
                       title: str, malicious_ids: Sequence[int] = ()) -> None:
    """Heatmap with clients on Y axis, rounds on X axis, raw trust as colour."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(max(8, trust_matrix.shape[1] * 0.5),
                                    max(4, trust_matrix.shape[0] * 0.35)))
    sns.heatmap(trust_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, ax=ax,
                cbar_kws={"label": "Trust score"}, linewidths=0.3)
    yt_labels = []
    for cid in trust_matrix.index:
        if cid in malicious_ids:
            yt_labels.append(f"C{cid} *")
        else:
            yt_labels.append(f"C{cid}")
    ax.set_yticklabels(yt_labels, rotation=0)
    ax.set_xlabel("Round")
    ax.set_ylabel("Client (* = malicious)")
    ax.set_title(title)
    _save(fig, out_path)


def plot_overhead_breakdown(df: pd.DataFrame, out_path: Path) -> None:
    """Stacked-bar plot of average per-round wall-time for each scenario."""
    _setup_style()
    agg_cols = ["time_local_train_sec", "time_encrypt_sec",
                "time_aggregate_sec", "time_decrypt_sec", "time_fuzzy_sec"]
    pretty = {
        "time_local_train_sec": "Local train",
        "time_encrypt_sec": "Encryption",
        "time_aggregate_sec": "Aggregation",
        "time_decrypt_sec": "Decryption",
        "time_fuzzy_sec": "Fuzzy + ZT",
    }
    means = df.groupby("scenario")[agg_cols].mean().rename(columns=pretty)
    means = means.loc[:, list(pretty.values())]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    means.plot(kind="bar", stacked=True, ax=ax,
               colormap="viridis", edgecolor="black", linewidth=0.4)
    ax.set_title("Mean per-round wall-time breakdown")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, out_path)


def plot_baseline_comparison(per_round: pd.DataFrame, summary: pd.DataFrame,
                             out_path: Path,
                             baseline_scenarios: Sequence[str],
                             proposed_scenarios: Sequence[str]) -> None:
    """Two-panel comparison figure: convergence curves + final test bars.

    Highlights how the proposed (HE + Fuzzy + ZT) variants stack up against
    standard literature baselines (FedMedian, Krum, TrimmedMean, FedProx).
    Each curve / bar is the mean over seeds; shaded band / error bar shows
    the standard deviation.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    keep = list(baseline_scenarios) + list(proposed_scenarios)
    pr = per_round[per_round["scenario"].isin(keep)].copy()
    palette = sns.color_palette("tab10", n_colors=max(1, len(keep)))
    color_map = {s: palette[i] for i, s in enumerate(keep)}
    for s in keep:
        sub = _aggregate_by_round(pr[pr["scenario"] == s], "val_macro_f1")
        if sub.empty:
            continue
        style = "--" if s in baseline_scenarios else "-"
        lw = 1.6 if s in baseline_scenarios else 2.4
        x = sub["round_idx"] + 1
        axes[0].plot(x, sub["mean"], style, label=s, color=color_map[s],
                     linewidth=lw,
                     marker="o" if s in proposed_scenarios else "x",
                     markersize=4)
        axes[0].fill_between(x,
                             sub["mean"] - sub["std"],
                             sub["mean"] + sub["std"],
                             alpha=0.15, color=color_map[s], linewidth=0)
    axes[0].set_title("Validation macro-F1 vs. round (mean +/- std over seeds)")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Macro F1")
    axes[0].legend(loc="best", fontsize=8, ncol=2)

    bar_df = summary[summary["scenario"].isin(keep)].copy()
    if not bar_df.empty:
        metric_cols = [c for c in ("test_accuracy", "test_macro_f1")
                       if c in bar_df.columns]
        rows = []
        for _, r in bar_df.iterrows():
            for m in metric_cols:
                rows.append({
                    "scenario": r["scenario"],
                    "metric": m,
                    "value": float(r[m]),
                    "std": float(r[f"{m}_std"]) if f"{m}_std" in bar_df.columns else 0.0,
                })
        long = pd.DataFrame(rows)
        sns.barplot(data=long, x="scenario", y="value", hue="metric",
                    ax=axes[1], palette="viridis",
                    edgecolor="black", linewidth=0.4, errorbar=None)
        scenarios_order = list(long["scenario"].drop_duplicates())
        metrics_order = list(long["metric"].drop_duplicates())
        n_metrics = len(metrics_order)
        bar_w = 0.8 / n_metrics
        for mi, metric in enumerate(metrics_order):
            for si, scen in enumerate(scenarios_order):
                row = long[(long["scenario"] == scen) & (long["metric"] == metric)]
                if row.empty:
                    continue
                mu = float(row["value"].iloc[0])
                sd = float(row["std"].iloc[0])
                if sd <= 0:
                    continue
                x = si - 0.4 + (mi + 0.5) * bar_w
                axes[1].errorbar(x, mu, yerr=sd, fmt="none", ecolor="black",
                                 capsize=3, linewidth=0.8)
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title("Final test metrics: baselines vs. proposed")
        axes[1].set_xlabel("Scenario")
        axes[1].set_ylabel("Score")
        axes[1].tick_params(axis="x", rotation=25)
        plt.setp(axes[1].get_xticklabels(), ha="right")
        for p in axes[1].patches:
            if p.get_height() > 0:
                axes[1].annotate(f"{p.get_height():.3f}",
                                 (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha="center", va="bottom", fontsize=7)
    _save(fig, out_path)


def plot_ablation_curve(sweep_df: pd.DataFrame, out_path: Path,
                        param_name: str, metric: str = "test_accuracy",
                        title: Optional[str] = None,
                        chosen_value: Optional[float] = None) -> None:
    """One line per scenario; ``param`` on x-axis, ``metric`` on y-axis.

    ``sweep_df`` is expected to have columns ``param_value``, ``scenario``,
    ``<metric>_mean``, ``<metric>_std``.  When ``chosen_value`` is given a
    vertical dashed line is drawn at that x so the figure shows precisely
    which point we selected as the operating value in the paper.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    scenarios = list(sweep_df["scenario"].drop_duplicates())
    palette = sns.color_palette("tab10", n_colors=max(1, len(scenarios)))
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    for i, s in enumerate(scenarios):
        sub = sweep_df[sweep_df["scenario"] == s].sort_values("param_value")
        if sub.empty:
            continue
        x = sub["param_value"].to_numpy(dtype=float)
        y = sub[mean_col].to_numpy(dtype=float)
        sd = sub[std_col].to_numpy(dtype=float) if std_col in sub.columns else np.zeros_like(y)
        ax.plot(x, y, marker="o", linewidth=2.0, color=palette[i], label=s)
        ax.fill_between(x, y - sd, y + sd, alpha=0.18, color=palette[i], linewidth=0)
    if chosen_value is not None:
        ax.axvline(float(chosen_value), color="gray", linestyle="--", linewidth=1.0,
                   label=f"chosen = {chosen_value}")
    ax.set_xlabel(param_name.replace("_", " "))
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"Ablation: {metric} vs. {param_name}")
    ax.legend(loc="best", fontsize=9)
    _save(fig, out_path)


def plot_robustness_bars(summary: pd.DataFrame, out_path: Path) -> None:
    """Bar plot comparing final test accuracy / macro-F1 across scenarios.

    When the summary contains ``<metric>_std`` columns (multi-seed run), the
    standard deviation is shown as error bars on top of each bar.
    """
    _setup_style()
    metric_cols = [c for c in ("test_accuracy", "test_macro_f1", "test_weighted_f1")
                   if c in summary.columns]
    if not metric_cols:
        return
    rows = []
    for _, r in summary.iterrows():
        for m in metric_cols:
            rows.append({
                "scenario": r["scenario"],
                "metric": m,
                "value": float(r[m]),
                "std": float(r[f"{m}_std"]) if f"{m}_std" in summary.columns else 0.0,
            })
    long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=long, x="scenario", y="value", hue="metric", ax=ax,
                palette="viridis", edgecolor="black", linewidth=0.4,
                errorbar=None)
    scenarios_order = list(long["scenario"].drop_duplicates())
    metrics_order = list(long["metric"].drop_duplicates())
    n_metrics = len(metrics_order)
    bar_w = 0.8 / n_metrics
    for mi, metric in enumerate(metrics_order):
        for si, scen in enumerate(scenarios_order):
            row = long[(long["scenario"] == scen) & (long["metric"] == metric)]
            if row.empty:
                continue
            mu = float(row["value"].iloc[0])
            sd = float(row["std"].iloc[0])
            if sd <= 0:
                continue
            x = si - 0.4 + (mi + 0.5) * bar_w
            ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor="black", capsize=3,
                        linewidth=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Final test-set metrics by scenario (error bars = std across seeds)")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Score")
    ax.legend(title="Metric")
    ax.tick_params(axis="x", rotation=15)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="bottom", fontsize=8)
    _save(fig, out_path)
