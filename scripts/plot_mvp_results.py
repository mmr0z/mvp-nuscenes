#!/usr/bin/env python3
"""Generate thesis-ready plots for the MVP CenterPoint experiment.

The script deliberately uses the longest available log for each epoch.  This
avoids double-counting interrupted/resumed runs that start again at epoch 4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = (
    REPO_ROOT
    / "third_party/CenterPoint/work_dirs/"
    "nusc_centerpoint_voxelnet_0075voxel_bevfusion_virtual_ft_e15"
)

COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#2F2F2F",
    "grey": "#8A8A8A",
    "light_grey": "#E6E6E6",
}

CLASS_LABELS = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "trailer": "trailer",
    "construction_vehicle": "construction_vehicle",
    "pedestrian": "pedestrian",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "traffic_cone": "traffic_cone",
    "barrier": "barrier",
}

ERROR_LABELS = {
    "trans_err": ("mATE", "m"),
    "scale_err": ("mASE", "1 − IoU"),
    "orient_err": ("mAOE", "rad"),
    "vel_err": ("mAVE", "m/s"),
    "attr_err": ("mAAE", "1 − accuracy"),
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="CenterPoint experiment directory (default: known MVP experiment).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <work-dir>/eval_latest/wykresy).",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=81,
        help="Number of logged points in the centered moving average.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mean_list(values) -> float:
    return float(sum(values) / len(values))


def load_training_rows(work_dir: Path):
    """Return one canonical, longest log segment per epoch."""
    rows_by_file_epoch = defaultdict(list)
    for log_path in sorted(work_dir.glob("*.log.json")):
        with log_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {log_path}:{line_number}") from exc
                if record.get("mode") != "train" or not record.get("loss"):
                    continue
                rows_by_file_epoch[(log_path, int(record["epoch"]))].append(record)

    epochs = sorted({epoch for _, epoch in rows_by_file_epoch})
    canonical = []
    selection = []
    for epoch in epochs:
        candidates = [
            (path, rows)
            for (path, candidate_epoch), rows in rows_by_file_epoch.items()
            if candidate_epoch == epoch
        ]
        path, rows = max(
            candidates,
            key=lambda item: (len(item[1]), max(r["iter"] for r in item[1]), item[0].name),
        )
        rows = sorted(rows, key=lambda row: int(row["iter"]))
        max_iter = max(int(row["iter"]) for row in rows)
        selection.append(
            {
                "epoch": epoch,
                "source": path.name,
                "records": len(rows),
                "max_iter": max_iter,
            }
        )
        for row in rows:
            canonical.append(
                {
                    "epoch": epoch,
                    "iter": int(row["iter"]),
                    "progress": (epoch - 1) + int(row["iter"]) / max_iter,
                    "lr": float(row["lr"]),
                    "loss_mean": mean_list(row["loss"]),
                    "hm_loss_mean": mean_list(row["hm_loss"]),
                    "loc_loss_mean": mean_list(row["loc_loss"]),
                    "time": finite_or_none(row.get("time")),
                    "memory": finite_or_none(row.get("memory")),
                    "source_log": path.name,
                }
            )
    return canonical, selection


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = min(window, len(values))
    kernel = np.ones(window, dtype=float) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def write_training_csv(rows, output_dir: Path) -> None:
    columns = [
        "epoch",
        "iter",
        "progress",
        "lr",
        "loss_mean",
        "hm_loss_mean",
        "loc_loss_mean",
        "time",
        "memory",
        "source_log",
    ]
    with (output_dir / "training_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_epochs(rows, output_dir: Path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["epoch"]].append(row)
    summaries = []
    for epoch in sorted(grouped):
        epoch_rows = grouped[epoch]
        summaries.append(
            {
                "epoch": epoch,
                "records": len(epoch_rows),
                "loss_mean": float(np.mean([r["loss_mean"] for r in epoch_rows])),
                "hm_loss_mean": float(np.mean([r["hm_loss_mean"] for r in epoch_rows])),
                "loc_loss_mean": float(np.mean([r["loc_loss_mean"] for r in epoch_rows])),
                "lr_start": epoch_rows[0]["lr"],
                "lr_end": epoch_rows[-1]["lr"],
            }
        )
    columns = list(summaries[0])
    with (output_dir / "training_epoch_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def plot_training(rows, epoch_summary, output_dir: Path, rolling_window: int) -> None:
    x = np.array([row["progress"] for row in rows])
    loss = np.array([row["loss_mean"] for row in rows])
    hm = np.array([row["hm_loss_mean"] for row in rows])
    loc = np.array([row["loc_loss_mean"] for row in rows])
    lr = np.array([row["lr"] for row in rows])
    epochs = np.array([row["epoch"] for row in epoch_summary])
    epoch_loss = np.array([row["loss_mean"] for row in epoch_summary])

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 10.8), sharex=True)
    fig.suptitle("Przebieg treningu MVP CenterPoint", fontsize=16, fontweight="bold")

    raw_stride = max(1, len(rows) // 3500)
    axes[0].plot(
        x[::raw_stride],
        loss[::raw_stride],
        color=COLORS["grey"],
        alpha=0.16,
        linewidth=0.55,
        label="wartości logowane",
    )
    axes[0].plot(
        x,
        moving_average(loss, rolling_window),
        color=COLORS["blue"],
        linewidth=2.0,
        label=f"średnia ruchoma ({rolling_window} punktów)",
    )
    axes[0].scatter(
        epochs - 0.5,
        epoch_loss,
        color=COLORS["black"],
        s=28,
        zorder=4,
        label="średnia epoki",
    )
    reduction = 100 * (epoch_loss[0] - epoch_loss[-1]) / epoch_loss[0]
    axes[0].set_ylabel("Średnia strata")
    axes[0].set_title(
        f"Łączna strata (średnia z 6 głowic zadaniowych); spadek epoka 1 → 6: {reduction:.1f}%"
    )
    axes[0].legend(ncol=3, loc="upper right")

    axes[1].plot(
        x,
        moving_average(hm, rolling_window),
        color=COLORS["orange"],
        linewidth=1.9,
        label="hm_loss (klasyfikacja)",
    )
    axes[1].plot(
        x,
        moving_average(loc, rolling_window),
        color=COLORS["green"],
        linewidth=1.9,
        label="loc_loss (regresja)",
    )
    axes[1].set_ylabel("Wartość straty")
    axes[1].set_title("Składowe funkcji straty")
    axes[1].legend(ncol=2, loc="upper right")

    axes[2].plot(x, lr, color=COLORS["purple"], linewidth=1.8)
    axes[2].fill_between(x, lr, color=COLORS["purple"], alpha=0.14)
    axes[2].set_ylabel("Learning rate")
    axes[2].set_xlabel("Postęp treningu [epoka]")
    axes[2].set_title("Harmonogram learning rate")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    max_epoch = int(max(epochs))
    for axis in axes:
        for boundary in range(1, max_epoch):
            axis.axvline(boundary, color="#BDBDBD", linewidth=0.75, linestyle="--")
        axis.set_xlim(0, max_epoch)
    axes[2].set_xticks(range(max_epoch + 1))
    fig.text(
        0.5,
        0.002,
        "Źródło: logi treningowe; krzywe wygładzone wyłącznie na potrzeby wizualizacji.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    save_figure(fig, output_dir, "01_przebieg_treningu")


def plot_epoch_losses(epoch_summary, output_dir: Path) -> None:
    epochs = np.array([row["epoch"] for row in epoch_summary])
    series = [
        ("loss", "loss_mean", COLORS["blue"]),
        ("hm_loss", "hm_loss_mean", COLORS["orange"]),
        ("loc_loss", "loc_loss_mean", COLORS["green"]),
    ]
    width = 0.23
    fig, axis = plt.subplots(figsize=(10.8, 5.9))
    for offset, (label, key, color) in zip((-width, 0, width), series):
        values = np.array([row[key] for row in epoch_summary])
        bars = axis.bar(epochs + offset, values, width=width, label=label, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axis.set_title("Średnie wartości funkcji straty w kolejnych epokach")
    axis.set_xlabel("Epoka")
    axis.set_ylabel("Średnia wartość")
    axis.set_xticks(epochs)
    axis.legend(ncol=3)
    axis.set_ylim(0, max(row["loc_loss_mean"] for row in epoch_summary) * 1.18)
    fig.tight_layout()
    save_figure(fig, output_dir, "02_srednie_straty_epok")


def plot_class_ap(metrics, output_dir: Path) -> None:
    mean_ap = metrics["mean_dist_aps"]
    ordered = sorted(mean_ap, key=mean_ap.get)
    values = np.array([100 * mean_ap[name] for name in ordered])
    global_map = 100 * metrics["mean_ap"]
    colors = [COLORS["red"] if value < global_map else COLORS["blue"] for value in values]

    fig, axis = plt.subplots(figsize=(10.8, 6.7))
    bars = axis.barh([CLASS_LABELS[name] for name in ordered], values, color=colors)
    axis.axvline(
        global_map,
        color=COLORS["black"],
        linestyle="--",
        linewidth=1.5,
        label=f"mAP wszystkich klas = {global_map:.2f}%",
    )
    axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4, fontsize=9)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Średnie AP [%]")
    axis.set_ylabel("Klasa nuScenes")
    axis.set_title("Średnie AP według klasy")
    axis.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, output_dir, "03_srednie_ap_wedlug_klasy")


def plot_ap_heatmap(metrics, output_dir: Path) -> None:
    thresholds = ["0.5", "1.0", "2.0", "4.0"]
    classes = sorted(metrics["mean_dist_aps"], key=metrics["mean_dist_aps"].get, reverse=True)
    matrix = np.array(
        [[100 * metrics["label_aps"][name][threshold] for threshold in thresholds] for name in classes]
    )
    cmap = LinearSegmentedColormap.from_list(
        "ap_quality", ["#B2182B", "#F7F7F7", "#2166AC"], N=256
    )
    fig, axis = plt.subplots(figsize=(9.4, 7.2))
    image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(thresholds)), [f"AP@{threshold} m" for threshold in thresholds])
    axis.set_yticks(range(len(classes)), [CLASS_LABELS[name] for name in classes])
    axis.set_xlabel("Próg dopasowania odległości środków obiektów")
    axis.set_ylabel("Klasa nuScenes")
    axis.set_title("AP klas dla różnych progów odległości")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value < 18 or value > 82 else COLORS["black"]
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", color=text_color)
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("AP [%]")
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_ap_progi_odleglosci")


def plot_tp_errors(metrics, output_dir: Path) -> None:
    classes = sorted(metrics["mean_dist_aps"], key=metrics["mean_dist_aps"].get, reverse=True)
    y = np.arange(len(classes))
    fig, axes = plt.subplots(1, 5, figsize=(18.5, 7.1), sharey=True)
    fig.suptitle(
        "Błędy True Positive według klasy (mniej = lepiej)",
        fontsize=15,
        fontweight="bold",
    )
    metric_colors = [
        COLORS["blue"],
        COLORS["sky"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["purple"],
    ]
    for axis, (metric, (short_name, unit)), color in zip(axes, ERROR_LABELS.items(), metric_colors):
        values = [finite_or_none(metrics["label_tp_errors"][name].get(metric)) for name in classes]
        numeric = np.array([0.0 if value is None else value for value in values])
        bars = axis.barh(y, numeric, color=color, alpha=0.86)
        max_value = max(numeric) if max(numeric) > 0 else 1
        axis.set_xlim(0, max_value * 1.23)
        axis.set_title(f"{short_name}\n[{unit}]")
        axis.set_xlabel("Błąd")
        for bar, value in zip(bars, values):
            label = "—" if value is None else f"{value:.3f}"
            x_position = 0.01 * max_value if value is None else value + 0.02 * max_value
            axis.text(x_position, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)
    axes[0].set_yticks(y, [CLASS_LABELS[name] for name in classes])
    axes[0].set_ylabel("Klasa nuScenes")
    axes[0].invert_yaxis()
    fig.text(
        0.5,
        0.01,
        "Znak „—” oznacza metrykę niezdefiniowaną dla danej klasy w protokole nuScenes.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95), w_pad=1.2)
    save_figure(fig, output_dir, "05_bledy_tp_wedlug_klasy")


def plot_reference_comparison(metrics, output_dir: Path) -> None:
    models = [
        "Ten checkpoint",
        "CenterPoint-Voxel\n(baseline)",
        "MVP CenterPoint-Voxel\n(publikacja)",
    ]
    values = {
        "mAP": [100 * metrics["mean_ap"], 59.5, 66.0],
        "NDS": [100 * metrics["nd_score"], 66.7, 69.9],
    }
    bar_colors = [COLORS["blue"], COLORS["grey"], COLORS["orange"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), sharey=True)
    fig.suptitle("Porównanie z wynikami referencyjnymi", fontsize=15, fontweight="bold")
    for axis, metric in zip(axes, ("mAP", "NDS")):
        bars = axis.bar(models, values[metric], color=bar_colors, width=0.68)
        axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values[metric]], padding=4)
        axis.set_title(metric)
        axis.set_ylabel("Wynik [%]")
        axis.set_ylim(0, 78)
        axis.tick_params(axis="x", labelsize=9)
    fig.text(
        0.5,
        0.01,
        "Wyniki publikowane mają charakter referencyjny; nie są kontrolowanym badaniem ablacyjnym.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_figure(fig, output_dir, "06_porownanie_referencyjne")


def plot_precision_recall(metrics, details, output_dir: Path) -> None:
    threshold = "2.0"
    classes = sorted(metrics["mean_dist_aps"], key=metrics["mean_dist_aps"].get, reverse=True)
    palette = [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#6A3D9A",
        "#B15928",
        "#1B9E77",
        "#555555",
    ]
    fig, axis = plt.subplots(figsize=(10.4, 7.2))
    for name, color in zip(classes, palette):
        curve = details[f"{name}:{threshold}"]
        recall = np.asarray(curve["recall"], dtype=float)
        precision = np.asarray(curve["precision"], dtype=float)
        ap = 100 * metrics["label_aps"][name][threshold]
        axis.plot(recall, precision, linewidth=1.8, color=color, label=f"{name} ({ap:.1f}%)")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Krzywe precision–recall dla progu odległości 2 m")
    axis.legend(title="Klasa (AP@2 m)", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_figure(fig, output_dir, "07_krzywe_precision_recall_2m")


def write_readme(output_dir: Path, work_dir: Path, selection, epoch_summary, metrics) -> None:
    first_loss = epoch_summary[0]["loss_mean"]
    last_loss = epoch_summary[-1]["loss_mean"]
    reduction = 100 * (first_loss - last_loss) / first_loss
    ordered_best = sorted(metrics["mean_dist_aps"], key=metrics["mean_dist_aps"].get, reverse=True)
    chosen_logs = "\n".join(
        f"- epoka {item['epoch']}: `{item['source']}` "
        f"({item['records']} punktów, iteracje do {item['max_iter']})"
        for item in selection
    )
    content = f"""# Wykresy wyników MVP CenterPoint

W katalogu znajdują się wersje PNG (300 DPI, szybki podgląd) oraz PDF (format wektorowy do składu pracy). Wszystkie wartości pochodzą z `{work_dir.name}`.

## Gotowe rysunki i proponowane podpisy

1. [`01_przebieg_treningu.png`](01_przebieg_treningu.png) — **Przebieg treningu modelu MVP CenterPoint.** Pokazano średnią stratę sześciu głowic zadaniowych, jej składowe klasyfikacyjną i regresyjną oraz harmonogram learning rate. Krzywe strat wygładzono średnią ruchomą; szare wartości surowe pozostawiono jako informację o zmienności.
2. [`02_srednie_straty_epok.png`](02_srednie_straty_epok.png) — **Średnie wartości funkcji straty w kolejnych epokach.** Zestawienie uwidacznia systematyczną zbieżność procesu uczenia oraz różnice skali między składowymi straty.
3. [`03_srednie_ap_wedlug_klasy.png`](03_srednie_ap_wedlug_klasy.png) — **Średnie AP dla klas zbioru nuScenes.** Linia przerywana oznacza globalne mAP = {100 * metrics['mean_ap']:.2f}%; czerwonym kolorem zaznaczono klasy poniżej tej wartości.
4. [`04_ap_progi_odleglosci.png`](04_ap_progi_odleglosci.png) — **AP klas dla progów dopasowania 0,5–4,0 m.** Mapa cieplna pokazuje wrażliwość wyniku na tolerancję błędu położenia środka obiektu.
5. [`05_bledy_tp_wedlug_klasy.png`](05_bledy_tp_wedlug_klasy.png) — **Błędy True Positive według klasy.** Zestawiono błędy translacji, skali, orientacji, prędkości i atrybutu; dla każdej z tych metryk niższa wartość oznacza lepszy wynik.
6. [`06_porownanie_referencyjne.png`](06_porownanie_referencyjne.png) — **Porównanie mAP i NDS z wynikami referencyjnymi.** Wykres obejmuje analizowany checkpoint, publikowany baseline CenterPoint-Voxel oraz publikowany wynik MVP CenterPoint-Voxel.
7. [`07_krzywe_precision_recall_2m.png`](07_krzywe_precision_recall_2m.png) — **Krzywe precision–recall klas dla progu 2 m.** Legenda zawiera wartość AP@2 m każdej klasy.

## Najważniejsze obserwacje

- Średnia strata treningowa spadła z {first_loss:.3f} w epoce 1 do {last_loss:.3f} w epoce 6, czyli o {reduction:.1f}%.
- Najwyższe średnie AP osiągnięto dla klas `{ordered_best[0]}` ({100 * metrics['mean_dist_aps'][ordered_best[0]]:.2f}%), `{ordered_best[1]}` ({100 * metrics['mean_dist_aps'][ordered_best[1]]:.2f}%) i `{ordered_best[2]}` ({100 * metrics['mean_dist_aps'][ordered_best[2]]:.2f}%).
- Najniższe średnie AP dotyczą klas `{ordered_best[-1]}` ({100 * metrics['mean_dist_aps'][ordered_best[-1]]:.2f}%) i `{ordered_best[-2]}` ({100 * metrics['mean_dist_aps'][ordered_best[-2]]:.2f}%).
- Checkpoint osiągnął mAP {100 * metrics['mean_ap']:.2f}% i NDS {100 * metrics['nd_score']:.2f}%; względem publikowanego baseline'u CenterPoint-Voxel jest to odpowiednio +{100 * metrics['mean_ap'] - 59.5:.2f} p.p. i +{100 * metrics['nd_score'] - 66.7:.2f} p.p.

## Metodyka łączenia logów treningowych

Trening był wznawiany, a logi zawierają nakładające się fragmenty epoki 4. Skrypt automatycznie wybiera dla każdej epoki plik z największą liczbą punktów, dzięki czemu krótkie, przerwane próby nie są liczone wielokrotnie:

{chosen_logs}

W logach nie zapisano mAP/NDS po każdej epoce. Z tego powodu nie należy tworzyć wykresu „mAP względem epoki” przez interpolację — dostępny jest wyłącznie wynik końcowej ewaluacji. `loss`, `hm_loss` i `loc_loss` w logu są listami dla sześciu głowic zadaniowych, więc na wykresach użyto ich średniej arytmetycznej.

## Dane pomocnicze i reprodukcja

- [`training_curve.csv`](training_curve.csv) — oczyszczony przebieg treningu punkt po punkcie.
- [`training_epoch_summary.csv`](training_epoch_summary.csv) — wartości uśrednione w epokach.
- Wersje PDF mają te same nazwy co pliki PNG.

Ponowne wygenerowanie:

```bash
MPLCONFIGDIR=.cache/matplotlib .venv.cu121/bin/python scripts/plot_mvp_results.py
```
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    eval_dir = work_dir / "eval_latest"
    output_dir = (args.output_dir or (eval_dir / "wykresy")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_json(eval_dir / "metrics_summary.json")
    details = load_json(eval_dir / "metrics_details.json")
    training_rows, selection = load_training_rows(work_dir)
    if not training_rows:
        raise RuntimeError(f"No training rows found in {work_dir}")

    write_training_csv(training_rows, output_dir)
    epoch_summary = summarize_epochs(training_rows, output_dir)
    plot_training(training_rows, epoch_summary, output_dir, args.rolling_window)
    plot_epoch_losses(epoch_summary, output_dir)
    plot_class_ap(metrics, output_dir)
    plot_ap_heatmap(metrics, output_dir)
    plot_tp_errors(metrics, output_dir)
    plot_reference_comparison(metrics, output_dir)
    plot_precision_recall(metrics, details, output_dir)
    write_readme(output_dir, work_dir, selection, epoch_summary, metrics)

    print(f"Generated 7 plots (PNG + PDF) in: {output_dir}")


if __name__ == "__main__":
    main()
