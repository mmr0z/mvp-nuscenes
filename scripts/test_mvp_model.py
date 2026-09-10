#!/usr/bin/env python3
"""Run a reproducible nuScenes evaluation for the trained MVP model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CENTERPOINT_ROOT = REPO_ROOT / "third_party" / "CenterPoint"
DEFAULT_CONFIG = (
    CENTERPOINT_ROOT
    / "configs"
    / "mvp"
    / "nusc_centerpoint_voxelnet_0075voxel_bevfusion_virtual_ft.py"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "mvp_centerpoint_epoch_6.pth"
DEFAULT_WORK_DIR = REPO_ROOT / "results" / "generated" / "validation"
DEFAULT_VAL_INFO = CENTERPOINT_ROOT / "data" / "nuScenes" / "infos_val_10sweeps_withvelo_filter_True.pkl"
DEFAULT_DATA_ROOT = CENTERPOINT_ROOT / "data" / "nuScenes"
DEFAULT_MIN_MAP = 0.60
DEFAULT_MIN_NDS = 0.65
METHOD_NAME = "MVP CenterPoint VoxelNet"
FUSION_TYPE = "Fuzja punktowa na wejściu: LiDAR + punkty wirtualne wygenerowane z kamer"
NUSCENES_CLASSES = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
)
REFERENCE_RESULTS = {
    "CenterPoint-Voxel (publikowany baseline)": {"mean_ap": 0.595, "nd_score": 0.667},
    "MVP CenterPoint-Voxel (wynik publikowany)": {"mean_ap": 0.660, "nd_score": 0.699},
}
TP_ERROR_LABELS = {
    "trans_err": ("mATE", "m", "średni błąd translacji"),
    "scale_err": ("mASE", "1 - IoU", "średni błąd skali"),
    "orient_err": ("mAOE", "rad", "średni błąd orientacji"),
    "vel_err": ("mAVE", "m/s", "średni błąd prędkości"),
    "attr_err": ("mAAE", "1 - accuracy", "średni błąd atrybutu"),
}


def path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained MVP CenterPoint model and check nuScenes quality thresholds."
    )
    parser.add_argument("--config", type=path_arg, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=path_arg, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--work-dir", type=path_arg, default=DEFAULT_WORK_DIR)
    parser.add_argument("--val-info", type=path_arg, default=DEFAULT_VAL_INFO)
    parser.add_argument("--data-root", type=path_arg, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--python", type=path_arg, default=Path(sys.executable).resolve())
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--min-map", type=float, default=DEFAULT_MIN_MAP)
    parser.add_argument("--min-nds", type=float, default=DEFAULT_MIN_NDS)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths and print CUDA status without running inference.",
    )
    parser.add_argument(
        "--metrics-only",
        type=path_arg,
        help="Check an existing metrics_summary.json without running inference.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_inputs(args: argparse.Namespace) -> None:
    require_file(args.config, "config")
    require_file(args.checkpoint, "checkpoint")
    require_file(args.val_info, "validation info")
    require_file(args.python, "Python interpreter")
    require_file(CENTERPOINT_ROOT / "tools" / "dist_test.py", "CenterPoint evaluation script")
    require_dir(args.data_root / "v1.0-trainval", "nuScenes trainval metadata")
    require_dir(args.data_root / "samples" / "LIDAR_TOP", "nuScenes LiDAR samples")
    require_dir(args.data_root / "samples" / "LIDAR_TOP_VIRTUAL", "MVP virtual LiDAR samples")
    require_dir(args.data_root / "sweeps" / "LIDAR_TOP_VIRTUAL", "MVP virtual LiDAR sweeps")

    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1")
    if not 0.0 <= args.min_map <= 1.0:
        raise ValueError("--min-map must be between 0 and 1")
    if not 0.0 <= args.min_nds <= 1.0:
        raise ValueError("--min-nds must be between 0 and 1")


def validate_thresholds(min_map: float, min_nds: float) -> None:
    if not 0.0 <= min_map <= 1.0:
        raise ValueError("--min-map must be between 0 and 1")
    if not 0.0 <= min_nds <= 1.0:
        raise ValueError("--min-nds must be between 0 and 1")


def finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def validate_metrics(metrics: dict[str, Any]) -> None:
    """Fail early when an incomplete or corrupt nuScenes summary is supplied."""
    for metric_name in ("mean_ap", "nd_score"):
        value = finite_number(metrics.get(metric_name), metric_name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{metric_name} must be between 0 and 1")

    for error_name in TP_ERROR_LABELS:
        value = finite_number(metrics.get("tp_errors", {}).get(error_name), error_name)
        if value < 0.0:
            raise ValueError(f"{error_name} must be non-negative")

    label_aps = metrics.get("label_aps")
    mean_dist_aps = metrics.get("mean_dist_aps")
    if not isinstance(label_aps, dict) or not isinstance(mean_dist_aps, dict):
        raise ValueError("label_aps and mean_dist_aps must be objects")
    missing_classes = set(NUSCENES_CLASSES) - set(label_aps)
    missing_classes.update(set(NUSCENES_CLASSES) - set(mean_dist_aps))
    if missing_classes:
        raise ValueError(
            "nuScenes class metrics are missing: " + ", ".join(sorted(missing_classes))
        )
    for class_name in NUSCENES_CLASSES:
        mean_ap = finite_number(mean_dist_aps[class_name], f"{class_name} mean AP")
        if not 0.0 <= mean_ap <= 1.0:
            raise ValueError(f"{class_name} mean AP must be between 0 and 1")
        for distance in ("0.5", "1.0", "2.0", "4.0"):
            ap = finite_number(
                label_aps[class_name].get(distance), f"{class_name} AP@{distance}m"
            )
            if not 0.0 <= ap <= 1.0:
                raise ValueError(f"{class_name} AP@{distance}m must be between 0 and 1")


def json_safe(value: Any) -> Any:
    """Convert non-finite floats to JSON null for strict RFC-compatible output."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def cuda_status(python: Path) -> tuple[bool, str]:
    command = [
        str(python),
        "-c",
        (
            "import torch; "
            "print(f'torch={torch.__version__} cuda={torch.version.cuda} "
            "available={torch.cuda.is_available()} devices={torch.cuda.device_count()}')"
        ),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    detail = (result.stdout or result.stderr).strip()
    return result.returncode == 0 and "available=True" in detail, detail


def evaluation_env() -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [
        str(REPO_ROOT),
        str(REPO_ROOT / "third_party"),
        str(REPO_ROOT / "third_party" / "CenterNet2"),
        str(REPO_ROOT / "third_party" / "CenterNet2" / "projects" / "CenterNet2"),
        str(CENTERPOINT_ROOT),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
    return env


def runtime_status(python: Path) -> tuple[bool, str]:
    command = [
        str(python),
        "-c",
        (
            "import torch, spconv; "
            "from det3d.models import build_detector; "
            "print(f'torch={torch.__version__} spconv={spconv.__version__}')"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=CENTERPOINT_ROOT,
        env=evaluation_env(),
        capture_output=True,
        text=True,
    )
    detail = (result.stdout if result.returncode == 0 else result.stderr).strip()
    return result.returncode == 0, detail


def evaluation_command(args: argparse.Namespace) -> list[str]:
    dist_test = CENTERPOINT_ROOT / "tools" / "dist_test.py"
    test_args = [
        str(dist_test),
        str(args.config),
        "--work_dir",
        str(args.work_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--gpus",
        str(args.gpus),
    ]
    if args.gpus == 1:
        return [str(args.python), *test_args]
    return [
        str(args.python),
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.gpus}",
        *test_args,
    ]


def run_evaluation(args: argparse.Namespace) -> Path:
    runtime_available, runtime_detail = runtime_status(args.python)
    print(f"CenterPoint runtime: {'OK' if runtime_available else 'ERROR'}")
    if not runtime_available:
        raise RuntimeError(runtime_detail)

    available, detail = cuda_status(args.python)
    print(f"CUDA: {detail}")
    if not available:
        raise RuntimeError("CUDA is unavailable; CenterPoint MVP evaluation requires a GPU.")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    command = evaluation_command(args)

    print(f"Running: {shlex.join(command)}")
    subprocess.run(command, cwd=CENTERPOINT_ROOT, env=evaluation_env(), check=True)
    return args.work_dir / "metrics_summary.json"


def check_metrics(metrics_path: Path, min_map: float, min_nds: float) -> dict[str, object]:
    require_file(metrics_path, "nuScenes metrics")
    with metrics_path.open() as metrics_file:
        metrics = json.load(metrics_file)
    validate_metrics(metrics)

    details_path = metrics_path.with_name("metrics_details.json")
    metrics_details = None
    if details_path.is_file():
        with details_path.open() as details_file:
            metrics_details = json.load(details_file)

    timing_path = metrics_path.with_name("inference_timing.json")
    inference_timing = None
    if timing_path.is_file():
        with timing_path.open() as timing_file:
            inference_timing = json.load(timing_file)

    mean_ap = float(metrics["mean_ap"])
    nd_score = float(metrics["nd_score"])
    passed = mean_ap >= min_map and nd_score >= min_nds
    report = {
        "passed": passed,
        "thresholds": {
            "min_map": min_map,
            "min_nds": min_nds,
        },
        "metrics_summary_path": str(metrics_path),
        "metrics_details_path": str(details_path) if metrics_details is not None else None,
        "inference_timing_path": str(timing_path) if inference_timing is not None else None,
        "metrics_summary": metrics,
        "metrics_details": metrics_details,
        "inference_timing": inference_timing,
    }

    print(f"mAP: {mean_ap:.4f} (minimum: {min_map:.4f})")
    print(f"NDS: {nd_score:.4f} (minimum: {min_nds:.4f})")
    print("TP errors:")
    for metric_name, metric_value in metrics["tp_errors"].items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("Per-class metrics:")
    for class_name, class_ap in metrics["mean_dist_aps"].items():
        aps = metrics["label_aps"][class_name]
        errors = metrics["label_tp_errors"][class_name]
        ap_values = " ".join(f"AP@{distance}={value:.4f}" for distance, value in aps.items())
        error_values = " ".join(f"{name}={value:.4f}" for name, value in errors.items())
        print(f"  {class_name}: AP={class_ap:.4f} {ap_values} {error_values}")

    if inference_timing is not None:
        print("Inference timing:")
        print(
            "  latency: "
            f"{inference_timing['latency_milliseconds_per_frame_per_gpu']:.2f} ms/frame"
        )
        print(f"  throughput: {inference_timing['frames_per_second_per_gpu']:.2f} FPS/GPU")
        print(
            "  full loop: "
            f"{inference_timing['full_inference_loop_seconds']:.2f} s "
            f"({inference_timing['full_inference_loop_frames_per_second']:.2f} FPS)"
        )

    print(f"MVP_MODEL_TEST={'PASS' if passed else 'FAIL'}")
    return report


def percent(value: float) -> str:
    return f"{value * 100:.2f}"


def write_metrics_csv(report: dict[str, object], csv_path: Path) -> None:
    """Write long-form metrics that can be imported directly into analysis software."""
    metrics = report["metrics_summary"]
    timing = report["inference_timing"]
    rows: list[dict[str, Any]] = []

    def add(
        category: str,
        metric: str,
        value: Any,
        unit: str,
        class_name: str = "",
        distance_threshold_m: Any = "",
        source: str = "this_checkpoint",
    ) -> None:
        rows.append(
            {
                "category": category,
                "source": source,
                "metric": metric,
                "class": class_name,
                "distance_threshold_m": distance_threshold_m,
                "value": value,
                "unit": unit,
            }
        )

    add("global", "mAP", metrics["mean_ap"], "ratio")
    add("global", "NDS", metrics["nd_score"], "ratio")
    for source, reference in REFERENCE_RESULTS.items():
        add("reference", "mAP", reference["mean_ap"], "ratio", source=source)
        add("reference", "NDS", reference["nd_score"], "ratio", source=source)
    for error_name, value in metrics["tp_errors"].items():
        short_name, unit, _ = TP_ERROR_LABELS[error_name]
        add("global_tp_error", short_name, value, unit)

    for class_name in NUSCENES_CLASSES:
        add("class_ap", "mean_AP", metrics["mean_dist_aps"][class_name], "ratio", class_name)
        for distance, value in metrics["label_aps"][class_name].items():
            add("class_ap", "AP", value, "ratio", class_name, distance)
        for error_name, value in metrics.get("label_tp_errors", {}).get(class_name, {}).items():
            short_name, unit, _ = TP_ERROR_LABELS[error_name]
            safe_value = None if not math.isfinite(float(value)) else value
            add("class_tp_error", short_name, safe_value, unit, class_name)

    if timing is not None:
        add("runtime", "latency", timing["latency_milliseconds_per_frame_per_gpu"], "ms/frame/GPU")
        add("runtime", "throughput", timing["frames_per_second_per_gpu"], "frames/s/GPU")
        add(
            "runtime",
            "full_loop_throughput",
            timing["full_inference_loop_frames_per_second"],
            "frames/s",
        )
        if timing.get("peak_vram_allocated_mib_per_gpu") is not None:
            add(
                "runtime",
                "peak_vram_allocated",
                timing["peak_vram_allocated_mib_per_gpu"],
                "MiB/GPU",
            )

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "category",
                "source",
                "metric",
                "class",
                "distance_threshold_m",
                "value",
                "unit",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(report: dict[str, object], summary_path: Path) -> None:
    metrics = report["metrics_summary"]
    timing = report["inference_timing"]
    errors = metrics["tp_errors"]
    cfg = metrics.get("cfg", {})
    class_ranking = sorted(
        metrics["mean_dist_aps"].items(), key=lambda item: item[1], reverse=True
    )
    strongest = ", ".join(f"{name} ({percent(value)}%)" for name, value in class_ranking[:3])
    weakest = ", ".join(f"{name} ({percent(value)}%)" for name, value in class_ranking[-3:])
    distance_macro_aps = {
        distance: sum(metrics["label_aps"][name][distance] for name in NUSCENES_CLASSES)
        / len(NUSCENES_CLASSES)
        for distance in ("0.5", "1.0", "2.0", "4.0")
    }

    lines = [
        "# Ewaluacja MVP — raport do analizy pracy dyplomowej",
        "",
        f"Model osiąga mAP **{percent(metrics['mean_ap'])}%** i NDS "
        f"**{percent(metrics['nd_score'])}%** na zbiorze walidacyjnym nuScenes.",
        "",
        "## Protokół eksperymentu",
        "",
        f"- Model: {METHOD_NAME}.",
        f"- Fuzja: {FUSION_TYPE}.",
        f"- Zbiór: nuScenes validation, {timing.get('dataset_samples', 'brak danych') if timing else 'brak danych'} ramek, po 10 kolejnych skanów LiDAR.",
        f"- Dopasowanie AP: odległość środków 3D przy progach {', '.join(str(v) for v in cfg.get('dist_ths', []))} m; maksymalnie {cfg.get('max_boxes_per_sample', 'brak danych')} detekcji na ramkę.",
        f"- Konfiguracja: `{Path(report['config']).name}`.",
        f"- Checkpoint: `{Path(report['checkpoint']).name}`.",
        "- Punkty wirtualne wygenerowano wcześniej z obrazów; ewaluator otrzymuje je jako rozszerzoną chmurę punktów.",
        "",
        "## Metryki główne",
        "",
        "| Metryka | Wynik [%] |",
        "| --- | ---: |",
        f"| mAP | {percent(metrics['mean_ap'])} |",
        f"| NDS | {percent(metrics['nd_score'])} |",
        "",
        "## Błędy True Positive nuScenes",
        "",
        "| Metryka | Wynik | Jednostka | Interpretacja |",
        "| --- | ---: | --- | --- |",
    ]

    for error_name, (short_name, unit, description) in TP_ERROR_LABELS.items():
        lines.append(
            f"| {short_name} | {errors[error_name]:.4f} | {unit} | "
            f"{description}; mniej = lepiej |"
        )

    lines.extend(
        [
            "",
            "## AP według klasy i tolerancji odległości",
            "",
            "Wartości podano w procentach. Średnie AP klasy jest średnią z czterech progów odległości.",
            "",
            "| Klasa | AP@0.5 m | AP@1.0 m | AP@2.0 m | AP@4.0 m | Średnie AP |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for class_name in NUSCENES_CLASSES:
        aps = metrics["label_aps"][class_name]
        lines.append(
            f"| {class_name} | {percent(aps['0.5'])} | {percent(aps['1.0'])} | "
            f"{percent(aps['2.0'])} | {percent(aps['4.0'])} | "
            f"{percent(metrics['mean_dist_aps'][class_name])} |"
        )
    lines.extend(
        [
            f"| **Makrośrednia klas** | **{percent(distance_macro_aps['0.5'])}** | **{percent(distance_macro_aps['1.0'])}** | **{percent(distance_macro_aps['2.0'])}** | **{percent(distance_macro_aps['4.0'])}** | **{percent(metrics['mean_ap'])}** |",
            "",
            "## Porównanie z wynikami referencyjnymi",
            "",
            "| Model | mAP [%] | NDS [%] | ΔmAP względem referencji [p.p.] | ΔNDS [p.p.] |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| **Ten checkpoint** | **{percent(metrics['mean_ap'])}** | **{percent(metrics['nd_score'])}** | — | — |",
        ]
    )
    for name, reference in REFERENCE_RESULTS.items():
        lines.append(
            f"| {name} | {percent(reference['mean_ap'])} | {percent(reference['nd_score'])} | "
            f"{percent(metrics['mean_ap'] - reference['mean_ap'])} | "
            f"{percent(metrics['nd_score'] - reference['nd_score'])} |"
        )
    lines.extend(
        [
            "",
            "Wartości referencyjne pochodzą z tabeli wyników walidacyjnych publikacji i repozytorium MVP.",
        ]
    )

    lines.extend(["", "## Wydajność inferencji", ""])
    if timing is None:
        lines.append("Brak pliku `inference_timing.json`; wydajność nie została zmierzona.")
    else:
        vram = timing.get("peak_vram_allocated_mib_per_gpu")
        vram_text = f"{vram:.0f}" if vram is not None else "—"
        lines.extend(
            [
                "| GPU | Ramki pomiarowe / wszystkie | Latencja [ms/ramkę/GPU] | Przepustowość [FPS/GPU] | Pełna pętla [s] | Peak VRAM allocated [MiB/GPU] |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                f"| {timing.get('gpu_name', 'brak danych')} | {timing['timed_samples']} / {timing['dataset_samples']} | {timing['latency_milliseconds_per_frame_per_gpu']:.2f} | {timing['frames_per_second_per_gpu']:.2f} | {timing['full_inference_loop_seconds']:.2f} | {vram_text} |",
                "",
                "Pomiar latencji obejmuje środkową 1/3 zbioru po rozgrzaniu GPU. Pełna pętla obejmuje ładowanie danych i inferencję CenterPoint, ale nie obejmuje generowania punktów wirtualnych z kamer ani obliczania metryk nuScenes.",
            ]
        )

    baseline = REFERENCE_RESULTS["CenterPoint-Voxel (publikowany baseline)"]
    published_mvp = REFERENCE_RESULTS["MVP CenterPoint-Voxel (wynik publikowany)"]
    lines.extend(
        [
            "",
            "## Wnioski do analizy",
            "",
            f"- Najwyższe średnie AP uzyskano dla: {strongest}.",
            f"- Najniższe średnie AP uzyskano dla: {weakest}; te klasy są głównym obszarem dalszej poprawy.",
            f"- Względem publikowanego baseline'u CenterPoint-Voxel checkpoint poprawia mAP o **{percent(metrics['mean_ap'] - baseline['mean_ap'])} p.p.** i NDS o **{percent(metrics['nd_score'] - baseline['nd_score'])} p.p.**",
            f"- Względem publikowanego wyniku MVP rezultat jest niższy o **{percent(published_mvp['mean_ap'] - metrics['mean_ap'])} p.p. mAP** i **{percent(published_mvp['nd_score'] - metrics['nd_score'])} p.p. NDS**.",
            "",
            "## Ograniczenia interpretacji",
            "",
            "- To pojedynczy checkpoint i pojedyncza ewaluacja; bez powtórzeń treningu nie można podać odchylenia standardowego ani przedziału ufności między seedami.",
            "- Porównania z tabelą publikacji są referencyjne, nie są kontrolowanym badaniem ablacyjnym wykonanym w identycznym środowisku.",
            "- Koszt generowania punktów wirtualnych przez model 2D nie wchodzi do podanej latencji; liczby nie reprezentują pełnego potoku kamera → MVP → detekcja 3D.",
            "- `use_camera=false` w metadanych nuScenes oznacza, że ewaluator widzi tylko wejście punktowe. Nie oznacza to braku informacji z kamer, ponieważ jest ona zakodowana we wcześniej wygenerowanych punktach wirtualnych.",
            "",
            "## Reprodukcja",
            "",
            "```bash",
            "conda activate mvp-clean",
            "python scripts/test_mvp_model.py",
            "```",
            "",
            "Do ponownego wygenerowania raportu bez inferencji:",
            "",
            "```bash",
            f"python scripts/test_mvp_model.py --metrics-only {report['metrics_summary_path']}",
            "```",
            "",
            "Dane maszynowe do dalszych wykresów i obliczeń znajdują się w `thesis_metrics.csv`; pełny raport źródłowy znajduje się w `mvp_test_report.json`.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    try:
        validate_thresholds(args.min_map, args.min_nds)
        if args.metrics_only:
            require_file(args.metrics_only, "nuScenes metrics")
            print("Metrics input check: OK")
        else:
            validate_inputs(args)
            print("Input check: OK")

        if args.check_only:
            runtime_available, runtime_detail = runtime_status(args.python)
            print(f"CenterPoint runtime: {'OK' if runtime_available else 'ERROR'}")
            if not runtime_available:
                raise RuntimeError(runtime_detail)
            _, detail = cuda_status(args.python)
            print(f"CUDA: {detail}")
            return 0

        metrics_path = args.metrics_only or run_evaluation(args)
        report = check_metrics(metrics_path, args.min_map, args.min_nds)
        report["config"] = str(args.config)
        report["checkpoint"] = str(args.checkpoint)
        report_path = metrics_path.parent / "mvp_test_report.json"
        with report_path.open("w") as report_file:
            json.dump(json_safe(report), report_file, indent=2, allow_nan=False)
            report_file.write("\n")
        print(f"Report: {report_path}")
        summary_path = metrics_path.parent / "summary.md"
        write_summary(report, summary_path)
        print(f"Summary: {summary_path}")
        csv_path = metrics_path.parent / "thesis_metrics.csv"
        write_metrics_csv(report, csv_path)
        print(f"Thesis metrics: {csv_path}")
        return 0 if report["passed"] else 1
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"MVP_MODEL_TEST=ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
