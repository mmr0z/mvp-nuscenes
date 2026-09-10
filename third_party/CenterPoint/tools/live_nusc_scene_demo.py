import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

CENTERPOINT_ROOT = Path(__file__).resolve().parents[1]
if str(CENTERPOINT_ROOT) not in sys.path:
    sys.path.insert(0, str(CENTERPOINT_ROOT))

from det3d.datasets import build_dataset
from det3d.torchie import Config
from det3d.torchie.parallel import collate_kitti
try:
    from tools.demo_utils import Box, _second_det_to_nusc_box
except ModuleNotFoundError:
    from demo_utils import Box, _second_det_to_nusc_box


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CenterPoint/MVP inference on one nuScenes scene and render an animated BEV preview."
    )
    parser.add_argument("config", help="Path to a CenterPoint config.")
    parser.add_argument("--checkpoint", help="Model checkpoint.")
    parser.add_argument(
        "--prediction-json",
        help="NuScenes detection JSON to visualize instead of running model inference.",
    )
    parser.add_argument("--scene", help="nuScenes scene name, e.g. scene-0103.")
    parser.add_argument("--scene-index", type=int, default=0, help="Index among available scenes when --scene is omitted.")
    parser.add_argument("--list-scenes", action="store_true", help="List scenes present in cfg.data.val and exit.")
    parser.add_argument("--score-threshold", type=float, default=0.35, help="Detection score threshold for drawing boxes.")
    parser.add_argument("--range", type=float, default=54.0, help="BEV range in meters from ego vehicle.")
    parser.add_argument("--size", type=int, default=900, help="Rendered frame size in pixels.")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback/video FPS.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth sample in the scene.")
    parser.add_argument("--show", action="store_true", help="Show a live OpenCV window while running.")
    parser.add_argument("--output", help="Optional MP4 output path.")
    parser.add_argument("--draw-gt", action="store_true", help="Draw ground-truth boxes in red.")
    parser.add_argument("--labels", action="store_true", help="Draw predicted class labels and scores.")
    return parser.parse_args()


def clone_detection(det):
    return {
        "box3d_lidar": np.asarray(det["box3d_lidar"]).copy(),
        "scores": np.asarray(det["scores"]).copy(),
        "label_preds": np.asarray(det["label_preds"]).copy(),
    }


def output_to_numpy(output):
    det = {"metadata": output["metadata"]}
    for key in ["box3d_lidar", "scores", "label_preds"]:
        value = output[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        det[key] = value
    return det


def gt_to_detection(info):
    boxes = info["gt_boxes"].astype(np.float32)
    return {
        "box3d_lidar": boxes.copy(),
        "scores": np.ones(len(boxes), dtype=np.float32),
        "label_preds": np.zeros(len(boxes), dtype=np.int64),
    }


def default_prediction_json(args, cfg):
    if not args.checkpoint:
        return None
    val_info_name = Path(cfg.data.val.info_path).with_suffix(".json").name
    return Path(args.checkpoint).parent / val_info_name


def load_prediction_json(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def global_annos_to_lidar_boxes(annos, info, class_to_label):
    global_from_lidar = np.linalg.inv(info["ref_from_car"].dot(info["car_from_global"]))
    lidar_from_global = np.linalg.inv(global_from_lidar)
    rotation = lidar_from_global[:3, :3]

    boxes = []
    for anno in annos:
        center_global = np.array([*anno["translation"], 1.0], dtype=np.float64)
        center_lidar = lidar_from_global.dot(center_global)[:3]
        orientation_global = Quaternion(anno["rotation"])
        orientation_lidar = Quaternion(matrix=rotation.dot(orientation_global.rotation_matrix))
        name = anno["detection_name"]
        label = class_to_label.get(name, -1)
        boxes.append(
            Box(
                center_lidar.tolist(),
                anno["size"],
                orientation_lidar,
                label=label,
                score=anno["detection_score"],
                name=name,
            )
        )
    return boxes


def scene_sample_tokens(nusc, scene):
    tokens = []
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        tokens.append(sample_token)
        sample_token = sample["next"]
    return tokens


def available_scenes(nusc, dataset):
    token_to_index = {info["token"]: idx for idx, info in enumerate(dataset._nusc_infos)}
    scenes = []
    for scene in nusc.scene:
        indices = [token_to_index[token] for token in scene_sample_tokens(nusc, scene) if token in token_to_index]
        if indices:
            scenes.append((scene, indices))
    return scenes


def select_scene(nusc, dataset, scene_name, scene_index):
    scenes = available_scenes(nusc, dataset)
    if scene_name:
        for scene, indices in scenes:
            if scene["name"] == scene_name:
                return scene, indices
        names = ", ".join(scene["name"] for scene, _ in scenes[:20])
        raise ValueError(f"Scene {scene_name!r} is not present in cfg.data.val. First available scenes: {names}")

    if scene_index < 0 or scene_index >= len(scenes):
        raise ValueError(f"--scene-index must be between 0 and {len(scenes) - 1}")
    return scenes[scene_index]


def xy_to_pixel(xy, limit, size):
    xy = np.asarray(xy)
    u = (xy[..., 0] + limit) / (2 * limit) * (size - 1)
    v = (limit - xy[..., 1]) / (2 * limit) * (size - 1)
    return np.stack([u, v], axis=-1).astype(np.int32)


def draw_grid(img, limit):
    size = img.shape[0]
    axis_color = (55, 55, 55)
    grid_color = (30, 30, 30)
    for meters in range(-int(limit), int(limit) + 1, 10):
        u = int((meters + limit) / (2 * limit) * (size - 1))
        v = int((limit - meters) / (2 * limit) * (size - 1))
        cv2.line(img, (u, 0), (u, size - 1), grid_color, 1)
        cv2.line(img, (0, v), (size - 1, v), grid_color, 1)
    center = xy_to_pixel([[0.0, 0.0]], limit, size)[0]
    cv2.line(img, (center[0], 0), (center[0], size - 1), axis_color, 1)
    cv2.line(img, (0, center[1]), (size - 1, center[1]), axis_color, 1)


def draw_points(img, points, limit):
    size = img.shape[0]
    xyz = points[:, :3]
    dist = np.linalg.norm(xyz[:, :2], axis=1)
    mask = (
        (np.abs(xyz[:, 0]) <= limit)
        & (np.abs(xyz[:, 1]) <= limit)
        & (dist > 3.0)
    )
    xyz = xyz[mask]
    dist = dist[mask]
    if len(xyz) == 0:
        return

    pix = xy_to_pixel(xyz[:, :2], limit, size)
    in_img = (
        (pix[:, 0] >= 0)
        & (pix[:, 0] < size)
        & (pix[:, 1] >= 0)
        & (pix[:, 1] < size)
    )
    pix = pix[in_img]
    dist = dist[in_img]
    vals = np.clip(230 - 170 * dist / limit, 45, 230).astype(np.uint8)
    img[pix[:, 1], pix[:, 0]] = np.stack([vals, vals, vals], axis=1)


def draw_box(img, box, limit, color, label=None, thickness=2):
    size = img.shape[0]
    corners = box.bottom_corners()[:2, :].T
    pts = xy_to_pixel(corners, limit, size)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    all_corners = box.corners()[:2, :].T
    front = np.mean(all_corners[[2, 3]], axis=0)
    center = np.mean(all_corners[[2, 3, 7, 6]], axis=0)
    front_line = xy_to_pixel([center, front], limit, size)
    cv2.line(img, tuple(front_line[0]), tuple(front_line[1]), color, thickness, lineType=cv2.LINE_AA)

    if label:
        text_pos = xy_to_pixel([box.center[:2]], limit, size)[0]
        if 0 <= text_pos[0] < size and 0 <= text_pos[1] < size:
            cv2.putText(
                img,
                label,
                tuple(text_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )


def render_frame(points, pred_boxes, gt_boxes, class_names, title, args):
    size = args.size
    img = np.zeros((size, size, 3), dtype=np.uint8)
    draw_grid(img, args.range)
    draw_points(img, points, args.range)

    if args.draw_gt and gt_boxes is not None:
        for box in gt_boxes:
            draw_box(img, box, args.range, (60, 60, 255), thickness=2)

    for box in pred_boxes:
        if box.score < args.score_threshold:
            continue
        label = None
        if args.labels:
            if 0 <= box.label < len(class_names):
                name = class_names[box.label]
            else:
                name = box.name if box.name else str(box.label)
            label = f"{name} {box.score:.2f}"
        draw_box(img, box, args.range, (255, 170, 40), label=label, thickness=2)

    cv2.circle(img, tuple(xy_to_pixel([[0.0, 0.0]], args.range, size)[0]), 4, (0, 255, 255), -1)
    cv2.putText(img, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"pred >= {args.score_threshold:.2f}   blue=prediction   red=GT",
        (14, size - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return img


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.val)
    nusc = NuScenes(version=dataset.version, dataroot=str(dataset._root_path), verbose=False)

    scenes = available_scenes(nusc, dataset)
    if args.list_scenes:
        for idx, (scene, indices) in enumerate(scenes):
            print(f"{idx:03d} {scene['name']} frames={len(indices)} description={scene['description']}")
        return

    scene, indices = select_scene(nusc, dataset, args.scene, args.scene_index)
    indices = indices[:: max(args.stride, 1)]
    if not indices:
        raise RuntimeError("No frames selected for this scene.")

    prediction_json = Path(args.prediction_json) if args.prediction_json else None
    cuda_available = torch.cuda.is_available()
    if prediction_json is None and not cuda_available:
        candidate = default_prediction_json(args, cfg)
        if candidate is not None and candidate.exists():
            prediction_json = candidate
            print(f"CUDA is not available; using saved predictions from {prediction_json}")

    model = None
    prediction_annos = None
    if prediction_json is not None:
        if not prediction_json.exists():
            raise FileNotFoundError(f"Prediction JSON does not exist: {prediction_json}")
        prediction_annos = load_prediction_json(prediction_json)
    else:
        if not cuda_available:
            raise RuntimeError(
                "CUDA is required for live model inference, but torch.cuda.is_available() is False. "
                "Use --prediction-json to visualize saved detections without GPU."
            )
        if not args.checkpoint:
            raise RuntimeError("--checkpoint is required when running live model inference.")

        from det3d.models import build_detector
        from det3d.torchie.apis import batch_processor
        from det3d.torchie.trainer import load_checkpoint

        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = model.cuda()
        model.eval()

    writer = None
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, args.fps, (args.size, args.size))

    delay_ms = max(1, int(1000 / args.fps))
    print(f"Rendering {scene['name']} ({len(indices)} frames)")
    class_to_label = {name: idx for idx, name in enumerate(dataset._class_names)}

    try:
        for frame_idx, dataset_idx in enumerate(indices):
            frame_start = time.time()
            sample = dataset[dataset_idx]
            points = sample["points"]

            info = dataset._nusc_infos[dataset_idx]
            if model is None:
                pred_boxes = global_annos_to_lidar_boxes(
                    prediction_annos.get(info["token"], []), info, class_to_label
                )
            else:
                data = collate_kitti([sample])
                with torch.no_grad():
                    output = batch_processor(model, data, train_mode=False, local_rank=0)[0]
                detection = output_to_numpy(output)
                pred_boxes = _second_det_to_nusc_box(clone_detection(detection))
            gt_boxes = _second_det_to_nusc_box(clone_detection(gt_to_detection(info)))

            elapsed_ms = (time.time() - frame_start) * 1000.0
            title = f"{scene['name']}  {frame_idx + 1}/{len(indices)}  {elapsed_ms:.0f} ms"
            frame = render_frame(points, pred_boxes, gt_boxes, dataset._class_names, title, args)

            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("MVP CenterPoint nuScenes scene", frame)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved video to {args.output}")
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
