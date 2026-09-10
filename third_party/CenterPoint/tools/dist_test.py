import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def save_json(data, root, filename):
    with open(os.path.join(root, filename), "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    model.eval()
    mode = "val"
    torch.cuda.reset_peak_memory_stats()

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    torch.cuda.synchronize()
    loop_start = time.perf_counter()
    timing_start_batch = int(len(data_loader) / 3)
    timing_end_batch = int(len(data_loader) * 2 / 3)
    timing_start = None
    timing_end = None
    timed_samples = 0
    processed_samples = 0

    for i, data_batch in enumerate(data_loader):
        if i == timing_start_batch:
            torch.cuda.synchronize()
            timing_start = time.perf_counter()

        if i == timing_end_batch:
            torch.cuda.synchronize()
            timing_end = time.perf_counter()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )
        processed_samples += len(outputs)
        if timing_start_batch <= i < timing_end_batch:
            timed_samples += len(outputs)

        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
            if args.local_rank == 0:
                prog_bar.update()

    torch.cuda.synchronize()
    synchronize()
    loop_end = time.perf_counter()

    if timing_start is None:
        timing_start = loop_start
    if timing_end is None:
        timing_end = loop_end

    local_timing = {
        "timed_seconds": timing_end - timing_start,
        "timed_samples": timed_samples,
        "processed_samples": processed_samples,
        "full_loop_seconds": loop_end - loop_start,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }

    all_predictions = all_gather(detections)
    all_timings = all_gather(local_timing)

    seconds_per_frame = local_timing["timed_seconds"] / max(local_timing["timed_samples"], 1)
    print("\n Total time per frame: ", seconds_per_frame)

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    save_pred(predictions, args.work_dir)
    total_timed_samples = sum(item["timed_samples"] for item in all_timings)
    total_rank_seconds = sum(item["timed_seconds"] for item in all_timings)
    timed_wall_seconds = max(item["timed_seconds"] for item in all_timings)
    full_loop_seconds = max(item["full_loop_seconds"] for item in all_timings)
    latency_seconds = total_rank_seconds / max(total_timed_samples, 1)
    timing = {
        "scope": "CenterPoint inference using LiDAR and precomputed MVP virtual points",
        "excludes": [
            "offline camera-to-virtual-point generation",
            "nuScenes metric calculation",
        ],
        "gpus": cfg.gpus,
        "samples_per_gpu": cfg.data.samples_per_gpu if not args.speed_test else 1,
        "dataset_samples": len(dataset),
        "timed_samples": total_timed_samples,
        "timed_wall_seconds": timed_wall_seconds,
        "latency_seconds_per_frame_per_gpu": latency_seconds,
        "latency_milliseconds_per_frame_per_gpu": latency_seconds * 1000,
        "frames_per_second_per_gpu": 1.0 / latency_seconds,
        "aggregate_frames_per_second": total_timed_samples / timed_wall_seconds,
        "full_inference_loop_seconds": full_loop_seconds,
        "full_inference_loop_frames_per_second": len(dataset) / full_loop_seconds,
        "gpu_name": torch.cuda.get_device_name(args.local_rank),
        "peak_vram_allocated_mib_per_gpu": max(
            item["peak_memory_allocated_bytes"] for item in all_timings
        ) / (1024 ** 2),
        "peak_vram_reserved_mib_per_gpu": max(
            item["peak_memory_reserved_bytes"] for item in all_timings
        ) / (1024 ** 2),
    }
    save_json(timing, args.work_dir, "inference_timing.json")

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
