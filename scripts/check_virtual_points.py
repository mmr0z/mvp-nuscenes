#!/usr/bin/env python
import argparse
import os
import pickle
from typing import Iterable

import numpy as np


REQUIRED_KEYS = ("virtual_points", "real_points", "real_points_indice")


def load_infos(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def virtual_path(lidar_path):
    tokens = lidar_path.split("/")
    return os.path.join(*tokens[:-2], tokens[-2] + "_VIRTUAL", tokens[-1] + ".pkl.npy")


def iter_lidar_paths(infos) -> Iterable[str]:
    seen = set()
    for info in infos:
        lidar_path = info["lidar_path"]
        if lidar_path not in seen:
            seen.add(lidar_path)
            yield lidar_path

        for sweep in info["sweeps"]:
            lidar_path = sweep["lidar_path"]
            if lidar_path not in seen:
                seen.add(lidar_path)
                yield lidar_path


def validate(path):
    data = np.load(path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError("not a dict")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError("missing keys: " + ", ".join(missing))


def main():
    parser = argparse.ArgumentParser(description="Validate MVP virtual point files referenced by nuScenes infos.")
    parser.add_argument("--info_path", action="append", required=True)
    parser.add_argument("--delete-bad", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    checked = 0
    missing = []
    bad = []

    for info_path in args.info_path:
        infos = load_infos(info_path)
        for lidar_path in iter_lidar_paths(infos):
            if args.limit and checked >= args.limit:
                break

            path = virtual_path(lidar_path)
            checked += 1

            if not os.path.isfile(path):
                missing.append(path)
                print(f"MISSING\t{path}")
                continue

            try:
                validate(path)
            except Exception as exc:
                bad.append((path, exc))
                print(f"BAD\t{path}\t{type(exc).__name__}: {exc}")
                if args.delete_bad:
                    os.remove(path)
                    print(f"DELETED\t{path}")

    print(f"checked={checked} missing={len(missing)} bad={len(bad)}")
    if missing or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
