#!/usr/bin/env python3
"""Portable single-GPU entry points for the bundled MVP model."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CP = ROOT / 'third_party/CenterPoint'
CONFIG = CP / 'configs/mvp/nusc_centerpoint_voxelnet_0075voxel_bevfusion_virtual_ft.py'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'validate', 'test', 'infer', 'testset', 'prepare'])
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'models/mvp_centerpoint_epoch_6.pth')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--from-scratch', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--version', choices=['v1.0-trainval', 'v1.0-test'], default='v1.0-trainval')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.epochs < 1:
        p.error('--epochs must be positive')
    checkpoint = a.checkpoint.expanduser().resolve()
    work = ROOT / 'results/generated' / a.mode
    env = os.environ.copy()
    paths = [ROOT, ROOT / "third_party", CP, ROOT / 'third_party/CenterNet2', ROOT / 'third_party/CenterNet2/projects/CenterNet2', ROOT / 'third_party/detectron2']
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get('PYTHONPATH', '')
    env['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    py = sys.executable
    cwd = CP
    if a.mode in ('validate', 'test'):
        cmd = [py, str(ROOT / 'scripts/test_mvp_model.py'), '--checkpoint', str(checkpoint), '--work-dir', str(work)]
        if a.mode == 'validate':
            cmd += ['--min-map', '0', '--min-nds', '0']
    elif a.mode == 'prepare':
        cmd = [py, '-c', 'from det3d.datasets.nuscenes.nusc_common import create_nuscenes_infos; '
               f'create_nuscenes_infos("data/nuScenes", version={a.version!r}, nsweeps=10, filter_zero=True)']
    else:
        cfg = CONFIG.read_text()
        if a.mode == 'train':
            cfg += f'\nload_from = {None if a.from_scratch else str(checkpoint)!r}\nresume_from = None\ntotal_epochs = {a.epochs}\n'
        elif a.mode == 'testset':
            cfg += '\nimport copy\ndata["test"] = copy.deepcopy(data["val"])\ndata["test"].update(version="v1.0-test", info_path="data/nuScenes/infos_test_10sweeps_withvelo.pkl", ann_file="data/nuScenes/infos_test_10sweeps_withvelo.pkl")\n'
        config = work / 'config.py'
        if not a.dry_run:
            work.mkdir(parents=True, exist_ok=True)
            config.write_text(cfg)
        if a.mode == 'train':
            cmd = [py, 'tools/train.py', str(config), '--work_dir', str(work), '--validate', '--seed', str(a.seed)]
        else:
            cmd = [py, 'tools/dist_test.py', str(config), '--work_dir', str(work), '--checkpoint', str(checkpoint)]
            if a.mode == 'testset':
                cmd += ['--testset']
    print('cwd:', cwd, flush=True)
    print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return 0
    if a.mode != 'prepare' and not (a.mode == 'train' and a.from_scratch) and not checkpoint.is_file():
        p.error(f'Missing checkpoint: {checkpoint}; run scripts/download_models.sh')
    return subprocess.run(cmd, cwd=cwd, env=env).returncode

if __name__ == '__main__':
    raise SystemExit(main())
