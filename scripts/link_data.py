#!/usr/bin/env python3
import argparse
from pathlib import Path
p = argparse.ArgumentParser(description='Link an existing nuScenes dataset to both model workspaces.')
p.add_argument('dataset', type=Path)
a = p.parse_args()
source = a.dataset.expanduser().resolve()
if not (source / 'samples').is_dir():
    p.error('Dataset must contain samples/')
root = Path(__file__).resolve().parents[1]
for target in (root / 'data/nuScenes', root / 'third_party/CenterPoint/data/nuScenes'):
    if target.exists() or target.is_symlink():
        if target.resolve() == source:
            continue
        p.error(f'Refusing to overwrite {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)
    print(f'{target} -> {source}')
