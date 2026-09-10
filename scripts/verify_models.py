#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1] / 'models'
for name, meta in json.loads((root / 'manifest.json').read_text()).items():
    path = root / name
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != meta['bytes'] or digest.hexdigest() != meta['sha256']:
        raise SystemExit(f'Checksum mismatch: {name}')
    print(f'OK: {name}')
