import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('metrics', ROOT / 'scripts/test_mvp_model.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

class WorkflowTests(unittest.TestCase):
    def test_reference_passes_quality_gate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = metrics.check_metrics(ROOT / 'results/reference/metrics_summary.json', .60, .65)
        self.assertTrue(result['passed'])

    def test_regression_fails_quality_gate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = metrics.check_metrics(ROOT / 'results/reference/metrics_summary.json', .99, .99)
        self.assertFalse(result['passed'])

    def test_corrupt_metrics_are_rejected(self):
        data = json.loads((ROOT / 'results/reference/metrics_summary.json').read_text())
        for value in [float('nan'), float('inf'), -1, 2, None]:
            with self.subTest(value=value):
                data['mean_ap'] = value
                with self.assertRaises(ValueError):
                    metrics.validate_metrics(data)

    def test_missing_class_is_rejected(self):
        data = json.loads((ROOT / 'results/reference/metrics_summary.json').read_text())
        del data['label_aps']['car']
        with self.assertRaises(ValueError):
            metrics.validate_metrics(data)

    def test_launchers_dry_run_without_gpu(self):
        for mode in ['train', 'validate', 'test', 'infer', 'testset', 'prepare']:
            with self.subTest(mode=mode):
                proc = subprocess.run([sys.executable, str(ROOT / 'scripts/run.py'), mode, '--dry-run'], capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn(str(ROOT / 'third_party/CenterPoint'), proc.stdout)

    def test_invalid_epoch_count(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'scripts/run.py'), 'train', '--epochs', '0', '--dry-run'], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)

    def test_data_link_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            (root / 'dataset/samples').mkdir(parents=True)
            (root / 'data/nuScenes').mkdir(parents=True)
            script = root / 'scripts/link_data.py'
            script.write_text((ROOT / 'scripts/link_data.py').read_text())
            proc = subprocess.run([sys.executable, str(script), str(root / 'dataset')], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertTrue((root / 'data/nuScenes').is_dir())
            self.assertFalse((root / 'data/nuScenes').is_symlink())

if __name__ == '__main__':
    unittest.main()
