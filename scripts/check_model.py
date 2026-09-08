"""Run the model invariants without requiring a separate test runner."""
from pathlib import Path
import runpy, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
checks=runpy.run_path(str(ROOT/'tests/test_model.py'))
for name,fn in checks.items():
    if name.startswith('test_'):
        fn()
        print('PASS',name)
