"""VRAM budget verification test suite (< 5.2 GB strict / < 5.5 GB hard ceiling)."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.verify_pipeline import run_verification_pipeline


if __name__ == "__main__":
    summary = run_verification_pipeline()
    assert summary["peak_vram_gb"] < 5.2, f"Peak VRAM {summary['peak_vram_gb']} GB >= 5.2 GB"
    print("test_vram_budget.py: PASSED")
