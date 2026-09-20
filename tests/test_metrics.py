import importlib.util
import sys
from pathlib import Path

# eval/ is a plain folder, not an installed package; load metrics.py by path
# so the test is robust regardless of how pytest is invoked.
_spec = importlib.util.spec_from_file_location(
    "evident_eval_metrics", Path(__file__).resolve().parents[1] / "eval" / "metrics.py"
)
metrics = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = metrics
_spec.loader.exec_module(metrics)


def test_recall_at_k():
    assert metrics.recall_at_k(["a", "b", "c"], ["b"], k=3) == 1.0
    assert metrics.recall_at_k(["a", "b", "c"], ["z"], k=3) == 0.0
    assert metrics.recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 0.5


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert metrics.reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_mrr():
    rankings = [(["a", "b"], ["a"]), (["a", "b"], ["b"])]
    assert metrics.mrr(rankings) == 0.75  # (1 + 0.5) / 2


def test_ece_perfect_is_zero():
    # confidence exactly matches accuracy in each bin
    conf = [0.95, 0.95, 0.05, 0.05]
    correct = [True, True, False, False]  # high-conf all right, low-conf all wrong
    assert metrics.expected_calibration_error(conf, correct, n_bins=10) < 0.11
