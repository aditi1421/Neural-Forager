import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def analysis():
    path = Path(__file__).parents[1] / "audit" / "analyze.py"
    spec = importlib.util.spec_from_file_location("audit_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_weights_mazes_equally_not_replicates(analysis):
    rows = []
    for seed, replicates, difference in [(1, 8, 0.2), (2, 2, 0.8)]:
        for n in range(replicates):
            for backend, value in [("spiking", difference), ("rate", 0)]:
                rows.append(dict(seed=seed, neural_seed=n, budget=12, condition="clean",
                                 backend=backend, accuracy=value))
    result = analysis.contrast(rows, "accuracy")
    assert result["difference"] == pytest.approx(0.5)
    assert result["maze_differences"] == pytest.approx({"1": 0.2, "2": 0.8})


def test_exact_sign_flip_and_holm_have_known_answers(analysis):
    rows = [dict(seed=s, neural_seed=31, backend=b, accuracy=a)
            for s in range(4) for b, a in [("spiking", 0.7), ("rate", 0.5)]]
    result = analysis.contrast(rows, "accuracy")
    assert result["p_spiking_greater"] == 1 / 16
    assert result["ci95"] == pytest.approx([0.2, 0.2])
    comparisons = {"a": {"p_spiking_greater": 0.01}, "b": {"p_spiking_greater": 0.03},
                   "c": {"p_spiking_greater": 0.5}}
    analysis.holm(comparisons)
    assert [v["p_holm"] for v in comparisons.values()] == pytest.approx([0.03, 0.06, 0.5])


def test_missing_or_duplicate_backend_is_rejected(analysis):
    row = dict(seed=1, neural_seed=31, backend="spiking", accuracy=1)
    with pytest.raises(AssertionError):
        analysis.contrast([row], "accuracy")
    with pytest.raises(AssertionError):
        analysis.contrast([row, row], "accuracy")
