import nengo
import numpy as np

from neural_forager.audit.dynamic import FilteredRateMemory
from neural_forager.research.sensors import SensoryObservation


def test_smoother_is_nonspiking_and_has_expected_step_response():
    memory = FilteredRateMemory(31, 0.008)
    try:
        assert type(memory.place.neuron_type) is nengo.LIFRate
        memory.currents[:] = 1
        memory.filtered_current[:] = 0
        for i in range(20):
            value = memory._stimulus((i + 1) * 0.001)
        np.testing.assert_allclose(value, 1 - np.exp(-0.020 / 0.008))
    finally:
        memory.close()


def test_smoother_restore_clears_dynamic_state_and_preserves_weights():
    memory = FilteredRateMemory(31, 0.020)
    try:
        obs = SensoryObservation((1, 1), (False, True, True, False), 2, True)
        memory.learn(10, obs, (True, True, True))
        snapshot = memory.snapshot()
        assert memory.filtered_current.max() > 0
        memory.restore(snapshot)
        assert memory.filtered_current.max() == 0
        for a, b in zip(snapshot.weights, memory.snapshot().weights):
            np.testing.assert_array_equal(a, b)
    finally:
        memory.close()
