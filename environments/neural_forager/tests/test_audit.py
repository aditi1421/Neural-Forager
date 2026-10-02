import nengo
import numpy as np
import pytest

from neural_forager.audit.memory import AuditMemory
from neural_forager.research.memory import Memory, MemorySnapshot
from neural_forager.research.sensors import SensoryObservation


@pytest.fixture
def pair():
    memories = [AuditMemory(31, 12, b) for b in ("spiking", "rate")]
    yield memories
    for m in memories:
        m.close()


def test_rate_control_has_identical_drives_connections_and_parameter_count(pair):
    a, b = pair
    assert type(a.place.neuron_type) is nengo.LIF
    assert type(b.place.neuron_type) is nengo.LIFRate
    np.testing.assert_array_equal(a.drive, b.drive)
    assert a.resources()["weight_scalars"] == b.resources()["weight_scalars"] == 8748
    assert a.resources()["cache_bytes"] == b.resources()["cache_bytes"] == Memory().cache.nbytes


def test_weight_transfer_is_exact_and_reads_do_not_change_weights(pair):
    a, b = pair
    obs = SensoryObservation((1, 1), (False, True, True, False), 2, True)
    a.learn(10, obs, (True, True, True))
    snap = a.snapshot()
    b.restore(snap)
    for x, y in zip(snap.weights, b.snapshot().weights):
        np.testing.assert_array_equal(x, y)
    b.recall(10)
    for x, y in zip(snap.weights, b.snapshot().weights):
        np.testing.assert_array_equal(x, y)


def test_current_noise_and_lesions_are_backend_independent(pair):
    a, b = pair
    for condition in ("current_noise", "lesion_50"):
        for m in pair:
            m.set_condition(condition, 7)
            m._cue(10)
        np.testing.assert_array_equal(a.alive, b.alive)
        for _ in range(3):
            np.testing.assert_array_equal(a._stimulus(0.0), b._stimulus(0.0))
    assert np.count_nonzero(~a.alive) == a.n_neurons // 2


def test_restore_refreshes_cache_from_weights_not_snapshot_targets(pair):
    a, _ = pair
    obs = SensoryObservation((1, 1), (False, True, True, False), 2, True)
    a.learn(10, obs, (True, True, True))
    snap = a.snapshot()
    zeroed = MemorySnapshot(snap.cache.copy(), snap.known.copy(), tuple(np.zeros_like(w) for w in snap.weights))
    a.refresh_on_restore = True
    a.restore(zeroed)
    assert np.max(np.abs(a.cache[10])) == 0
    assert np.max(np.abs(snap.cache[10])) > 0.5


def test_activity_counts_spikes_only_for_spiking_backend(pair):
    a, b = pair
    for m in pair:
        m.recall(10)
    assert a.resources()["spike_count"] > 0
    assert a.resources()["spike_count"] == pytest.approx(round(a.resources()["spike_count"]))
    assert b.resources()["spike_count"] is None
    assert b.resources()["integrated_rate"] > 0


def test_short_read_does_not_reuse_samples_from_previous_retrieval(pair):
    a, _ = pair
    a.samples.extend([np.ones(9) * 999 for _ in range(25)])
    a.set_condition("short_read", 7)
    np.testing.assert_array_equal(a.recall(10), np.zeros(9))
    assert len(a.samples) == 20
