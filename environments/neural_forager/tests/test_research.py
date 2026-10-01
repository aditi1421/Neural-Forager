from dataclasses import fields

import numpy as np

from neural_forager.research.config import ResearchConfig
from neural_forager.research.controller import MemoryController
from neural_forager.research.memory import Memory, NeuralMemory
from neural_forager.research.runner import ResearchRunner
from neural_forager.research.sensors import SensoryObservation, SensoryWorld


def test_research_observation_has_no_true_pose_or_change_label():
    assert {f.name for f in fields(SensoryObservation)} == {"odometry", "legal", "landmark", "reward_present"}
    world = SensoryWorld(ResearchConfig(seed=7))
    world.intervene("drift")
    observation = world.observe()
    assert observation.odometry != world.maze.position
    assert observation.legal == world.maze.observe().legal


def test_survey_is_a_local_walk_and_ends_at_home():
    world = SensoryWorld(ResearchConfig(seed=11))
    observations = world.survey()
    for a, b in zip(observations, observations[1:]):
        delta = (b.odometry[0] - a.odometry[0], b.odometry[1] - a.odometry[1])
        assert abs(delta[0]) + abs(delta[1]) == 1
    assert world.maze.position == world.maze.home
    assert len({o.odometry for o in observations}) == int((~world.maze.walls).sum())
    assert any(o.reward_present for o in observations)


def test_paired_worlds_get_identical_changes_and_sensors():
    a, b = [SensoryWorld(ResearchConfig(seed=19)) for _ in range(2)]
    assert a.intervene("combined") == b.intervene("combined")
    assert a.observe() == b.observe()


def test_gates_protect_other_neural_connections_and_restore_isolated_weights():
    brain = NeuralMemory(31)
    obs = SensoryObservation((1, 1), (False, True, False, True), 2, True)
    try:
        brain.learn(10, obs, (True, True, True))
        snapshot = brain.snapshot()
        changed = SensoryObservation((1, 1), obs.legal, obs.landmark, False)
        brain.learn(10, changed, (False, True, False))
        after = brain.snapshot()
        assert np.array_equal(after.weights[0], snapshot.weights[0])
        assert not np.array_equal(after.weights[1], snapshot.weights[1])
        assert np.array_equal(after.weights[2], snapshot.weights[2])
        brain.restore(snapshot)
        restored = brain.snapshot()
        assert all(np.array_equal(a, b) for a, b in zip(snapshot.weights, restored.weights))
        assert brain.writes.sum() == 0
    finally:
        brain.close()


def test_classical_diagnosis_repairs_pose_without_truth_input():
    config = ResearchConfig(seed=7)
    memory = Memory()
    runner = ResearchRunner(config)
    snapshot, truth = runner.calibrate(memory)
    result = runner.episode("tabular", "drift", memory, snapshot, truth)
    assert result["first_diagnosis"]["label"] == "drift"
    assert result["pose_accuracy"] > 0.8
    assert result["memory_damage"] == 0


def test_reward_edit_does_not_count_as_corruption():
    config = ResearchConfig(seed=7)
    memory = Memory()
    runner = ResearchRunner(config)
    snapshot, truth = runner.calibrate(memory)
    result = runner.episode("tabular", "reward", memory, snapshot, truth)
    assert result["success"]
    assert result["first_diagnosis"]["label"] == "reward"
    assert result["memory_damage"] == 0
    assert result["writes"][1] > 0


def test_stable_control_does_not_raise_false_alarm():
    config = ResearchConfig(seed=7)
    memory = Memory()
    runner = ResearchRunner(config)
    snapshot, truth = runner.calibrate(memory)
    result = runner.episode("tabular", "stable", memory, snapshot, truth)
    assert result["success"] and result["first_diagnosis"] is None
    assert result["memory_damage"] == 0


def test_unknown_landmark_does_not_erase_landmark_memory():
    config = ResearchConfig(seed=7)
    memory = Memory()
    runner = ResearchRunner(config)
    snapshot, _ = runner.calibrate(memory)
    controller = MemoryController(config, "ungated", memory)
    world = SensoryWorld(config)
    obs = world.observe(dropout=False)
    missing = SensoryObservation(obs.odometry, obs.legal, None, obs.reward_present)
    old = memory.cache[10, 5:].copy()
    controller.observe(missing, 0)
    assert np.array_equal(memory.cache[10, 5:], old)
