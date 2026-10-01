import numpy as np

from neural_forager.agents import NeuralAgent
from neural_forager.config import ExperimentConfig
from neural_forager.experiment import Experiment
from neural_forager.world import Observation


def test_pes_stores_reward_and_frozen_weights_do_not_change():
    observation = Observation((1, 1), (True, True, True, True))
    next_observation = Observation((2, 1), (True, True, True, True))
    for kind in ["neural", "frozen"]:
        brain = NeuralAgent(ExperimentConfig(agent=kind, size=7))
        try:
            before = brain.values(observation)[1]
            for _ in range(12):
                brain.last_values = brain.values(observation)
                brain.learn(observation, 1, 1.0, next_observation, True)
            after = brain.values(observation)[1]
            if kind == "neural":
                assert after > before + 0.1
                assert brain.weight_change > 0
            else:
                assert after == before == 0
                assert brain.weight_change == 0
            assert np.isfinite(after)
        finally:
            brain.close()


def test_neural_motor_selects_a_strong_legal_preference():
    brain = NeuralAgent(ExperimentConfig(size=7))
    try:
        for action in range(4):
            utilities = np.full(4, -0.5)
            utilities[action] = 1
            assert brain.select(utilities, np.arange(4)) == action
        assert brain.select(np.array([10, 0, 0, 0]), np.array([1, 2, 3])) != 0
    finally:
        brain.close()


def test_reproducible_baseline_and_scheduled_changes():
    results = []
    for _ in range(2):
        with Experiment(ExperimentConfig(agent="tabular", steps=90)) as experiment:
            results.append(experiment.run())
            assert experiment.done
            assert experiment.world.t == 90
            assert [e["step"] for e in experiment.world.events] == [30, 60]
            experiment.step()
            assert experiment.world.t == 90
    assert results[0]["history"] == results[1]["history"]
    assert results[0]["trials"] == results[1]["trials"]


def test_mixed_trials_are_not_given_misleading_efficiency():
    with Experiment(ExperimentConfig(agent="random", steps=60, trial_limit=20)) as experiment:
        experiment.step()
        experiment.intervene("food")
        while not experiment.trials:
            experiment.step()
        assert experiment.trials[0]["mixed"]
        assert experiment.trials[0]["efficiency"] is None


def test_coverage_includes_observed_terminal_food_cell():
    with Experiment(ExperimentConfig(agent="tabular", steps=60)) as experiment:
        assert experiment.world.home in experiment.agent.seen
        from neural_forager.world import DIRECTIONS
        action = int(np.flatnonzero(experiment.world.observe().legal)[0])
        dx, dy = DIRECTIONS[action]
        food = (1 + dx, 1 + dy)
        experiment.world.food = food
        experiment.agent.q[experiment.agent.index(experiment.world.observe()), action] = 1
        experiment.step()
        assert food in experiment.agent.seen
        assert experiment.world.position == experiment.world.home
