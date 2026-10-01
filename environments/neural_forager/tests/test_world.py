from dataclasses import asdict

import numpy as np
import pytest

from neural_forager.config import ExperimentConfig
from neural_forager.world import Maze


@pytest.mark.parametrize("seed", range(20))
def test_connected_world_and_safe_interventions(seed):
    maze = Maze(ExperimentConfig(seed=seed))
    reachable = len(maze.distances(maze.home))
    assert reachable == int((~maze.walls).sum())
    path = maze.shortest_path(maze.home, maze.food)
    event = maze.change("route")
    assert len(maze.distances(maze.home)) == reachable
    if event["applied"]:
        assert event["edge"] in {maze.edge(a, b) for a, b in zip(path, path[1:])}
    old_food = maze.food
    maze.change("food")
    assert maze.food != old_food
    assert maze.food in maze.distances(maze.home)


def test_world_and_interventions_are_seeded_independently_of_agent():
    a, b = [Maze(ExperimentConfig(seed=12)) for _ in range(2)]
    b.step(int(np.flatnonzero(b.observe().legal)[0]))
    assert np.array_equal(a.walls, b.walls)
    for kind in ["route", "food"]:
        a.change(kind)
        b.change(kind)
        assert a.food == b.food
        assert a.blocked == b.blocked


def test_observations_do_not_expose_global_state():
    maze = Maze(ExperimentConfig())
    assert set(asdict(maze.observe())) == {"position", "legal"}
    assert len(maze.observe().legal) == 4


def test_collision_and_trial_restart():
    maze = Maze(ExperimentConfig())
    start = maze.position
    result = maze.step(0)
    assert result.collision and maze.position == start
    assert result.reward < 0
    assert maze.t == 1
    maze.restart_trial()
    assert maze.trial_steps == 0 and maze.t == 1
    with pytest.raises(ValueError):
        maze.step(7)


def test_food_reward_and_timeout():
    maze = Maze(ExperimentConfig(trial_limit=20))
    action = int(np.flatnonzero(maze.observe().legal)[0])
    from neural_forager.world import DIRECTIONS
    dx, dy = DIRECTIONS[action]
    maze.food = (maze.position[0] + dx, maze.position[1] + dy)
    result = maze.step(action)
    assert result.terminal and result.collected and result.reward == 1
    maze.restart_trial()
    for _ in range(20):
        result = maze.step(0)
    assert result.terminal and not result.collected


def test_invalid_configuration():
    with pytest.raises(ValueError):
        ExperimentConfig(size=8)
    with pytest.raises(ValueError):
        ExperimentConfig(steps=-1)
