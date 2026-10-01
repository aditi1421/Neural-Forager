from dataclasses import dataclass

import numpy as np

from ..config import ExperimentConfig
from ..world import DIRECTIONS, Maze
from .config import ResearchConfig, Scenario


@dataclass(frozen=True)
class SensoryObservation:
    odometry: tuple[int, int]
    legal: tuple[bool, bool, bool, bool]
    landmark: int | None
    reward_present: bool


class SensoryWorld:
    """True pose and intervention labels never appear in agent observations."""

    def __init__(self, config: ResearchConfig):
        self.config = config
        self.maze = Maze(ExperimentConfig(seed=config.seed, size=config.size))
        self.landmarks = np.random.default_rng(config.seed + 27000).integers(0, 4, (9, 9))
        self.bias = (0, 0)
        self.odometry = self.maze.home
        self.moves = 0
        self.event: dict = {"kind": "stable", "applied": True}

    def observe(self, dropout: bool = True) -> SensoryObservation:
        local = self.maze.observe()
        x, y = self.maze.position
        # Counter-based sensor randomness avoids consuming agent-dependent RNG.
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.moves, x, y, 531]))
        landmark = int(self.landmarks[y, x])
        if dropout and rng.random() < self.config.landmark_dropout:
            landmark = None
        return SensoryObservation(self.odometry, local.legal, landmark, self.maze.position == self.maze.food)

    def move(self, action: int) -> None:
        before = self.maze.position
        self.maze.step(action)
        after = self.maze.position
        self.odometry = (self.odometry[0] + after[0] - before[0],
                         self.odometry[1] + after[1] - before[1])
        self.moves += 1

    def intervene(self, scenario: Scenario) -> dict:
        if scenario in ("route", "combined"):
            self.event = self.maze.change("route")
        elif scenario == "reward":
            self.event = self.maze.change("food")
        else:
            self.event = {"kind": scenario, "applied": True}
        if scenario in ("drift", "combined"):
            rng = np.random.default_rng(self.config.seed + 36000)
            choices = [(2, 0), (0, 2), (2, 2), (4, 0), (0, 4)]
            self.bias = choices[int(rng.integers(len(choices)))]
            self.odometry = tuple(a + b for a, b in zip(self.odometry, self.bias))
        return {**self.event, "scenario": scenario, "bias": self.bias}

    def survey(self) -> list[SensoryObservation]:
        """A DFS using local sensing only; no hidden map is sent to the learner."""
        visited = {self.maze.home}
        reverse_actions: list[int] = []
        observations = [self.observe(dropout=False)]
        while True:
            obs = self.observe(dropout=False)
            x, y = obs.odometry
            unexplored = [a for a, (dx, dy) in enumerate(DIRECTIONS)
                          if obs.legal[a] and (x + dx, y + dy) not in visited]
            if unexplored:
                action = unexplored[0]
                reverse_actions.append((action + 2) % 4)
            elif reverse_actions:
                action = reverse_actions.pop()
            else:
                break
            self.move(action)
            observations.append(self.observe(dropout=False))
            visited.add(self.odometry)
        self.moves = self.maze.t = self.maze.trial_steps = self.maze.collected = 0
        return observations

    def truth(self) -> np.ndarray:
        """Evaluator only: true per-location geometry, reward, and landmark."""
        table = np.zeros((81, 9))
        for y in range(9):
            for x in range(9):
                if self.maze.walls[y, x]:
                    continue
                neighbors = self.maze.neighbors((x, y))
                table[y * 9 + x, :4] = [1 if (x + dx, y + dy) in neighbors else -1 for dx, dy in DIRECTIONS]
                table[y * 9 + x, 4] = float((x, y) == self.maze.food)
                table[y * 9 + x, 5 + int(self.landmarks[y, x])] = 1
        return table
