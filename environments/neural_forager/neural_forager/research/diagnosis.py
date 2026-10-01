from collections import deque
from dataclasses import dataclass

import numpy as np

from ..world import DIRECTIONS, Maze
from .config import ResearchConfig
from .memory import Memory
from .sensors import SensoryObservation


@dataclass(frozen=True)
class Diagnosis:
    label: str
    offset: tuple[int, int]
    confidence: float
    ambiguous: bool


class Diagnostician:
    """Conventional finite-hypothesis inference over *learned* neural readbacks.

    Scores are engineering likelihood surrogates, not calibrated biological
    probabilities. Alternatives are stable world, one changed edge, or pose bias.
    """
    def __init__(self, config: ResearchConfig, allow_localization: bool = True):
        self.config = config
        self.allow_localization = allow_localization
        self.offset = (0, 0)
        self.history: deque[SensoryObservation] = deque(maxlen=config.history)

    @staticmethod
    def index(position: tuple[int, int]) -> int | None:
        x, y = position
        return y * 9 + x if 0 <= x < 9 and 0 <= y < 9 else None

    def cost(self, memory: Memory, offset: tuple[int, int], edge=None) -> float:
        cost = 0.0
        for observation in self.history:
            position = tuple(a + b for a, b in zip(observation.odometry, offset))
            index = self.index(position)
            if index is None or not memory.known[index]:
                cost += 12
                continue
            values = memory.cache[index]
            predicted = values[:4] > 0
            for action, (dx, dy) in enumerate(DIRECTIONS):
                neighbor = (position[0] + dx, position[1] + dy)
                prediction = bool(predicted[action])
                if edge is not None and Maze.edge(position, neighbor) == edge:
                    prediction = not prediction
                cost += 1.8 * (prediction != observation.legal[action])
            if observation.landmark is not None:
                cost += 3.5 * (int(np.argmax(values[5:])) != observation.landmark)
        return float(cost)

    def infer(self, observation: SensoryObservation, memory: Memory) -> Diagnosis:
        self.history.append(observation)
        stable = self.cost(memory, self.offset)
        # Candidate changed edges are inferred from mismatching observations,
        # never supplied by the world or evaluator.
        edges = set()
        for obs in self.history:
            p = tuple(a + b for a, b in zip(obs.odometry, self.offset))
            index = self.index(p)
            if index is None or not memory.known[index]:
                continue
            predicted = memory.cache[index, :4] > 0
            for action, (dx, dy) in enumerate(DIRECTIONS):
                if bool(predicted[action]) != obs.legal[action]:
                    edges.add(Maze.edge(p, (p[0] + dx, p[1] + dy)))
        route = min((self.cost(memory, self.offset, edge) for edge in sorted(edges)), default=stable) + 2
        alternatives = []
        if self.allow_localization:
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if abs(dx) + abs(dy) <= 4 and (dx, dy) != self.offset:
                        alternatives.append((self.cost(memory, (dx, dy)) + 3, (dx, dy)))
        drift, offset = min(alternatives, default=(10000, self.offset))
        scores = np.array([stable, route, drift])
        probability = np.exp(-(scores - scores.min()))
        probability /= probability.sum()
        winner = int(np.argmin(scores))
        confidence = float(probability[winner])
        ambiguous = confidence < self.config.confidence
        label = ("stable", "route", "drift")[winner]
        if not ambiguous and label == "drift":
            self.offset = offset
        if not ambiguous and label == "stable":
            position = tuple(a + b for a, b in zip(observation.odometry, self.offset))
            index = self.index(position)
            if index is not None and memory.known[index]:
                reward_expected = memory.cache[index, 4] > 0.4
                if reward_expected != observation.reward_present:
                    label = "reward"
        return Diagnosis("abstain" if ambiguous else label, self.offset, confidence, ambiguous)
