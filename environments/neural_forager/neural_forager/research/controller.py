from collections import deque

import numpy as np

from ..world import DIRECTIONS
from .config import Method, ResearchConfig
from .diagnosis import Diagnostician
from .memory import Memory
from .sensors import SensoryObservation


class MemoryController:
    """Shared planner for all controls; differences are plasticity and pose repair."""
    def __init__(self, config: ResearchConfig, method: Method, memory: Memory):
        self.config = config
        self.method = method
        self.memory = memory
        self.diagnostician = Diagnostician(config, allow_localization=method != "no_localization")
        self.visits = np.zeros(81, dtype=int)
        self.rng = np.random.default_rng(config.seed + 41000)
        self.position = (1, 1)
        self.last_diagnosis = None
        self.abstentions = 0
        self.events: list[dict] = []

    def observe(self, observation: SensoryObservation, step: int) -> None:
        diagnosis = self.diagnostician.infer(observation, self.memory)
        self.last_diagnosis = diagnosis
        self.abstentions += int(diagnosis.ambiguous)
        if diagnosis.label not in ("stable", "abstain"):
            self.events.append({"step": step, "label": diagnosis.label, "confidence": diagnosis.confidence})
        self.position = tuple(a + b for a, b in zip(observation.odometry, diagnosis.offset))
        index = Diagnostician.index(self.position)
        if index is None:
            return
        recalled = self.memory.recall(index)
        self.visits[index] += 1
        selective = self.method in ("selective", "tabular")
        if selective and diagnosis.ambiguous:
            return
        if selective:
            target = self.memory.targets(observation)
            gates = (bool(np.any(np.abs(recalled[:4] - target[:4]) > 0.35)),
                     bool(abs(recalled[4] - target[4]) > 0.25),
                     observation.landmark is not None and bool(np.any(np.abs(recalled[5:] - target[5:]) > 0.35)))
        else:
            gates = (True, True, observation.landmark is not None)
        self.memory.learn(index, observation, gates)

    def choose(self, observation: SensoryObservation) -> int:
        legal = np.flatnonzero(observation.legal)
        start = Diagnostician.index(self.position)
        if start is None or not self.memory.known[start] or self.last_diagnosis.ambiguous:
            return int(self.rng.choice(legal))
        # BFS uses decoded memory, with current local sensing overriding stale
        # affordances at the start. No call to the world's path planner occurs.
        paths: dict[int, tuple[int, int]] = {start: (-1, 0)}
        queue = deque([start])
        while queue:
            index = queue.popleft()
            x, y = index % 9, index // 9
            affordance = observation.legal if index == start else self.memory.cache[index, :4] > 0
            for action, (dx, dy) in enumerate(DIRECTIONS):
                if not affordance[action]:
                    continue
                neighbor = Diagnostician.index((x + dx, y + dy))
                if neighbor is None or neighbor in paths or not self.memory.known[neighbor]:
                    continue
                first = action if index == start else paths[index][0]
                paths[neighbor] = (first, paths[index][1] + 1)
                queue.append(neighbor)
        candidates = [i for i in paths if i != start]
        if not candidates:
            return int(self.rng.choice(legal))
        goals = [i for i in candidates if self.memory.cache[i, 4] > 0.4]
        if goals:
            destination = min(goals, key=lambda i: (paths[i][1], i))
        else:
            destination = min(candidates, key=lambda i: (self.visits[i], paths[i][1], i))
        action = paths[destination][0]
        return int(action if action in legal else self.rng.choice(legal))
