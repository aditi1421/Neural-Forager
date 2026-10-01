"""Privileged world state stays here; agents receive only Observation."""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ACTION_NAMES = ("north", "east", "south", "west")


@dataclass(frozen=True)
class Observation:
    position: tuple[int, int]
    legal: tuple[bool, bool, bool, bool]


@dataclass(frozen=True)
class Transition:
    observation: Observation
    reward: float
    collected: bool
    terminal: bool
    collision: bool
    trial_steps: int


class Maze:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.size = config.size
        self.walls = np.ones((self.size, self.size), dtype=bool)
        self.home = (1, 1)
        self.blocked: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self._carve()
        self.position = self.home
        distances = self.distances(self.home)
        candidates = sorted(distances, key=distances.get)[-max(3, len(distances) // 5):]
        self.food = candidates[int(self.rng.integers(len(candidates)))]
        self.t = 0
        self.trial_steps = 0
        self.collected = 0
        self.phase = "learn"
        self.events: list[dict] = []

    def _carve(self):
        self.walls[1, 1] = False
        stack = [self.home]
        while stack:
            x, y = stack[-1]
            options = [(x + dx * 2, y + dy * 2, dx, dy) for dx, dy in DIRECTIONS
                       if 0 < x + dx * 2 < self.size - 1
                       and 0 < y + dy * 2 < self.size - 1
                       and self.walls[y + dy * 2, x + dx * 2]]
            if not options:
                stack.pop()
                continue
            nx, ny, dx, dy = options[int(self.rng.integers(len(options)))]
            self.walls[y + dy, x + dx] = self.walls[ny, nx] = False
            stack.append((nx, ny))
        # Add loops so route closures need not strand the creature.
        connectors = []
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                if self.walls[y, x] and (
                    (not self.walls[y, x - 1] and not self.walls[y, x + 1])
                    or (not self.walls[y - 1, x] and not self.walls[y + 1, x])
                ):
                    connectors.append((x, y))
        self.rng.shuffle(connectors)
        for x, y in connectors[:max(3, self.size // 2)]:
            self.walls[y, x] = False

    @staticmethod
    def edge(a, b):
        return tuple(sorted((a, b)))

    def neighbors(self, position):
        x, y = position
        return [(x + dx, y + dy) for dx, dy in DIRECTIONS
                if 0 <= x + dx < self.size and 0 <= y + dy < self.size
                and not self.walls[y + dy, x + dx]
                and self.edge(position, (x + dx, y + dy)) not in self.blocked]

    def distances(self, start):
        result = {start: 0}
        queue = deque([start])
        while queue:
            point = queue.popleft()
            for neighbor in self.neighbors(point):
                if neighbor not in result:
                    result[neighbor] = result[point] + 1
                    queue.append(neighbor)
        return result

    def shortest_path(self, start, end):
        distances = self.distances(end)
        if start not in distances:
            return []
        path = [start]
        while path[-1] != end:
            path.append(min(self.neighbors(path[-1]), key=lambda p: distances.get(p, 10000)))
        return path

    def observe(self):
        neighbors = self.neighbors(self.position)
        x, y = self.position
        return Observation(self.position, tuple((x + dx, y + dy) in neighbors for dx, dy in DIRECTIONS))

    def change(self, kind: str):
        if kind == "route":
            path = self.shortest_path(self.home, self.food)
            # Block an edge of the old shortest route, preserving connectivity.
            options = list(zip(path, path[1:]))
            self.rng.shuffle(options)
            chosen = None
            for a, b in options:
                edge = self.edge(a, b)
                self.blocked.add(edge)
                distances = self.distances(self.home)
                if len(distances) == int((~self.walls).sum()):
                    chosen = edge
                    break
                self.blocked.remove(edge)
            self.phase = "reroute"
            event = {"step": self.t, "kind": "route", "applied": chosen is not None,
                     "edge": chosen, "message": "Passage closed" if chosen else "No safe route closure available"}
        elif kind == "food":
            distances = self.distances(self.home)
            options = [p for p in distances if p != self.food and p != self.home and distances[p] >= self.size // 2]
            self.food = options[int(self.rng.integers(len(options)))]
            self.phase = "relocate"
            event = {"step": self.t, "kind": "food", "applied": True, "message": "Food relocated"}
        else:
            raise ValueError("Unknown change")
        self.events.append(event)
        return event

    def step(self, action: int):
        if action not in range(4):
            raise ValueError("Action must be 0..3")
        legal = self.observe().legal[action]
        if legal:
            dx, dy = DIRECTIONS[action]
            self.position = self.position[0] + dx, self.position[1] + dy
        self.t += 1
        self.trial_steps += 1
        collected = self.position == self.food
        self.collected += int(collected)
        terminal = collected or self.trial_steps >= self.config.trial_limit
        return Transition(self.observe(), 1.0 if collected else (-0.01 if legal else -0.08),
                          collected, terminal, not legal, self.trial_steps)

    def restart_trial(self):
        self.position = self.home
        self.trial_steps = 0

    def snapshot(self):
        return {"size": self.size, "walls": self.walls.astype(int).tolist(),
                "position": self.position, "home": self.home, "food": self.food,
                "blocked": list(self.blocked), "step": self.t, "phase": self.phase,
                "collected": self.collected, "events": self.events[-12:]}
