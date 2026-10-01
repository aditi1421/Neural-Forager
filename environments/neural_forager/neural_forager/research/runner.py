from time import perf_counter

import numpy as np

from .config import Method, ResearchConfig, Scenario
from .controller import MemoryController
from .memory import Memory, MemorySnapshot, NeuralMemory
from .sensors import SensoryWorld


class ResearchRunner:
    scenarios: tuple[Scenario, ...] = ("stable", "reward", "route", "drift", "combined")
    methods: tuple[Method, ...] = ("selective", "ungated", "no_localization", "tabular")

    def __init__(self, config: ResearchConfig):
        self.config = config

    @staticmethod
    def categorical(table: np.ndarray) -> np.ndarray:
        return np.column_stack([table[:, :4] > 0, table[:, 4] > 0.4, np.argmax(table[:, 5:], axis=1)])

    def calibrate(self, memory: Memory) -> tuple[MemorySnapshot, np.ndarray]:
        world = SensoryWorld(self.config)
        observations = world.survey()
        for _ in range(2):
            for observation in observations:
                index = observation.odometry[1] * 9 + observation.odometry[0]
                memory.learn(index, observation, (True, True, True))
        for index in np.flatnonzero(memory.known):
            memory.recall(int(index))
        return memory.snapshot(), world.truth()

    def episode(self, method: Method, scenario: Scenario, memory: Memory,
                snapshot: MemorySnapshot, original_truth: np.ndarray) -> dict:
        memory.restore(snapshot)
        world = SensoryWorld(self.config)
        intervention = world.intervene(scenario)
        controller = MemoryController(self.config, method, memory)
        trace = []
        start = perf_counter()
        success = False
        for step in range(self.config.horizon + 1):
            observation = world.observe()
            controller.observe(observation, step)
            diagnosis = controller.last_diagnosis
            trace.append({"step": step, "true_position": world.maze.position,
                          "estimated_position": controller.position,
                          "odometry": observation.odometry,
                          "landmark": observation.landmark, "legal": observation.legal,
                          "diagnosis": diagnosis.label, "confidence": diagnosis.confidence,
                          "pose_correct": controller.position == world.maze.position,
                          "writes": memory.writes.tolist()})
            if observation.reward_present:
                success = True
                break
            if step < self.config.horizon:
                world.move(controller.choose(observation))
        for index in np.flatnonzero(snapshot.known):
            memory.recall(int(index))
        old_truth = self.categorical(original_truth)
        new_truth = self.categorical(world.truth())
        before = self.categorical(snapshot.cache)
        after = self.categorical(memory.cache)
        eligible = snapshot.known[:, None] & (before == old_truth) & (new_truth == old_truth)
        damaged = eligible & (after != new_truth)
        channel_damage = {}
        for channel, section in (("geometry", slice(0, 4)), ("reward", slice(4, 5)), ("landmark", slice(5, 6))):
            count = int(eligible[:, section].sum())
            channel_damage[channel] = float(damaged[:, section].sum() / count) if count else None
        first = controller.events[0] if controller.events else None
        # An intervention can be latent until its first consequence is sensed.
        # Recovery latency here is measured from injection, not detection.
        return {"method": method, "scenario": scenario, "seed": self.config.seed,
                "neural_seed": self.config.neural_seed, "config": self.config.model_dump(),
                "success": success, "latency": world.moves,
                "capped_latency": world.moves if success else self.config.horizon,
                "memory_damage": float(damaged.sum() / max(1, eligible.sum())),
                "damage_count": int(damaged.sum()), "eligible_count": int(eligible.sum()),
                "channel_damage": channel_damage,
                "calibration_accuracy": float((before[snapshot.known] == old_truth[snapshot.known]).mean()),
                "pose_accuracy": float(np.mean([row["pose_correct"] for row in trace])),
                "first_diagnosis": first, "events": controller.events,
                "abstentions": controller.abstentions,
                "writes": memory.writes.tolist(), "intervention": intervention,
                "seconds": perf_counter() - start, "trace": trace}

    def run(self, scenarios: tuple[Scenario, ...] | None = None,
            methods: tuple[Method, ...] | None = None) -> list[dict]:
        scenarios = scenarios or self.scenarios
        methods = methods or self.methods
        neural = NeuralMemory(self.config.neural_seed)
        tabular = Memory()
        try:
            neural_snapshot, truth = self.calibrate(neural)
            tabular_snapshot, _ = self.calibrate(tabular)
            results = []
            for scenario in scenarios:
                for method in methods:
                    memory, snapshot = (tabular, tabular_snapshot) if method == "tabular" else (neural, neural_snapshot)
                    results.append(self.episode(method, scenario, memory, snapshot, truth))
            return results
        finally:
            neural.close()
