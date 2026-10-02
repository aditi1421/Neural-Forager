from time import perf_counter

import numpy as np

from ..research.config import ResearchConfig
from ..research.memory import Memory
from ..research.runner import ResearchRunner
from .config import AuditConfig
from .memory import AuditMemory


class AuditRunner:
    def __init__(self, config: AuditConfig):
        self.config = config
        self.base = ResearchRunner(ResearchConfig(seed=config.seed, neural_seed=config.neural_seed,
                                                 horizon=config.horizon))

    def run(self) -> dict:
        records, navigation, calibration = [], [], []
        tabular = Memory()
        tab_snapshot, truth = self.base.calibrate(tabular)
        known = np.flatnonzero(tab_snapshot.known)
        target_categories = self.base.categorical(truth)[known]
        tab_error = float(np.mean((self.base.categorical(tab_snapshot.cache)[known] != target_categories)))
        for budget in self.config.budgets:
            memories = {b: AuditMemory(self.config.neural_seed, budget, b) for b in ("spiking", "rate")}
            snapshots = {}
            try:
                for backend, memory in memories.items():
                    started = perf_counter()
                    snapshots[backend], _ = self.base.calibrate(memory)
                    calibration.append({"backend": backend, "budget": budget,
                        "seconds": perf_counter() - started,
                        "accuracy": float(np.mean(self.base.categorical(snapshots[backend].cache)[known] == target_categories))})
                for trained_by, snapshot in snapshots.items():
                    for condition in AuditMemory.conditions:
                        for backend, memory in memories.items():
                            memory.refresh_on_restore = False
                            memory.set_condition(condition, self.config.seed)
                            memory.restore(snapshot)
                            started = perf_counter()
                            decoded = np.array([memory.recall(int(i)) for i in known])
                            seconds = perf_counter() - started
                            categories = self.base.categorical(decoded)
                            correct = categories == target_categories
                            records.append({"seed": self.config.seed, "neural_seed": self.config.neural_seed,
                                "budget": budget, "trained_by": trained_by, "backend": backend,
                                "condition": condition, "accuracy": float(correct.mean()),
                                "geometry_accuracy": float(correct[:, :4].mean()),
                                "reward_accuracy": float(correct[:, 4].mean()),
                                "landmark_accuracy": float(correct[:, 5].mean()),
                                "reward_site_detected": bool(np.all(categories[truth[known, 4] > 0.4, 4] == 1)),
                                "mse": float(np.mean((decoded - truth[known]) ** 2)),
                                "seconds": seconds, "resources": memory.resources(),
                                "cell_indices": known.tolist(), "decoded": decoded.tolist()})
                if budget == 12:
                    for condition in ("clean", "current_noise", "lesion_50"):
                        for scenario in ("drift", "combined"):
                            for backend, memory in memories.items():
                                memory.refresh_on_restore = True
                                memory.set_condition(condition, self.config.seed)
                                result = self.base.episode("selective", scenario, memory, snapshots[backend], truth)
                                result.update(backend=backend, condition=condition, budget=budget)
                                result["resources"] = memory.resources()
                                navigation.append(result)
            finally:
                for memory in memories.values():
                    memory.close()
        return {"config": self.config.model_dump(), "memory": records, "navigation": navigation,
                "calibration": calibration, "tabular": {"accuracy": 1 - tab_error,
                    "value_bytes": tabular.cache.nbytes, "value_scalars": tabular.cache.size}}
