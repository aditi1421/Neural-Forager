from time import perf_counter

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..research.config import ResearchConfig
from ..research.runner import ResearchRunner
from .memory import AuditMemory


class DynamicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = Field(default=7, ge=0)
    neural_seed: int = Field(default=31, ge=0)


class DynamicSuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seeds: tuple[int, ...] = tuple(range(401, 409))
    neural_seeds: tuple[int, ...] = (31, 47)


class FilteredRateMemory(AuditMemory):
    """A non-spiking first-order current smoother before the rate nonlinearity."""

    def __init__(self, seed: int, tau: float):
        self.filtered_current = np.zeros(972)
        self.filter_alpha = -np.expm1(-0.001 / tau)
        super().__init__(seed, 12, "rate")

    def _stimulus(self, t):
        current = super()._stimulus(t)
        self.filtered_current += self.filter_alpha * (current - self.filtered_current)
        return self.filtered_current

    def restore(self, snapshot):
        self.filtered_current[:] = 0
        super().restore(snapshot)


class DynamicRunner:
    def __init__(self, config: DynamicConfig):
        self.config = config

    def run(self) -> dict:
        base = ResearchRunner(ResearchConfig(seed=self.config.seed, neural_seed=self.config.neural_seed))
        receivers = {"spiking": AuditMemory(self.config.neural_seed, 12, "spiking"),
                     "rate": AuditMemory(self.config.neural_seed, 12, "rate"),
                     "rate_filter_8ms": FilteredRateMemory(self.config.neural_seed, 0.008),
                     "rate_filter_20ms": FilteredRateMemory(self.config.neural_seed, 0.020)}
        rows = []
        try:
            for source in ("spiking", "rate"):
                snapshot, truth = base.calibrate(receivers[source])
                indices = np.flatnonzero(snapshot.known)
                categories = base.categorical(truth)[indices]
                for condition in ("clean", "current_noise"):
                    for name, memory in receivers.items():
                        memory.set_condition(condition, self.config.seed)
                        memory.restore(snapshot)
                        start = perf_counter()
                        decoded = np.array([memory.recall(int(i)) for i in indices])
                        rows.append({"seed": self.config.seed, "neural_seed": self.config.neural_seed,
                            "trained_by": source, "backend": name, "condition": condition,
                            "mse": float(np.mean((decoded - truth[indices]) ** 2)),
                            "accuracy": float(np.mean(base.categorical(decoded) == categories)),
                            "seconds": perf_counter() - start, "indices": indices.tolist(),
                            "decoded": decoded.tolist(), "resources": memory.resources()})
                # Every training source must start clean and from zero weights,
                # rather than from the other source's transferred/noisy state.
                if source == "spiking":
                    receivers["rate"].close()
                    receivers["rate"] = AuditMemory(self.config.neural_seed, 12, "rate")
            return {"config": self.config.model_dump(), "memory": rows}
        finally:
            for memory in receivers.values():
                memory.close()
