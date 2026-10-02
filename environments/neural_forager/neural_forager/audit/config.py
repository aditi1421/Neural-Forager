from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Backend = Literal["spiking", "rate"]
Condition = Literal["clean", "current_noise", "lesion_50", "short_read"]


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = Field(default=7, ge=0)
    neural_seed: int = Field(default=31, ge=0)
    horizon: int = Field(default=120, ge=10, le=1000)
    budgets: tuple[int, ...] = (1, 12)


class AuditSuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seeds: tuple[int, ...] = tuple(range(301, 317))
    neural_seeds: tuple[int, ...] = (31, 47)
    horizon: int = Field(default=120, ge=10, le=1000)
    budgets: tuple[int, ...] = (1, 12)
