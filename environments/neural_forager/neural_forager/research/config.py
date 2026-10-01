from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Scenario = Literal["stable", "reward", "route", "drift", "combined"]
Method = Literal["selective", "ungated", "no_localization", "tabular"]


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = Field(default=7, ge=0)
    neural_seed: int = Field(default=31, ge=0)
    horizon: int = Field(default=120, ge=10, le=1000)
    landmark_dropout: float = Field(default=0.2, ge=0, le=1)
    history: int = Field(default=4, ge=1, le=10)
    confidence: float = Field(default=0.75, gt=0.5, lt=1)
    size: Literal[9] = 9


class ResearchSuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seeds: tuple[int, ...] = tuple(range(201, 213))
    neural_seeds: tuple[int, ...] = (31, 47)
    horizon: int = Field(default=120, ge=10, le=1000)
