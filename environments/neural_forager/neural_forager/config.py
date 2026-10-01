from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AgentKind = Literal["neural", "frozen", "tabular", "random"]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(default=7, ge=0, le=1000000)
    size: int = Field(default=9, ge=7, le=15)
    steps: int = Field(default=1800, ge=30, le=20000)
    trial_limit: int = Field(default=120, ge=20, le=1000)
    agent: AgentKind = "neural"
    changes: bool = True
    neurons_per_place: int = Field(default=16, ge=4, le=64)
    epsilon: float = Field(default=0.12, ge=0, le=1)
    gamma: float = Field(default=0.97, ge=0, lt=1)

    @model_validator(mode="after")
    def odd_size(self):
        if self.size % 2 != 1:
            raise ValueError("size must be odd")
        return self
