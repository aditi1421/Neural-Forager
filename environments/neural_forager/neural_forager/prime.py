"""Prime adapter: a local neural program, with no model endpoint calls."""
import asyncio
import json

import verifiers as vf
from pydantic import Field

from .config import AgentKind, ExperimentConfig
from .experiment import Experiment
from .research.config import ResearchConfig, ResearchSuiteConfig
from .research.runner import ResearchRunner


class ForagerTasksetConfig(vf.TasksetConfig):
    steps: int = Field(default=600, ge=30, le=20000)
    agent: AgentKind = "neural"
    research: ResearchSuiteConfig | None = None


class ForagerEnvConfig(vf.EnvConfig):
    taskset: ForagerTasksetConfig = Field(default_factory=ForagerTasksetConfig)
    harness: vf.HarnessConfig = Field(default_factory=vf.HarnessConfig)


class ForagerTaskset(vf.Taskset):
    config_type = ForagerTasksetConfig

    def __init__(self, config):
        if config.research is not None:
            source = [{"prompt": [{"role": "user", "content": f"Evaluate selective memory on maze {seed}, neural seed {neural_seed}."}],
                       "research": ResearchConfig(seed=seed, neural_seed=neural_seed,
                                                  horizon=config.research.horizon).model_dump()}
                      for seed in config.research.seeds for neural_seed in config.research.neural_seeds]
            super().__init__(source=source, config=config, rewards=[self.retention],
                             metrics=[self.recovery_rate, self.recovery_latency])
            return
        source = [{"prompt": [{"role": "user", "content": f"Forage in held-out maze {seed}."}],
                   "seed": seed, "steps": config.steps, "agent": config.agent} for seed in range(101, 151)]
        super().__init__(source=source, config=config, rewards=[self.food_rate], metrics=[self.coverage])

    @staticmethod
    def food_rate(task, state):
        return state["forager"]["summary"]["food"] / state["forager"]["summary"]["steps"]

    @staticmethod
    def coverage(task, state):
        return state["forager"]["summary"]["coverage"]

    @staticmethod
    def retention(task, state):
        rows = [r for r in state["research"] if r["method"] == "selective" and r["scenario"] != "stable"]
        return 1 - sum(r["memory_damage"] for r in rows) / len(rows)

    @staticmethod
    def recovery_rate(task, state):
        rows = [r for r in state["research"] if r["method"] == "selective" and r["scenario"] != "stable"]
        return sum(r["success"] for r in rows) / len(rows)

    @staticmethod
    def recovery_latency(task, state):
        rows = [r for r in state["research"] if r["method"] == "selective" and r["scenario"] != "stable"]
        return sum(r["capped_latency"] for r in rows) / len(rows)


class ForagerProgram:
    @staticmethod
    def run(task):
        if "research" in task:
            return ResearchRunner(ResearchConfig.model_validate(task["research"])).run()
        config = ExperimentConfig(seed=task["seed"], steps=task["steps"], agent=task["agent"])
        with Experiment(config) as experiment:
            return experiment.run()

    @staticmethod
    async def execute(task, state):
        result = await asyncio.to_thread(ForagerProgram.run, task)
        if "research" in task:
            state["research"] = result
            summary = [{key: row[key] for key in ["seed", "neural_seed", "scenario", "method", "success",
                       "capped_latency", "memory_damage", "pose_accuracy", "first_diagnosis"]} for row in result]
            state["completion"] = [{"role": "assistant", "content": json.dumps(summary)}]
            return state
        state["forager"] = result
        state["completion"] = [{"role": "assistant", "content": json.dumps(result["summary"])}]
        return state
