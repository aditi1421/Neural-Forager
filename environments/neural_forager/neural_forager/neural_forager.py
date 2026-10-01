import verifiers as vf
from .prime import ForagerEnvConfig, ForagerProgram, ForagerTaskset


def load_environment(config: ForagerEnvConfig | None = None) -> vf.Env:
    config = ForagerEnvConfig.from_config(config)
    return vf.Env(taskset=ForagerTaskset(config.taskset),
                  harness=vf.Harness(program=ForagerProgram.execute, config=config.harness))
