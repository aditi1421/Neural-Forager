from time import perf_counter

from .agents import make_agent
from .config import ExperimentConfig
from .world import ACTION_NAMES, Maze


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.world = Maze(config)
        self.agent = make_agent(config)
        self.agent.seen.add(self.world.observe().position)
        self.trials = []
        self.history = []
        self.phase_steps = {"learn": 0, "reroute": 0, "relocate": 0}
        self.phase_food = dict.fromkeys(self.phase_steps, 0)
        self.started = perf_counter()
        self.action = None
        self.trial_optimal = len(self.world.shortest_path(self.world.home, self.world.food)) - 1
        self.trial_phase = self.world.phase
        self.trial_mixed = False
        self.closed = False

    @property
    def done(self):
        return self.world.t >= self.config.steps

    def intervene(self, kind):
        event = self.world.change(kind)
        self.trial_mixed = self.world.trial_steps > 0
        if not self.trial_mixed:
            self.trial_phase = self.world.phase
            self.trial_optimal = len(self.world.shortest_path(self.world.home, self.world.food)) - 1
        return event

    def step(self):
        if self.done:
            return self.snapshot()
        if self.config.changes:
            if self.world.t == self.config.steps // 3:
                self.intervene("route")
            elif self.world.t == self.config.steps * 2 // 3:
                self.intervene("food")
        observation = self.world.observe()
        action = self.agent.choose(observation)
        transition = self.world.step(action)
        self.agent.seen.add(transition.observation.position)
        self.agent.learn(observation, action, transition.reward, transition.observation, transition.terminal)
        self.action = ACTION_NAMES[action]
        self.phase_steps[self.world.phase] += 1
        self.phase_food[self.world.phase] += int(transition.collected)
        self.history.append({"step": self.world.t, "food": self.world.collected,
                             "td_error": self.agent.td_error, "phase": self.world.phase})
        if transition.terminal:
            self.trials.append({"step": self.world.t, "length": transition.trial_steps,
                                "success": transition.collected, "phase": self.trial_phase,
                                "mixed": self.trial_mixed, "optimal": self.trial_optimal,
                                "efficiency": self.trial_optimal / transition.trial_steps
                                if transition.collected and not self.trial_mixed else None})
            self.world.restart_trial()
            self.trial_phase = self.world.phase
            self.trial_mixed = False
            self.trial_optimal = len(self.world.shortest_path(self.world.home, self.world.food)) - 1
        return self.snapshot()

    def snapshot(self):
        return {"config": self.config.model_dump(), "world": self.world.snapshot(),
                "brain": self.agent.telemetry(), "trials": self.trials[-80:],
                "history": self.history[-300:], "action": self.action,
                "done": self.done, "summary": self.summary()}

    def summary(self):
        successes = [trial for trial in self.trials if trial["success"]]
        efficiencies = [t["efficiency"] for t in successes if t["efficiency"] is not None]
        return {"agent": self.config.agent, "seed": self.config.seed, "steps": self.world.t,
                "food": self.world.collected, "trials": len(self.trials),
                "success_rate": len(successes) / max(1, len(self.trials)),
                "mean_efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else None,
                "coverage": len(self.agent.seen) / int((~self.world.walls).sum()),
                "phase_food": self.phase_food.copy(), "phase_steps": self.phase_steps.copy(),
                "elapsed_seconds": round(perf_counter() - self.started, 3),
                "weight_change": self.agent.weight_change,
                "interventions": self.world.events.copy()}

    def run(self):
        while not self.done:
            self.step()
        return {"config": self.config.model_dump(), "summary": self.summary(),
                "trials": self.trials, "history": self.history}

    def close(self):
        if not self.closed:
            self.agent.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
