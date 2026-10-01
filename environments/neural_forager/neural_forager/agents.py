"""Neural value memory with PES; no planner or access to the hidden maze.

Python supplies TD teaching signals and epsilon exploration. Place-action values
live only in Nengo connection weights. The tabular agent is a separate baseline.
"""
from collections import deque

import nengo
import numpy as np

from .config import ExperimentConfig
from .world import Observation


class Agent:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed + 10000)
        self.last_values = np.zeros(4)
        self.last_utilities = np.zeros(4)
        self.td_error = 0.0
        self.decisions = 0
        self.seen = set()
        self.value_display = np.zeros((config.size, config.size))
        self.activity = []
        self.weight_change = 0.0

    def index(self, observation: Observation):
        x, y = observation.position
        return y * self.config.size + x

    def values(self, observation: Observation):
        return np.zeros(4)

    def choose(self, observation: Observation):
        self.seen.add(observation.position)
        values = self.values(observation)
        self.last_values = values.copy()
        self.value_display[observation.position[1], observation.position[0]] = float(max(values))
        legal = np.flatnonzero(observation.legal)
        utilities = values + self.rng.uniform(-0.015, 0.015, 4)
        if self.rng.random() < self.config.epsilon or self.config.agent == "random":
            utilities[:] = -0.5
            utilities[int(self.rng.choice(legal))] = 1.0
        utilities[~np.array(observation.legal)] = -1.4
        self.last_utilities = utilities
        self.decisions += 1
        return self.select(utilities, legal)

    def select(self, utilities, legal):
        return int(legal[np.argmax(utilities[legal])])

    def learn(self, observation, action, reward, next_observation, terminal):
        pass

    def close(self):
        pass

    def telemetry(self):
        return {"q": self.last_values.tolist(), "utilities": self.last_utilities.tolist(),
                "td_error": self.td_error, "seen": [list(p) for p in sorted(self.seen)],
                "values": self.value_display.tolist(), "activity": self.activity,
                "neurons": 0, "weight_change": self.weight_change}


class TabularAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.q = np.zeros((config.size**2, 4))
        self.traces = np.zeros_like(self.q)

    def values(self, observation):
        return self.q[self.index(observation)].copy()

    def learn(self, observation, action, reward, next_observation, terminal):
        index = self.index(observation)
        bootstrap = 0.0 if terminal else max(self.values(next_observation)[np.array(next_observation.legal)])
        delta = reward + self.config.gamma * bootstrap - self.q[index, action]
        self.td_error = float(delta)
        self.traces *= self.config.gamma * 0.8
        self.traces[index, :] = 0
        self.traces[index, action] = 1
        self.q += 0.22 * delta * self.traces
        if terminal:
            self.traces[:] = 0
        self.weight_change = float(np.linalg.norm(self.q))


class NeuralAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.n_places = config.size**2
        self.n_neurons = self.n_places * config.neurons_per_place
        self.currents = np.zeros(self.n_neurons)
        self.drive = self.rng.uniform(1.8, 2.6, self.n_neurons)
        self.error = np.zeros(4)
        self.motor_input = np.zeros(4)
        self.q_samples = deque(maxlen=20)
        self.motor_samples = deque(maxlen=20)
        self.spike_samples = deque(maxlen=40)
        self.traces = np.zeros((self.n_places, 4))
        self.learning_enabled = config.agent != "frozen"
        self.model = nengo.Network(seed=config.seed, label="Neural Forager")
        with self.model:
            stimulus = nengo.Node(lambda t: self.currents)
            self.place = nengo.Ensemble(self.n_neurons, 1, gain=np.full(self.n_neurons, 2.0),
                                       bias=np.zeros(self.n_neurons), label="Place cells")
            nengo.Connection(stimulus, self.place.neurons, synapse=None)
            self.q_readout = nengo.Node(self._capture_q, size_in=4, size_out=0)
            self.memory = nengo.Connection(
                self.place.neurons, self.q_readout, transform=np.zeros((4, self.n_neurons)),
                synapse=0.01, learning_rule_type=nengo.PES(learning_rate=0.16, pre_synapse=0.01))
            error = nengo.Node(lambda t: self.error)
            nengo.Connection(error, self.memory.learning_rule, synapse=None)
            utilities = nengo.Node(lambda t: self.motor_input)
            self.motor = nengo.networks.EnsembleArray(64, 4, radius=1.5, label="Action populations")
            nengo.Connection(utilities, self.motor.input, synapse=None)
            motor_readout = nengo.Node(self._capture_motor, size_in=4, size_out=0)
            nengo.Connection(self.motor.output, motor_readout, synapse=0.01)
            raster = nengo.Node(self._capture_spikes, size_in=self.n_neurons, size_out=0)
            nengo.Connection(self.place.neurons, raster, synapse=None)
        self.sim = nengo.Simulator(self.model, dt=0.001, progress_bar=False)

    def _capture_q(self, t, values):
        self.q_samples.append(values.copy())

    def _capture_motor(self, t, values):
        self.motor_samples.append(values.copy())

    def _capture_spikes(self, t, values):
        active = values.reshape(self.n_places, self.config.neurons_per_place).sum(axis=1)
        self.spike_samples.append(active.copy())

    def _cue(self, index):
        self.currents[:] = 0
        start = index * self.config.neurons_per_place
        end = start + self.config.neurons_per_place
        self.currents[start:end] = self.drive[start:end]

    def values(self, observation):
        self._cue(self.index(observation))
        self.sim.run_steps(45, progress_bar=False)
        return np.clip(np.mean(self.q_samples, axis=0), -1.5, 1.5)

    def select(self, utilities, legal):
        # Contrast normalization lets small learned value differences survive
        # the finite spiking population's decoding noise.
        centered = utilities - np.mean(utilities[legal])
        scale = max(0.03, float(np.max(np.abs(centered[legal]))))
        self.motor_input[:] = np.clip(centered / scale, -1.4, 1.4)
        self.sim.run_steps(40, progress_bar=False)
        decoded = np.mean(self.motor_samples, axis=0)
        self.last_utilities = decoded.copy()
        # The motor interface enforces sensed walls; it has no global map.
        return int(legal[np.argmax(decoded[legal])])

    def learn(self, observation, action, reward, next_observation, terminal):
        index = self.index(observation)
        bootstrap = 0.0 if terminal else max(self.values(next_observation)[np.array(next_observation.legal)])
        delta = float(np.clip(reward + self.config.gamma * bootstrap - self.last_values[action], -1, 1))
        self.td_error = delta
        self.traces *= self.config.gamma * 0.8
        self.traces[index, :] = 0
        self.traces[index, action] = 1
        if self.learning_enabled:
            # Eligibility is conventional bookkeeping. Re-cue recent places to
            # apply the same TD error through the actual Nengo PES connection.
            # Pruning keeps the local simulation interactive.
            for place in np.flatnonzero(np.max(self.traces, axis=1) > 0.12):
                self._cue(int(place))
                self.sim.run_steps(20, progress_bar=False)
                self.error[:] = -delta * self.traces[place]
                self.sim.run_steps(12, progress_bar=False)
                self.error[:] = 0
        if terminal:
            self.traces[:] = 0
        self._cue(index)
        self.sim.run_steps(20, progress_bar=False)
        rates = np.mean(self.spike_samples, axis=0) / self.config.neurons_per_place
        self.activity = rates.reshape(self.config.size, self.config.size).tolist()
        self.weight_change = float(np.linalg.norm(self.sim.signals[self.sim.model.sig[self.memory]["weights"]]))

    def telemetry(self):
        result = super().telemetry()
        result["neurons"] = self.n_neurons + 4 * 64
        return result

    def close(self):
        self.sim.close()


def make_agent(config: ExperimentConfig) -> Agent:
    if config.agent in ("neural", "frozen"):
        return NeuralAgent(config)
    if config.agent == "tabular":
        return TabularAgent(config)
    return Agent(config)
