"""PES memories, plus a matched conventional memory control.

The planner's cache contains decoded neural retrievals, never target copies.
Plasticity gates act on three separate Nengo connections.
"""
from collections import deque
from dataclasses import dataclass

import nengo
import numpy as np

from .sensors import SensoryObservation


@dataclass
class MemorySnapshot:
    cache: np.ndarray
    known: np.ndarray
    weights: tuple[np.ndarray, ...]


class Memory:
    def __init__(self):
        self.cache = np.zeros((81, 9))
        self.known = np.zeros(81, dtype=bool)
        self.writes = np.zeros(3, dtype=int)

    @staticmethod
    def targets(observation: SensoryObservation) -> np.ndarray:
        target = np.zeros(9)
        target[:4] = np.array(observation.legal, dtype=float) * 2 - 1
        target[4] = float(observation.reward_present)
        if observation.landmark is not None:
            target[5 + observation.landmark] = 1
        return target

    @staticmethod
    def mask(gates: tuple[bool, bool, bool]) -> np.ndarray:
        return np.repeat(gates, [4, 1, 4]).astype(float)

    def recall(self, index: int) -> np.ndarray:
        return self.cache[index].copy()

    def learn(self, index: int, observation: SensoryObservation, gates: tuple[bool, bool, bool]) -> None:
        mask = self.mask(gates)
        self.cache[index] += 0.7 * mask * (self.targets(observation) - self.cache[index])
        self.known[index] = True
        self.writes += np.array(gates, dtype=int)

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(self.cache.copy(), self.known.copy(), ())

    def restore(self, snapshot: MemorySnapshot) -> None:
        self.cache[:] = snapshot.cache
        self.known[:] = snapshot.known
        self.writes[:] = 0

    def close(self) -> None:
        pass


class NeuralMemory(Memory):
    def __init__(self, seed: int):
        super().__init__()
        self.neurons_per_place = 12
        self.n_neurons = 81 * self.neurons_per_place
        self.currents = np.zeros(self.n_neurons)
        self.drive = np.random.default_rng(seed).uniform(1.8, 2.6, self.n_neurons)
        self.target = np.zeros(9)
        self.gate = np.zeros(9)
        self.output = np.zeros(9)
        self.samples = deque(maxlen=25)
        self.network = nengo.Network(seed=seed, label="Separate spatial, reward and landmark memories")
        self.connections = []
        with self.network:
            stimulus = nengo.Node(lambda t: self.currents)
            place = nengo.Ensemble(self.n_neurons, 1, gain=np.full(self.n_neurons, 2.0), bias=np.zeros(self.n_neurons))
            nengo.Connection(stimulus, place.neurons, synapse=None)
            readout = nengo.Node(self._capture, size_in=9, size_out=0)
            error = nengo.Node(lambda t: (self.output - self.target) * self.gate)
            for start, stop in [(0, 4), (4, 5), (5, 9)]:
                connection = nengo.Connection(place.neurons, readout[start:stop],
                    transform=np.zeros((stop - start, self.n_neurons)), synapse=0.008,
                    learning_rule_type=nengo.PES(learning_rate=0.25, pre_synapse=0.008))
                nengo.Connection(error[start:stop], connection.learning_rule, synapse=None)
                self.connections.append(connection)
        self.sim = nengo.Simulator(self.network, progress_bar=False)

    def _capture(self, t, value):
        self.output[:] = value
        self.samples.append(value.copy())

    def _cue(self, index: int) -> None:
        self.currents[:] = 0
        start = index * self.neurons_per_place
        self.currents[start:start + self.neurons_per_place] = self.drive[start:start + self.neurons_per_place]

    def recall(self, index: int) -> np.ndarray:
        self._cue(index)
        self.sim.run_steps(65, progress_bar=False)
        self.cache[index] = np.mean(self.samples, axis=0)
        return self.cache[index].copy()

    def learn(self, index: int, observation: SensoryObservation, gates: tuple[bool, bool, bool]) -> None:
        if not any(gates):
            return
        self.recall(index)  # Clear previous place's filtered activity before plasticity.
        self.target[:] = self.targets(observation)
        self.gate[:] = self.mask(gates)
        self.sim.run_steps(100, progress_bar=False)
        self.gate[:] = 0
        self.recall(index)
        self.known[index] = True
        self.writes += np.array(gates, dtype=int)

    def snapshot(self) -> MemorySnapshot:
        weights = tuple(self.sim.signals[self.sim.model.sig[c]["weights"]].copy() for c in self.connections)
        return MemorySnapshot(self.cache.copy(), self.known.copy(), weights)

    def restore(self, snapshot: MemorySnapshot) -> None:
        self.sim.reset()
        self.gate[:] = 0
        self.output[:] = 0
        self.currents[:] = 0
        self.samples.clear()
        for connection, weights in zip(self.connections, snapshot.weights):
            self.sim.signals[self.sim.model.sig[connection]["weights"]][:] = weights
        super().restore(snapshot)

    def close(self) -> None:
        self.sim.close()
