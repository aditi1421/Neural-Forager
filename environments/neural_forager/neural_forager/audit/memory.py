from collections import deque

import nengo
import numpy as np

from ..research.memory import Memory, MemorySnapshot, NeuralMemory
from .config import Backend, Condition


class AuditMemory(NeuralMemory):
    """Original PES circuit with a matched LIFRate replacement and read stresses.

    The old implementation stays unchanged for reproducing the first study.
    Its targets, gates, learning schedule, snapshots, and weight restoration are
    inherited. Both audit backends use this identical constructor and stimulus.
    """

    conditions: tuple[Condition, ...] = ("clean", "current_noise", "lesion_50", "short_read")

    def __init__(self, seed: int, neurons_per_place: int, backend: Backend):
        Memory.__init__(self)
        if neurons_per_place < 1:
            raise ValueError("neurons_per_place must be positive")
        self.backend = backend
        self.seed = seed
        self.neurons_per_place = neurons_per_place
        self.n_neurons = 81 * neurons_per_place
        self.currents = np.zeros(self.n_neurons)
        self.drive = np.random.default_rng(seed).uniform(1.8, 2.6, self.n_neurons)
        self.target, self.gate, self.output = np.zeros(9), np.zeros(9), np.zeros(9)
        self.samples = deque(maxlen=25)
        self.alive = np.ones(self.n_neurons, dtype=bool)
        self.noise_std = 0.0
        self.read_steps = 65
        self.noise_rng = np.random.default_rng(0)
        self.refresh_on_restore = False
        self.conditions_seed = 0
        self.condition: Condition = "clean"
        self.activity_integral = 0.0
        self.simulated_steps = 0
        self.network = nengo.Network(seed=seed, label=f"Matched {backend} memory")
        self.connections = []
        with self.network:
            stimulus = nengo.Node(self._stimulus)
            self.place = nengo.Ensemble(self.n_neurons, 1,
                neuron_type=nengo.LIF() if backend == "spiking" else nengo.LIFRate(),
                gain=np.full(self.n_neurons, 2.0), bias=np.zeros(self.n_neurons))
            nengo.Connection(stimulus, self.place.neurons, synapse=None)
            activity = nengo.Node(self._activity, size_in=self.n_neurons, size_out=0)
            nengo.Connection(self.place.neurons, activity, synapse=None)
            readout = nengo.Node(self._capture, size_in=9, size_out=0)
            error = nengo.Node(lambda t: (self.output - self.target) * self.gate)
            for start, stop in [(0, 4), (4, 5), (5, 9)]:
                connection = nengo.Connection(self.place.neurons, readout[start:stop],
                    transform=np.zeros((stop - start, self.n_neurons)), synapse=0.008,
                    learning_rule_type=nengo.PES(learning_rate=0.25, pre_synapse=0.008))
                nengo.Connection(error[start:stop], connection.learning_rule, synapse=None)
                self.connections.append(connection)
        self.sim = nengo.Simulator(self.network, progress_bar=False)

    def _stimulus(self, t):
        current = self.currents * self.alive
        if self.noise_std:
            current = current * (1 + self.noise_std * self.noise_rng.standard_normal(self.n_neurons))
        return current

    def _activity(self, t, value):
        # Spiking outputs are impulses in Hz; this integral counts spikes only
        # for LIF. For LIFRate it is integrated rate, never an event count.
        self.activity_integral += float(np.sum(value)) * 0.001
        self.simulated_steps += 1

    def set_condition(self, condition: Condition, maze_seed: int) -> None:
        self.condition = condition
        self.conditions_seed = maze_seed
        self.noise_std = 0.5 if condition == "current_noise" else 0.0
        self.read_steps = 20 if condition == "short_read" else 65
        sequence = [maze_seed, self.seed, self.neurons_per_place, self.conditions.index(condition), 915]
        self.noise_rng = np.random.default_rng(np.random.SeedSequence(sequence))
        rng = np.random.default_rng(np.random.SeedSequence([*sequence, 1]))
        self.alive[:] = True
        if condition == "lesion_50":
            self.alive[rng.choice(self.n_neurons, self.n_neurons // 2, replace=False)] = False

    def recall(self, index: int) -> np.ndarray:
        self._cue(index)
        self.samples.clear()
        self.sim.run_steps(self.read_steps, progress_bar=False)
        self.cache[index] = np.mean(self.samples, axis=0)
        return self.cache[index].copy()

    def restore(self, snapshot: MemorySnapshot) -> None:
        super().restore(snapshot)
        self.set_condition(self.condition, self.conditions_seed)
        self.activity_integral = 0.0
        self.simulated_steps = 0
        if self.refresh_on_restore:
            # Prevent a clean Python cache from concealing damage to the neural
            # representation used by diagnosis/planning during navigation.
            for index in np.flatnonzero(snapshot.known):
                self.recall(int(index))

    def resources(self) -> dict:
        weights = [self.sim.signals[self.sim.model.sig[c]["weights"]] for c in self.connections]
        return {"neurons": self.n_neurons, "weight_scalars": sum(w.size for w in weights),
                "weight_bytes": sum(w.nbytes for w in weights), "cache_bytes": self.cache.nbytes,
                "simulation_steps": self.simulated_steps,
                "spike_count": self.activity_integral if self.backend == "spiking" else None,
                "integrated_rate": self.activity_integral if self.backend == "rate" else None}
