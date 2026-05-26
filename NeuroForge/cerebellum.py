"""Cerebellum — forward models, timing, and fine-tuning.

Large parallel granule cell layer for input expansion.
Purkinje cells for supervised learning (error-driven).
Feed-forward prediction of sensory consequences.
Timing and rhythmic control via delay lines.
"""

import numpy as np


class GranuleCellLayer:
    """Granule cell layer — sparse expansion of input space.

    Each input dimension is expanded into multiple granule cells
    with random receptive fields, enabling the cerebellum to
    discriminate fine spatiotemporal patterns.
    """

    def __init__(self, input_dim, expansion_factor=8, num_granules=None):
        self.input_dim = input_dim
        self.expansion = expansion_factor
        if num_granules is None:
            num_granules = input_dim * expansion_factor
        self.num_granules = num_granules

        # Random projection weights (fixed — no learning)
        self.W = np.random.randn(input_dim, num_granules) * np.sqrt(2.0 / input_dim)
        self.b = -0.5 * np.ones(num_granules)  # sparse activity threshold

    def forward(self, x):
        """Expand and sparsify input through granule cells."""
        excitation = np.dot(x, self.W) + self.b
        # Threshold for sparse activity
        activation = np.maximum(0, excitation)
        # Top-k sparsification
        k = max(1, self.num_granules // 20)
        if len(activation) > k:
            threshold = np.sort(activation)[-k]
            activation[activation < threshold] = 0.0
        return activation


class PurkinjeCell:
    """Purkinje cell — supervised learning from climbing fibre error signal.

    Implements cerebellar learning rule:
      ΔW = -η * error * granule_input
    """

    def __init__(self, input_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.lr = learning_rate
        self.W = np.random.randn(input_dim) * 0.1
        self.b = 0.0

    def forward(self, granule_activity):
        """Predict output from granule cell input."""
        return np.dot(granule_activity, self.W) + self.b

    def update(self, granule_activity, target):
        """Learning from climbing fibre error signal (target - prediction)."""
        prediction = self.forward(granule_activity)
        error = target - prediction
        self.W += self.lr * error * granule_activity
        self.b += self.lr * error
        return np.mean(np.abs(error))


class DeepCerebellarNuclei:
    """Deep cerebellar nuclei — integrate Purkinje output and provide
    the cerebellum's actual output to the rest of the brain.
    """

    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros(output_dim)
        self.lr = learning_rate

    def forward(self, purkinje_input):
        return np.dot(purkinje_input, self.W) + self.b

    def update(self, purkinje_input, target):
        pred = self.forward(purkinje_input)
        error = target - pred
        grad = np.outer(purkinje_input, error)
        self.W += self.lr * grad
        self.b += self.lr * np.mean(error, axis=0) if error.ndim > 0 else error
        return np.mean(np.abs(error))


class Cerebellum:
    """Full cerebellar circuit: input expansion → Purkinje learning → output.

    Acts as a forward model: predicts next sensory state given
    current state and motor command.
    """

    def __init__(self, input_dim, output_dim=None, expansion_factor=8):
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.state_dim = output_dim  # what the cerebellum predicts (h23 dimension)

        self.granule = GranuleCellLayer(input_dim, expansion_factor)
        self.purkinje = PurkinjeCell(self.granule.num_granules)
        self.dcn = DeepCerebellarNuclei(1, self.state_dim)

        self.delay_buffer = []
        self.max_delay = 5

    def forward(self, current_state, motor_command=None):
        if motor_command is not None:
            combined = np.concatenate([current_state, motor_command])
        else:
            combined = current_state

        self.delay_buffer.append(combined.copy())
        if len(self.delay_buffer) > self.max_delay:
            self.delay_buffer.pop(0)

        delayed = self.delay_buffer[0] if self.delay_buffer else combined
        granule_out = self.granule.forward(delayed)
        purkinje_out = self.purkinje.forward(granule_out)
        prediction = self.dcn.forward(np.array([purkinje_out]))
        return prediction

    def update(self, current_state, actual_next_state, motor_command=None):
        if motor_command is not None:
            combined_size = len(current_state) + len(motor_command)
        else:
            combined_size = len(current_state)
        if motor_command is not None:
            combined = np.concatenate([current_state, motor_command])
        else:
            combined = current_state

        self.delay_buffer.append(combined.copy())
        if len(self.delay_buffer) > self.max_delay:
            self.delay_buffer.pop(0)

        delayed = self.delay_buffer[0] if self.delay_buffer else combined
        granule_out = self.granule.forward(delayed)

        purkinje_pred = self.purkinje.forward(granule_out)
        pe = self.purkinje.update(granule_out, np.mean(actual_next_state))

        dcn_pred = self.dcn.forward(np.array([purkinje_pred]))
        # actual_next_state is just the state (not combined), DCN predicts state_dim
        dcn_error = np.mean(np.abs(actual_next_state - dcn_pred))
        self.dcn.update(np.array([purkinje_pred]), actual_next_state)

        return pe + dcn_error

    def reset(self):
        self.delay_buffer = []
