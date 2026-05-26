"""Amygdala — emotional tagging and memory modulation."""

import numpy as np


class Amygdala:
    """Emotional valence tagging system.

    Attaches a positive/negative valence to every experience,
    modulates hippocampal consolidation strength, and provides
    fast emotional responses via a direct thalamus→amygdala pathway.
    """

    def __init__(self, input_dim=48, learning_rate=0.1, valence_decay=0.9):
        self.input_dim = input_dim
        self.lr = learning_rate
        self.valence_decay = valence_decay

        # Valence state: running estimate of current emotional context
        self.current_valence = 0.0  # -1 (negative) to +1 (positive)
        self.arousal = 0.0  # 0 (calm) to 1 (excited)

        # Fast emotional pathway: thalamus → amygdala direct weights
        self.thalamic_W = np.random.randn(input_dim, 1) * 0.1
        self.thalamic_b = 0.0

        # Emotional memory: stores valence of past experiences
        self.memory_valence = []  # list of (state_hash, valence)
        self.max_memories = 500

    def fast_pathway(self, thalamic_input):
        """Direct thalamus→amygdala emotional response (bypasses cortex)."""
        raw = np.dot(thalamic_input, self.thalamic_W) + self.thalamic_b
        valence = np.tanh(raw.item())
        self.arousal = np.abs(valence)
        return valence

    def slow_pathway(self, cortical_state, prediction_error):
        """Cortex→amygdala appraisal based on prediction outcome.

        Positive prediction error (surprising success) → positive valence.
        Negative prediction error (unexpected failure) → negative valence.
        """
        pe = np.mean(np.abs(prediction_error))
        self.current_valence = self.valence_decay * self.current_valence + \
            (1 - self.valence_decay) * np.clip(0.5 - pe, -1.0, 1.0)
        # Update thalamic weights from cortical appraisal
        error = self.current_valence - self.fast_pathway(cortical_state[:self.input_dim])
        grad = error * (1 - self.arousal ** 2) * cortical_state[:self.input_dim]
        self.thalamic_W.ravel()[:] += self.lr * grad[:self.input_dim].reshape(-1)
        self.thalamic_b += self.lr * error
        return self.current_valence

    def tag_memory(self, state, target):
        """Store valence-tagged memory for future modulation."""
        h = hash(state.tobytes()[:64])
        self.memory_valence.append((h % 10000, self.current_valence))
        if len(self.memory_valence) > self.max_memories:
            self.memory_valence.pop(0)

    def get_consolidation_modulation(self, state):
        """Return multiplier for hippocampal storage strength.

        Emotional events (high |valence| or high arousal) get stronger
        consolidation — biologically, stress hormones enhance memory.
        """
        base = 1.0
        emotional_boost = 1.0 + 0.5 * (np.abs(self.current_valence) + self.arousal)
        return base * emotional_boost

    def get_learning_rate_modulation(self):
        """Return multiplier for neocortical learning rate.

        High arousal narrows attention (less learning of peripheral details).
        """
        return 1.0 - 0.3 * self.arousal
