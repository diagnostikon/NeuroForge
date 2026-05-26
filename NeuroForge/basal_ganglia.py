"""
Basal Ganglia — action selection and dopamine-modulated reinforcement learning.

Components:
  - Striatum: evaluates candidate actions via cortical state
  - Dopamine: reward prediction error signal
  - Action selection: direct pathway (go) and indirect pathway (no-go)
"""

import numpy as np


class Dopamine:
    """Reward prediction error signal.

    dopamine = actual_reward - expected_reward
    Positive dopamine = better than expected → reinforce.
    Negative dopamine = worse than expected → suppress.
    """

    def __init__(self, base_level=0.1, learning_rate=0.1):
        self.base = base_level
        self.lr = learning_rate
        self.expected = base_level

    def update(self, reward, td_error=None):
        """Compute dopamine signal from reward and optional TD error.

        Returns dopamine (scalar) — positive reinforces, negative suppresses.
        """
        if td_error is not None:
            rpe = td_error
        else:
            rpe = reward - self.expected
        self.expected += self.lr * rpe
        return np.clip(rpe, -1.0, 1.0)


class Striatum:
    """Evaluates candidate actions based on cortical state.

    For each candidate word, computes a Q-value from the current
    cortical state.  The PFC uses these Q-values to bias planning.
    """

    def __init__(self, input_dim, num_actions, learning_rate=0.01):
        self.W = np.random.randn(input_dim, num_actions) * 0.01
        self.lr = learning_rate

    def q_values(self, cortical_state):
        """Return Q-values for all actions from current state."""
        return np.dot(cortical_state, self.W)

    def update(self, state, action_idx, dopamine):
        """Update Q-values using dopamine as reinforcement signal."""
        q = self.q_values(state)
        # One-hot action
        action_vec = np.zeros(self.W.shape[1])
        action_vec[action_idx] = 1.0
        # Update: Q(s,a) += lr * dopamine * (1 - Q(s,a))
        grad = dopamine * (action_vec - q)
        self.W += self.lr * np.outer(state, grad)
        return q


class BasalGanglia:
    """Complete BG loop: evaluate actions, compute dopamine, update values.

    Integrates with the PFC to bias word selection based on learned values.
    """

    def __init__(self, input_dim, num_actions, learning_rate=0.01):
        self.striatum = Striatum(input_dim, num_actions, learning_rate)
        self.dopamine = Dopamine()
        self.last_q = None
        self.last_state = None

    def evaluate(self, cortical_state):
        """Return Q-values for each action given the current cortical state."""
        self.last_state = cortical_state
        self.last_q = self.striatum.q_values(cortical_state)
        return self.last_q

    def reinforce(self, action_idx, reward):
        """Apply dopamine-modulated update after receiving a reward."""
        da = self.dopamine.update(reward)
        self.striatum.update(self.last_state, action_idx, da)
        return da
