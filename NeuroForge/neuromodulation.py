"""Neuromodulatory systems — global brain state control.

Dopamine (DA):      reward prediction error, motivation, working memory gating
Serotonin (5-HT):   mood, patience, social behaviour
Noradrenaline (NE): arousal, attention, novelty detection
Acetylcholine (ACh): learning rate modulation, memory encoding
"""

import numpy as np


class DopamineSystem:
    """Dopamine — reward prediction error and motivation.

    Phasic DA signals reward prediction error (RPE):
      RPE = actual_reward - expected_reward
    Tonic DA sets overall motivation/energy level.
    """

    def __init__(self, base_level=0.1, learning_rate=0.1):
        self.base = base_level
        self.tonic = base_level
        self.phasic = 0.0
        self.lr = learning_rate
        self.expected_reward = 0.0

    def update(self, reward, td_error=None):
        """Compute phasic dopamine from reward prediction error."""
        if td_error is not None:
            rpe = td_error
        else:
            rpe = reward - self.expected_reward
            self.expected_reward += self.lr * rpe
        self.phasic = np.clip(rpe, -1.0, 1.0)
        self.tonic = self.base + 0.2 * np.clip(reward, -1.0, 1.0)
        return self.phasic

    def get_working_memory_gate(self):
        """DA modulates working memory gating — high DA opens the gate."""
        return float(np.clip(0.5 + self.phasic, 0.0, 1.0))

    def get_motivation(self):
        return float(np.clip(self.tonic + self.phasic, 0.0, 1.0))


class SerotoninSystem:
    """Serotonin — mood stability and patience.

    High 5-HT → patient, content, resistant to distraction.
    Low 5-HT → impulsive, reactive, negative mood.
    """

    def __init__(self, base_level=0.5):
        self.base = base_level
        self.level = base_level
        self.mood = 0.0  # -1 (negative) to +1 (positive)

    def update(self, reward, prediction_accuracy):
        """5-HT rises with sustained reward and predictable outcomes."""
        self.level += 0.05 * (reward - self.level)
        self.level = np.clip(self.level, 0.0, 1.0)
        self.mood += 0.1 * (reward - self.mood)
        self.mood = np.clip(self.mood, -1.0, 1.0)

    def get_patience(self):
        return float(self.level)

    def get_mood(self):
        return float(self.mood)


class NoradrenalineSystem:
    """Noradrenaline — arousal, alertness, novelty detection.

    High NE → focused, alert, narrow attention.
    Low NE → relaxed, diffuse attention, drowsy.
    Novelty triggers phasic NE bursts.
    """

    def __init__(self, base_level=0.3):
        self.base = base_level
        self.tonic = base_level
        self.phasic = 0.0
        self.novelty_trace = 0.0

    def update(self, prediction_error, novelty=None):
        """NE responds to prediction error (novelty/surprise)."""
        if novelty is not None:
            self.novelty_trace = novelty
        else:
            self.novelty_trace = np.clip(np.mean(np.abs(prediction_error)), 0.0, 1.0)
        self.phasic = self.novelty_trace
        self.tonic = self.base + 0.3 * self.novelty_trace
        return self.phasic

    def get_arousal(self):
        return float(np.clip(self.tonic + self.phasic, 0.0, 1.0))

    def get_attention_breadth(self):
        """High NE → narrow focus (low breadth)."""
        return float(1.0 - self.get_arousal())


class AcetylcholineSystem:
    """Acetylcholine — learning rate modulation and memory encoding.

    High ACh → enhanced encoding, higher learning rates for new patterns.
    Low ACh → consolidation mode, replay of existing memories.
    """

    def __init__(self, base_level=0.3):
        self.base = base_level
        self.level = base_level

    def update(self, novelty, uncertainty):
        """ACh rises with uncertainty — need to learn new patterns."""
        self.level += 0.1 * (0.5 * novelty + 0.5 * uncertainty - self.level)
        self.level = np.clip(self.level, 0.0, 1.0)

    def get_learning_modulation(self):
        """Returns multiplier for learning rate (0.5-1.5)."""
        return float(0.5 + self.level)

    def get_encoding_strength(self):
        return float(self.level)


class Neuromodulator:
    """Global neuromodulatory hub — coordinates all systems."""

    def __init__(self):
        self.da = DopamineSystem()
        self.serotonin = SerotoninSystem()
        self.ne = NoradrenalineSystem()
        self.ach = AcetylcholineSystem()

    def update_all(self, reward=None, prediction_error=None,
                   novelty=None, uncertainty=None, td_error=None):
        """Update all neuromodulatory systems from sensory feedback."""
        if td_error is not None or reward is not None:
            r = reward if reward is not None else 0.0
            self.da.update(r, td_error=td_error)
            accuracy = 1.0 - np.clip(np.mean(np.abs(prediction_error)), 0.0, 1.0) if prediction_error is not None else 0.5
            self.serotonin.update(r, accuracy)

        if prediction_error is not None or novelty is not None:
            self.ne.update(prediction_error, novelty=novelty)
            nv = novelty if novelty is not None else (
                np.clip(np.mean(np.abs(prediction_error)), 0.0, 1.0) if prediction_error is not None else 0.0)
            unc = uncertainty if uncertainty is not None else (1.0 - nv)
            self.ach.update(nv, unc)

    def get_state(self):
        """Return dict of all neuromodulator levels."""
        return {
            'dopamine_phasic': float(self.da.phasic),
            'dopamine_tonic': float(self.da.tonic),
            'serotonin': float(self.serotonin.level),
            'mood': float(self.serotonin.mood),
            'noradrenaline': float(self.ne.get_arousal()),
            'acetylcholine': float(self.ach.level),
            'motivation': float(self.da.get_motivation()),
            'patience': float(self.serotonin.get_patience()),
        }
