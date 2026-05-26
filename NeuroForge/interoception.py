"""Interoception — simulated homeostatic drives and intrinsic motivation.

Provides the brain with simulated "body" signals that generate
intrinsic goals: curiosity, fatigue, social need, etc.
These drives bias action selection and learning.
"""

import numpy as np


class HomeostaticDrive:
    """A single homeostatic drive with set-point and error signal."""

    def __init__(self, name, set_point=0.5, decay=0.99, volatility=0.02):
        self.name = name
        self.set_point = set_point
        self.level = set_point
        self.decay = decay
        self.volatility = volatility

    def update(self, satisfaction=0.0):
        """Drift toward set-point, modified by satisfaction."""
        self.level = self.decay * self.level + (1 - self.decay) * self.set_point
        self.level += satisfaction * (1 - self.decay)
        self.level += np.random.randn() * self.volatility
        self.level = np.clip(self.level, 0.0, 1.0)

    def get_drive(self):
        """Drive strength = distance from set-point (0 = satisfied)."""
        return np.abs(self.level - self.set_point)

    def get_direction(self):
        """+1 if below set-point (need more), -1 if above (need less)."""
        return 1.0 if self.level < self.set_point else -1.0


class Interoception:
    """Simulated interoceptive system with multiple homeostatic drives."""

    def __init__(self):
        self.drives = {
            'curiosity': HomeostaticDrive('curiosity', set_point=0.7, decay=0.995, volatility=0.03),
            'social_need': HomeostaticDrive('social_need', set_point=0.6, decay=0.99, volatility=0.02),
            'competence': HomeostaticDrive('competence', set_point=0.8, decay=0.99, volatility=0.01),
            'novelty_seeking': HomeostaticDrive('novelty_seeking', set_point=0.6, decay=0.99, volatility=0.04),
        }
        self.overall_mood = 0.5
        self.energy = 0.8  # fatigue-like signal

    def update(self, prediction_error=None, social_interaction=False,
               learning_progress=0.0, novelty=0.0):
        """Update all drives based on recent experience."""
        pe = np.mean(np.abs(prediction_error)) if prediction_error is not None else 0.0

        # Curiosity: satisfied by prediction error (learning happens)
        self.drives['curiosity'].update(satisfaction=-pe * 2 + 0.3)

        # Social need: satisfied by interaction
        self.drives['social_need'].update(satisfaction=0.2 if social_interaction else -0.1)

        # Competence: satisfied by learning progress
        self.drives['competence'].update(satisfaction=learning_progress)

        # Novelty seeking: satisfied by novel input
        self.drives['novelty_seeking'].update(satisfaction=novelty * 2 - 0.5)

        # Energy decreases with effort, recovers with rest
        self.energy = np.clip(self.energy * 0.998 + 0.002, 0.1, 1.0)

        # Overall mood = average drive satisfaction
        satisfaction = sum(1.0 - d.get_drive() for d in self.drives.values()) / len(self.drives)
        self.overall_mood = 0.95 * self.overall_mood + 0.05 * satisfaction

    def get_drive_vector(self):
        """Return drive signals as a vector for input to other systems."""
        return np.array([d.get_drive() for d in self.drives.values()])

    def get_dominant_drive(self):
        """Return the name and strength of the strongest drive."""
        max_drive = max(self.drives.items(), key=lambda x: x[1].get_drive())
        return max_drive[0], max_drive[1].get_drive()

    def get_intrinsic_reward(self):
        """Compute intrinsic reward from drive satisfaction.

        Used as a reward signal for the basal ganglia / dopamine system,
        even in the absence of external reward.
        """
        total = 0.0
        for drive in self.drives.values():
            total += (1.0 - drive.get_drive()) ** 2
        return total / len(self.drives)

    def get_state(self):
        """Full interoceptive state report."""
        return {
            'drives': {name: d.get_drive() for name, d in self.drives.items()},
            'mood': float(self.overall_mood),
            'energy': float(self.energy),
            'dominant_drive': self.get_dominant_drive()[0],
            'intrinsic_reward': float(self.get_intrinsic_reward()),
        }
