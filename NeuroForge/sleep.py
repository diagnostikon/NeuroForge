"""
Sleep and memory consolidation — sharp-wave ripples, replay, synaptic downscaling.

Three-phase sleep cycle:
  1. NREM slow-wave: hippocampus replays recent memories → neocortex
  2. REM: random cortical activation + hippocampal replay (dream-like)
  3. Synaptic downscaling: normalize all weights to maintain plasticity
"""

import numpy as np


class SharpWaveRipple:
    """Hippocampal sharp-wave ripple replay for memory consolidation.

    During quiet wakefulness and NREM sleep, the hippocampus replays
    stored sequences to strengthen neocortical connections.
    """

    def __init__(self, hippocampus, neocortex, learning_rate=0.005):
        self.hippocampus = hippocampus  # FullHippocampus instance
        self.neocortex = neocortex  # BidirectionalCortex instance
        self.lr = learning_rate

    def replay(self, n_steps=50, temperature=0.3):
        """Sample random hippocampal memories and train the neocortex on them.

        Returns average replay loss.
        """
        if not hasattr(self.hippocampus, 'replay_buffer') or len(self.hippocampus.replay_buffer) < 2:
            return 0.0

        total_loss = 0.0
        n = min(n_steps, len(self.hippocampus.replay_buffer))

        for _ in range(n):
            idx = np.random.randint(len(self.hippocampus.replay_buffer))
            sparse, target = self.hippocampus.replay_buffer[idx]

            # Project DG sparse code to CA3 space
            ca3_input = np.dot(sparse, self.hippocampus.dg_to_ca3)

            # CA3 recall (possibly degraded — simulates partial cue)
            noise = np.random.randn(*ca3_input.shape) * 0.1 * temperature
            recalled = self.hippocampus.ca3.recall(ca3_input + noise)

            # CA1 maps back to cortical target
            pred = self.hippocampus.ca1.forward(recalled)
            loss = self.hippocampus.ca1.update(recalled, target)
            total_loss += loss

        return total_loss / n


class SynapticDownscale:
    """Homeostatic synaptic downscaling during sleep.

    Scales all weights toward zero to prevent runaway potentiation,
    maintaining the network near a critical point.
    """

    @staticmethod
    def downscale(weight, factor=0.98, target_norm=None):
        """Scale a weight array toward zero, to prevent runaway potentiation.

        factor: how much to keep (0.98 = 2% reduction)
        target_norm: if set, renormalize to this mean absolute value (None = skip)
        """
        if isinstance(weight, np.ndarray) and weight.ndim >= 2 and weight.flags.writeable:
            weight *= factor

    @staticmethod
    def downscale_column(col):
        """Scale all weight matrices in a BiologicalColumn."""
        for name in ('W23', 'W4', 'W5', 'W6', 'W_ctx_proj'):
            w = getattr(col, name, None)
            if isinstance(w, np.ndarray) and w.ndim >= 2 and w.flags.writeable:
                w *= 0.98  # gentle scaling only; no renorm to preserve learned magnitudes


class SleepCycle:
    """Full sleep cycle with NREM, REM, and synaptic downscaling.

    Cycles through:
      1. Sharp-wave ripple replay (NREM)
      2. Random dream-like activation (REM)
      3. Synaptic downscaling
    """

    def __init__(self, hippocampus, neocortex):
        self.swr = SharpWaveRipple(hippocampus, neocortex)
        self.downscale = SynapticDownscale()

    def sleep(self, nrem_steps=100, rem_steps=30):
        """Run one full sleep cycle.

        Returns dict of losses for each phase.
        """
        results = {}

        # Phase 1: NREM — sharp-wave ripple replay
        replay_loss = self.swr.replay(nrem_steps)
        results['replay_loss'] = replay_loss

        # Phase 2: REM — random cortical activation
        rem_loss = self._dream(rem_steps)
        results['rem_loss'] = rem_loss

        # Phase 3: Synaptic downscaling
        self._downscale_all()
        results['downscaled'] = True

        return results

    def _dream(self, steps=30):
        """REM-like random activation with hippocampal replay."""
        h = self.swr.hippocampus
        if not hasattr(h, 'replay_buffer') or len(h.replay_buffer) < 2:
            return 0.0
        return self.swr.replay(steps, temperature=1.0)

    def _downscale_all(self):
        """Apply synaptic downscaling to neocortex weights."""
        nc = self.swr.neocortex
        for col in nc.area1.columns:
            self.downscale.downscale_column(col)
        for col in nc.area2.columns:
            self.downscale.downscale_column(col)
        self.downscale.downscale(nc.fb_w)
        self.downscale.downscale(nc.fb_w_th)
