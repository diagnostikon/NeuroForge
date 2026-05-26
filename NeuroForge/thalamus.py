"""
Thalamus — more than a relay.

Features:
  - Thalamic Reticular Nucleus (TRN): attentional gating
  - Burst vs. tonic firing modes depending on arousal/attention
  - Gating of cortical columns: which columns receive input
"""

import numpy as np


class TRN:
    """Thalamic Reticular Nucleus — attentional spotlight.

    The TRN gates which thalamocortical channels are active,
    implementing an attentional bottleneck.
    """

    def __init__(self, num_channels, base_inhibition=0.1):
        self.num_channels = num_channels
        self.inhibition = np.ones(num_channels) * base_inhibition
        self.base = base_inhibition

    def gate(self, channel_signals, attention_weights=None):
        """Apply TRN gating to channel signals.

        channel_signals: (num_channels,) or (batch, num_channels) or (num_channels, dim)
        attention_weights: (num_channels,) or None (uniform)
        Returns gated signals.
        """
        if attention_weights is None:
            attention_weights = np.ones(self.num_channels) / self.num_channels
        effective_inhibition = self.inhibition * (1.0 - attention_weights)
        gate_factor = 1.0 - effective_inhibition
        # Handle 2D channel_signals (num_channels, dim) by broadcasting
        if channel_signals.ndim >= 2:
            gate_factor = gate_factor.reshape(-1, *([1] * (channel_signals.ndim - 1)))
        return channel_signals * gate_factor

    def set_attention(self, channel_idx, strength):
        """Boost or suppress a specific channel."""
        self.inhibition[channel_idx] = self.base * (1.0 - strength)


class ThalamicRelay:
    """Thalamic relay with burst and tonic firing modes.

    Tonic mode: stable, rate-coded (normal waking state).
    Burst mode: high-frequency bursts (novelty, attention, sleep spindles).
    """

    def __init__(self, input_dim, output_dim, mode='tonic'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode  # 'tonic' or 'burst'
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.burst_threshold = 0.3
        self.burst_frequency = 0.0  # cycles per step

    def forward(self, x, arousal=0.5):
        """Forward pass with firing mode dependent on arousal.

        High arousal → tonic (linear relay).
        Low arousal → burst (thresholded, all-or-nothing).
        """
        raw = np.dot(x, self.W)
        if self.mode == 'burst' or arousal < 0.3:
            # Burst mode: thresholded, binary-like
            return np.where(raw > self.burst_threshold, raw, 0.0) * 2.0
        else:
            # Tonic mode: linear relay with slight nonlinearity
            return np.tanh(raw)


class Thalamus:
    """Full thalamic model with TRN gating and relay nuclei.

    Routes sensory/cortical input to the appropriate columns,
    with attentional gating via TRN and dynamic firing modes.
    """

    def __init__(self, input_dim, num_columns, mode='tonic'):
        self.num_columns = num_columns
        self.trn = TRN(num_columns)
        self.relays = [ThalamicRelay(input_dim, input_dim, mode)
                       for _ in range(num_columns)]
        self.arousal = 0.5  # 0 = asleep, 1 = highly alert

    def process(self, cortical_input, attention_weights=None):
        """Route input through TRN gates and relays.

        Returns list of column-specific inputs (one per column).
        """
        # Each relay processes the input independently
        relayed = np.array([r.forward(cortical_input, self.arousal)
                            for r in self.relays])
        # TRN gates the relayed signals
        gated = self.trn.gate(relayed, attention_weights)
        return gated

    def set_arousal(self, level):
        """Set arousal level (0 = sleep, 1 = alert)."""
        self.arousal = np.clip(level, 0.0, 1.0)
        for r in self.relays:
            r.mode = 'tonic' if self.arousal > 0.3 else 'burst'

    def attend_to(self, column_idx, strength=0.8):
        """Direct attentional spotlight to a specific column."""
        self.trn.set_attention(column_idx, strength)
