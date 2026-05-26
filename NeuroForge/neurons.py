import numpy as np


class CorticalColumn:
    def __init__(self, input_size, output_size, learning_rate=0.01,
                 context_decay=0.9, context_buffer=0, use_softmax=True):
        self.context_buffer = context_buffer
        self.input_size = input_size
        self.use_softmax = use_softmax
        if context_buffer > 0:
            self.context = np.zeros(input_size * context_buffer)
            total_input_size = input_size * (1 + context_buffer)
        else:
            self.context = np.zeros(input_size)
            total_input_size = input_size * 2
        self.weights = np.random.randn(total_input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)
        self.learning_rate = learning_rate
        self.context_decay = context_decay
        self.mom = 0.0
        self.probs = np.zeros(output_size)

    def forward(self, x):
        combined = np.concatenate([x, self.context])
        self.input = combined
        self.logits = np.dot(combined, self.weights) + self.bias
        if self.use_softmax:
            exp_pred = np.exp(self.logits - np.max(self.logits))
            exp_sum = exp_pred.sum()
            if exp_sum == 0:
                self.probs = np.ones_like(exp_pred) / len(exp_pred)
            else:
                self.probs = exp_pred / exp_sum
        else:
            self.probs = np.clip(self.logits, -5, 5)
        if self.context_buffer > 0:
            self.context = np.roll(self.context, self.input_size)
            self.context[:self.input_size] = x
        else:
            self.context = self.context_decay * self.context + (1 - self.context_decay) * x
        return self.probs

    def update(self, target):
        error = self.probs - target
        grad = np.outer(self.input, error)
        self.weights -= self.learning_rate * grad
        self.bias -= self.learning_rate * error
        if self.use_softmax:
            loss = -np.sum(target * np.log(np.clip(self.probs, 1e-12, 1.0)))
        else:
            loss = 0.5 * np.mean(error ** 2)
        return loss

    def reset_context(self):
        self.context = np.zeros(len(self.context))


class CorticalLayer:
    def __init__(self, input_size, output_size, num_columns=5, learning_rate=0.01,
                 context_decay=0.9, context_buffer=0, use_softmax=True):
        self.columns = [
            CorticalColumn(input_size, output_size, learning_rate, context_decay,
                           context_buffer, use_softmax)
            for _ in range(num_columns)
        ]
        self.output_size = output_size * num_columns

    def forward(self, x):
        return np.concatenate([col.forward(x) for col in self.columns])

    def update(self, target):
        loss = sum(col.update(target) for col in self.columns)
        return loss / len(self.columns)

    def reset_context(self):
        for col in self.columns:
            col.reset_context()


class Thalamus:
    """Routes different filtered views of input to each column.

    Projects input_size vector → num_columns × column_input_size via
    a learnable weight matrix. Each column gets its own slice,
    giving it a slightly different receptive field.

    Supports gradient-based learning when `update()` is called.
    """
    def __init__(self, input_size, num_columns, column_input_size,
                 learning_rate=0.01):
        self.input_size = input_size
        self.num_columns = num_columns
        self.column_input_size = column_input_size
        self.learning_rate = learning_rate
        self.W = np.random.randn(input_size, num_columns * column_input_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(num_columns * column_input_size)
        self.mom = 0.9
        self._v_W = np.zeros_like(self.W)
        self._v_b = np.zeros_like(self.b)

    def forward(self, x):
        return np.dot(x, self.W) + self.b

    def get_column_slice(self, thalamic_out, col_idx):
        start = col_idx * self.column_input_size
        return thalamic_out[start:start + self.column_input_size]

    def update(self, x, column_d_thals):
        """Gradient-based thalamus update with momentum.

        column_d_thals: list of per-column gradients (or None if skipped).
        Each gradient is the dL/d(thalamic_slice) from the column's L4.
        """
        d_out = np.zeros(self.num_columns * self.column_input_size)
        for i, grad in enumerate(column_d_thals):
            if grad is not None:
                s = i * self.column_input_size
                d_out[s:s + self.column_input_size] = grad
        dW = np.outer(x, d_out)
        db = d_out
        self._v_W = self.mom * self._v_W - self.learning_rate * np.clip(dW, -1.0, 1.0)
        self._v_b = self.mom * self._v_b - self.learning_rate * np.clip(db, -1.0, 1.0)
        self.W += self._v_W
        self.b += self._v_b


class ThalamicCorticalLayer:
    """CorticalLayer where each column receives its own thalamic projection.

    A Thalamus preprocesses raw input → per-column views, then each
    CorticalColumn processes its own slice independently.
    """
    def __init__(self, input_size, output_size, num_columns=5, learning_rate=0.01,
                 context_decay=0.9, context_buffer=0, use_softmax=True):
        self.num_columns = num_columns
        self.output_size = output_size * num_columns
        self.column_output_size = output_size
        self.thalamus = Thalamus(input_size, num_columns, output_size,
                                  learning_rate=learning_rate * 0.1)
        self.columns = [
            CorticalColumn(output_size, output_size, learning_rate,
                           context_decay, context_buffer, use_softmax)
            for _ in range(num_columns)
        ]

    def forward(self, x):
        thal_out = self.thalamus.forward(x)
        col_outputs = []
        for i, col in enumerate(self.columns):
            col_input = self.thalamus.get_column_slice(thal_out, i)
            col_outputs.append(col.forward(col_input))
        return np.concatenate(col_outputs)

    def update(self, target):
        loss = sum(col.update(target) for col in self.columns)
        return loss / len(self.columns)

    def reset_context(self):
        for col in self.columns:
            col.reset_context()


class BiologicalColumn:
    """Cortical column with L4 → L23 → L5/L6 microcircuit.

    Biologically-plausible mechanisms:
      - Predictive Coding: L4 computes error = thal_input - td_prediction,
        not concatenation. Area2 learns to cancel predictable input.
      - Feedback Alignment: replaces backprop (W.T) with fixed random
        feedback matrices B4, B23, B5. Error signals propagate locally
        without symmetric weights.
      - Dale's Law: each output neuron has a fixed sign (excitatory or
        inhibitory). Weights are projected after each update.
      - Homeostatic excitability (set via BiologicalLayer).
      - Axonal delays: each layer has configurable delay lines.
      - Active dendrites: NMDA spikes (coincidence detection),
        calcium spikes (broad depolarization).
      - STDP eligibility traces: pre/post spike timing for learning.
      - Molecular signaling: CaMKII autophosphorylation, PKA cascade, CREB.

    L6 generates untrained feedback to lower areas.
    """
    def __init__(self, column_input_size, hidden_size, output_size,
                 learning_rate=0.01, context_decay=0.9, context_buffer=0,
                 use_softmax=True, lateral_size=0, td_size=0,
                 l23_activation='tanh', predictive_coding=True,
                 axonal_delay=0, dendrite_calcium=True, use_stdp=False):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.use_softmax = use_softmax
        self.lateral_size = lateral_size
        self.td_size = td_size
        self.column_input_size = column_input_size
        self.l23_activation = l23_activation
        self.predictive_coding = predictive_coding
        self.axonal_delay = axonal_delay
        self.dendrite_calcium = dendrite_calcium
        self.use_stdp = use_stdp

        # --- Axonal delay lines ---
        # Each layer's output is delayed by `axonal_delay` timesteps
        if axonal_delay > 0:
            self._delay_L4 = np.zeros((axonal_delay, hidden_size))
            self._delay_L23 = np.zeros((axonal_delay, hidden_size))
            self._delay_L5 = np.zeros((axonal_delay, ctx_dim))
            self._delay_idx = 0
        else:
            self._delay_L4 = None
            self._delay_L23 = None
            self._delay_L5 = None

        # --- Active dendrites: NMDA / calcium dynamics ---
        # NMDA conductance: g_nmda = sigmoid(h4) * (1 + depol_factor)
        self.nmda_gate = 0.0
        self.calcium_level = 0.0           # slow calcium pool
        self.calcium_threshold = 0.3       # triggers calcium spike
        self.calcium_decay = 0.95          # per-step decay
        self.calcium_spike = 0.0

        # --- STDP eligibility traces ---
        # Pre-synaptic spike trace (exponential decay)
        self._stdp_trace_pre = np.zeros(hidden_size)
        # Post-synaptic spike trace
        self._stdp_trace_post = np.zeros(output_size)
        self.stdp_tau_pre = 20.0            # time constant for pre trace
        self.stdp_tau_post = 20.0           # time constant for post trace
        self.stdp_A_plus = 0.01             # LTP amplitude
        self.stdp_A_minus = 0.012           # LTD amplitude

        # --- Molecular signaling ---
        # CaMKII: autophosphorylation (persistent activity, memory)
        self.camkii = np.ones(hidden_size) * 0.1
        self.camkii_autophos = 0.0          # global autophosphorylation level
        self.camkii_decay = 0.999
        self.camkii_activation_rate = 0.01
        # PKA: G-protein cascade (modulated by neuromodulators)
        self.pka = 0.3
        self.pka_decay = 0.99
        self.pka_activation = 0.02
        # CREB: transcription factor (long-term memory gene expression)
        self.creb = 0.0
        self.creb_decay = 0.998
        self.creb_activation_rate = 0.005
        # Dopamine-modulated consolidation tag
        self._consolidation_tag = 0.0

        self.context_buffer = context_buffer
        self.context_decay = context_decay
        if context_buffer > 0:
            self.context = np.zeros(hidden_size * context_buffer)
            ctx_dim = hidden_size * (1 + context_buffer)
        else:
            self.context = np.zeros(hidden_size)
            ctx_dim = hidden_size * 2
        self.ctx_dim = ctx_dim

        # L4: with predictive coding, W4 input = column_input_size (error signal)
        # without PC: W4 input = column_input_size + td_size (concatenation)
        if predictive_coding:
            l4_dim = column_input_size
        else:
            l4_dim = column_input_size + td_size
        self.W4 = np.random.randn(l4_dim, hidden_size) * np.sqrt(2.0 / max(l4_dim, 1))
        self.b4 = np.zeros(hidden_size)

        # L23: [h4, lateral] → hidden
        l23_dim = hidden_size + lateral_size
        self.W23 = np.random.randn(l23_dim, hidden_size) * np.sqrt(2.0 / max(l23_dim, 1))
        self.b23 = np.zeros(hidden_size)

        self.W5 = np.random.randn(ctx_dim, output_size) * np.sqrt(2.0 / max(ctx_dim, 1))
        self.b5 = np.zeros(output_size)

        self.W6 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b6 = np.zeros(hidden_size)

        # Learned near-identity context projection
        self.W_ctx_proj = np.eye(hidden_size) + np.random.randn(hidden_size, hidden_size) * 0.01

        # Per-layer learning rates
        self.lr4 = self.learning_rate * 1.5
        self.lr23 = self.learning_rate * 1.2
        self.lr5 = self.learning_rate * 1.0

        # Momentum
        self.mom = 0.3
        self._v_W4 = np.zeros_like(self.W4)
        self._v_b4 = np.zeros_like(self.b4)
        self._v_W23 = np.zeros_like(self.W23)
        self._v_b23 = np.zeros_like(self.b23)
        self._v_W5 = np.zeros_like(self.W5)
        self._v_b5 = np.zeros_like(self.b5)
        self._v_W_ctx = np.zeros_like(self.W_ctx_proj)

        # --- Feedback Alignment: fixed random feedback matrices ---
        # B4: same shape as W4 — provides d_l4_in from dh4 without using W4.T
        self._B4 = np.random.randn(*self.W4.shape) * 0.1
        # B23: maps dh23 → dh4 (first hidden_size rows correspond to h4 input)
        self._B23 = np.random.randn(l23_dim, hidden_size) * 0.1
        # B5: maps err → dh23_raw (first hidden_size rows of W5)
        self._B5 = np.random.randn(ctx_dim, output_size) * 0.1
        # B_ctx: maps dh23_raw → dh23 through context projection
        self._B_ctx = np.random.randn(hidden_size, hidden_size) * 0.1

        # --- Dale's Law: per-neuron sign constraints ---
        # Each output neuron in a layer has a fixed sign.
        # W4[:, j] connects all L4 inputs → output neuron j, sign of j is fixed
        self._dale_W4 = np.random.choice([-1, 1], size=hidden_size)
        self._dale_W23 = np.random.choice([-1, 1], size=hidden_size)
        self._dale_W5 = np.random.choice([-1, 1], size=output_size)
        self._apply_dale()

        # cached for forward/backprop
        self._l4_in = None
        self._error_pc = None
        self._h4 = None
        self._h4_raw = None           # before NMDA modulation
        self._combined_23 = None
        self._h23 = None
        self._h23_nmda = None        # after NMDA spike modulation
        self._calcium_event = False  # calcium spike this step
        self._ctx_in = None
        self.output = None
        self.feedback = None
        self._d_thal = None
        self._d_td = None
        self._d_lat = None
        self._stdp_eligible = False   # was STDP active this update?

    def _apply_dale(self):
        """Project weights to enforce Dale's law: each neuron has one sign."""
        for W, signs in [(self.W4, self._dale_W4), (self.W23, self._dale_W23),
                         (self.W5, self._dale_W5)]:
            for j in range(W.shape[1]):
                if signs[j] > 0:
                    W[:, j] = np.maximum(0, W[:, j])
                else:
                    W[:, j] = np.minimum(0, W[:, j])

    def forward(self, thalamic_input, lateral_input=None, td_input=None):
        if self.predictive_coding:
            error = (thalamic_input - td_input) if td_input is not None else thalamic_input
            self._error_pc = error
            self._l4_in = error
            h4 = np.dot(error, self.W4) + self.b4
        else:
            parts = [thalamic_input]
            if td_input is not None:
                parts.append(td_input)
            elif self.td_size > 0:
                parts.append(np.zeros(self.td_size))
            l4_in = np.concatenate(parts)
            self._l4_in = l4_in
            self._error_pc = None
            h4 = np.dot(l4_in, self.W4) + self.b4
        np.clip(h4, -20, 20, out=h4)
        self._h4_raw = h4

        # --- Active dendrites: NMDA spike modulation ---
        # NMDA receptors require both presynaptic input AND postsynaptic depolarization.
        # g_nmda = sigmoid(h4) * (1 + depol), where depol comes from backpropagating
        # action potentials (approximated by L23 activity from previous step).
        if self.dendrite_calcium:
            depol = np.tanh(self.calcium_spike * 2.0)  # backpropagating AP influence
            nmda_g = 1.0 / (1.0 + np.exp(-h4)) * (1.0 + depol)
            self.nmda_gate = np.mean(nmda_g)
            h4 = h4 * (1.0 + 0.3 * nmda_g)  # NMDA boost
        else:
            self.nmda_gate = 0.0
        np.clip(h4, -20, 20, out=h4)

        # --- Axonal delay: L4 → L23 ---
        if self.axonal_delay > 0:
            self._delay_L4[self._delay_idx % self.axonal_delay] = h4
            idx = (self._delay_idx - self.axonal_delay + 1) % self.axonal_delay
            h4_delayed = self._delay_L4[idx]
        else:
            h4_delayed = h4
        self._h4 = h4_delayed

        # L23: integrate L4 + lateral
        parts23 = [h4_delayed]
        if lateral_input is not None:
            parts23.append(lateral_input)
        elif self.lateral_size > 0:
            parts23.append(np.zeros(self.lateral_size))
        l23_in = np.concatenate(parts23)
        np.clip(l23_in, -20, 20, out=l23_in)
        self._combined_23 = l23_in
        l23_z = np.dot(l23_in, self.W23) + self.b23
        if self.l23_activation == 'relu':
            h23 = np.maximum(0, l23_z)
        else:
            h23 = np.tanh(l23_z)
        np.clip(h23, -20, 20, out=h23)
        self._h23_nmda = h23

        # --- Calcium spike detection ---
        if self.dendrite_calcium:
            nmda_activity = np.mean(nmda_g) if self.dendrite_calcium else 0.0
            self.calcium_level = self.calcium_decay * self.calcium_level + \
                (1 - self.calcium_decay) * nmda_activity
            if self.calcium_level > self.calcium_threshold:
                self.calcium_spike = min(1.0, self.calcium_spike + 0.2)
                self._calcium_event = True
            else:
                self.calcium_spike *= 0.9
                self._calcium_event = False

        # --- Axonal delay: L23 → L5 ---
        if self.axonal_delay > 0:
            self._delay_L23[self._delay_idx % self.axonal_delay] = h23
            idx = (self._delay_idx - self.axonal_delay + 1) % self.axonal_delay
            h23 = self._delay_L23[idx]
            self._delay_idx = (self._delay_idx + 1) % max(self.axonal_delay, 1)
        self._h23 = h23

        # --- STDP eligibility traces ---
        # Pre-synaptic: every active L4 neuron leaves a trace
        pre_act = np.abs(self._h4) > 0.1
        self._stdp_trace_pre *= (1.0 - 1.0 / self.stdp_tau_pre)
        self._stdp_trace_pre[pre_act] += 0.1

        # L5: output with context
        h23_proj = np.dot(h23, self.W_ctx_proj)
        ctx_in = np.concatenate([h23_proj, self.context])
        self._ctx_in = ctx_in

        # --- Axonal delay: L5 → output ---
        if self.axonal_delay > 0:
            self._delay_L5[self._delay_idx % self.axonal_delay] = ctx_in
            ctx_in = self._delay_L5[idx]

        logits = np.dot(ctx_in, self.W5) + self.b5
        if self.use_softmax:
            exp = np.exp(logits - np.max(logits))
            s = exp.sum()
            self.output = exp / s if s > 0 else np.ones_like(exp) / len(exp)
        else:
            self.logits_lin = np.clip(logits, -10, 10)
            self.output = 1.0 / (1.0 + np.exp(-self.logits_lin))

        # --- Post-synaptic STDP trace ---
        post_act = np.abs(self.output) > 0.1
        self._stdp_trace_post *= (1.0 - 1.0 / self.stdp_tau_post)
        self._stdp_trace_post[post_act] += 0.1

        if self.context_buffer > 0:
            self.context = np.roll(self.context, self.hidden_size)
            self.context[:self.hidden_size] = h23
        else:
            self.context = self.context_decay * self.context + (1 - self.context_decay) * h23

        # L6: feedback to lower areas (untrained)
        self.feedback = np.tanh(np.dot(h23, self.W6) + self.b6)
        return self.output

    def update(self, target, dopamine=0.0):
        if self.use_softmax:
            err = self.output - target
            loss = -np.sum(target * np.log(np.clip(self.output, 1e-12, 1.0)))
        else:
            err = self.output - target
            loss = 0.5 * np.mean(err ** 2)
            sigmoid_deriv = self.output * (1 - self.output) + 1e-8
            err = err * sigmoid_deriv

        # --- Molecular signaling updates ---
        # CaMKII: activated by calcium, autophosphorylates for persistence
        ca_act = np.abs(self.calcium_level) + self.calcium_spike
        self.camkii = self.camkii * self.camkii_decay + \
            self.camkii_activation_rate * ca_act * (1 - self.camkii)
        self.camkii = np.clip(self.camkii, 0.0, 1.0)
        self.camkii_autophos = 0.99 * self.camkii_autophos + 0.01 * np.mean(self.camkii)

        # PKA: modulated by dopamine (reward signal)
        da_mod = np.clip(dopamine, 0.0, 1.0)
        self.pka = self.pka * self.pka_decay + self.pka_activation * da_mod
        self.pka = np.clip(self.pka, 0.0, 1.0)

        # CREB: activated by CaMKII (long-term memory)
        self.creb = self.creb * self.creb_decay + \
            self.creb_activation_rate * self.camkii_autophos
        self.creb = np.clip(self.creb, 0.0, 1.0)

        # Consolidation tag: memories tagged for long-term storage
        self._consolidation_tag = 0.9 * self._consolidation_tag + \
            0.1 * (self.camkii_autophos * self.creb)

        # --- STDP: spike-timing dependent plasticity ---
        if self.use_stdp:
            for i in range(len(self._stdp_trace_pre)):
                for j in range(len(self._stdp_trace_post)):
                    delta_t = self._stdp_trace_pre[i] - self._stdp_trace_post[j]
                    if delta_t > 0:
                        dw = self.stdp_A_plus * np.exp(-abs(delta_t) / self.stdp_tau_pre)
                        self.W5[i % self.hidden_size, j % self.W5.shape[1]] += dw
                    elif delta_t < 0:
                        dw = -self.stdp_A_minus * np.exp(-abs(delta_t) / self.stdp_tau_post)
                        self.W5[abs(i) % self.hidden_size, j % self.W5.shape[1]] += dw
            self._stdp_eligible = True
        else:
            self._stdp_eligible = False

        # --- Feedback Alignment: use fixed B5 instead of W5.T ---
        lr_mod = 1.0 + 0.5 * self.pka + 0.5 * self.camkii_autophos
        dh23_raw = np.dot(err, self._B5[:self.hidden_size].T)
        dW_ctx_proj = np.outer(self._h23, dh23_raw)
        dh23_raw = np.dot(dh23_raw, self._B_ctx.T)
        dW5 = np.outer(self._ctx_in, err)
        db5 = err

        # L23 activation backprop (local — uses activation function)
        if self.l23_activation == 'relu':
            dh23 = dh23_raw * (self._h23 > 0).astype(float)
        else:
            dh23 = dh23_raw * (1 - self._h23 ** 2)

        # --- Feedback Alignment: use fixed B23 instead of W23.T ---
        dW23 = np.outer(self._combined_23, dh23)
        db23 = dh23
        dh4 = np.dot(dh23, self._B23[:self.hidden_size].T)
        if self.lateral_size > 0:
            self._d_lat = np.dot(dh23, self._B23[self.hidden_size:].T)
        else:
            self._d_lat = None

        # --- Feedback Alignment: use fixed B4 instead of W4.T ---
        dW4 = np.outer(self._l4_in, dh4)
        db4 = dh4
        d_l4_in = np.dot(dh4, self._B4.T)

        # Gradient w.r.t. L4 input components (thal vs td)
        if self.predictive_coding:
            self._d_thal = d_l4_in
            self._d_td = -d_l4_in
        elif self.td_size > 0:
            self._d_thal = d_l4_in[:self.column_input_size]
            self._d_td = d_l4_in[self.column_input_size:]
        else:
            self._d_thal = d_l4_in[:self.column_input_size]
            self._d_td = None

        # Gradient clipping
        grad_norm = np.sqrt(
            np.sum(dW5 ** 2) + np.sum(db5 ** 2) +
            np.sum(dW_ctx_proj ** 2) +
            np.sum(dW23 ** 2) + np.sum(db23 ** 2) +
            np.sum(dW4 ** 2) + np.sum(db4 ** 2)
        )
        clip_scale = min(1.0, 5.0 / max(grad_norm, 1e-12))

        # Apply updates with momentum and molecular modulation
        self._v_W5 = self.mom * self._v_W5 - self.lr5 * lr_mod * dW5 * clip_scale
        self.W5 += self._v_W5
        self._v_b5 = self.mom * self._v_b5 - self.lr5 * lr_mod * db5 * clip_scale
        self.b5 += self._v_b5
        self._v_W_ctx = self.mom * self._v_W_ctx - self.lr5 * 0.1 * lr_mod * dW_ctx_proj * clip_scale
        self.W_ctx_proj += self._v_W_ctx
        self._v_W23 = self.mom * self._v_W23 - self.lr23 * lr_mod * dW23 * clip_scale
        self.W23 += self._v_W23
        self._v_b23 = self.mom * self._v_b23 - self.lr23 * lr_mod * db23 * clip_scale
        self.b23 += self._v_b23
        self._v_W4 = self.mom * self._v_W4 - self.lr4 * lr_mod * dW4 * clip_scale
        self.W4 += self._v_W4
        self._v_b4 = self.mom * self._v_b4 - self.lr4 * lr_mod * db4 * clip_scale
        self.b4 += self._v_b4

        # Weight clipping to prevent runaway growth
        for W in [self.W4, self.W23, self.W5, self.W_ctx_proj]:
            np.clip(W, -10, 10, out=W)
        for b in [self.b4, self.b23, self.b5]:
            np.clip(b, -10, 10, out=b)

        # Dale's Law: project weights to enforce sign constraints
        self._apply_dale()

        return loss

    def reset_context(self):
        self.context = np.zeros(len(self.context))


class BiologicalLayer:
    """Layer of BiologicalColumns with optional lateral inhibition.

    After forward, columns compete via k-WTA: the top-k by output
    norm remain active; others are zeroed.  This enforces sparse,
    decorrelated representations.
    """
    def __init__(self, column_input_size, hidden_size, output_size,
                 num_columns=5, learning_rate=0.01, context_decay=0.9,
                 context_buffer=0, use_softmax=True, lateral_k=0, td_size=0,
                 l23_activation='tanh', learn_lateral=True,
                 axonal_delay=0, dendrite_calcium=True, use_stdp=False):
        self.num_columns = num_columns
        self.column_input_size = column_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_dim = output_size * num_columns
        self.lateral_k = lateral_k
        self.td_size = td_size
        self.learn_lateral = learn_lateral
        self._active_mask = None

        # Homeostatic plasticity: columns that lose k-WTA become more excitable
        self.excitability = np.ones(num_columns)
        self.exc_lr = 0.02

        # Learned lateral connectivity matrix: W_lat[i,j] = weight from col j → col i
        # Used to compute lateral_input_i = sum_j W_lat[i,j] * prev_h23_j
        if learn_lateral:
            self.W_lat = np.random.randn(num_columns, num_columns) * 0.001
            self.lat_lr = learning_rate * 0.05
            self._prev_h23s = None
            lat_size = hidden_size
        else:
            self.W_lat = None
            self._prev_h23s = None
            lat_size = 0

        self.columns = [
            BiologicalColumn(column_input_size, hidden_size, output_size,
                             learning_rate, context_decay, context_buffer,
                             use_softmax, lateral_size=lat_size, td_size=td_size,
                             l23_activation=l23_activation,
                             predictive_coding=True,
                             axonal_delay=axonal_delay,
                             dendrite_calcium=dendrite_calcium,
                             use_stdp=use_stdp)
            for _ in range(num_columns)
        ]

    def forward(self, thalamic_inputs, td_inputs=None):
        """thalamic_inputs: array of shape (num_columns, column_input_size)

        td_inputs: None, 1D array (td_size,) broadcast to all columns,
                   or 2D array (num_columns, td_size) per-column td.
        """
        if td_inputs is not None and td_inputs.ndim == 1:
            td_inputs = np.tile(td_inputs, (self.num_columns, 1))

        # Compute learned lateral from previous step's h23
        if self.learn_lateral and self._prev_h23s is not None:
            # lateral_input_i = sum_j W_lat[i,j] * prev_h23_j
            # prev_h23s: (num_columns, hidden_size)
            # W_lat: (num_columns, num_columns)
            # lat_inputs: (num_columns, hidden_size) = W_lat @ prev_h23s
            lat_inputs = np.dot(self.W_lat, self._prev_h23s)
        else:
            lat_inputs = None

        outputs = []
        for i, col in enumerate(self.columns):
            td = td_inputs[i] if td_inputs is not None else None
            lat = lat_inputs[i] if lat_inputs is not None else None
            out = col.forward(thalamic_inputs[i], lateral_input=lat, td_input=td)
            outputs.append(out)

        # Store current h23 for next step's lateral
        if self.learn_lateral:
            self._prev_h23s = np.array([col._h23 for col in self.columns])

        result = np.concatenate(outputs)

        if self.lateral_k > 0:
            result, mask = self._lateral_inhibition(result)
            self._active_mask = mask
        else:
            self._active_mask = np.ones(self.num_columns, dtype=bool)
        return result

    def _lateral_inhibition(self, concat_output):
        """k-WTA with homeostatic plasticity: losers become more excitable.

        The excitability factor biases column selection so that consistently
        suppressed columns eventually get a chance to activate.
        """
        col_outs = concat_output.reshape(self.num_columns, -1)
        norms = np.linalg.norm(col_outs, axis=1)
        k = min(self.lateral_k, self.num_columns)
        if k >= self.num_columns:
            return concat_output, np.ones(self.num_columns, dtype=bool)

        # Homeostatic: adjust norms by excitability
        adjusted = norms * self.excitability
        threshold = np.partition(adjusted, -k)[-k]
        mask = adjusted >= threshold
        col_outs[~mask] = 0.0

        # Update excitability: winners become less excitable, losers more
        self.excitability[mask] -= self.exc_lr
        self.excitability[~mask] += self.exc_lr
        self.excitability = np.clip(self.excitability, 0.5, 2.0)

        return col_outs.ravel(), mask

    def update(self, target, dopamine=0.0):
        if self._active_mask is None:
            self._active_mask = np.ones(self.num_columns, dtype=bool)
        active_count = max(np.sum(self._active_mask), 1)

        # If target spans all columns, slice per-column; else broadcast
        full_target_len = self.num_columns * self.output_size
        if len(target) == full_target_len:
            col_targets = target.reshape(self.num_columns, self.output_size)
        else:
            col_targets = [target] * self.num_columns

        loss = 0.0
        d_laterals = []
        for i, col in enumerate(self.columns):
            loss += col.update(col_targets[i], dopamine=dopamine)
            if col._d_lat is not None and self._active_mask[i]:
                d_laterals.append(col._d_lat)

        # Update learned lateral weights with momentum
        if self.learn_lateral and d_laterals and self._prev_h23s is not None:
            dW_lat = np.zeros_like(self.W_lat)
            active_indices = [i for i in range(self.num_columns) if self._active_mask[i]]
            for idx, i in enumerate(active_indices):
                dW_lat[i] = np.dot(d_laterals[idx], self._prev_h23s.T)
            self.W_lat -= self.lat_lr * np.clip(dW_lat, -1.0, 1.0)

        return loss / active_count

    def reset_context(self):
        for col in self.columns:
            col.reset_context()
        if self.learn_lateral:
            self._prev_h23s = None
        self.excitability = np.ones(self.num_columns)


class CorticalArea:
    """Thalamus + BiologicalLayer = a cortical area with lateral competition.

    Routes raw input → Thalamus → per-column views → BiologicalLayer.
    """
    def __init__(self, input_size, hidden_size, output_size,
                 num_columns=5, learning_rate=0.01, context_decay=0.9,
                 context_buffer=0, use_softmax=True, lateral_k=0, td_size=0,
                 l23_activation='tanh', learn_lateral=False,
                 axonal_delay=0, dendrite_calcium=True, use_stdp=False):
        self.num_columns = num_columns
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.output_dim = output_size * num_columns
        self.thalamus = Thalamus(input_size, num_columns, output_size,
                                  learning_rate=learning_rate * 0.1)
        self.layer = BiologicalLayer(output_size, hidden_size, output_size,
                                       num_columns, learning_rate, context_decay,
                                       context_buffer, use_softmax, lateral_k,
                                       td_size=td_size,
                                       l23_activation=l23_activation,
                                       learn_lateral=learn_lateral,
                                       axonal_delay=axonal_delay,
                                       dendrite_calcium=dendrite_calcium,
                                       use_stdp=use_stdp)

    def forward(self, x, td_inputs=None, thal_mod=None):
        """Forward pass through thalamus → biological layer.

        thal_mod: optional additive modulation of thalamic output
        (L6→Thalamus return path — higher area modulates lower area's
        thalamic gating).
        """
        thal_out_raw = self.thalamus.forward(x)
        self._thal_out_raw = thal_out_raw  # cache for L6→Thalamus gradient
        self._thal_mod_used = thal_mod   # store modulation that was applied
        if thal_mod is not None:
            # L6→Thalamus gain control: multiplicative modulation bounded to [0.9, 1.1]
            # Biologically: thalamic reticular nucleus controls gain via GABergic projection
            self._thal_gain = 1.0 + 0.05 * np.tanh(thal_mod)
            thal_out = thal_out_raw * self._thal_gain
        else:
            self._thal_gain = None
            thal_out = thal_out_raw
        col_inputs = thal_out.reshape(self.num_columns, -1)
        return self.layer.forward(col_inputs, td_inputs)

    def update(self, target, dopamine=0.0):
        return self.layer.update(target, dopamine=dopamine)

    def reset_context(self):
        self.layer.reset_context()

    @property
    def columns(self):
        return self.layer.columns


class StackedCortex:
    """Two-layer cortex with hierarchical targets.
    Layer1 predicts next token (softmax, CE). Layer2 predicts next
    Layer1 state (linear, MSE). Optional predictive coding feedback.
    """
    def __init__(self, input_size, hidden_size, output_size,
                 num_columns=5, learning_rate=0.01, context_decay=0.9,
                 context_buffer=0):
        self.learning_rate = learning_rate
        self.num_columns = num_columns
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.h1_dim = hidden_size * num_columns

        self.layer1 = CorticalLayer(input_size, hidden_size, num_columns,
                                     learning_rate, context_decay, context_buffer,
                                     use_softmax=True)
        self.layer2 = CorticalLayer(self.h1_dim, self.h1_dim, num_columns,
                                     learning_rate * 0.5, context_decay, context_buffer,
                                     use_softmax=False)
        self.fb_w = np.random.randn(self.h1_dim, self.h1_dim) * 0.001
        self.fb_b = np.zeros(self.h1_dim)
        self._prev_l2_rep = None

    def forward(self, x):
        self.out1 = self.layer1.forward(x)
        self.out2 = self.layer2.forward(self.out1)
        return self.out2

    def update(self, x_next):
        loss1 = self.layer1.update(x_next)
        h1_curr = self.layer1.forward(x_next)
        loss2 = self.layer2.update(h1_curr)
        h2_concat = self.layer2.forward(h1_curr)
        if self._prev_l2_rep is not None:
            td_pred = self._prev_l2_rep @ self.fb_w + self.fb_b
            self.fb_error = np.clip(h1_curr - td_pred, -1.0, 1.0)
            grad = np.outer(self._prev_l2_rep, self.fb_error) / self.h1_dim
            self.fb_w -= self.learning_rate * 0.01 * np.clip(grad, -0.1, 0.1)
            self.fb_b -= self.learning_rate * 0.01 * np.clip(self.fb_error, -0.1, 0.1)
        else:
            self.fb_error = np.zeros_like(h1_curr)
        self._prev_l2_rep = h2_concat.reshape(self.num_columns, -1).mean(axis=0)
        self.out1 = h1_curr
        self.out2 = h2_concat
        return (loss1 + loss2) / 2

    def forward_with_feedback(self, x):
        self.out1 = self.layer1.forward(x)
        if self._prev_l2_rep is not None:
            td_pred = self._prev_l2_rep @ self.fb_w + self.fb_b
            self.fb_error = np.clip(self.out1 - td_pred, -1.0, 1.0)
            l2_input = self.out1 - 0.1 * self.fb_error
        else:
            self.fb_error = np.zeros_like(self.out1)
            l2_input = self.out1
        self.out2 = self.layer2.forward(l2_input)
        self._prev_l2_rep = self.out2.reshape(self.num_columns, -1).mean(axis=0)
        return self.out2

    def reset_context(self):
        self.layer1.reset_context()
        self.layer2.reset_context()
        self._prev_l2_rep = None


if __name__ == "__main__":
    print("=" * 55)
    print("THALAMUS: column-specific projections")
    print("=" * 55)
    np.random.seed(42)
    thal = Thalamus(input_size=2, num_columns=4, column_input_size=2)
    one_hot_a = np.array([1, 0])
    one_hot_b = np.array([0, 1])
    out_a = thal.forward(one_hot_a)
    out_b = thal.forward(one_hot_b)
    print(f"  Thalamic output for A: {out_a.round(3)}")
    print(f"  Thalamic output for B: {out_b.round(3)}")
    for i in range(4):
        col_slice_a = thal.get_column_slice(out_a, i)
        col_slice_b = thal.get_column_slice(out_b, i)
        diff = np.linalg.norm(col_slice_a - col_slice_b)
        print(f"  Column {i}: A={col_slice_a.round(3)}, B={col_slice_b.round(3)}, ||Δ||={diff:.3f}")

    print()
    print("--- ThalamicCorticalLayer on A,A,B sequence ---")
    seq = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])] * 200
    tcl = ThalamicCorticalLayer(input_size=3, output_size=3, num_columns=4,
                                 learning_rate=0.3, context_buffer=0)
    for i in range(len(seq) - 1):
        tcl.forward(seq[i])
        loss = tcl.update(seq[i + 1])
        if i % 150 == 0:
            print(f"  Step {i}, avg_loss={loss:.4f}")
    tcl.reset_context()
    for inp, expected in [(seq[0], seq[1]), (seq[1], seq[2]), (seq[2], seq[0])]:
        out = tcl.forward(inp)
        col_outs = out.reshape(4, 3)
        avg = col_outs.mean(axis=0)
        pred = np.argmax(avg)
        exp = np.argmax(expected)
        match = "✓" if pred == exp else "✗"
        print(f"  {inp} -> pred {pred}, expected {exp} {match}")

    print()
    print("=" * 55)
    print("BIOLOGICAL COLUMN: L4→L23→L5 microcircuit")
    print("=" * 55)
    np.random.seed(42)
    bcol = BiologicalColumn(column_input_size=3, hidden_size=8, output_size=3,
                            learning_rate=0.3, use_softmax=True)
    seq3 = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    bseq = seq3 * 200
    for i in range(len(bseq) - 1):
        bcol.forward(bseq[i])
        loss = bcol.update(bseq[i + 1])
        if i % 150 == 0:
            print(f"  Step {i}, loss={loss:.4f}")
    bcol.reset_context()
    for inp, expected in [(seq3[0], seq3[1]), (seq3[1], seq3[2]), (seq3[2], seq3[0])]:
        out = bcol.forward(inp)
        pred = np.argmax(out)
        exp = np.argmax(expected)
        match = "✓" if pred == exp else "✗"
        print(f"  {inp} -> pred {pred}, expected {exp} {match}")

    print()
    print("--- BiologicalLayer (3 cols) on A,A,B sequence ---")
    np.random.seed(42)
    bl = BiologicalLayer(column_input_size=3, hidden_size=8, output_size=3,
                          num_columns=3, learning_rate=0.3)
    for i in range(len(bseq) - 1):
        col_in = np.array([bseq[i]] * 3)
        bl.forward(col_in)
        loss = bl.update(bseq[i + 1])
        if i % 150 == 0:
            print(f"  Step {i}, avg_loss={loss:.4f}")
    bl.reset_context()
    for inp, expected in [(seq3[0], seq3[1]), (seq3[1], seq3[2]), (seq3[2], seq3[0])]:
        col_in = np.array([inp] * 3)
        out = bl.forward(col_in)
        col_outs = out.reshape(3, 3)
        avg = col_outs.mean(axis=0)
        pred = np.argmax(avg)
        exp = np.argmax(expected)
        match = "✓" if pred == exp else "✗"
        print(f"  {inp} -> pred {pred}, expected {exp} {match}")

    print()
    print("--- CorticalArea (Thalamus + BiologicalLayer) on A,A,B ---")
    np.random.seed(42)
    ca = CorticalArea(input_size=3, hidden_size=8, output_size=3,
                       num_columns=3, learning_rate=0.3)
    for i in range(len(bseq) - 1):
        ca.forward(bseq[i])
        loss = ca.update(bseq[i + 1])
        if i % 150 == 0:
            print(f"  Step {i}, avg_loss={loss:.4f}")
    ca.reset_context()
    for inp, expected in [(seq3[0], seq3[1]), (seq3[1], seq3[2]), (seq3[2], seq3[0])]:
        out = ca.forward(inp)
        col_outs = out.reshape(3, 3)
        avg = col_outs.mean(axis=0)
        pred = np.argmax(avg)
        exp = np.argmax(expected)
        match = "✓" if pred == exp else "✗"
        print(f"  {inp} -> pred {pred}, expected {exp} {match}")

    print()
    print("--- CorticalArea + lateral k-WTA (k=2) on A,A,B ---")
    np.random.seed(42)
    ca_kwta = CorticalArea(input_size=3, hidden_size=8, output_size=3,
                            num_columns=3, learning_rate=0.3, lateral_k=2)
    for i in range(len(bseq) - 1):
        ca_kwta.forward(bseq[i])
        loss = ca_kwta.update(bseq[i + 1])
        if i % 150 == 0:
            print(f"  Step {i}, avg_loss={loss:.4f}")
    ca_kwta.reset_context()
    for inp, expected in [(seq3[0], seq3[1]), (seq3[1], seq3[2]), (seq3[2], seq3[0])]:
        out = ca_kwta.forward(inp)
        col_outs = out.reshape(3, 3)
        avg = col_outs.mean(axis=0)
        pred = np.argmax(avg)
        exp = np.argmax(expected)
        match = "✓" if pred == exp else "✗"
        print(f"  {inp} -> pred {pred}, expected {exp} {match}")
        # Show sparsity
        norms = np.linalg.norm(col_outs, axis=1)
        print(f"    column norms: {norms.round(3)} (active={np.sum(norms>0)})")

    print()
    col = CorticalColumn(
        input_size=3, output_size=3, learning_rate=0.3, context_decay=0.8
    )

    zero = np.array([1, 0, 0])
    one = np.array([0, 1, 0])
    two = np.array([0, 0, 1])

    sequence = [zero, one, two] * 300
    target_confidence = 0.95

    print("Training on 0 -> 1 -> 2 -> 0 -> ...\n")
    for i in range(len(sequence) - 1):
        probs = col.forward(sequence[i])
        loss = col.update(sequence[i + 1])
        if i % 150 == 0:
            print(f"Step {i}, loss={loss:.4f}")

    print("\n--- Final Tests ---")
    col.reset_context()
    tests = [
        ("After 0, expect 1:", zero, one),
        ("After 1, expect 2:", one, two),
        ("After 2, expect 0:", two, zero),
    ]
    for label, inp, expected in tests:
        col.reset_context()
        probs = col.forward(inp)
        pred_idx = np.argmax(probs)
        exp_idx = np.argmax(expected)
        match = "✓" if pred_idx == exp_idx else "✗"
        print(f"  {label} prob={probs}, pred={pred_idx}, expected={exp_idx} {match}")

    print("\n" + "=" * 55)
    print("STEP 2: CorticalLayer with 4 columns on same 3-step sequence")
    print("=" * 55)

    layer = CorticalLayer(
        input_size=3, output_size=3, num_columns=4, learning_rate=0.3, context_decay=0.8
    )

    print("Training... (layer output size:", layer.output_size, ")\n")
    for i in range(len(sequence) - 1):
        layer.forward(sequence[i])
        loss = layer.update(sequence[i + 1])
        if i % 150 == 0:
            print(f"Step {i}, avg_loss={loss:.4f}")

    print("\n--- Layer Final Tests ---")
    for label, inp, expected in tests:
        layer.reset_context()
        out = layer.forward(inp)
        col_outputs = out.reshape(4, 3)
        avg_probs = col_outputs.mean(axis=0)
        pred_idx = np.argmax(avg_probs)
        exp_idx = np.argmax(expected)
        match = "✓" if pred_idx == exp_idx else "✗"
        print(
            f"  {label} avg_probs={avg_probs.round(3)}, pred={pred_idx}, expected={exp_idx} {match}"
        )
