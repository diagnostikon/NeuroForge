import numpy as np

from neurons import CorticalLayer, CorticalArea
from thalamus import Thalamus

A = np.array([1, 0])
B = np.array([0, 1])

class SimpleStack:
    def __init__(
        self,
        input_size,
        num_columns=5,
        learning_rate=0.3,
        context_decay=0.9,
        context_buffer=0,
    ):
        self.num_columns = num_columns
        self.input_size = input_size
        self.h1_dim = input_size * num_columns

        self.layer1 = CorticalLayer(
            input_size,
            input_size,
            num_columns,
            learning_rate,
            context_decay,
            context_buffer,
            use_softmax=True,
        )
        self.layer2 = CorticalLayer(
            self.h1_dim,
            self.h1_dim,
            num_columns,
            learning_rate * 0.3,
            context_decay,
            context_buffer,
            use_softmax=False,
        )

    def update(self, x_next):
        loss1 = self.layer1.update(x_next)
        h1_curr = self.layer1.forward(x_next)
        loss2 = self.layer2.update(h1_curr)
        self.layer2.forward(h1_curr)
        self._l1l = loss1
        self._l2l = loss2
        return (loss1 + loss2) / 2

    def predict(self, x):
        h1 = self.layer1.forward(x)
        self.layer2.forward(h1)
        return h1.reshape(self.num_columns, self.input_size).mean(axis=0)

    def reset_context(self):
        self.layer1.reset_context()
        self.layer2.reset_context()


class PredictiveStack:
    def __init__(
        self,
        input_size,
        num_columns=5,
        learning_rate=0.3,
        context_decay=0.9,
        context_buffer=0,
    ):
        self.num_columns = num_columns
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.h1_dim = input_size * num_columns

        self.layer1 = CorticalLayer(
            input_size,
            input_size,
            num_columns,
            learning_rate,
            context_decay,
            context_buffer,
            use_softmax=True,
        )
        self.layer2 = CorticalLayer(
            self.h1_dim,
            self.h1_dim,
            num_columns,
            learning_rate * 0.3,
            context_decay,
            context_buffer,
            use_softmax=False,
        )
        self._prev_h2_avg = None

    def update(self, x_next):
        loss1 = self.layer1.update(x_next)
        h1_curr = self.layer1.forward(x_next)

        if self._prev_h2_avg is not None:
            self.l1_error = np.clip(h1_curr - self._prev_h2_avg, -1.0, 1.0)
        else:
            self.l1_error = np.zeros_like(h1_curr)

        loss2 = self.layer2.update(h1_curr)
        h2_concat = self.layer2.forward(h1_curr)
        self._prev_h2_avg = h2_concat.reshape(self.num_columns, -1).mean(axis=0)
        self.l1_out = h1_curr
        self._l1l = loss1
        self._l2l = loss2
        return (loss1 + loss2) / 2

    def predict(self, x):
        h1 = self.layer1.forward(x)
        h2_concat = self.layer2.forward(h1)
        if self._prev_h2_avg is not None:
            self.l1_error = np.clip(h1 - self._prev_h2_avg, -1.0, 1.0)
        else:
            self.l1_error = np.zeros_like(h1)
        self._prev_h2_avg = h2_concat.reshape(self.num_columns, -1).mean(axis=0)
        self.l1_out = h1
        return h1.reshape(self.num_columns, self.input_size).mean(axis=0)

    def reset_context(self):
        self.layer1.reset_context()
        self.layer2.reset_context()
        self._prev_h2_avg = None


class BidirectionalCortex:
    def __init__(self, input_size, hidden_size, output_size,
                 num_columns=5, learning_rate=0.5, context_buffer=3,
                 lateral_k=3, l23_activation='tanh',
                 predictive_coding=True):
        self.num_columns = num_columns
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.h1_dim = output_size * num_columns
        self.predictive_coding = predictive_coding

        lr1 = learning_rate
        lr2 = learning_rate * 0.2

        td_dim = output_size if predictive_coding else hidden_size

        self.area1 = CorticalArea(input_size, hidden_size, output_size,
                                   num_columns, lr1,
                                   context_buffer=context_buffer,
                                   use_softmax=True, lateral_k=lateral_k,
                                   td_size=td_dim,
                                   l23_activation=l23_activation,
                                   learn_lateral=True)
        self.area2 = CorticalArea(self.h1_dim, hidden_size, output_size,
                                   num_columns, lr2,
                                   context_buffer=context_buffer,
                                   use_softmax=False, lateral_k=lateral_k,
                                   l23_activation=l23_activation,
                                   learn_lateral=True)

        self.fb_w = np.random.randn(hidden_size, td_dim) * np.sqrt(2.0 / hidden_size)
        self.fb_b = np.zeros(td_dim)
        self.fb_lr = learning_rate * 0.3

        self.fb_w_th = np.random.randn(hidden_size, output_size * num_columns) * np.sqrt(2.0 / hidden_size)
        self.fb_b_th = np.zeros(output_size * num_columns)
        self.fb_lr_th = learning_rate * 0.05

        self.thalamus = Thalamus(input_dim=output_size, num_columns=num_columns)

        self.layer1 = self.area1
        self.layer2 = self.area2

        self._prev_td = None
        self._prev_avg_h23 = None
        self._prev_x = None
        self._thal_mod = None
        self._mem_blend = np.zeros(self.h1_dim)
        self._td_errors = []

    @property
    def l1_error(self):
        if self._td_errors:
            return self._td_errors[-1] * np.ones(self.h1_dim)
        return np.zeros(self.h1_dim)

    def reset_td_errors(self):
        self._td_errors = []

    def update(self, x_next):
        loss1 = self.area1.update(x_next)

        thal_grads = [c._d_thal for c in self.area1.columns
                      if c._d_thal is not None]
        td_grads = [c._d_td for c in self.area1.columns
                    if c._d_td is not None]

        if thal_grads and self._prev_x is not None:
            self.area1.thalamus.update(self._prev_x, thal_grads)

        if td_grads and self._prev_avg_h23 is not None:
            avg_td_err = np.mean(td_grads, axis=0)
            grad_w = np.outer(self._prev_avg_h23, avg_td_err)
            self.fb_w -= self.fb_lr * np.clip(grad_w, -1.0, 1.0)
            self.fb_b -= self.fb_lr * np.clip(avg_td_err, -1.0, 1.0)
            self._td_errors.append(np.mean(np.abs(avg_td_err)))

        if thal_grads and self._prev_avg_h23 is not None:
            col_in_size = self.area1.layer.column_input_size
            d_out = np.zeros(self.num_columns * col_in_size)
            for i, grad in enumerate(thal_grads):
                s = i * col_in_size
                d_out[s:s + len(grad)] = grad
            thal_raw = getattr(self.area1, '_thal_out_raw', None)
            thal_mod_used = getattr(self.area1, '_thal_mod_used', None)
            if thal_raw is not None and thal_mod_used is not None:
                gain_deriv = 0.05 * (1.0 - np.tanh(thal_mod_used) ** 2)
                thal_rms = np.sqrt(np.mean(thal_raw ** 2)) + 1e-8
                d_out = d_out * (thal_raw / thal_rms) * gain_deriv
            grad_w_th = np.outer(self._prev_avg_h23, d_out)
            self.fb_w_th -= self.fb_lr_th * np.clip(grad_w_th, -1.0, 1.0)
            self.fb_b_th -= self.fb_lr_th * np.clip(d_out, -1.0, 1.0)

        h1_curr = self.area1.forward(x_next, td_inputs=self._prev_td,
                                     thal_mod=self._thal_mod)

        loss2 = self.area2.update(h1_curr)
        h2_curr = self.area2.forward(h1_curr)

        self._prev_avg_h23 = np.mean(
            [c._h23 for c in self.area2.columns], axis=0
        )
        self._prev_td = np.dot(self._prev_avg_h23, self.fb_w) + self.fb_b

        attn = np.abs(self._prev_avg_h23[:self.num_columns])
        attn = attn / (attn.sum() + 1e-12)
        thal_mod_raw = np.dot(self._prev_avg_h23, self.fb_w_th) + self.fb_b_th
        thal_per_col = thal_mod_raw.reshape(self.num_columns, -1)
        thal_per_col = self.thalamus.trn.gate(thal_per_col, attention_weights=attn)
        self._thal_mod = thal_per_col.ravel()
        self._prev_x = x_next

        self._l1l = loss1
        self._l2l = loss2
        self._last_td_mag = np.mean(np.abs(self._prev_td)) if self._prev_td is not None else 0.0
        return (loss1 + loss2) / 2

    def predict(self, x):
        # TRN attention: h23 states gate which columns are boosted
        if self._prev_avg_h23 is not None:
            attn = np.abs(self._prev_avg_h23[:self.num_columns])
            attn = attn / (attn.sum() + 1e-12)
            gated = self.thalamus.process(x, attention_weights=attn)
            thal_gate = np.mean(gated, axis=0) if gated.ndim > 1 else gated
        else:
            thal_gate = x

        h1 = self.area1.forward(thal_gate, td_inputs=self._prev_td,
                                thal_mod=self._thal_mod)

        h2 = self.area2.forward(h1)
        self._prev_avg_h23 = np.mean(
            [c._h23 for c in self.area2.columns], axis=0
        )
        self._prev_td = np.dot(self._prev_avg_h23, self.fb_w) + self.fb_b

        attn = np.abs(self._prev_avg_h23[:self.num_columns])
        attn = attn / (attn.sum() + 1e-12)
        thal_mod_raw = np.dot(self._prev_avg_h23, self.fb_w_th) + self.fb_b_th
        thal_per_col = thal_mod_raw.reshape(self.num_columns, -1)
        thal_per_col = self.thalamus.trn.gate(thal_per_col, attention_weights=attn)
        self._thal_mod = thal_per_col.ravel()
        self._prev_x = x
        self.l1_out = h1
        col_outs = h1.reshape(self.num_columns, self.output_size)
        return col_outs.mean(axis=0)

    def reset_context(self):
        self.area1.reset_context()
        self.area2.reset_context()
        self._prev_td = None
        self._prev_avg_h23 = None
        self._prev_x = None
        self._thal_mod = None
        self._td_errors = []


def train_model(model, seq):
    h1 = model.layer1.forward(seq[0])
    h2_concat = model.layer2.forward(h1)
    if hasattr(model, "_prev_h2_avg"):
        model._prev_h2_avg = h2_concat.reshape(model.num_columns, -1).mean(axis=0)
    for i in range(1, len(seq) - 1):
        model.update(seq[i])


def measure_losses(model, seq, window=100):
    l1l, l2l = [], []
    start = max(0, len(seq) - window - 1)
    h1 = model.layer1.forward(seq[start])
    h2_concat = model.layer2.forward(h1)
    if hasattr(model, "_prev_h2_avg"):
        model._prev_h2_avg = h2_concat.reshape(model.num_columns, -1).mean(axis=0)
    if hasattr(model, "_prev_avg_h23"):
        model._prev_avg_h23 = np.mean(
            [c._h23 for c in model.area2.columns], axis=0
        )
        model._prev_td = np.dot(model._prev_avg_h23, model.fb_w) + model.fb_b
        model._thal_mod = np.dot(model._prev_avg_h23, model.fb_w_th) + model.fb_b_th
        model._prev_x = seq[start]
    for i in range(start + 1, len(seq) - 1):
        model.update(seq[i])
        l1l.append(model._l1l)
        l2l.append(model._l2l)
    return np.mean(l1l), np.mean(l2l)


def seq_accuracy(model, seq, warmup=30):
    correct, total = 0, 0
    model.reset_context()
    for i in range(len(seq) - 1):
        probs = model.predict(seq[i])
        if i >= warmup:
            if np.argmax(probs) == np.argmax(seq[i + 1]):
                correct += 1
            total += 1
    return correct / total if total else 0.0
