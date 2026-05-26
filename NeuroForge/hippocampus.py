"""
Hippocampal formation — episodic memory with subfield structure.

Two APIs are provided:
  - Hippocampus / SparseHippocampus  (flat key‑value store, backward‑compatible)
  - FullHippocampus (DG + CA3 + CA1, biologically detailed)
"""

import numpy as np

# ======================================================================
# Backward‑compatible flat key‑value store (used by LanguageLearner)
# ======================================================================

class Hippocampus:
    """Episodic memory store for the neocortex.

    Stores experiences as key-value pairs:
        key   = the cortical state (e.g., average h23 of Area1)
        value = the sensory outcome (e.g., the next token that actually occurred)

    Retrieves the most similar past experiences to prime the neocortex,
    just like the brain recalls relevant memories to guide prediction.
    """

    def __init__(self, key_dim, value_dim, max_memories=10000, similarity="cosine"):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_memories = max_memories
        self.similarity = similarity  # 'cosine' or 'dot'

        # Memory banks
        self.keys = np.empty((0, key_dim))
        self.values = np.empty((0, value_dim))

        # Pre-allocated storage for speed (will grow as needed)
        self._keys_list = []
        self._values_list = []

    def store(self, key, value):
        """Encode a new experience."""
        self._keys_list.append(key.ravel())
        self._values_list.append(value.ravel())
        if len(self._keys_list) >= 100:
            self._compact()
        if len(self._keys_list) > self.max_memories:
            self._keys_list.pop(0)
            self._values_list.pop(0)

    def _compact(self):
        """Move list buffers into numpy arrays, appending to existing."""
        if len(self._keys_list) == 0:
            return
        new_keys = np.array(self._keys_list)
        new_values = np.array(self._values_list)
        if len(self.keys) > 0:
            self.keys = np.concatenate([self.keys, new_keys])
            self.values = np.concatenate([self.values, new_values])
        else:
            self.keys = new_keys
            self.values = new_values
        if len(self.keys) > self.max_memories:
            self.keys = self.keys[-self.max_memories:]
            self.values = self.values[-self.max_memories:]
        self._keys_list = []
        self._values_list = []

    def retrieve(self, query, k=5):
        """Find the k most similar memories to the query."""
        if len(self._keys_list) > 0:
            self._compact()
        if len(self.keys) == 0:
            return []
        query = query.ravel()
        if self.similarity == "cosine":
            query_norm = np.linalg.norm(query) + 1e-12
            key_norms = np.linalg.norm(self.keys, axis=1) + 1e-12
            similarities = np.dot(self.keys, query) / (key_norms * query_norm)
        else:
            similarities = np.dot(self.keys, query)
        if k >= len(self.keys):
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -k)[-k:]
            indices = indices[np.argsort(similarities[indices])[::-1]]
        return [(similarities[i], self.values[i]) for i in indices]

    def replay_batch(self, batch_size=32):
        """Randomly sample a batch of memories for offline consolidation."""
        if len(self._keys_list) > 0:
            self._compact()
        if len(self.keys) == 0:
            return None, None
        n = min(batch_size, len(self.keys))
        indices = np.random.choice(len(self.keys), n, replace=False)
        return self.keys[indices], self.values[indices]


class SparseHippocampus:
    """Hippocampus wrapper that sparsifies keys before storage/retrieval.

    Motivated by the dentate gyrus: sparse representations reduce overlap
    between similar states (pattern separation), preventing catastrophic
    interference in the memory store.
    """

    def __init__(self, key_dim, value_dim, max_memories=10000, sparsity=10):
        self._mem = Hippocampus(key_dim, value_dim, max_memories, similarity="cosine")
        self.sparsity = sparsity

    def _sparsify(self, x):
        """Keep only the top‑k entries; zero out the rest."""
        x = x.copy().ravel()
        if self.sparsity >= len(x):
            return x
        threshold = np.sort(np.abs(x))[-self.sparsity]
        x[np.abs(x) < threshold] = 0.0
        return x

    def store(self, key, value):
        self._mem.store(self._sparsify(key), value)

    def retrieve(self, query, k=5):
        return self._mem.retrieve(self._sparsify(query), k)

    def retrieve_top1(self, query):
        """Return (value, similarity, margin) for the single best match.

        similarity: raw dot-product / cosine overlap with the best key.
        margin:     how much better the top match is than the runner-up
                    (1.0 = completely dominant, 0.0 = tied).
        """
        query = self._sparsify(query)
        top2 = self._mem.retrieve(query, k=2)
        if not top2:
            return None, 0.0, 0.0
        sim1, val1 = top2[0]
        if len(top2) == 1:
            return val1, float(sim1), 1.0
        sim2 = top2[1][0]
        denom = max(sim1, 1e-12)
        margin = (sim1 - sim2) / denom
        return val1, float(sim1), float(margin)

    def replay_batch(self, batch_size=32):
        return self._mem.replay_batch(batch_size)


# ======================================================================
# Biologically detailed subfield model
# ======================================================================

class DentateGyrus:
    """Pattern separation: maps dense cortical input to sparse, orthogonal code."""

    def __init__(self, input_dim, granule_cells=500, sparsity=0.02, learning_rate=0.01):
        self.input_dim = input_dim
        self.granule_cells = granule_cells
        self.sparsity = sparsity
        self.lr = learning_rate
        self.W = np.random.randn(input_dim, granule_cells) * 0.1
        self.W *= np.random.rand(input_dim, granule_cells) < 0.1
        self.b = np.zeros(granule_cells)

    def forward(self, x):
        activation = np.dot(x, self.W) + self.b
        k = max(1, int(self.granule_cells * self.sparsity))
        indices = np.argpartition(activation, -k)[-k:]
        output = np.zeros(self.granule_cells)
        output[indices] = activation[indices]
        return output


class CA3:
    """Auto-associative memory: recurrent network that can complete patterns."""

    def __init__(self, num_cells, learning_rate=0.01):
        self.num_cells = num_cells
        self.lr = learning_rate
        self.W = np.zeros((num_cells, num_cells))
        self.threshold = 0.5

    def store(self, pattern):
        """Hebbian outer-product learning."""
        p = pattern.reshape(-1, 1)
        self.W += self.lr * np.dot(p, p.T)

    def recall(self, partial_pattern, steps=5):
        """Iterative pattern completion from a partial cue."""
        state = partial_pattern.copy()
        for _ in range(steps):
            state = np.dot(self.W, state)
            state = np.where(state > self.threshold, 1, 0)
        return state


class CA1:
    """Integrates CA3 output and projects back to neocortex."""

    def __init__(self, input_dim, output_dim, learning_rate=0.01):
        self.lr = learning_rate
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)

    def forward(self, x):
        return np.tanh(np.dot(x, self.W) + self.b)

    def update(self, x, target):
        pred = self.forward(x)
        error = pred - target
        self.W -= self.lr * np.outer(x, error)
        self.b -= self.lr * error
        return np.mean(error**2)


class FullHippocampus:
    """
    Complete hippocampal formation.

    Encoding: cortex (h23) → DentateGyrus (sparse) → CA3 (stored)
    Recall:   partial cue → CA3 (completion) → CA1 (cortical output)
    Replay:   random reactivation of CA3 patterns → CA1 → neocortex
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        granule_cells=200,
        ca3_cells=200,
        learning_rate=0.01,
    ):
        self.ca3_dim = ca3_cells
        self.dg = DentateGyrus(input_dim, granule_cells)
        self.ca3 = CA3(ca3_cells, learning_rate)
        self.ca1 = CA1(ca3_cells, output_dim, learning_rate)
        # Projection: DG output → CA3 input (dense random projection)
        self.dg_to_ca3 = np.random.randn(granule_cells, ca3_cells) * 0.1
        self.replay_buffer = []
        self.max_buffer = 1000

    def encode(self, cortical_state, target_value):
        """Store a complete experience: cortical state → target."""
        sparse = self.dg.forward(cortical_state)
        ca3_input = np.dot(sparse, self.dg_to_ca3)
        self.ca3.store(ca3_input)
        self.replay_buffer.append((sparse, target_value))
        if len(self.replay_buffer) > self.max_buffer:
            self.replay_buffer.pop(0)

    def retrieve(self, cortical_state):
        """Recall target from a partial or noisy cortical state."""
        sparse = self.dg.forward(cortical_state)
        ca3_input = np.dot(sparse, self.dg_to_ca3)
        completed = self.ca3.recall(ca3_input)
        predicted_target = self.ca1.forward(completed)
        return predicted_target

    def replay(self, steps=10):
        """Replay random memories to train CA1."""
        if len(self.replay_buffer) < 2:
            return 0.0
        total_loss = 0.0
        for _ in range(steps):
            idx = np.random.randint(len(self.replay_buffer))
            sparse, target = self.replay_buffer[idx]
            recalled = self.ca3.recall(sparse)
            loss = self.ca1.update(recalled, target)
            total_loss += loss
        return total_loss / steps
