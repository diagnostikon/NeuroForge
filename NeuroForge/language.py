"""
language.py — bridge between BidirectionalCortex and natural language.

Supports two hippocampus implementations:
  - 'sparse': SparseHippocampus (flat key-value, backward-compatible)
  - 'full':   FullHippocampus (DG+CA3+CA1, biologically detailed)

Also integrates sleep consolidation, basal-ganglia reward, amygdala
emotional tagging, neuromodulation, and global workspace broadcast.
"""

import pickle
import numpy as np

from hippocampus import Hippocampus, SparseHippocampus, FullHippocampus
from neocortex import BidirectionalCortex, seq_accuracy
from sleep import SleepCycle
from basal_ganglia import BasalGanglia
from amygdala import Amygdala
from neuromodulation import Neuromodulator
from global_workspace import GlobalWorkspace
from cerebellum import Cerebellum
from interoception import Interoception


class ReplayBuffer:
    """Circular buffer of word-level transitions for experience replay.

    Stores (input_word_idx, target_word_idx) pairs.  During replay, these
    are fed through the cortex as if they were fresh training examples,
    reinforcing rare transitions and preventing catastrophic forgetting.
    """
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def store(self, in_idx, target_idx):
        if len(self.buffer) < self.capacity:
            self.buffer.append((in_idx, target_idx))
        else:
            self.buffer[self.pos] = (in_idx, target_idx)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]


class LanguageLearner:
    """Wraps a cortex + hippocampus and handles word ↔ one‑hot conversion.

    hippocampus_type: 'sparse' (SparseHippocampus, backward-compatible)
                      'full'   (FullHippocampus with DG+CA3+CA1)
    """

    def __init__(
        self,
        vocab,
        hidden_size=24,
        num_columns=7,
        learning_rate=0.1,
        context_buffer=4,
        use_memory=True,
        lateral_k=0,
        l23_activation='relu',
        hippocampus_type='sparse',
        use_amygdala=True,
        use_neuromodulation=True,
        use_workspace=True,
        use_cerebellum=True,
        use_interoception=True,
    ):
        self.vocab = vocab  # list of words
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(vocab)

        self.cortex = BidirectionalCortex(
            input_size=self.vocab_size,
            hidden_size=hidden_size,
            output_size=self.vocab_size,
            num_columns=num_columns,
            learning_rate=learning_rate,
            context_buffer=context_buffer,
            lateral_k=lateral_k,
            l23_activation=l23_activation,
        )

        self.use_memory = use_memory
        self.hippocampus_type = hippocampus_type
        if use_memory:
            if hippocampus_type == 'full':
                self.hippocampus = FullHippocampus(
                    input_dim=hidden_size,
                    hidden_dim=hidden_size,
                    output_dim=self.vocab_size,
                    granule_cells=500,
                    ca3_cells=200,
                    learning_rate=0.01,
                )
            else:  # 'sparse'
                self.hippocampus = SparseHippocampus(
                    key_dim=hidden_size,
                    value_dim=self.vocab_size,
                    max_memories=2000,
                    sparsity=10,
                )
        else:
            self.hippocampus = None

        self._last_area1_h23_avg = None
        self.replay = ReplayBuffer(capacity=500) if use_memory else None

        # Sleep cycle (for consolidation)
        self.sleep_cycle = None

        # Basal ganglia (for reward-based learning)
        self.basal_ganglia = None

        # --- New brain modules ---
        self.use_amygdala = use_amygdala
        self.use_neuromodulation = use_neuromodulation
        self.use_workspace = use_workspace
        self.use_cerebellum = use_cerebellum
        self.use_interoception = use_interoception

        self.amygdala = Amygdala(input_dim=hidden_size) if use_amygdala else None
        self.neuromod = Neuromodulator() if use_neuromodulation else None
        self.workspace = GlobalWorkspace() if use_workspace else None
        self.cerebellum = Cerebellum(input_dim=hidden_size + len(vocab), output_dim=hidden_size, expansion_factor=4) if use_cerebellum else None
        self.interoception = Interoception() if use_interoception else None

        self.training_step = 0

    def one_hot(self, word):
        """Convert a word to its one‑hot vector."""
        vec = np.zeros(self.vocab_size)
        if word in self.word2idx:
            vec[self.word2idx[word]] = 1.0
        return vec

    # ------------------------------------------------------------------
    # Persistence: save / load trained model to disk (avoids retraining)
    # ------------------------------------------------------------------
    def save(self, path="language_learner.pkl"):
        """Serialize the entire learner (cortex, hippocampus, vocab) to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path="language_learner.pkl"):
        """Deserialize a previously saved LanguageLearner."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def word_from_probs(self, probs, temperature=0.8):
        """Sample a word from the probability distribution (with temperature)."""
        probs = np.asarray(probs).ravel()
        if temperature != 1.0:
            probs = np.power(probs + 1e-10, 1.0 / max(temperature, 0.01))
            probs /= probs.sum()
        idx = np.random.choice(len(probs), p=probs)
        return idx, np.eye(self.vocab_size)[idx]

    def receive_broadcast(self, content, source):
        """Receive broadcast from global workspace — integrates new info."""
        if isinstance(content, str) and content in self.word2idx:
            _ = self.cortex.predict(self.one_hot(content))

    def _normalize_h23(self, h23):
        """Normalize a cortical state vector to unit length.

        The hippocampus stores h23 states as sparse keys.  Without
        normalization, the raw magnitude of h23 varies per input word,
        making similarity comparisons unreliable.  Normalizing ensures
        that dot‑product similarity reflects angular distance only.
        """
        n = np.linalg.norm(h23)
        return h23 / (n + 1e-12) if n > 0 else h23

    def _probs_from_memory(self, h23_state, override_strength=0.9):
        """Check hippocampus for a strong episodic match.

        Returns (probs, used_memory, confidence) where:
          probs       = probability distribution (vocab_size,) either from
                        memory (one-shot), or zeros (no strong match).
          used_memory = True if memory had a strong enough match.
          confidence  = the confidence score [0, 1].

        When the hippocampus has a clear best match (high similarity +
        large margin over runner-up), it returns a near-deterministic
        distribution peaked on the retrieved value — bypassing the
        neocortex entirely.  This is true one‑shot episodic recall.
        """
        if self.hippocampus is None or h23_state is None:
            return np.zeros(self.vocab_size), False, 0.0

        query = self._normalize_h23(h23_state)

        if self.hippocampus_type == 'full':
            # FullHippocampus: retrieve returns a single predicted vector
            pred = self.hippocampus.retrieve(query)
            if pred is None or np.all(pred == 0):
                return np.zeros(self.vocab_size), False, 0.0
            # Scale confidence from CA3 recall quality
            sim = np.max(np.abs(pred))
            confidence = min(0.5, sim)  # FullHippocampus is less precise
            idx = np.argmax(pred)
            probs = np.zeros(self.vocab_size)
            probs[idx] = override_strength
            probs += (1.0 - override_strength) / self.vocab_size
            probs /= probs.sum()
            return probs, confidence >= 0.4, confidence
        else:
            val, sim, margin = self.hippocampus.retrieve_top1(query)
            if val is None:
                return np.zeros(self.vocab_size), False, 0.0
            confidence = min(1.0, sim * (0.5 + 0.5 * margin))
            if confidence >= 0.6:
                probs = np.zeros(self.vocab_size)
                probs[np.argmax(val)] = override_strength
                probs += (1.0 - override_strength) / self.vocab_size
                probs /= probs.sum()
                return probs, True, confidence
            return np.zeros(self.vocab_size), False, confidence

    def train_on_sentence(self, words, blend_memory=0.0, mem_confidence=0.6):
        """Feed a list of words one by one, updating the cortex.

        Integrates amygdala emotional tagging and neuromodulation
        (ACh modulates learning rate; DA/NE modulate consolidation).
        """
        total_loss = 0.0
        for i in range(len(words) - 1):
            x = self.one_hot(words[i])
            y_target = self.one_hot(words[i + 1])

            h23_before = self._last_area1_h23_avg

            probs = self.cortex.predict(x)

            # Memory blending
            if (blend_memory > 0 and self._last_area1_h23_avg is not None
                    and self.hippocampus is not None):
                memories = self._retrieve_memories(
                    self._normalize_h23(self._last_area1_h23_avg), k=5)
                if memories:
                    mem_blend = np.mean([m[1] for m in memories], axis=0)
                    mem_blend /= mem_blend.sum() + 1e-12
                    probs = (1.0 - blend_memory) * probs + blend_memory * mem_blend
                    probs /= probs.sum()

            # Hippocampus one-shot override
            mem_probs, used_memory, conf = self._probs_from_memory(h23_before)
            if used_memory:
                probs = mem_probs

            area1_h23_before = np.mean(
                [c._h23 for c in self.cortex.area1.columns], axis=0)

            # Amygdala emotional tagging
            if self.amygdala is not None:
                pred_error = y_target - probs
                self.amygdala.slow_pathway(area1_h23_before[:self.cortex.hidden_size], pred_error)
                consol_mod = self.amygdala.get_consolidation_modulation(area1_h23_before)

            # Hippocampus storage with emotional modulation
            if self.hippocampus is not None:
                h23_norm = self._normalize_h23(area1_h23_before).copy()
                if self.hippocampus_type == 'full':
                    self.hippocampus.encode(h23_norm, y_target.copy())
                else:
                    self.hippocampus.store(h23_norm, y_target.copy())

                # Amygdala tags memory
                if self.amygdala is not None:
                    self.amygdala.tag_memory(area1_h23_before.copy(), y_target.copy())

            # Neocortex backward with ACh-modulated learning rate
            lr_mod = 1.0
            if self.neuromod is not None:
                lr_mod = self.neuromod.ach.get_learning_modulation()
                for col in self.cortex.area1.columns:
                    if hasattr(col, 'learning_rate'):
                        col.learning_rate *= lr_mod

            loss = self.cortex.update(y_target)
            total_loss += loss

            if self.neuromod is not None and lr_mod != 1.0:
                for col in self.cortex.area1.columns:
                    if hasattr(col, 'learning_rate'):
                        col.learning_rate /= lr_mod

            # Cerebellum forward model
            if self.cerebellum is not None:
                h23_norm_curr = self._normalize_h23(
                    np.mean([c._h23 for c in self.cortex.area1.columns], axis=0)
                )
                if i > 0:
                    self.cerebellum.update(h23_norm_before, h23_norm_curr, motor_command=x)
                h23_norm_before = h23_norm.copy()

            self._last_area1_h23_avg = np.mean(
                [c._h23 for c in self.cortex.area1.columns], axis=0).copy()

            # Neuromodulation update
            if self.neuromod is not None:
                self.neuromod.update_all(
                    reward=0.5 - loss,
                    prediction_error=y_target - probs,
                    novelty=1.0 - np.max(probs),
                )

            # Interoception update
            if self.interoception is not None:
                self.interoception.update(
                    prediction_error=y_target - probs,
                    social_interaction='<USER>' in words or '<ASST>' in words,
                    learning_progress=1.0 - loss,
                )

            # Global workspace proposal
            if self.workspace is not None:
                self.workspace.propose(
                    content=words[i + 1],
                    salience=np.max(probs),
                    source='neocortex',
                )

            if self.replay is not None:
                self.replay.store(
                    self.word2idx[words[i]], self.word2idx[words[i + 1]])

            self.training_step += 1

        return total_loss / (len(words) - 1) if len(words) > 1 else 0.0

    def train_on_corpus(self, sentences, epochs=5, blend_memory=0.2,
                        replay_every=5, replay_steps=20,
                        sleep_every=10, nrem_steps=50, rem_steps=20):
        """Train on a list of sentences (each sentence is a list of words).

        Every `replay_every` epochs, consolidate by replaying sampled
        transitions from the replay buffer.  Every `sleep_every` epochs,
        run a full sleep cycle (NREM→REM→downscale) for memory consolidation.

        Only FullHippocampus supports sleep; SparseHippocampus silently
        skips the sleep phase.
        """
        for epoch in range(epochs):
            total_loss = 0.0
            for sent in sentences:
                self.cortex.reset_context()
                self._last_area1_h23_avg = None
                loss = self.train_on_sentence(sent, blend_memory=blend_memory)
                total_loss += loss
            print(
                f"Epoch {epoch + 1}/{epochs}, avg loss: {total_loss / len(sentences):.4f}"
            )

            # Replay consolidation
            if replay_every > 0 and (epoch + 1) % replay_every == 0 and self.replay is not None:
                replay_loss = self.replay_consolidate(steps=replay_steps,
                                                      blend_memory=blend_memory)
                if replay_loss > 0:
                    print(f"         replay loss: {replay_loss:.4f}")

            # Sleep consolidation (FullHippocampus only)
            if sleep_every > 0 and (epoch + 1) % sleep_every == 0:
                sleep_results = self.sleep_consolidate(
                    nrem_steps=nrem_steps, rem_steps=rem_steps)
                if sleep_results:
                    rl = sleep_results.get('replay_loss', 0.0)
                    if rl > 0:
                        print(f"         sleep replay: {rl:.4f}")

    def replay_consolidate(self, steps=20, blend_memory=0.0):
        """Replay sampled transitions from the replay buffer.

        Biologically, this mimics hippocampal replay during rest/sleep:
        the hippocampus reactivates recent experiences, and the neocortex
        replays them to strengthen memory traces.
        """
        if self.replay is None or len(self.replay.buffer) < 2:
            return 0.0

        total_loss = 0.0
        for _ in range(steps):
            in_idx, target_idx = self.replay.sample(1)[0]
            x = np.eye(self.vocab_size)[in_idx]
            y_target = np.eye(self.vocab_size)[target_idx]

            self.cortex.reset_context()
            self._last_area1_h23_avg = None

            probs = self.cortex.predict(x)

            if blend_memory > 0 and self._last_area1_h23_avg is not None and self.hippocampus is not None:
                memories = self._retrieve_memories(
                    self._normalize_h23(self._last_area1_h23_avg), k=5)
                if memories:
                    mem_blend = np.mean([m[1] for m in memories], axis=0)
                    mem_blend /= mem_blend.sum() + 1e-12
                    probs = (1.0 - blend_memory) * probs + blend_memory * mem_blend
                    probs /= probs.sum()

            # Check hippocampus for one-shot override (on output only)
            h23_before = self._last_area1_h23_avg
            mem_probs, used_memory, _ = self._probs_from_memory(h23_before)
            if used_memory:
                probs = mem_probs

            loss = self.cortex.update(y_target)
            total_loss += loss

        return total_loss / steps

    def init_sleep_cycle(self):
        """Initialize the sleep cycle for memory consolidation."""
        if self.hippocampus is not None and self.hippocampus_type == 'full':
            self.sleep_cycle = SleepCycle(self.hippocampus, self.cortex)
            return True
        return False

    def sleep_consolidate(self, nrem_steps=50, rem_steps=20):
        """Run one sleep cycle: NREM replay → REM dream → downscale.

        Only available with FullHippocampus (requires replay buffer).
        """
        if self.sleep_cycle is None:
            if not self.init_sleep_cycle():
                return {}
        return self.sleep_cycle.sleep(nrem_steps, rem_steps)

    def init_basal_ganglia(self, learning_rate=0.01):
        """Initialize the basal ganglia for reward-based learning."""
        h23_dim = len(self._normalize_h23(
            np.zeros(self.cortex.hidden_size))) if hasattr(self.cortex, 'hidden_size') else 48
        self.basal_ganglia = BasalGanglia(
            input_dim=h23_dim,
            num_actions=self.vocab_size,
            learning_rate=learning_rate,
        )
        return True

    def reward_response(self, response_words, reward=1.0):
        """Reward the model for generating a good response.

        Feeds the response through the cortex, then reinforces each
        word choice via the basal ganglia's dopamine signal.
        """
        if self.basal_ganglia is None:
            self.init_basal_ganglia()
        if self.basal_ganglia is None:
            return 0.0
        total_da = 0.0
        for w in response_words:
            if w not in self.word2idx:
                continue
            h23 = self._last_area1_h23_avg
            if h23 is None:
                h23 = np.mean([c._h23 for c in self.cortex.area1.columns], axis=0)
            if h23 is None:
                continue
            q = self.basal_ganglia.evaluate(self._normalize_h23(h23))
            action_idx = self.word2idx[w]
            da = self.basal_ganglia.reinforce(action_idx, reward)
            total_da += da
        return total_da / len(response_words) if response_words else 0.0

    def _retrieve_memories(self, query, k=5):
        """Retrieve top-k memories, works with both Sparse and Full hippocampus."""
        if self.hippocampus is None:
            return []
        if self.hippocampus_type == 'full':
            # FullHippocampus: retrieve single prediction, pad to look like k=1
            pred = self.hippocampus.retrieve(query)
            if pred is not None:
                return [(1.0, pred)]
            return []
        # SparseHippocampus
        return self.hippocampus.retrieve(query, k=k) if hasattr(self.hippocampus, 'retrieve') else []

    def generate(self, seed_word, max_words=20, temperature=0.7, blend_memory=0.3,
                 suppress_period=False):
        """Generate a sequence of words starting from seed_word.

        Uses hippocampus-primary prediction: if the hippocampus has a
        strong episodic match for the current state, it overrides the
        neocortex (one‑shot recall).  Otherwise falls back to neocortex
        with optional memory blending.

        suppress_period: if True, don't penalize early periods
                         (useful for dialog where ASST may answer in 1-2 words).
        """
        self.cortex.reset_context()
        self._last_area1_h23_avg = None
        words = [seed_word]

        x = self.one_hot(seed_word)
        probs = self.cortex.predict(x)
        area1_h23_avg = np.mean([c._h23 for c in self.cortex.area1.columns], axis=0)
        self._last_area1_h23_avg = area1_h23_avg

        for step in range(max_words):
            # ---- Memory blending (gentle bias from retrieved memories) ----
            if (self.hippocampus is not None
                    and self._last_area1_h23_avg is not None
                    and blend_memory > 0):
                memories = self._retrieve_memories(
                    self._normalize_h23(self._last_area1_h23_avg), k=5)
                if memories:
                    mem_vals = np.array([m[1] for m in memories])
                    bias_probs = mem_vals.mean(axis=0)
                    bias_probs /= bias_probs.sum() + 1e-12
                    probs = (1.0 - blend_memory) * probs + blend_memory * bias_probs
                    probs /= probs.sum()

            # ---- Hippocampus-primary: check for one-shot override ----
            mem_probs, used_memory, _ = self._probs_from_memory(
                self._last_area1_h23_avg)
            if used_memory:
                probs = mem_probs
            elif step < 2 and not suppress_period:
                period_idx = self.word2idx.get(".", -1)
                if period_idx >= 0:
                    probs[period_idx] *= 0.1
                    probs /= probs.sum()

            idx, next_onehot = self.word_from_probs(probs, temperature)
            word = self.idx2word[idx]
            words.append(word)
            if word in (".", "!", "?"):
                break
            if word in ("<USER>", "<ASST>"):
                break

            x = next_onehot
            probs = self.cortex.predict(x)
            area1_h23_avg = np.mean(
                [c._h23 for c in self.cortex.area1.columns], axis=0)
            self._last_area1_h23_avg = area1_h23_avg

        return words


# ======================================================================
# Corpus: training sentences (stories + dialogs)
# ======================================================================

# --- Plain story sentences (grammar basics) ---
STORIES = [
    "the cat sat on the mat .",
    "the dog ran in the park .",
    "a bird sang on the tree .",
    "the cat saw a bird .",
    "the dog chased the cat .",
    "the bird flew away .",
    "the boy sat on the mat .",
    "a girl ran in the park .",
]

# --- Dialog sentences (turn-taking) ---
# <USER> introduces a user utterance, <ASST> introduces an assistant response.
# The model learns: USER speaks → ASST responds → turn passes back.
DIALOGS = [
        # --- Greetings (5) ---
        "<USER> hello <ASST> hello .",
        "<USER> hi <ASST> hi there .",
        "<USER> hey <ASST> hey how are you .",
        "<USER> good morning <ASST> good morning to you .",
        "<USER> good evening <ASST> good evening friend .",
        # --- Name (4) ---
        "<USER> what is your name <ASST> my name is cat .",
        "<USER> what are you called <ASST> i am called cat .",
        "<USER> who are you <ASST> i am a helpful cat .",
        "<USER> tell me your name <ASST> my name is cat .",
        # --- How-are-you (3) ---
        "<USER> how are you <ASST> i am good .",
        "<USER> how do you do <ASST> i am doing well .",
        "<USER> how is it going <ASST> it is going well .",
        # --- Likes: cats (3) ---
        "<USER> do you like cats <ASST> yes i like cats .",
        "<USER> do you love cats <ASST> yes i love cats .",
        "<USER> are cats nice <ASST> cats are very nice .",
        # --- Likes: dogs (3) ---
        "<USER> do you like dogs <ASST> yes dogs are good .",
        "<USER> do you love dogs <ASST> yes i love dogs .",
        "<USER> are dogs nice <ASST> dogs are very nice .",
        # --- Statement responses (10) ---
        "<USER> i saw a bird <ASST> birds are pretty .",
        "<USER> the cat sat on the mat <ASST> that is nice .",
        "<USER> the dog ran in the park <ASST> dogs are playful .",
        "<USER> i like cats <ASST> cats are lovely .",
        "<USER> i like dogs <ASST> dogs are loyal .",
        "<USER> i have a cat <ASST> cats are great pets .",
        "<USER> i have a dog <ASST> dogs are wonderful .",
        "<USER> the bird flew away <ASST> birds fly high .",
        "<USER> the cat saw a bird <ASST> cats watch birds .",
        "<USER> the dog chased the cat <ASST> dogs chase cats .",
        # --- Weather / nature (4) ---
        "<USER> it is raining outside <ASST> take an umbrella .",
        "<USER> it is cold today <ASST> wear a warm coat .",
        "<USER> it is a sunny day <ASST> let us go outside .",
        "<USER> the sun is shining <ASST> what a nice day .",
        # --- Feelings (4) ---
        "<USER> i feel happy today <ASST> that is wonderful .",
        "<USER> i feel sad <ASST> i hope you feel better .",
        "<USER> i am tired <ASST> get some rest .",
        "<USER> i am excited <ASST> that is great .",
        # --- Food (5) ---
        "<USER> i like pizza <ASST> pizza is delicious .",
        "<USER> i like cookies <ASST> cookies are sweet .",
        "<USER> do you like ice cream <ASST> ice cream is tasty .",
        "<USER> what is your favorite food <ASST> i like fish .",
        "<USER> i want pasta <ASST> pasta is yummy .",
        # --- Animals / pets (4) ---
        "<USER> do you have a pet <ASST> i have a fish .",
        "<USER> i like birds <ASST> birds sing nicely .",
        "<USER> cats are funny <ASST> yes cats are amusing .",
        "<USER> dogs are playful <ASST> dogs love to run .",
        # --- Story follow-ups (6) ---
        "<USER> the boy sat on the mat <ASST> boys sit too .",
        "<USER> a girl ran in the park <ASST> girls like parks .",
        "<USER> a bird sang on the tree <ASST> birds sing songs .",
        "<USER> the boy is nice <ASST> that is kind .",
        "<USER> the girl is kind <ASST> kindness is good .",
        "<USER> the cat is soft <ASST> cats are fluffy .",
        # --- Time / age (3) ---
        "<USER> how old are you <ASST> i am one year old .",
        "<USER> what time is it <ASST> it is time to chat .",
        "<USER> what day is it <ASST> it is a lovely day .",
        # --- Yes / no questions (3) ---
        "<USER> is it raining <ASST> i do not know .",
        "<USER> are you happy <ASST> yes i am happy .",
        "<USER> is the sun shining <ASST> i think so .",
        # --- Miscellaneous (5) ---
        "<USER> tell me a story <ASST> once upon a time .",
        "<USER> sing a song <ASST> la la la .",
        "<USER> let us play a game <ASST> what game do you like .",
        "<USER> read a book <ASST> reading is fun .",
        "<USER> tell me a joke <ASST> why did the cat laugh .",
    ]

if __name__ == "__main__":
    corpus = STORIES + DIALOGS

    # Build vocabulary including speaker tokens
    words = set()
    for sent in corpus:
        for w in sent.split():
            words.add(w)
    vocab = sorted(list(words))
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Words: {vocab}")

    # Convert to word lists
    sentences = [s.split() for s in corpus]

    # Create learner — scaled up for longer dependencies
    # LR tuned for PC/FA/Dale constraints on 43-class dialog task
    import os
    save_path = "language_learner.pkl"
    if os.path.exists(save_path):
        print(f"\nLoading saved model from {save_path} ...")
        learner = LanguageLearner.load(save_path)
    else:
        learner = LanguageLearner(
            vocab,
            hidden_size=48,
            num_columns=5,
            learning_rate=0.005,
            context_buffer=6,
            use_memory=True,
            lateral_k=0,
            l23_activation='relu',
        )
        print("\nTraining...")
        learner.train_on_corpus(sentences, epochs=60, blend_memory=0.15,
                                replay_every=10, replay_steps=15)
        learner.save(save_path)
        print(f"Saved to {save_path}")

    # Test plain next-word generation
    print("\n--- Plain generation ---")
    for seed in ["the", "a", "cat", "dog", "bird", "i", "my", "hello", "what"]:
        gen = learner.generate(seed, max_words=10, temperature=1.0, blend_memory=0.3)
        print(f"  {seed}: {' '.join(gen)}")

    # Test dialog generation: seed with <USER> utterance, let ASST complete
    print("\n--- Dialog generation (USER asks, ASST responds) ---")
    user_utterances = [
        ["<USER>", "hello"],
        ["<USER>", "what", "is", "your", "name"],
        ["<USER>", "how", "are", "you"],
        ["<USER>", "do", "you", "like", "cats"],
        ["<USER>", "i", "saw", "a", "bird"],
        ["<USER>", "the", "cat", "sat", "on", "the", "mat"],
    ]
    for utt in user_utterances:
        learner.cortex.reset_context()
        learner._last_area1_h23_avg = None
        # Feed the user utterance
        for w in utt:
            _ = learner.cortex.predict(learner.one_hot(w))
            learner._last_area1_h23_avg = np.mean(
                [c._h23 for c in learner.cortex.area1.columns], axis=0
            )
        # Seed with <ASST> and generate
        response = learner.generate("<ASST>", max_words=12, temperature=1.0,
                                    blend_memory=0.3)
        user_words = " ".join(w for w in utt if not w.startswith("<"))
        asst_words = " ".join(w for w in response if not w.startswith("<"))
        print(f"  USER: {user_words}")
        print(f"  ASST: {asst_words}")
        print()
