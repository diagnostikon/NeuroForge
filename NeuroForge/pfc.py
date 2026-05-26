import numpy as np

from language import LanguageLearner


class PrefrontalCortex:
    """
    Prefrontal planner — the leap from babbling to thinking.

    Triage decision flow:
      1. Hippocampus one-shot  — if the current state matches a stored
         memory with high confidence, recall the answer directly
         (episodic / System 1).
      2. Goal‑directed planning — simulate a handful of candidate
         next words, check which one moves the cortical state closest
         to the desired *goal* state, and pick the best
         (deliberate / System 2).
      3. Neocortex fallback    — if planning is uncertain, fall back
         to the raw predictor.

    The *goal state* is a vector representing "what a good response
    looks like" in cortical state space.  It is computed from a few
    example (context_word, good_response_word) pairs — e.g. greeting
    politely, answering a question, etc.
    """

    # Token that should never be emitted as response words
    STRUCTURAL_TOKENS = {"<USER>", "<ASST>"}

    # ------------------------------------------------------------------
    # Configuration (tunable)
    # ------------------------------------------------------------------
    # Hippocampus one‑shot threshold
    MEM_CONFIDENCE = 0.6
    # If the best candidate's score is within this fraction of the
    # neocortex's top prob, we go with the plan; otherwise fall back.
    PLAN_CONFIDENCE_RATIO = 0.8
    # Period suppression: penalty multiplier for "." so it's not the
    # default lazy prediction.  Only applied during planning, never during
    # training — the model still learns real period transitions.
    PERIOD_PENALTY = 0.3

    def __init__(self, learner: LanguageLearner,
                 goal_samples=None, goal_weight=0.5, reward_weight=0.2):
        self.learner = learner
        self.cortex = learner.cortex
        self.hippocampus = learner.hippocampus
        self.vocab = learner.vocab
        self.word2idx = learner.word2idx
        self.idx2word = learner.idx2word
        self.goal_weight = goal_weight
        self.reward_weight = reward_weight

        if goal_samples is not None:
            self.goal_state = self._compute_goal_state(goal_samples)
        else:
            self.goal_state = self._compute_goal_state(
                [("<ASST>", "hello"), ("<ASST>", "hi"),
                 ("<ASST>", "i"), ("<ASST>", "my"),
                 ("<ASST>", "yes")]
            )
        self.goal_norm = np.linalg.norm(self.goal_state) + 1e-12

    # ------------------------------------------------------------------
    # Goal state — the "feeling" of a good response
    # ------------------------------------------------------------------
    def _compute_goal_state(self, samples):
        """Average cortical state AFTER uttering a good response word."""
        states = []
        for context_word, response_word in samples:
            ctx = self.learner.one_hot(context_word)
            resp = self.learner.one_hot(response_word)
            self.cortex.reset_context()
            self.cortex.predict(ctx)
            self.cortex.predict(resp)
            h23 = np.mean([c._h23 for c in self.cortex.area1.columns], axis=0)
            states.append(h23)
        return np.mean(states, axis=0)

    # ------------------------------------------------------------------
    # State save / restore  (candidate simulation without side effects)
    # ------------------------------------------------------------------
    def _save_state(self):
        cols1 = self.cortex.area1.columns
        cols2 = self.cortex.area2.columns
        return {
            "ctx": self.cortex._prev_x,
            "td": self.cortex._prev_td,
            "thal": self.cortex._thal_mod,
            "h23_avg": self.cortex._prev_avg_h23,
            "a1_ctx": [c.context.copy() for c in cols1],
            "a1_h23": [c._h23.copy() for c in cols1],
            "a2_ctx": [c.context.copy() for c in cols2],
            "a2_h23": [c._h23.copy() for c in cols2],
        }

    def _restore_state(self, state):
        cols1 = self.cortex.area1.columns
        cols2 = self.cortex.area2.columns
        self.cortex._prev_x = state["ctx"]
        self.cortex._prev_td = state["td"]
        self.cortex._thal_mod = state["thal"]
        self.cortex._prev_avg_h23 = state["h23_avg"]
        for i, c in enumerate(cols1):
            c.context[:] = state["a1_ctx"][i]
            c._h23[:] = state["a1_h23"][i]
        for i, c in enumerate(cols2):
            c.context[:] = state["a2_ctx"][i]
            c._h23[:] = state["a2_h23"][i]

    # ------------------------------------------------------------------
    # Forward simulation  (non‑destructive)
    # ------------------------------------------------------------------
    def _simulate_state_after(self, utterance_words):
        """
        Return the cortical state (avg h23) that would result from
        feeding utterance_words through the cortex, WITHOUT mutating
        the live cortex state.
        """
        saved = self._save_state()
        self.cortex.reset_context()
        for w in utterance_words:
            self.cortex.predict(self.learner.one_hot(w))
        h23 = np.mean([c._h23 for c in self.cortex.area1.columns], axis=0)
        self._restore_state(saved)
        return h23

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _goal_similarity(self, state):
        sim = (np.dot(state, self.goal_state)
               / (np.linalg.norm(state) + 1e-12) / self.goal_norm)
        return (sim + 1) / 2  # cosine [-1,1] → [0,1]

    def _score_candidate(self, next_state, raw_prob=None, q_value=None):
        sim = self._goal_similarity(next_state)
        score = self.goal_weight * sim
        if raw_prob is not None:
            score += (1 - self.goal_weight - self.reward_weight) * raw_prob
        if q_value is not None:
            score += self.reward_weight * q_value
        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan_next_word(self, context_words, k_memories=5, k_random=3):
        """
        context_words: words fed so far (including <USER>/<ASST> tokens).
        Returns (best_word, score, mode) where mode is:
          "one-shot" — hippocampus retrieved a direct answer.
          "planned"  — simulation‑based planning picked the word.
          "fallback" — fell back to raw neocortex prediction.
        """
        # ---------------------------------------------------------------
        # 1. Hippocampus one‑shot override
        # ---------------------------------------------------------------
        if self.hippocampus is not None:
            current_h23 = np.mean(
                [c._h23 for c in self.cortex.area1.columns], axis=0)
            if self.learner._last_area1_h23_avg is not None:
                current_h23 = self.learner._last_area1_h23_avg
            mem_probs, used_memory, conf = self.learner._probs_from_memory(
                current_h23, override_strength=0.95)
            if used_memory:
                idx = np.argmax(mem_probs)
                word = self.idx2word[idx]
                if word != ".":
                    return word, float(conf), "one-shot"
                # Period as one-shot is suspicious — defer to planning

        # ---------------------------------------------------------------
        # 2. Gather candidates (excluding structural tokens)
        # ---------------------------------------------------------------
        candidates = []

        # Get current Q-values from basal ganglia (if available)
        bg = getattr(self.learner, 'basal_ganglia', None)
        q_values = None
        if bg is not None:
            current_h23_q = np.mean(
                [c._h23 for c in self.cortex.area1.columns], axis=0)
            if self.learner._last_area1_h23_avg is not None:
                current_h23_q = self.learner._last_area1_h23_avg
            q_values = bg.evaluate(self.learner._normalize_h23(current_h23_q))

        # 2a. From hippocampus (similar past experiences)
        if self.hippocampus is not None:
            query = self.learner._normalize_h23(
                np.mean([c._h23 for c in self.cortex.area1.columns], axis=0))
            if self.learner.hippocampus_type == 'full':
                # FullHippocampus: single retrieval
                pred = self.hippocampus.retrieve(query)
                if pred is not None:
                    idx = np.argmax(pred)
                    word = self.idx2word[idx]
                    if word not in self.STRUCTURAL_TOKENS:
                        candidates.append((word, 0.8))
            else:
                # SparseHippocampus
                memories = self.hippocampus.retrieve(query, k=k_memories)
                for sim, val in memories:
                    idx = np.argmax(val)
                    word = self.idx2word[idx]
                    if word not in self.STRUCTURAL_TOKENS:
                        candidates.append((word, float(sim)))

        # 2b. From neocortex predictor (sample from its distribution,
        #     excluding structural tokens)
        probs = self.cortex.predict(
            self.learner.one_hot(context_words[-1]))
        probs = probs.ravel()
        # Zero out structural tokens and renormalize
        for tok in self.STRUCTURAL_TOKENS:
            if tok in self.word2idx:
                probs[self.word2idx[tok]] = 0.0
        probs /= probs.sum() + 1e-12
        if len(candidates) < k_memories + k_random:
            for _ in range(k_random + k_memories - len(candidates)):
                idx = np.random.choice(len(probs), p=probs)
                candidates.append((self.idx2word[idx], float(probs[idx])))

        # ---------------------------------------------------------------
        # 3. Simulate & score each candidate toward the goal
        # ---------------------------------------------------------------
        best_word = context_words[-1]
        best_score = -1e9
        for word, raw in candidates:
            sim_utt = context_words + [word]
            next_state = self._simulate_state_after(sim_utt)
            q_val = None
            if q_values is not None and word in self.word2idx:
                q_val = q_values[self.word2idx[word]]
            score = self._score_candidate(next_state, raw, q_val)
            # Penalize period unless it's the only reasonable choice
            if word == "." and len(candidates) > 1:
                score *= self.PERIOD_PENALTY
            if score > best_score:
                best_score = score
                best_word = word

        # ---------------------------------------------------------------
        # 4. Decide: trust the plan or fall back to raw predictor?
        # ---------------------------------------------------------------
        top_prob = float(probs.max())
        plan_confidence = best_score / (top_prob + 1e-12)
        if plan_confidence >= self.PLAN_CONFIDENCE_RATIO:
            return best_word, best_score, "planned"
        # Fallback: pick the top neocortex word (structural tokens already zeroed)
        fallback_idx = int(np.argmax(probs))
        fallback_word = self.idx2word[fallback_idx]
        if fallback_word == "." and len(candidates) > 1:
            non_period = [c for c in candidates if c[0] != "."]
            if non_period:
                fallback_word = max(non_period, key=lambda x: x[1])[0]
        return fallback_word, float(probs.max()), "fallback"

    def generate_planned(self, seed_words, max_words=10):
        """Generate a full sentence using planning at each step.

        Returns (words_list, modes_list) showing which mode chose each word.
        """
        words = list(seed_words)
        modes = []
        for _ in range(max_words):
            word, _, mode = self.plan_next_word(words)
            modes.append(mode)
            if word in (".", "!", "?"):
                words.append(word)
                break
            if word in ("<USER>", "<ASST>"):
                break
            # Actually feed the word so the cortex state advances
            self.cortex.predict(self.learner.one_hot(word))
            words.append(word)
        return words, modes
