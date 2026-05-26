"""Master Brain — top-level orchestrator that wires ALL brain modules together.

This is the central nervous system: initializes, coordinates, and runs
every module in the Rebirth AI brain.
"""

import numpy as np

from config import Config
from language import LanguageLearner
from pfc import PrefrontalCortex
from amygdala import Amygdala
from neuromodulation import Neuromodulator
from global_workspace import GlobalWorkspace
from cerebellum import Cerebellum
from interoception import Interoception
from basal_ganglia import BasalGanglia


class MasterBrain:
    """Complete brain with all regions wired together.

    Architecture (simplified flow):
      Input → Thalamus → Neocortex → Hippocampus → PFC → Output
                  ↕           ↕            ↕           ↕
              Amygdala ← Neuromodulation ← Global Workspace
                  ↕           ↕
              Interoception ←→ BG (action selection)
                  ↕
              Cerebellum (forward model)
    """

    def __init__(self, vocab, config=None):
        self.cfg = config or Config()
        self.vocab = vocab

        # --- Core language system ---
        self.learner = LanguageLearner(
            vocab=vocab,
            hidden_size=self.cfg.hidden_size,
            num_columns=self.cfg.num_columns,
            learning_rate=self.cfg.learning_rate,
            context_buffer=self.cfg.context_buffer,
            use_memory=True,
            hippocampus_type=self.cfg.hippocampus_type,
        )

        # --- Prefrontal Cortex (planning) ---
        self.pfc = PrefrontalCortex(
            self.learner,
            goal_weight=self.cfg.goal_weight,
            reward_weight=self.cfg.reward_weight,
        )

        # --- Basal Ganglia (action selection) ---
        self.learner.init_basal_ganglia(
            learning_rate=self.cfg.bg_learning_rate
        )

        # --- Amygdala (emotion) ---
        self.amygdala = Amygdala(
            input_dim=self.cfg.hidden_size,
            learning_rate=self.cfg.amygdala_lr,
            valence_decay=self.cfg.valence_decay,
        )

        # --- Neuromodulation (DA/5-HT/NE/ACh) ---
        self.neuromod = Neuromodulator()

        # --- Global Workspace (consciousness-like broadcast) ---
        self.workspace = GlobalWorkspace(
            capacity=self.cfg.workspace_capacity,
            broadcast_decay=self.cfg.broadcast_decay,
        )

        # --- Cerebellum (forward model) ---
        self.cerebellum = Cerebellum(
            input_dim=self.cfg.hidden_size + len(vocab),  # state + motor command
            output_dim=self.cfg.hidden_size,  # predicts next state only
            expansion_factor=4,
        )

        # --- Interoception (homeostatic drives) ---
        self.interoception = Interoception()

        # --- State tracking ---
        self.training_step = 0
        self.current_state_h23 = None
        self.last_prediction_error = None
        self.last_reward = 0.0
        self.conversation_history = []

    # ---------------------------------------------------------------
    # Core step: process one input, produce output
    # ---------------------------------------------------------------
    def step(self, input_word, external_reward=None):
        """Process a single input word through the full brain.

        Returns dict with all system outputs.
        """
        # 1. Neocortex predicts next word
        probs = self.learner.cortex.predict(self.learner.one_hot(input_word))
        self.learner._last_area1_h23_avg = np.mean(
            [c._h23 for c in self.learner.cortex.area1.columns], axis=0
        )
        self.current_state_h23 = self.learner._last_area1_h23_avg

        # 2. Cerebellum forward model prediction
        h23_norm = self.learner._normalize_h23(self.current_state_h23)
        cerebellar_pred = self.cerebellum.forward(
            h23_norm,
            motor_command=np.eye(len(self.vocab))[self.vocab.index(input_word)] if input_word in self.vocab else None
        )

        # 3. Amygdala emotional appraisal (fast pathway via thalamus)
        thal_input = np.mean([c._h23 for c in self.learner.cortex.area1.columns], axis=0)
        emotional_valence = self.amygdala.fast_pathway(thal_input[:self.cfg.hidden_size])

        # 4. Compute prediction error
        self.last_prediction_error = np.zeros_like(probs)  # set during training

        # 5. Global workspace — propose current state
        self.workspace.propose(
            content=input_word,
            salience=np.max(probs) + 0.5 * np.abs(emotional_valence),
            source='neocortex'
        )

        # 6. Neuromodulation update
        novelty = 1.0 - np.max(probs) if probs is not None else 0.5
        self.neuromod.update_all(
            reward=external_reward,
            prediction_error=self.last_prediction_error,
            novelty=novelty,
            uncertainty=1.0 - np.max(probs) if probs is not None else 0.5,
        )

        # 7. Interoception update
        self.interoception.update(
            prediction_error=self.last_prediction_error,
            social_interaction=True,
            learning_progress=1.0 - np.max(probs) if probs is not None else 0.0,
            novelty=novelty,
        )

        # 8. Amygdala slow pathway (cortical appraisal)
        self.amygdala.slow_pathway(thal_input[:self.cfg.hidden_size], self.last_prediction_error)

        # 9. Global workspace competition
        attention_mod = {
            'neocortex': self.neuromod.ne.get_attention_breadth(),
            'hippocampus': self.neuromod.ach.get_encoding_strength(),
        }
        workspace_winner = self.workspace.compete(attention_modulation=attention_mod)
        self.workspace.broadcast_to([self.learner])

        self.training_step += 1

        # 10. Intrinsic reward from interoception
        intrinsic_reward = self.interoception.get_intrinsic_reward()

        return {
            'probs': probs,
            'predicted_idx': int(np.argmax(probs)) if probs is not None else 0,
            'emotional_valence': float(emotional_valence),
            'neuromod_state': self.neuromod.get_state(),
            'interoception': self.interoception.get_state(),
            'workspace_focus': self.workspace.get_current_focus(),
            'intrinsic_reward': float(intrinsic_reward),
            'cerebellar_prediction': cerebellar_pred,
        }

    def train_step(self, input_word, target_word, external_reward=None):
        """Train on a single (input, target) pair through the full brain."""
        # Neocortex update
        x = self.learner.one_hot(input_word)
        target = self.learner.one_hot(target_word)

        probs = self.learner.cortex.predict(x)

        # Amygdala tags the memory
        h23 = np.mean([c._h23 for c in self.learner.cortex.area1.columns], axis=0)
        self.amygdala.tag_memory(h23, target)

        # Compute consolidation modulation
        consol_mod = self.amygdala.get_consolidation_modulation(h23)

        # Hippocampus storage with emotional modulation
        h23_norm = self.learner._normalize_h23(h23)
        if self.learner.hippocampus_type == 'full':
            self.learner.hippocampus.encode(h23_norm, target.copy())
        else:
            self.learner.hippocampus.store(h23_norm, target.copy())

        # Neocortex backward pass with learning rate modulated by ACh
        lr_mod = self.neuromod.ach.get_learning_modulation()
        original_lr = self.learner.cortex.area1.layer.learning_rate
        self.learner.cortex.area1.layer.learning_rate *= lr_mod
        loss = self.learner.cortex.update(target)
        self.learner.cortex.area1.layer.learning_rate = original_lr

        # Cerebellum learns forward model
        h23_norm_curr = self.learner._normalize_h23(
            np.mean([c._h23 for c in self.learner.cortex.area1.columns], axis=0)
        )
        self.cerebellum.update(
            h23_norm,
            h23_norm_curr,
            motor_command=x,
        )

        # Prediction error for neuromodulation
        pred_error = target - probs

        # Reward (external or intrinsic)
        if external_reward is None:
            external_reward = self.interoception.get_intrinsic_reward()

        # BG reward signal
        if self.learner.basal_ganglia is not None:
            self.learner.basal_ganglia.reinforce(
                self.learner.word2idx.get(target_word, 0),
                external_reward,
            )

        # Neuromodulation update
        reward = external_reward if external_reward is not None else 0.0
        self.neuromod.update_all(
            reward=reward,
            prediction_error=pred_error,
            novelty=1.0 - np.max(probs),
            uncertainty=np.var(probs) if probs is not None else 0.5,
            td_error=np.max(pred_error),
        )

        # Interoception update
        self.interoception.update(
            prediction_error=pred_error,
            social_interaction=True,
            learning_progress=np.mean(np.abs(pred_error)),
            novelty=1.0 - np.max(probs),
        )

        # Amygdala slow pathway
        self.amygdala.slow_pathway(h23, pred_error)

        return {
            'loss': float(loss),
            'emotional_valence': float(self.amygdala.current_valence),
            'consolidation_modulation': float(consol_mod),
            'neuromod_state': self.neuromod.get_state(),
            'intrinsic_reward': float(self.interoception.get_intrinsic_reward()),
        }

    def plan_response(self, context_words):
        """Plan next response word using PFC with full brain context."""
        # Get PFC plan
        word, score, mode = self.pfc.plan_next_word(context_words)

        # Neuromodulator gating: low patience → more random exploration
        patience = self.neuromod.serotonin.get_patience()
        if np.random.random() > patience:
            # Explore: sample from neocortex distribution
            last_word = context_words[-1] if context_words else '<ASST>'
            probs = self.learner.cortex.predict(self.learner.one_hot(last_word))
            probs = probs.ravel()
            for tok in ['<USER>', '<ASST>']:
                if tok in self.learner.word2idx:
                    probs[self.learner.word2idx[tok]] = 0.0
            probs /= probs.sum() + 1e-12
            idx = np.random.choice(len(probs), p=probs)
            word = self.learner.idx2word[idx]
            mode = 'explore'

        # Basal Ganglia bias: if confidence is low, use BG Q-values
        bg = self.learner.basal_ganglia
        if bg is not None and mode in ('fallback', 'explore'):
            h23 = np.mean([c._h23 for c in self.learner.cortex.area1.columns], axis=0)
            q = bg.evaluate(self.learner._normalize_h23(h23))
            best_q_idx = int(np.argmax(q))
            best_q_word = self.learner.idx2word[best_q_idx]
            if best_q_word not in ('<USER>', '<ASST>'):
                word = best_q_word
                mode = 'bg_biased'

        return word, score, mode

    def generate_response(self, user_input_words, max_words=12):
        """Generate a full response to user input through the complete brain."""
        # Feed user context
        self.learner.cortex.reset_context()
        self.learner._last_area1_h23_avg = None

        context = ['<USER>'] + user_input_words
        for w in context:
            result = self.step(w)

        # Generate ASST response
        response = ['<ASST>']
        for _ in range(max_words):
            word, score, mode = self.plan_response(context + response)

            # Feed chosen word
            _ = self.learner.cortex.predict(self.learner.one_hot(word))
            self.learner._last_area1_h23_avg = np.mean(
                [c._h23 for c in self.learner.cortex.area1.columns], axis=0
            )

            response.append(word)
            if word in ('.', '!', '?'):
                break
            if word in ('<USER>', '<ASST>'):
                break

        return response

    def get_brain_state(self):
        """Full diagnostic of all brain systems."""
        return {
            'neuromodulation': self.neuromod.get_state(),
            'interoception': self.interoception.get_state(),
            'amygdala': {
                'valence': float(self.amygdala.current_valence),
                'arousal': float(self.amygdala.arousal),
                'memories': len(self.amygdala.memory_valence),
            },
            'workspace': self.workspace.report_self(),
            'training_step': self.training_step,
            'hippocampus_type': self.learner.hippocampus_type,
            'hippocampus_memories': (
                len(self.learner.hippocampus.replay_buffer)
                if hasattr(self.learner.hippocampus, 'replay_buffer')
                else 'N/A'
            ),
            'pfc_goal_weight': self.pfc.goal_weight,
        }
