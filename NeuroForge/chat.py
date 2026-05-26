"""Interactive chat with FULL brain — all systems active."""

import os

import numpy as np

from language import LanguageLearner, STORIES, DIALOGS
from pfc import PrefrontalCortex
from master_brain import MasterBrain
from config import Config

SAVE_PATH = "language_learner.pkl"

corpus = STORIES + DIALOGS
words = set()
for sent in corpus:
    for w in sent.split():
        words.add(w)
vocab = sorted(words)
sentences = [s.split() for s in corpus]


def main():
    cfg = Config()

    if os.path.exists(SAVE_PATH):
        try:
            old = LanguageLearner.load(SAVE_PATH)
            if old.vocab_size != len(vocab):
                print(f"Vocab mismatch ({old.vocab_size} vs {len(vocab)}), retraining ...")
                raise FileNotFoundError
            learner = old
            del old
            print(f"Loading saved model from {SAVE_PATH} ...")
        except (FileNotFoundError, AttributeError, EOFError, KeyError):
            learner = None
    else:
        learner = None

    if learner is None:
        print("Training new full-brain model ...")
        learner = LanguageLearner(
            vocab,
            hidden_size=cfg.hidden_size,
            num_columns=cfg.num_columns,
            context_buffer=cfg.context_buffer,
            learning_rate=cfg.learning_rate,
            use_memory=True,
            lateral_k=0,
            l23_activation="relu",
            hippocampus_type='full',
            use_amygdala=True,
            use_neuromodulation=True,
            use_workspace=True,
            use_cerebellum=True,
            use_interoception=True,
        )
        learner.train_on_corpus(
            sentences, epochs=cfg.epochs, blend_memory=cfg.blend_memory,
            replay_every=cfg.replay_every, replay_steps=cfg.replay_steps,
            sleep_every=cfg.sleep_every, nrem_steps=50, rem_steps=20,
        )
        learner.save(SAVE_PATH)
        print(f"Saved to {SAVE_PATH}")

    # Wire full brain
    brain = MasterBrain(vocab)
    brain.learner = learner
    brain.pfc = PrefrontalCortex(learner, reward_weight=cfg.reward_weight)
    learner.init_basal_ganglia()
    learner.init_sleep_cycle()

    print(f"\n{'='*60}")
    print(f"  Rebirth AI — Full Brain Online")
    print(f"  {len(vocab)} words, {len(sentences)} sentences")
    print(f"  All systems: Neocortex + Hippocampus + PFC + Thalamus")
    print(f"  Amygdala + Neuromodulation + Basal Ganglia + Cerebellum")
    print(f"  Global Workspace + Interoception + Sleep Cycle")
    print(f"{'='*60}")
    print("  Commands: /quit /reset /brain\n")

    learner.cortex.reset_context()
    learner._last_area1_h23_avg = None

    while True:
        try:
            user_input = input("USER> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("/quit", "/exit", "/q"):
            break
        if user_input.lower() == "/reset":
            learner.cortex.reset_context()
            learner._last_area1_h23_avg = None
            print("[reset]")
            continue
        if user_input.lower() == "/brain":
            state = brain.get_brain_state()
            print("\n--- Brain State ---")
            print(f"  Neuromod: {state['neuromodulation']}")
            print(f"  Mood: {state['interoception']['mood']:.3f}")
            print(f"  Energy: {state['interoception']['energy']:.3f}")
            print(f"  Drives: {state['interoception']['drives']}")
            print(f"  Amygdala valence: {state['amygdala']['valence']:.3f}")
            print(f"  Workspace items: {state['workspace']['items_in_workspace']}")
            print(f"  Hippocampus memories: {state['hippocampus_memories']}")
            print(f"  Training step: {state['training_step']}")
            print()
            continue
        if not user_input:
            continue

        known = [w for w in user_input.lower().split() if w in learner.word2idx]
        if not known:
            known = ["hello"]

        # Full brain processes user input
        context = ["<USER>"] + known
        for w in context:
            brain.step(w)

        # Generate response through full brain
        response = brain.generate_response(known, max_words=12)

        asst = " ".join(w for w in response if not w.startswith("<"))
        print("ASST>", asst)


if __name__ == "__main__":
    main()
