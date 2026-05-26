# Rebirth AI (THIS IS THE PROJECT THE WEIGHTS ARE NOT INCLUDED!)

A biologically-plausible artificial brain built in pure Python/NumPy.

## Architecture

Rebirth AI implements a complete cognitive architecture based on the mammalian brain:

| Region | File | Status |
|--------|------|--------|
| **Neocortex** | `neocortex.py` | ✅ BidirectionalCortex with Predictive Coding, Feedback Alignment, Dale's Law |
| **Hippocampus** | `hippocampus.py` | ✅ FullHippocampus (DG→CA3→CA1) + SparseHippocampus |
| **Prefrontal Cortex** | `pfc.py` | ✅ Goal-directed planning, one-shot recall, simulation |
| **Thalamus** | `thalamus.py` | ✅ TRN attentional gating, tonic/burst modes |
| **Amygdala** | `amygdala.py` | ✅ Emotional valence tagging, memory modulation |
| **Basal Ganglia** | `basal_ganglia.py` | ✅ Dopamine reward, action selection via Q-learning |
| **Neuromodulation** | `neuromodulation.py` | ✅ DA / 5-HT / NE / ACh systems |
| **Cerebellum** | `cerebellum.py` | ✅ Forward model with granule cells + Purkinje learning |
| **Interoception** | `interoception.py` | ✅ Homeostatic drives, intrinsic motivation |
| **Global Workspace** | `global_workspace.py` | ✅ Consciousness-like broadcast, metacognition |
| **Sleep** | `sleep.py` | ✅ NREM replay, REM dream, synaptic downscale |
| **Neurons** | `neurons.py` | ✅ BiologicalColumn, CorticalArea, k-WTA, homeostatic plasticity |

### Information Flow

```
Input → Thalamus (TRN gate) → Neocortex (PC/FA) → Hippocampus (store/recall)
            ↕                        ↕                    ↕
        Amygdala ─── Neuromodulation ─── Global Workspace
            ↕                        ↕
       Interoception ←─── Basal Ganglia (action selection)
            ↕
       Cerebellum (forward model)
```

## Quick Start

```bash
# One-command setup (creates venv, installs numpy, runs tests)
./setup.sh

# Or manually:
pip install numpy
python tests.py

# Train and chat
python chat.py
```

### Interactive Commands

- `USER> hello` — Chat with the brain
- `/reset` — Reset context
- `/brain` — Show full brain state (neuromodulation, drives, amygdala, workspace)
- `/quit` — Exit

### Train from Scratch

```bash
rm language_learner.pkl
python chat.py
```

## Training

The default corpus has 60+ dialogs across 10 categories (greetings, name, feelings, weather, food, pets, etc.).

```python
from language import LanguageLearner

learner = LanguageLearner(
    vocab,
    hippocampus_type='full',     # FullHippocampus (DG+CA3+CA1)
    use_amygdala=True,           # Emotional tagging
    use_neuromodulation=True,    # DA/5-HT/NE/ACh
    use_workspace=True,          # Global workspace
    use_cerebellum=True,         # Forward models
    use_interoception=True,      # Homeostatic drives
)
learner.train_on_corpus(
    sentences, epochs=60,
    replay_every=10, sleep_every=10,
)
```

## File Overview

| File | Purpose |
|------|---------|
| `neurons.py` | BiologicalColumn, CorticalArea, k-WTA, thalamic units |
| `neocortex.py` | BidirectionalCortex with TRN-gated Thalamus |
| `hippocampus.py` | SparseHippocampus + FullHippocampus (DG/CA3/CA1) |
| `pfc.py` | PrefrontalCortex planner |
| `thalamus.py` | TRN, ThalamicRelay, Thalamus |
| `amygdala.py` | Emotional valence tagging |
| `basal_ganglia.py` | Dopamine, Striatum, action selection |
| `neuromodulation.py` | Dopamine, Serotonin, Noradrenaline, Acetylcholine |
| `cerebellum.py` | Granule cells, Purkinje learning, forward model |
| `interoception.py` | Homeostatic drives, intrinsic motivation |
| `global_workspace.py` | Consciousness-like broadcast |
| `sleep.py` | SharpWaveRipple, SleepCycle |
| `language.py` | LanguageLearner, ReplayBuffer, training loop |
| `master_brain.py` | Top-level orchestrator for all modules |
| `chat.py` | Interactive conversation demo |
| `config.py` | Central hyperparameters |
| `tests.py` | Unit + integration tests |
| `setup.sh` | One-command setup (venv + deps + tests) |
| `LICENSE` | MIT License |

## Biological Mechanisms

- **Predictive Coding** (Rao & Ballard 1999): Top-down predictions subtracted from bottom-up input at L4
- **Feedback Alignment** (Lillicrap 2016): Random feedback weights bypass the weight transport problem
- **Dale's Law**: Each neuron has fixed excitatory/inhibitory sign
- **Homeostatic Plasticity**: k-WTA with adaptive threshold maintenance
- **Dentate Gyrus**: Pattern separation via random sparse projections
- **CA3**: Auto-associative Hebbian memory with pattern completion
- **CA1**: Maps CA3 recall to cortical output space
- **TRN**: Attentional gating of thalamocortical channels
- **Dopamine RPE**: Reward prediction error drives learning
- **Sleep Consolidation**: NREM replay → REM dream → synaptic downscaling

## Requirements

- Python 3.8+
- NumPy
- No GPU required (runs on CPU)

## License

MIT — see [LICENSE](LICENSE) for the full text.
