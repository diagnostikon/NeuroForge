"""Tests for ALL brain modules."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# === 1. NEURONS ===
from neurons import CorticalLayer, CorticalArea, BiologicalColumn

def test_biological_column():
    col = BiologicalColumn(column_input_size=8, hidden_size=16, output_size=8)
    # Check key weight matrices exist
    assert hasattr(col, 'W23')
    assert hasattr(col, 'W4')
    assert hasattr(col, 'W5')
    assert hasattr(col, 'W6')
    # Forward pass works
    x = np.random.randn(8)
    out = col.forward(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"
    print("  ✓ BiologicalColumn")

def test_cortical_layer():
    layer = CorticalLayer(8, 8, 3, 0.1, 0.9, 0, use_softmax=True)
    x = np.random.randn(8)
    out = layer.forward(x)
    assert out.shape[0] == 8 * 3
    # Use one-hot target for valid cross-entropy
    target = np.eye(8)[np.random.randint(8)]
    loss = layer.update(target)
    assert np.isfinite(loss)
    print(f"  ✓ CorticalLayer (loss={loss:.4f})")

def test_cortical_area():
    area = CorticalArea(8, 16, 8, 3, 0.1, use_softmax=True)
    x = np.random.randn(8)
    out = area.forward(x)
    assert out.shape[0] == 8 * 3
    target = np.eye(8)[np.random.randint(8)]
    loss = area.update(target)
    assert np.isfinite(loss)
    print(f"  ✓ CorticalArea (loss={loss:.4f})")

# === 2. NEOCORTEX ===
from neocortex import BidirectionalCortex, SimpleStack, PredictiveStack

def test_simplestack():
    s = SimpleStack(2, 5, 0.3)
    x = np.array([1.0, 0.0])
    out = s.predict(x)
    assert out.shape == (2,)
    print("  ✓ SimpleStack")

def test_predictivestack():
    p = PredictiveStack(2, 5, 0.3)
    x = np.array([1.0, 0.0])
    out = p.predict(x)
    assert out.shape == (2,)
    print("  ✓ PredictiveStack")

def test_bidirectional_cortex():
    bc = BidirectionalCortex(2, 8, 2, 5, 0.3)
    x = np.array([1.0, 0.0])
    out = bc.predict(x)
    assert out.shape == (2,)
    assert hasattr(bc, 'thalamus'), "Missing thalamus"
    loss = bc.update(x)
    assert loss > 0
    print("  ✓ BidirectionalCortex + Thalamus")

# === 3. HIPPOCAMPUS ===
from hippocampus import SparseHippocampus, FullHippocampus

def test_sparse_hippocampus():
    hc = SparseHippocampus(key_dim=16, value_dim=8, max_memories=100, sparsity=5)
    for i in range(50):
        k = np.random.randn(16)
        v = np.random.randn(8)
        hc.store(k, v)
    query = np.random.randn(16)
    results = hc.retrieve(query, k=3)
    assert len(results) <= 3
    val, sim, margin = hc.retrieve_top1(query)
    if val is not None:
        assert val.shape == (8,)
    print("  ✓ SparseHippocampus")

def test_full_hippocampus():
    hc = FullHippocampus(input_dim=16, hidden_dim=16, output_dim=8)
    for i in range(5):
        state = np.random.randn(16)
        target = np.random.randn(8)
        hc.encode(state, target)
    query = np.random.randn(16)
    pred = hc.retrieve(query)
    assert pred is not None
    assert pred.shape == (8,)
    print("  ✓ FullHippocampus (DG+CA3+CA1)")

# === 4. PFC ===
from language import LanguageLearner

def test_pfc():
    vocab = ['hello', '.', '<USER>', '<ASST>']
    ll = LanguageLearner(vocab, hidden_size=8, num_columns=2, use_memory=True,
                         hippocampus_type='full',
                         use_amygdala=False, use_neuromodulation=False,
                         use_workspace=False, use_cerebellum=False, use_interoception=False)
    ll.train_on_corpus([['<USER>', 'hello', '<ASST>', 'hello', '.']], epochs=3)
    from pfc import PrefrontalCortex
    pfc = PrefrontalCortex(ll)
    ll.cortex.reset_context()
    for w in ['<USER>', 'hello']:
        ll.cortex.predict(ll.one_hot(w))
    word, score, mode = pfc.plan_next_word(['<USER>', 'hello', '<ASST>'])
    assert isinstance(word, str)
    print(f"  ✓ PFC (mode={mode}, word={word})")

# === 5. AMYGDALA ===
from amygdala import Amygdala

def test_amygdala():
    amy = Amygdala(input_dim=16)
    state = np.random.randn(16)
    pe = np.random.randn(8) * 0.1
    valence = amy.slow_pathway(state, pe)
    assert -1.0 <= valence <= 1.0
    thal_in = np.random.randn(16)
    fast_val = amy.fast_pathway(thal_in)
    assert -1.0 <= fast_val <= 1.0
    mod = amy.get_consolidation_modulation(state)
    assert mod >= 1.0
    print(f"  ✓ Amygdala (valence={valence:.3f}, mod={mod:.3f})")

# === 6. NEUROMODULATION ===
from neuromodulation import Neuromodulator

def test_neuromodulation():
        nm = Neuromodulator()
        nm.update_all(reward=0.8, prediction_error=np.array([0.1]), novelty=0.5)
        state = nm.get_state()
        assert 'dopamine_phasic' in state
        assert 'serotonin' in state
        assert 'noradrenaline' in state
        assert 'acetylcholine' in state
        print(f"  ✓ Neuromodulation (DA={state['dopamine_phasic']:.3f}, 5-HT={state['serotonin']:.3f})")

# === 7. BASAL GANGLIA ===
from basal_ganglia import BasalGanglia

def test_basal_ganglia():
    bg = BasalGanglia(input_dim=16, num_actions=8)
    state = np.random.randn(16)
    q = bg.evaluate(state)
    assert q.shape == (8,)
    da = bg.reinforce(3, reward=1.0)
    assert 0 <= da <= 1.0
    print(f"  ✓ BasalGanglia (Q={q.shape}, DA={da:.3f})")

# === 8. CEREBELLUM ===
from cerebellum import Cerebellum

def test_cerebellum():
    cb = Cerebellum(input_dim=8, output_dim=8, expansion_factor=4)
    state = np.random.randn(8)
    pred = cb.forward(state)
    assert pred.shape == (8,)
    next_state = np.random.randn(8)
    loss = cb.update(state, next_state)
    assert loss >= 0
    print(f"  ✓ Cerebellum (pred={pred.shape}, loss={loss:.3f})")

# === 9. INTEROCEPTION ===
from interoception import Interoception

def test_interoception():
    intero = Interoception()
    intero.update(prediction_error=np.array([0.1]), social_interaction=True, learning_progress=0.5, novelty=0.3)
    state = intero.get_state()
    assert 'drives' in state
    assert 'mood' in state
    reward = intero.get_intrinsic_reward()
    assert 0 <= reward <= 1.0
    print(f"  ✓ Interoception (mood={state['mood']:.3f}, reward={reward:.3f})")

# === 10. GLOBAL WORKSPACE ===
from global_workspace import GlobalWorkspace

def test_global_workspace():
    ws = GlobalWorkspace(capacity=5)
    ws.propose('hello', salience=0.9, source='neocortex')
    ws.propose('world', salience=0.3, source='hippocampus')
    winner = ws.compete()
    assert winner is not None
    focus = ws.get_current_focus()
    assert focus is not None
    report = ws.report_self()
    assert report['items_in_workspace'] > 0
    print(f"  ✓ GlobalWorkspace (focus={focus['content']}, items={report['items_in_workspace']})")

# === 11. SLEEP ===
from sleep import SleepCycle, SharpWaveRipple

def test_sleep():
    from hippocampus import FullHippocampus
    from neocortex import BidirectionalCortex
    hc = FullHippocampus(input_dim=8, hidden_dim=8, output_dim=4)
    nc = BidirectionalCortex(4, 8, 4, 3, 0.1)
    swr = SharpWaveRipple(hc, nc)
    loss = swr.replay(n_steps=5)
    assert loss >= 0
    sc = SleepCycle(hc, nc)
    results = sc.sleep(nrem_steps=5, rem_steps=3)
    assert 'replay_loss' in results
    print(f"  ✓ Sleep (replay_loss={results['replay_loss']:.3f})")

# === 12. THALAMUS ===
from thalamus import Thalamus

def test_thalamus():
    th = Thalamus(input_dim=8, num_columns=3)
    x = np.random.randn(8)
    gated = th.process(x)
    assert gated.shape == (3, 8)
    th.set_arousal(0.8)
    th.attend_to(0, strength=0.9)
    print(f"  ✓ Thalamus (gated={gated.shape})")

# === 13. MASTER BRAIN ===
from master_brain import MasterBrain

def test_master_brain():
    vocab = ['hello', '.', '<USER>', '<ASST>']
    brain = MasterBrain(vocab)
    # Override hidden_size config to match actual learner
    brain.cfg.hidden_size = brain.learner.cortex.hidden_size
    brain.amygdala = Amygdala(input_dim=brain.cfg.hidden_size)
    brain.cerebellum = Cerebellum(input_dim=brain.cfg.hidden_size + len(vocab),
                                   output_dim=brain.cfg.hidden_size, expansion_factor=2)
    brain.learner.train_on_corpus(
        [['<USER>', 'hello', '<ASST>', 'hello', '.']], epochs=3)
    brain.learner.cortex.reset_context()
    brain.learner._last_area1_h23_avg = None
    result = brain.step('hello')
    assert 'probs' in result
    assert 'emotional_valence' in result
    assert 'neuromod_state' in result
    assert 'interoception' in result
    assert 'workspace_focus' in result
    print(f"  ✓ MasterBrain (valence={result['emotional_valence']:.3f})")

# === 14. INTEGRATION: full training ===
def test_full_training():
    vocab = ['hello', '.', '<USER>', '<ASST>', 'hi']
    sentences = [['<USER>', 'hello', '<ASST>', 'hello', '.'],
                 ['<USER>', 'hi', '<ASST>', 'hello', '.']]
    ll = LanguageLearner(vocab, hidden_size=12, num_columns=2, use_memory=True,
                         hippocampus_type='full',
                         use_amygdala=False, use_neuromodulation=False,
                         use_workspace=False, use_cerebellum=False, use_interoception=False)
    ll.train_on_corpus(sentences, epochs=5, blend_memory=0.0)
    gen = ll.generate('hello', max_words=3)
    assert len(gen) >= 1
    print(f"  ✓ Full training + generation ({' '.join(gen)})")

# === RUN ALL ===
def run_all():
    tests = [
        ("Neurons", test_biological_column),
        ("Neurons", test_cortical_layer),
        ("Neurons", test_cortical_area),
        ("Neocortex", test_simplestack),
        ("Neocortex", test_predictivestack),
        ("Neocortex", test_bidirectional_cortex),
        ("Hippocampus", test_sparse_hippocampus),
        ("Hippocampus", test_full_hippocampus),
        ("PFC", test_pfc),
        ("Amygdala", test_amygdala),
        ("Neuromodulation", test_neuromodulation),
        ("BasalGanglia", test_basal_ganglia),
        ("Cerebellum", test_cerebellum),
        ("Interoception", test_interoception),
        ("GlobalWorkspace", test_global_workspace),
        ("Sleep", test_sleep),
        ("Thalamus", test_thalamus),
        ("MasterBrain", test_master_brain),
        ("Integration", test_full_training),
    ]
    passed = 0
    failed = 0
    for group, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {group}/{test_fn.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0

if __name__ == '__main__':
    success = run_all()
    sys.exit(0 if success else 1)
