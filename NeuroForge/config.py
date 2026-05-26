"""Central configuration for all brain modules."""

class Config:
    # --- Neocortex ---
    hidden_size = 48
    num_columns = 5
    learning_rate = 0.005
    context_buffer = 6
    lateral_k = 0
    l23_activation = 'relu'
    predictive_coding = True

    # --- Hippocampus ---
    hippocampus_type = 'full'
    granule_cells = 500
    ca3_cells = 200
    hc_learning_rate = 0.01
    max_memories = 2000

    # --- Prefrontal Cortex ---
    goal_weight = 0.5
    reward_weight = 0.2
    mem_confidence = 0.6
    period_penalty = 0.3

    # --- Thalamus ---
    arousal = 0.5
    trn_base_inhibition = 0.1

    # --- Amygdala ---
    amygdala_lr = 0.1
    valence_decay = 0.9

    # --- Neuromodulation ---
    dopamine_lr = 0.1
    serotonin_base = 0.5
    noradrenaline_base = 0.3
    acetylcholine_base = 0.3

    # --- Basal Ganglia ---
    bg_learning_rate = 0.01
    bg_discount = 0.9

    # --- Cerebellum ---
    cerebellum_lr = 0.01
    granule_expansion = 8

    # --- Sleep ---
    nrem_steps = 50
    rem_steps = 20
    downscale_factor = 0.98

    # --- Global Workspace ---
    workspace_capacity = 10
    broadcast_decay = 0.8

    # --- Training ---
    epochs = 60
    blend_memory = 0.15
    replay_every = 10
    replay_steps = 15
    sleep_every = 10
