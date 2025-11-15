# configs.py — Hyperparameters & constants

class Config:
    env_name = "SuperMarioBros-1-1-v0"
    num_envs = 1
    total_episodes = 1000
    update_interval = 128
    gamma = 0.99
    lr = 2.5e-4
    clip_eps = 0.1
    epochs = 4
    batch_size = 64
    gae_lambda = 0.95
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    # Device / precision
    device = "cuda"
    dtype = "float16"   # Mixed precision
    compile_mode = "reduce-overhead"
