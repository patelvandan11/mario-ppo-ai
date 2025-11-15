# # agent.py — PPO agent definition
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from configs import Config

# class ActorCritic(nn.Module):
#     def __init__(self, action_dim):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(4, 32, 8, stride=4),   # 4 stacked frames
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, stride=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#         self.fc = nn.Linear(7 * 7 * 64, 512)
#         self.actor = nn.Linear(512, action_dim)
#         self.critic = nn.Linear(512, 1)

#     def forward(self, x):
#         x = x / 255.0
#         x = self.features(x)
#         x = F.relu(self.fc(x))
#         return self.actor(x), self.critic(x)

# class PPOAgent:
#     def __init__(self, action_dim):
#         self.config = Config()
#         self.device = torch.device(self.config.device)
#         self.policy = ActorCritic(action_dim).to(self.device)
#         self.policy = torch.compile(self.policy, mode=self.config.compile_mode)
#         self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)
#         self.scaler = torch.cuda.amp.GradScaler()

#     def act(self, state):
#         with torch.no_grad():
#             logits, value = self.policy(state)
#             dist = torch.distributions.Categorical(logits=logits)
#             action = dist.sample()
#         return action.item(), dist.log_prob(action), value

#     def evaluate(self, states, actions):
#         logits, values = self.policy(states)
#         dist = torch.distributions.Categorical(logits=logits)
#         log_probs = dist.log_prob(actions)
#         entropy = dist.entropy().mean()
#         return log_probs, values.squeeze(), entropy
# import gymnasium as gym  # Note: 'gymnasium' not 'gym'

# # Environment creation with render mode specified upfront
# env = gym.make("LunarLander-v3", render_mode="human")

# # Reset with seed parameter
# observation, info = env.reset(seed=14, options={})

# # Training loop with terminated/truncated distinction
# done = False
# while not done:
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     # Episode ends if either terminated OR truncated
#     done = terminated or truncated

# env.close()


# lunar_lander_extended.py
import argparse
import csv
import os
import time
from typing import Callable, List, Tuple

import gymnasium as gym
import imageio
import numpy as np

# --------- Policies ---------
def random_policy(obs) -> int:
    """Return a random action from the action space."""
    return env_action_space.sample()

def heuristic_policy(obs) -> int:
    """
    Simple heuristic for LunarLander:
    obs layout: [pos_x, pos_y, vel_x, vel_y, angle, angular_velocity, leg1_contact, leg2_contact]
    Actions: 0 = do nothing, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine
    This is a crude controller — useful for demonstration / testing.
    """
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1, leg2 = obs
    # If either leg has contact, do nothing (let it settle)
    if leg1 or leg2:
        return 0

    # Try to stabilize angle: if tilt left (angle > 0) fire right engine (action 3)
    if angle > 0.08:
        return 3  # fire right engine
    if angle < -0.08:
        return 1  # fire left engine

    # If falling too fast, use main engine
    if vel_y < -0.6:
        return 2  # main engine

    # If horizontal velocity large, try to correct by small side engines
    if vel_x > 0.2:
        return 1  # fire left engine to slow rightward movement
    if vel_x < -0.2:
        return 3  # fire right engine to slow leftward movement

    return 0  # do nothing

# --------- Episode runner ---------
def run_episode(env: gym.Env,
                policy_fn: Callable,
                record_frames: bool = False) -> Tuple[float, int, bool, bool, List[np.ndarray]]:
    """
    Runs a single episode. Returns (total_reward, steps, terminated, truncated, frames)
    If record_frames is True, collects frames from env.render() (requires rgb_array render_mode).
    """
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    frames = []

    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Capture frame if requested and env supports rgb_array
        if record_frames:
            try:
                frame = env.render()  # returns RGB array if render_mode="rgb_array"
                if frame is not None:
                    frames.append(frame)
            except Exception:
                # env.render() may raise if render mode isn't rgb_array; ignore gracefully
                pass

        if terminated or truncated:
            break

    return total_reward, steps, terminated, truncated, frames

# --------- CSV logging util ---------
def append_csv(path: str, row: dict, fieldnames: List[str]):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# --------- Main runner / CLI ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced LunarLander-v3 runner")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for environment")
    parser.add_argument("--render-mode", type=str, choices=["human", "rgb_array"], default="human",
                        help="Render mode for the environment. Use rgb_array if you want to save GIFs.")
    parser.add_argument("--policy", type=str, choices=["random", "heuristic"], default="heuristic",
                        help="Policy to use for actions.")
    parser.add_argument("--log-file", type=str, default="lunar_results.csv",
                        help="CSV file to append per-episode statistics.")
    parser.add_argument("--save-gif", type=str, default=None,
                        help="If set, directory path where episode GIFs will be saved (requires rgb_array render_mode).")
    parser.add_argument("--render-every", type=int, default=1,
                        help="Render (or record frames) every N episodes. 1 = every episode.")
    args = parser.parse_args()

    # Select policy function
    if args.policy == "random":
        policy_fn = random_policy
    else:
        policy_fn = heuristic_policy

    # Create environment with chosen render mode
    env = gym.make("LunarLander-v3", render_mode=args.render_mode)
    # Expose action space globally for the policy (random_policy uses it).
    global env_action_space
    env_action_space = env.action_space

    # Ensure GIF directory exists if requested
    if args.save_gif:
        os.makedirs(args.save_gif, exist_ok=True)

    csv_fields = ["episode", "seed", "total_reward", "steps", "terminated", "truncated", "policy", "timestamp"]
    print(f"Starting {args.episodes} episodes | render_mode={args.render_mode} | policy={args.policy}")

    try:
        for ep in range(1, args.episodes + 1):
            ep_seed = args.seed + ep  # different seed per episode for variety
            # Reset with seed for reproducibility (Gymnasium reset handles seeding)
            # Note: passing seed to reset inside run_episode would require modifying it; we set global seed before episode
            env.reset(seed=ep_seed)

            record = (args.render_mode == "rgb_array") and (ep % args.render_every == 0) and (args.save_gif is not None)

            total_reward, steps, terminated, truncated, frames = run_episode(env, policy_fn, record_frames=record)

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            row = {
                "episode": ep,
                "seed": ep_seed,
                "total_reward": total_reward,
                "steps": steps,
                "terminated": terminated,
                "truncated": truncated,
                "policy": args.policy,
                "timestamp": ts,
            }
            append_csv(args.log_file, row, csv_fields)

            print(f"Episode {ep:03d} | reward={total_reward:.2f} | steps={steps} | "
                  f"term={terminated} truncated={truncated} | seed={ep_seed}")

            # Save GIF if requested and frames were captured
            if record and frames:
                gif_path = os.path.join(args.save_gif, f"lunar_ep{ep:03d}.gif")
                try:
                    imageio.mimsave(gif_path, frames, fps=30)
                    print(f"  -> Saved GIF: {gif_path}")
                except Exception as e:
                    print(f"  -> Failed to save GIF: {e}")

    except KeyboardInterrupt:
        print("Interrupted by user — closing environment.")
    finally:
        env.close()
        print("Environment closed. CSV log:", os.path.abspath(args.log_file))
