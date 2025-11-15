# lunar_lander_user_label.py
import argparse
import csv
import os
import time
from collections import deque
from typing import Callable, Deque, List, Tuple

import gymnasium as gym
import imageio
import numpy as np

# ---------- Detection params (same as before, tweak if needed) ----------
TAIL_WINDOW = 12
SUSTAIN_LEG_STEPS = 6
CONFIDENCE_THRESH = 0.65
MAX_VEL_Y = 0.4
MAX_VEL_X = 0.5
MAX_ANGLE = 0.25
MAX_ABS_X = 0.6
MAX_NEG_REWARD_SPIKE = -100

# ---------- Policies ----------
def random_policy(obs) -> int:
    return env_action_space.sample()

def heuristic_policy(obs) -> int:
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1, leg2 = obs
    if leg1 or leg2:
        return 0
    if angle > 0.08:
        return 3
    if angle < -0.08:
        return 1
    if vel_y < -0.6:
        return 2
    if vel_x > 0.2:
        return 1
    if vel_x < -0.2:
        return 3
    return 0

# ---------- Improved landing detection (same as before) ----------
def evaluate_landing(tail_obs: Deque[np.ndarray], tail_rewards: Deque[float]) -> Tuple[bool, float, dict]:
    details = {}
    if len(tail_obs) == 0:
        return False, 0.0, details

    final = tail_obs[-1]
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1, leg2 = final

    both_legs_last = [1.0 if (o[6] == 1.0 and o[7] == 1.0) else 0.0 for o in tail_obs]
    consec_both_legs = 0
    for v in reversed(both_legs_last):
        if v == 1.0:
            consec_both_legs += 1
        else:
            break
    details["consec_both_legs"] = consec_both_legs

    mean_vel_y = float(np.mean([o[3] for o in tail_obs]))
    mean_vel_x = float(np.mean([o[2] for o in tail_obs]))
    mean_angle = float(np.mean([o[4] for o in tail_obs]))
    details.update({"mean_vel_y": mean_vel_y, "mean_vel_x": mean_vel_x, "mean_angle": mean_angle})

    abs_x = abs(pos_x)
    details["abs_x"] = abs_x

    min_tail_reward = float(min(tail_rewards)) if tail_rewards else 0.0
    details["min_tail_reward"] = min_tail_reward

    legs_score = min(1.0, consec_both_legs / SUSTAIN_LEG_STEPS)
    vy_score = max(0.0, 1.0 - min(1.0, abs(mean_vel_y) / MAX_VEL_Y))
    vx_score = max(0.0, 1.0 - min(1.0, abs(mean_vel_x) / MAX_VEL_X))
    vel_score = (vy_score * 0.7) + (vx_score * 0.3)
    angle_score = max(0.0, 1.0 - min(1.0, abs(mean_angle) / MAX_ANGLE))
    pos_score = max(0.0, 1.0 - min(1.0, abs_x / MAX_ABS_X))

    reward_penalty = 0.0
    if min_tail_reward <= MAX_NEG_REWARD_SPIKE:
        reward_penalty = 0.6

    landing_confidence = (0.40 * legs_score + 0.30 * vel_score + 0.15 * angle_score + 0.10 * pos_score)
    landing_confidence = max(0.0, landing_confidence - reward_penalty)

    details.update({
        "legs_score": legs_score,
        "vel_score": vel_score,
        "angle_score": angle_score,
        "pos_score": pos_score,
        "reward_penalty": reward_penalty,
        "landing_confidence": landing_confidence,
    })

    landed_successfully = landing_confidence >= CONFIDENCE_THRESH
    return bool(landed_successfully), float(landing_confidence), details

# ---------- Episode runner ----------
def run_episode(env: gym.Env, policy_fn: Callable, record_frames: bool = False):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    frames = []

    tail_obs: Deque[np.ndarray] = deque(maxlen=TAIL_WINDOW)
    tail_rewards: Deque[float] = deque(maxlen=TAIL_WINDOW)

    while True:
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        tail_obs.append(next_obs)
        tail_rewards.append(reward)

        if record_frames:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception:
                pass

        obs = next_obs
        if terminated or truncated:
            break

    landed_successfully, landing_confidence, landing_details = evaluate_landing(tail_obs, tail_rewards)
    return {
        "total_reward": total_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "frames": frames,
        "landed_successfully_auto": landed_successfully,
        "landing_confidence": landing_confidence,
        "landing_details": landing_details,
    }

# ---------- CSV logging ----------
def append_csv(path: str, row: dict, fieldnames: List[str]):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LunarLander with user label override for landed")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render-mode", type=str, choices=["human", "rgb_array"], default="human")
    parser.add_argument("--policy", type=str, choices=["random", "heuristic"], default="random")
    parser.add_argument("--log-file", type=str, default="lunar_results.csv")
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--no-prompt", action="store_true", help="Do not prompt user per episode; use auto decision")
    args = parser.parse_args()

    if args.policy == "random":
        policy_fn = random_policy
    else:
        policy_fn = heuristic_policy

    env = gym.make("LunarLander-v3", render_mode=args.render_mode)
    global env_action_space
    env_action_space = env.action_space

    if args.save_gif:
        os.makedirs(args.save_gif, exist_ok=True)

    csv_fields = ["episode", "seed", "total_reward", "steps", "terminated", "truncated",
                  "landed_successfully", "landed_user_label", "landing_confidence", "timestamp"]
    print(f"Starting {args.episodes} episodes | render_mode={args.render_mode} | policy={args.policy}")

    try:
        for ep in range(1, args.episodes + 1):
            ep_seed = args.seed + ep
            env.reset(seed=ep_seed)

            record = (args.render_mode == "rgb_array") and (ep % args.render_every == 0) and (args.save_gif is not None)
            result = run_episode(env, policy_fn, record_frames=record)

            # default auto result
            auto_label = result["landed_successfully_auto"]
            conf = result["landing_confidence"]

            user_label = "auto"  # will be 'user_yes' or 'user_no' when overridden
            final_label = auto_label

            # Prompt user unless disabled
            if not args.no_prompt:
                print("\n--- Episode summary ---")
                print(f"Episode {ep} | reward={result['total_reward']:.2f} | steps={result['steps']} | "
                      f"term={result['terminated']} truncated={result['truncated']}")
                print(f"Auto-detected landed={auto_label} (confidence={conf:.3f})")
                # Show short landing details for tuning/debug
                ld = result["landing_details"]
                sample_details = {k: ld[k] for k in ["consec_both_legs", "mean_vel_y", "mean_vel_x", "mean_angle", "abs_x"] if k in ld}
                print("Details:", sample_details)
                # Prompt
                ans = input("Mark as landed? (y = yes / n = no / Enter = keep auto) >>> ").strip().lower()
                if ans == "y":
                    final_label = True
                    user_label = "user_yes"
                elif ans == "n":
                    final_label = False
                    user_label = "user_no"
                else:
                    final_label = auto_label
                    user_label = "auto"
            else:
                # No prompt; keep auto
                final_label = auto_label
                user_label = "auto"

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            row = {
                "episode": ep,
                "seed": ep_seed,
                "total_reward": result["total_reward"],
                "steps": result["steps"],
                "terminated": result["terminated"],
                "truncated": result["truncated"],
                "landed_successfully": bool(final_label),
                "landed_user_label": user_label,
                "landing_confidence": round(conf, 3),
                "timestamp": ts,
            }
            append_csv(args.log_file, row, csv_fields)

            print(f"Saved: ep={ep} landed={row['landed_successfully']} label={row['landed_user_label']} conf={row['landing_confidence']:.3f}")

            # Save gif if requested
            if record and result["frames"]:
                gif_path = os.path.join(args.save_gif, f"lunar_ep{ep:03d}.gif")
                try:
                    imageio.mimsave(gif_path, result["frames"], fps=30)
                    print(f"  -> Saved GIF: {gif_path}")
                except Exception as e:
                    print(f"  -> Failed to save GIF: {e}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        print("Done. CSV log:", os.path.abspath(args.log_file))
