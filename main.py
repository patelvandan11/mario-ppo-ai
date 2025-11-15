# main.py — minimal working example for gym_super_mario_bros rendering
import warnings
# Optional: silence irrelevant deprecation warnings if you want cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Compatibility shim for numpy bool8 (if still using numpy>=2.0)
import numpy as _np
if not hasattr(_np, "bool8"):
    _np.bool8 = _np.bool_

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Old -> New API wrapper (keeps compatibility with newer gym wrappers)
class OldToNewAPIWrapper(gym.Wrapper):
    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
            return obs, reward, terminated, truncated, info
        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if not (isinstance(result, tuple) and len(result) == 2):
            obs = result
            info = {}
            return obs, info
        return result

def make_mario_env():
    # Create base Mario env (do NOT pass render_mode here)
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # Wrap for API compatibility first
    env = OldToNewAPIWrapper(env)
    # Apply Joypad simplified controls
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env


# def m():
    
if __name__ == "__main__":
    env = make_mario_env()

    # Start one episode and render frames
    obs, info = env.reset()
    done = False

    try:
        while not done:
            # sample a random action (replace with model.predict(obs) when testing a model)
            action = env.action_space.sample()
            # For vectorized wrappers you may need to pass [action]; here it's a regular env
            step_result = env.step(action)
            # Our wrapper converts to new API: step returns (obs, reward, terminated, truncated, info)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # fallback for old API
                obs, reward, done, info = step_result

            # Render — call env.render() since we did not set render_mode at creation
            env.render()

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        env.close()
        print("Environment closed.")
