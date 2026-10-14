"""Step 5-6 — Custom reward wrappers (info.txt:56-58).

- TimeFeeWrapper: tiny -fee per step (built-in has none). Should still land.
- HoverBonusWrapper: +1/frame when upright+slow. BREAKS it (hover hack).
- FixedHoverWrapper: same signal but as improvement + time fee. FIXES it.

Use with train.py --wrapper <name>
Eval always with wrapper OFF (clean env) per info.txt:63.
"""
import gymnasium as gym
import numpy as np


class TimeFeeWrapper(gym.Wrapper):
    """Small per-step fee so waiting costs something."""
    def __init__(self, env, fee: float = 0.02):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.fee
        return obs, reward, terminated, truncated, info


class HoverBonusWrapper(gym.Wrapper):
    """Naive +bonus/frame when upright+slow. Demonstrates reward hacking:
    hovering (up to 1000 pts) beats landing, so PPO learns to hover.
    """
    def __init__(self, env, bonus: float = 1.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, ang_vel, leg_l, leg_r = obs
        upright_slow = abs(angle) < 0.2 and abs(vx) < 0.5 and abs(vy) < 0.5
        if upright_slow and not (terminated or truncated):
            reward += self.bonus
        return obs, reward, terminated, truncated, info


class FixedHoverWrapper(gym.Wrapper):
    """Fixed version: pay for GETTING BETTER (potential difference),
    not for sitting nice, + small time fee. Hovering no longer pays.
    """
    def __init__(self, env, bonus: float = 1.0, fee: float = 0.02):
        super().__init__(env)
        self.bonus = bonus
        self.fee = fee
        self._prev_potential = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self._potential(obs)
        return obs, info

    def _potential(self, obs):
        x, y, vx, vy, angle, ang_vel, leg_l, leg_r = obs
        upright_slow = abs(angle) < 0.2 and abs(vx) < 0.5 and abs(vy) < 0.5
        return self.bonus if upright_slow else 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pot = self._potential(obs) if not (terminated or truncated) else 0.0
        shaping = pot - self._prev_potential  # improvement only
        self._prev_potential = pot
        reward = reward + shaping - self.fee
        return obs, reward, terminated, truncated, info


WRAPPERS = {
    "none": lambda e: e,
    "time_fee": lambda e: TimeFeeWrapper(e),
    "hover": lambda e: HoverBonusWrapper(e),
    "fixed": lambda e: FixedHoverWrapper(e),
}
