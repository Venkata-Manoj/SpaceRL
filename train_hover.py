"""Quick test: hover wrapper (reward hacking). Trains PPO with naive +1/frame when upright+slow.

Expected: PPO learns to hover (up to 1000 pts from the bonus) instead of landing.
Result: avg score may go up but landing rarely happens. Keep both clips as your 'break then fix' story.
"""
import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from rewards import HoverBonusWrapper


def main(timesteps: int, seed: int, model_path: str, wrapper: str = "hover"):
    from rewards import WRAPPERS
    env = gym.make("LunarLander-v3")
    env = WRAPPERS[wrapper](env)
    env = Monitor(env)

    model = PPO("MlpPolicy", env, verbose=1, seed=seed,
                tensorboard_log="./logs/")
    print(f"Training with {wrapper} wrapper for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, log_interval=10)
    model.save(model_path)
    print(f"Saved to {model_path}.zip")

    # Eval with wrapper OFF (per info.txt:63 — trust one run is bad)
    eval_env = gym.make("LunarLander-v3")
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10,
                                    deterministic=True)
    print(f"EVAL 10 eps: mean={mean_r:.1f} +/- {std_r:.1f} "
          f"(solved if >=200, wrapper OFF)")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default="models/ppo_hover")
    p.add_argument("--wrapper", type=str, default="hover",
                   choices=["none", "hover"])
    args = p.parse_args()
    main(args.timesteps, args.seed, args.model, args.wrapper)