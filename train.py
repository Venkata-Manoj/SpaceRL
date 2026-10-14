"""Step 3 — Train PPO 1M steps (info.txt:53).

Watches ep_rew_mean climb from negative toward +200 (solved).
Saves to models/ppo_lunar_1M.zip
Usage:
  python train.py               # full 1M
  python train.py --timesteps 200000   # quick test
"""
import argparse
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


def main(timesteps: int, seed: int, model_path: str):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    env = gym.make("LunarLander-v3")
    env = Monitor(env)

    model = PPO("MlpPolicy", env, verbose=1, seed=seed,
                tensorboard_log="./logs/")
    ckpt = CheckpointCallback(save_freq=100_000, save_path="models/checkpoints/",
                              name_prefix="ppo_lunar")
    print(f"Training PPO for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, log_interval=10, callback=ckpt)
    model.save(model_path)
    print(f"Saved to {model_path}.zip")

    # Quick eval, wrapper off (clean env), 10 episodes — solved = avg >= 200
    eval_env = gym.make("LunarLander-v3")
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10,
                                    deterministic=True)
    print(f"EVAL 10 eps: mean={mean_r:.1f} +/- {std_r:.1f} "
          f"(solved if >=200)")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default="models/ppo_lunar_1M")
    args = p.parse_args()
    main(args.timesteps, args.seed, args.model)
