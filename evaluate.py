"""Step 4 — Watch it land + record (info.txt:54).

Usage:
  python evaluate.py                        # watch 5 eps, human window
  python evaluate.py --record               # + save mp4 to videos/
  python evaluate.py --episodes 10 --model models/ppo_lunar_1M
"""
import argparse
import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO


def main(model_path: str, episodes: int, record: bool, seed: int):
    video_folder = "videos/landing"
    os.makedirs(video_folder, exist_ok=True)

    render_mode = "rgb_array" if record else "human"
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    if record:
        env = RecordVideo(env, video_folder=video_folder,
                          episode_trigger=lambda e: True,
                          disable_logger=True)

    print(f"Loading {model_path}.zip ...")
    model = PPO.load(model_path, env=env)

    totals = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        totals.append(total)
        print(f"ep {ep}: total={total:.1f} {'LAND?' if total >= 200 else ''}")

    mean = sum(totals) / len(totals)
    print(f"AVG {episodes} eps: {mean:.1f} (solved if >=200, wrapper OFF)")
    env.close()
    if record:
        print(f"Videos saved to {video_folder}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="models/ppo_lunar_1M")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--record", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.model, args.episodes, args.record, args.seed)
