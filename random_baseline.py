"""Step 2 — Watch it fail (info.txt:50).

Random policy on LunarLander-v3. Prints the 8 observation numbers
so you can match them to the table in the Gymnasium docs.
"""
import gymnasium as gym

OBS_NAMES = ["x", "y", "vx", "vy", "angle", "ang_vel", "leg_L", "leg_R"]
ACTIONS = ["nothing", "left", "main", "right"]

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=42)
print("obs:", dict(zip(OBS_NAMES, [round(float(v), 3) for v in obs])))

episodes, max_steps = 3, 1000
for ep in range(episodes):
    obs, info = env.reset() if ep > 0 else (obs, info)
    total = 0.0
    for t in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if t % 100 == 0:
            print(f"ep={ep} step={t} act={ACTIONS[action]} rew={reward:+.2f} "
                  f"obs={[round(float(v), 2) for v in obs]}")
        if terminated or truncated:
            print(f"episode {ep} ended at step {t} total={total:.1f} "
                  f"terminated={terminated} truncated={truncated}")
            obs, info = env.reset()
            break
env.close()
print("Done — you should have seen tumbling/crashing. Next: train.py")
