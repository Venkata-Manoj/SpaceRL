# SpaceRL — Train an AI to Land a Rocket (LunarLander-v3 + PPO + Reward Shaping)

Learn reinforcement learning by training PPO to land Lunar Lander, then break it with reward hacking and fix it.

Spec: see `info.txt`.

## Progress (maps to info.txt)

- [x] Step 1: Setup — Python 3.11, `venv`, `gymnasium[box2d]`, `stable-baselines3`
- [ ] Step 2: Watch random policy fail (`python random_baseline.py`)
- [ ] Step 3: Train PPO 1M steps (`python train.py`)
- [ ] Step 4: Watch it land + record (`python evaluate.py --record`)
- [ ] Step 5: Custom reward wrapper (`rewards.py`)
- [ ] Step 6: Break it (hover hack) then fix it — keep both clips
- [ ] Publish: GitHub + results table + videos

## Quickstart

```powershell
cd E:\SpaceRL
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python random_baseline.py
```

## How it works (one para)

Env gives 8 numbers per frame, agent picks 1 of 4 actions (nothing/left/main/right).
PPO nudges a 2x64 MLP toward higher return. Built-in reward = improvement in
`-100*dist -100*speed -100*tilt + legs` minus fuel, +100 land / -100 crash.
Solved = avg >= 200 over 10+ evals, wrapper off.

## Repo layout

```
info.txt            original brief
random_baseline.py  Step 2 — random actions, watch it crash
train.py            Step 3 — PPO training (coming next)
evaluate.py         Step 4 — eval + video (coming next)
rewards.py          Step 5-6 — custom wrappers (coming next)
models/             saved .zip (gitignored)
videos/             mp4/gif (gitignored)
```

## References

- Gymnasium: Basic Usage + Lunar Lander + Custom Wrappers
- Stable-Baselines3: Install + Examples (PPO MlpPolicy, save/load, VecVideoRecorder)
- HF Deep RL Course Unit 1 (use v3, not v2)
