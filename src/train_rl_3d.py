"""
train_rl_3d.py

Train PPO on Drone3DEnv. Creates VecNormalize-wrapped training & eval envs,
saves model checkpoints, final model and VecNormalize stats, and logs to TensorBoard.

Usage:
    python src/train_rl_3d.py
"""

import os
import time
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnMaxEpisodes,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from drone_env_3d import Drone3DEnv

# ----------------------------
# Configuration / Hyperparams
# ----------------------------
DEFAULT_TOTAL_TIMESTEPS = 300_000    # you can increase later (e.g. 1_000_000)
EVAL_FREQ = 10_000                   # evaluate every n steps
N_EVAL_EPISODES = 5
CHECKPOINT_FREQ = 50_000             # save intermediate checkpoints
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TB_LOG_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "tb_logs",
    f"ppo_drone_{int(time.time())}",
)

PPO_PARAMS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log=TB_LOG_DIR,
)

# ----------------------------
# Utility helpers
# ----------------------------
def make_env_fn(demo_mode=False, randomize_static=False):
    """
    Return a function that creates the environment instance (for DummyVecEnv).

    - randomize_static=True  -> trees regenerated each reset (for training robustness)
    - randomize_static=False -> fixed trees (for evaluation / visualization)
    """
    def _init():
        env = Drone3DEnv(
            demo_mode=demo_mode,
            safety_radius=3.0,
            wind_enabled=True,
            randomize_static=randomize_static,  # <--- key change
        )
        env = Monitor(env)
        return env

    return _init


class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback to save VecNormalize stats at training end.
    """

    def __init__(self, venv: VecNormalize, save_path: str, verbose=0):
        super().__init__(verbose)
        self.venv = venv
        self.save_path = save_path

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.venv.save(self.save_path)
            if self.verbose > 0:
                print(f"[SaveVecNormalizeCallback] Saved VecNormalize to {self.save_path}")
        except Exception as e:
            print("[SaveVecNormalizeCallback] Failed to save VecNormalize:", e)


# ----------------------------
# Main training function
# ----------------------------
def train(total_timesteps=DEFAULT_TOTAL_TIMESTEPS, resume=False):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    # TRAIN env: randomize_static=True  → new tree layout each episode
    train_raw = DummyVecEnv([make_env_fn(demo_mode=False, randomize_static=True)])
    train_venv = VecNormalize(train_raw, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # EVAL env: usually keep randomize_static=False for stable metrics
    eval_raw = DummyVecEnv([make_env_fn(demo_mode=False, randomize_static=False)])
    eval_venv = VecNormalize(eval_raw, norm_obs=True, norm_reward=False, clip_obs=10.0)

    vec_path = os.path.join(SAVE_DIR, "vecnormalize.pkl")

    if resume and os.path.exists(vec_path):
        try:
            train_venv = VecNormalize.load(vec_path, train_raw)
            eval_venv = VecNormalize.load(vec_path, eval_raw)
            print("[train] Loaded VecNormalize from", vec_path)
        except Exception as e:
            print("[train] Could not load VecNormalize (proceeding fresh):", e)
            train_venv = VecNormalize(train_raw, norm_obs=True, norm_reward=True, clip_obs=10.0)
            eval_venv = VecNormalize(eval_raw, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        try:
            eval_venv.obs_rms = train_venv.obs_rms
        except Exception:
            pass

    # ----------------------------
    # Callbacks
    # ----------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // train_venv.num_envs,
        save_path=SAVE_DIR,
        name_prefix="ppo_checkpoint",
    )

    stop_callback = StopTrainingOnMaxEpisodes(
        max_episodes=999999,
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env=eval_venv,
        best_model_save_path=SAVE_DIR,
        log_path=os.path.join(SAVE_DIR, "eval_logs"),
        eval_freq=EVAL_FREQ // train_venv.num_envs,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    save_vec_cb = SaveVecNormalizeCallback(
        train_venv,
        os.path.join(SAVE_DIR, "vecnormalize.pkl"),
        verbose=1,
    )

    # ----------------------------
    # Create or load model
    # ----------------------------
    model_path = os.path.join(SAVE_DIR, "ppo_drone_final.zip")
    if resume and os.path.exists(model_path):
        print("[train] Resuming training from", model_path)
        model = PPO.load(model_path, env=train_venv)
    else:
        model = PPO("MlpPolicy", train_venv, **PPO_PARAMS)

    # ----------------------------
    # Start training
    # ----------------------------
    print("[train] Starting training for", total_timesteps, "timesteps")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, save_vec_cb, stop_callback],
        )
    finally:
        try:
            print("[train] Saving final model...")
            model.save(model_path)
            train_venv.save(vec_path)
            print("[train] Saved model and VecNormalize.")
        except Exception as e:
            print("[train] Error saving final model/vec:", e)


# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Total timesteps to train",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous model/vecnormalize if available",
    )
    args = parser.parse_args()

    train(total_timesteps=args.timesteps, resume=args.resume)
