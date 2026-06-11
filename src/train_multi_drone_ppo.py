"""
train_multi_drone_ppo.py
========================
PPO training script for the multi-drone coverage environment.
Integrates with Firefly Algorithm (FA) waypoints as subgoals for a
hierarchical control architecture.

Usage:
    python train_multi_drone_ppo.py --timesteps 500000
    python train_multi_drone_ppo.py --timesteps 1000000 --resume
"""

import os
import sys
import time
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------------------
# Ensure local modules are importable when running from any working directory
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from multi_drone_coverage_env import MultiDroneCoverageEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS: int = 500_000
EVAL_FREQ: int = 15_000
N_EVAL_EPISODES: int = 3
CHECKPOINT_FREQ: int = 75_000
SAVE_DIR: str = os.path.join(_SRC_DIR, "..", "models")

PPO_PARAMS: dict = {
    "learning_rate": 2.5e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "verbose": 1,
}

# Derived save paths
_MODEL_FINAL_PATH = os.path.join(SAVE_DIR, "ppo_multi_drone_final")
_VECNORM_PATH = os.path.join(SAVE_DIR, "multi_drone_vecnorm.pkl")
_VECNORM_EVAL_PATH = os.path.join(SAVE_DIR, "multi_drone_vecnorm_eval.pkl")
_BEST_MODEL_DIR = os.path.join(SAVE_DIR, "best_model")
_CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env_fn(n_drones: int = 6, randomize_static: bool = False):
    """
    Returns a *factory* (zero-argument callable) that creates and wraps a
    ``MultiDroneCoverageEnv`` inside ``Monitor`` for episode-level logging.

    Parameters
    ----------
    n_drones : int
        Number of drones in the environment.
    randomize_static : bool
        Whether to randomise static obstacle positions at each reset.
        Use ``True`` for training (better generalisation) and ``False`` for
        deterministic evaluation.
    """

    def _factory():
        env = MultiDroneCoverageEnv(
            n_drones=n_drones,
            randomize_static=randomize_static,
        )
        env = Monitor(env)
        return env

    return _factory


# ---------------------------------------------------------------------------
# Custom callback
# ---------------------------------------------------------------------------

class CoverageLogCallback(BaseCallback):
    """
    Custom SB3 callback that records domain-specific scalar metrics to
    TensorBoard every ``log_freq`` steps.

    Metrics logged
    --------------
    - ``train/coverage_pct``      – fraction of the map covered [0, 100]
    - ``train/total_collisions``  – cumulative collision count
    - ``train/total_path_length`` – cumulative path length across all drones
    - ``train/total_battery_used``– cumulative battery consumed across all drones
    """

    def __init__(self, log_freq: int = 1_000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self._episode_coverage: list[float] = []
        self._episode_collisions: list[int] = []
        self._episode_path_length: list[float] = []
        self._episode_battery: list[float] = []

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        """Called after every environment step. Returns True to continue."""
        # ``self.locals`` is populated by SB3 with the latest step data.
        infos = self.locals.get("infos", [{}])

        for info in infos:
            if "coverage_pct" in info:
                self._episode_coverage.append(float(info["coverage_pct"]))
            if "total_collisions" in info:
                self._episode_collisions.append(int(info["total_collisions"]))
            if "total_path_length" in info:
                self._episode_path_length.append(float(info["total_path_length"]))
            if "total_battery_used" in info:
                self._episode_battery.append(float(info["total_battery_used"]))

        # Log aggregated stats at the configured frequency
        if self.num_timesteps % self.log_freq == 0:
            if self._episode_coverage:
                self.logger.record(
                    "train/coverage_pct", np.mean(self._episode_coverage)
                )
            if self._episode_collisions:
                self.logger.record(
                    "train/total_collisions", np.mean(self._episode_collisions)
                )
            if self._episode_path_length:
                self.logger.record(
                    "train/total_path_length", np.mean(self._episode_path_length)
                )
            if self._episode_battery:
                self.logger.record(
                    "train/total_battery_used", np.mean(self._episode_battery)
                )

            # Reset accumulators after logging
            self._episode_coverage.clear()
            self._episode_collisions.clear()
            self._episode_path_length.clear()
            self._episode_battery.clear()

        return True  # returning False would abort training


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def train(total_timesteps: int = DEFAULT_TIMESTEPS, resume: bool = False) -> None:
    """
    Train (or resume training) a PPO agent on ``MultiDroneCoverageEnv``.

    Parameters
    ----------
    total_timesteps : int
        Total environment interaction steps for this training run.
    resume : bool
        If ``True``, attempt to load a previously saved model and
        ``VecNormalize`` statistics before continuing training.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(_BEST_MODEL_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Multi-Drone PPO Training")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Resume          : {resume}")
    print(f"  Save directory  : {os.path.abspath(SAVE_DIR)}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Build training environment
    # ------------------------------------------------------------------
    train_env = DummyVecEnv([make_env_fn(n_drones=6, randomize_static=True)])

    if resume and os.path.isfile(_VECNORM_PATH):
        print(f"[Resume] Loading VecNormalize stats from: {_VECNORM_PATH}")
        train_env = VecNormalize.load(_VECNORM_PATH, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

    # ------------------------------------------------------------------
    # Build evaluation environment (no reward normalisation for fair eval)
    # ------------------------------------------------------------------
    eval_env = DummyVecEnv([make_env_fn(n_drones=6, randomize_static=False)])

    if resume and os.path.isfile(_VECNORM_EVAL_PATH):
        eval_env = VecNormalize.load(_VECNORM_EVAL_PATH, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,  # evaluate on raw rewards
            clip_obs=10.0,
            training=False,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=_CHECKPOINT_DIR,
        name_prefix="ppo_multi_drone",
        save_vecnormalize=True,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=_BEST_MODEL_DIR,
        log_path=SAVE_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    coverage_log_cb = CoverageLogCallback(log_freq=1_000, verbose=0)

    callbacks = [checkpoint_cb, eval_cb, coverage_log_cb]

    # ------------------------------------------------------------------
    # Model creation or resumption
    # ------------------------------------------------------------------
    model_zip = _MODEL_FINAL_PATH + ".zip"

    if resume and os.path.isfile(model_zip):
        print(f"[Resume] Loading model from: {model_zip}")
        model = PPO.load(
            _MODEL_FINAL_PATH,
            env=train_env,
            **{k: v for k, v in PPO_PARAMS.items() if k != "verbose"},
            verbose=PPO_PARAMS["verbose"],
        )
    else:
        if resume:
            print(
                f"[Warning] Resume requested but no model found at '{model_zip}'. "
                "Starting fresh."
            )
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            tensorboard_log=os.path.join(SAVE_DIR, "tensorboard_logs"),
            **PPO_PARAMS,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("\n[Train] Starting model.learn() ...")
    t0 = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not resume,
        progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\n[Train] Training complete in {elapsed/60:.1f} minutes.")

    # ------------------------------------------------------------------
    # Persist final artefacts
    # ------------------------------------------------------------------
    model.save(_MODEL_FINAL_PATH)
    train_env.save(_VECNORM_PATH)
    eval_env.save(_VECNORM_EVAL_PATH)

    print(f"[Save] Model      -> {_MODEL_FINAL_PATH}.zip")
    print(f"[Save] VecNorm    -> {_VECNORM_PATH}")
    print(f"[Save] Eval norm  -> {_VECNORM_EVAL_PATH}")

    # Clean up
    train_env.close()
    eval_env.close()
    print("\n[Done] All artefacts saved successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on the multi-drone coverage environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from a previously saved model and VecNormalize checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(total_timesteps=args.timesteps, resume=args.resume)
