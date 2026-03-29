import locale
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_TIMESTEP_LIMIT = 10_000_000
DEFAULT_SAVE_EVERY_TS = 50_000
DEFAULT_N_PROC = 4
TS_PER_ITERATION = 10_000
EXP_BUFFER_SIZE = 30_000
PPO_BATCH_SIZE = 10_000
PPO_MINIBATCH_SIZE = 5_000
PPO_EPOCHS = 3
PPO_ENT_COEF = 0.01
POLICY_LR = 3e-4
CRITIC_LR = 3e-4
POLICY_LAYER_SIZES = (256, 256, 256)
CRITIC_LAYER_SIZES = (256, 256, 256)


def prepare_runtime_locale():
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        os.environ["LC_ALL"] = "C"
        os.environ["LANG"] = "C"
        locale.setlocale(locale.LC_ALL, "")


def _numeric_checkpoint_dirs(parent: Path):
    if not parent.exists() or not parent.is_dir():
        return []

    return [
        child for child in parent.iterdir()
        if child.is_dir()
        and child.name.isdigit()
        and (child / "BOOK_KEEPING_VARS.json").exists()
    ]


def _resolve_checkpoint_parent(parent: Path):
    numeric_dirs = _numeric_checkpoint_dirs(parent)
    if not numeric_dirs:
        raise FileNotFoundError(f"No checkpoint folders found under {parent}")
    return str(max(numeric_dirs, key=lambda path: int(path.name)))


def resolve_checkpoint_folder(checkpoint="latest"):
    if checkpoint is None:
        return None

    if checkpoint != "latest":
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT_DIR / checkpoint_path

        if (checkpoint_path / "BOOK_KEEPING_VARS.json").exists():
            return str(checkpoint_path)

        if checkpoint_path.is_dir():
            return _resolve_checkpoint_parent(checkpoint_path)

        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    candidates = []
    candidates.extend(_numeric_checkpoint_dirs(MODELS_DIR))
    candidates.extend(
        checkpoint_dir
        for run_dir in ROOT_DIR.glob("models-*")
        for checkpoint_dir in _numeric_checkpoint_dirs(run_dir)
    )

    if not candidates:
        return None

    latest_checkpoint = max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, int(path.name)),
    )
    return str(latest_checkpoint)


def run_learner(
    env_create_func,
    *,
    checkpoint_load_folder="latest",
    timestep_limit=DEFAULT_TIMESTEP_LIMIT,
    save_every_ts=DEFAULT_SAVE_EVERY_TS,
    n_proc=DEFAULT_N_PROC,
    log_to_wandb=False,
    device="auto",
):
    prepare_runtime_locale()
    from rlgym_ppo import Learner

    MODELS_DIR.mkdir(exist_ok=True)

    checkpoint_path = resolve_checkpoint_folder(checkpoint_load_folder)
    effective_save_every_ts = save_every_ts
    if timestep_limit is not None:
        effective_save_every_ts = min(save_every_ts, timestep_limit)

    learner = Learner(
        env_create_func,
        n_proc=n_proc,
        min_inference_size=n_proc,
        ts_per_iteration=TS_PER_ITERATION,
        exp_buffer_size=EXP_BUFFER_SIZE,
        ppo_batch_size=PPO_BATCH_SIZE,
        ppo_minibatch_size=PPO_MINIBATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        ppo_ent_coef=PPO_ENT_COEF,
        policy_lr=POLICY_LR,
        critic_lr=CRITIC_LR,
        policy_layer_sizes=POLICY_LAYER_SIZES,
        critic_layer_sizes=CRITIC_LAYER_SIZES,
        standardize_returns=True,
        standardize_obs=False,
        metrics_logger=None,
        checkpoints_save_folder=str(MODELS_DIR),
        add_unix_timestamp=False,
        checkpoint_load_folder=checkpoint_path,
        save_every_ts=effective_save_every_ts,
        timestep_limit=timestep_limit,
        log_to_wandb=log_to_wandb,
        device=device,
    )
    learner.learn()
