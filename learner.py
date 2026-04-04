import locale
import os
import time
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
PPO_ENT_COEF = 0.03  # bumped from 0.01 to encourage re-exploration after obs change
POLICY_LR = 3e-4
CRITIC_LR = 3e-4
POLICY_LAYER_SIZES = (256, 256, 256)
CRITIC_LAYER_SIZES = (256, 256, 256)
DEFAULT_TRAIN_SEGMENT_SECONDS = 3 * 60 * 60
DEFAULT_COOLDOWN_SECONDS = 15 * 60


_DRIBBLE_OBS_BEFORE_WAYPOINT = 92   # DefaultObs with zero_padding=1
_DRIBBLE_OBS_AFTER_WAYPOINT = 96    # + 4 waypoint features (dx, dy, bearing_cos, bearing_sin)
_DRIBBLE_OBS_AFTER_CARRY_QUALITY = 97  # + 1 carry_quality feature


def migrate_dribble_waypoint_obs(checkpoint_path: str) -> None:
    """
    Expand the first layer of policy/critic from obs_size=92 to 96 (4 new waypoint features).
    New input columns are initialised to zero so the policy starts where it left off
    and gradually learns to use the waypoint signal.  Idempotent — skipped if already migrated.
    Recreates fresh Adam optimizer files so rlgym_ppo can load them without error.
    """
    import torch
    import torch.optim as optim
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF
    from rlgym_ppo.ppo.value_estimator import ValueEstimator

    migrated_any = False
    for filename in ("PPO_POLICY.pt", "PPO_VALUE_NET.pt"):
        filepath = Path(checkpoint_path) / filename
        if not filepath.exists():
            continue
        state_dict = torch.load(str(filepath), map_location="cpu", weights_only=True)
        w = state_dict.get("model.0.weight")
        if w is None or w.shape[1] != _DRIBBLE_OBS_BEFORE_WAYPOINT:
            continue  # Already migrated or unexpected shape
        n_extra = _DRIBBLE_OBS_AFTER_WAYPOINT - _DRIBBLE_OBS_BEFORE_WAYPOINT
        padding = torch.zeros(w.shape[0], n_extra, dtype=w.dtype)
        state_dict["model.0.weight"] = torch.cat([w, padding], dim=1)
        torch.save(state_dict, str(filepath))
        print(f"[obs-migration] {filename}: input {_DRIBBLE_OBS_BEFORE_WAYPOINT} → {_DRIBBLE_OBS_AFTER_WAYPOINT}")
        migrated_any = True

    if migrated_any:
        # Recreate fresh Adam optimizer files — old ones are stale after weight shape change.
        policy_state = torch.load(str(Path(checkpoint_path) / "PPO_POLICY.pt"), map_location="cpu", weights_only=True)
        action_count = policy_state["model.6.weight"].shape[0]

        policy = DiscreteFF(_DRIBBLE_OBS_AFTER_WAYPOINT, action_count, POLICY_LAYER_SIZES, "cpu")
        policy.load_state_dict(policy_state)
        policy_opt = optim.Adam(policy.parameters(), lr=POLICY_LR)
        torch.save(policy_opt.state_dict(), str(Path(checkpoint_path) / "PPO_POLICY_OPTIMIZER.pt"))

        value_net = ValueEstimator(_DRIBBLE_OBS_AFTER_WAYPOINT, CRITIC_LAYER_SIZES, "cpu")
        value_state = torch.load(str(Path(checkpoint_path) / "PPO_VALUE_NET.pt"), map_location="cpu", weights_only=True)
        value_net.load_state_dict(value_state)
        value_opt = optim.Adam(value_net.parameters(), lr=CRITIC_LR)
        torch.save(value_opt.state_dict(), str(Path(checkpoint_path) / "PPO_VALUE_NET_OPTIMIZER.pt"))

        print(f"[obs-migration] recreated fresh optimizer state files")


def migrate_dribble_carry_quality_obs(checkpoint_path: str) -> None:
    """
    Expand the first layer of policy/critic from obs_size=96 to 97 (carry_quality feature).
    Also runs migrate_dribble_waypoint_obs first so a single call handles both migrations
    when resuming from a pre-waypoint checkpoint.
    New input column is initialised to zero. Idempotent — skipped if already at 97.
    """
    migrate_dribble_waypoint_obs(checkpoint_path)

    import torch
    import torch.optim as optim
    from rlgym_ppo.ppo.discrete_policy import DiscreteFF
    from rlgym_ppo.ppo.value_estimator import ValueEstimator

    migrated_any = False
    for filename in ("PPO_POLICY.pt", "PPO_VALUE_NET.pt"):
        filepath = Path(checkpoint_path) / filename
        if not filepath.exists():
            continue
        state_dict = torch.load(str(filepath), map_location="cpu", weights_only=True)
        w = state_dict.get("model.0.weight")
        if w is None or w.shape[1] != _DRIBBLE_OBS_AFTER_WAYPOINT:
            continue  # Already migrated or unexpected shape
        padding = torch.zeros(w.shape[0], 1, dtype=w.dtype)
        state_dict["model.0.weight"] = torch.cat([w, padding], dim=1)
        torch.save(state_dict, str(filepath))
        print(f"[obs-migration] {filename}: input {_DRIBBLE_OBS_AFTER_WAYPOINT} → {_DRIBBLE_OBS_AFTER_CARRY_QUALITY}")
        migrated_any = True

    if migrated_any:
        policy_state = torch.load(str(Path(checkpoint_path) / "PPO_POLICY.pt"), map_location="cpu", weights_only=True)
        action_count = policy_state["model.6.weight"].shape[0]

        policy = DiscreteFF(_DRIBBLE_OBS_AFTER_CARRY_QUALITY, action_count, POLICY_LAYER_SIZES, "cpu")
        policy.load_state_dict(policy_state)
        policy_opt = optim.Adam(policy.parameters(), lr=POLICY_LR)
        torch.save(policy_opt.state_dict(), str(Path(checkpoint_path) / "PPO_POLICY_OPTIMIZER.pt"))

        value_net = ValueEstimator(_DRIBBLE_OBS_AFTER_CARRY_QUALITY, CRITIC_LAYER_SIZES, "cpu")
        value_state = torch.load(str(Path(checkpoint_path) / "PPO_VALUE_NET.pt"), map_location="cpu", weights_only=True)
        value_net.load_state_dict(value_state)
        value_opt = optim.Adam(value_net.parameters(), lr=CRITIC_LR)
        torch.save(value_opt.state_dict(), str(Path(checkpoint_path) / "PPO_VALUE_NET_OPTIMIZER.pt"))

        print(f"[obs-migration] recreated fresh optimizer state files")


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


def resolve_checkpoint_folder(checkpoint="latest", models_dir: Path = MODELS_DIR):
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
    candidates.extend(_numeric_checkpoint_dirs(models_dir))
    candidates.extend(
        checkpoint_dir
        for run_dir in ROOT_DIR.glob("models-*")
        if run_dir != models_dir
        for checkpoint_dir in _numeric_checkpoint_dirs(run_dir)
    )
    # Also scan the base models/ dir in case it holds legacy/public checkpoints
    base_models = ROOT_DIR / "models"
    if base_models != models_dir:
        candidates.extend(_numeric_checkpoint_dirs(base_models))

    if not candidates:
        return None

    latest_checkpoint = max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, int(path.name)),
    )
    return str(latest_checkpoint)


def get_cumulative_timesteps(checkpoint="latest"):
    checkpoint_path = resolve_checkpoint_folder(checkpoint)
    if checkpoint_path is None:
        return 0

    bookkeeping_path = Path(checkpoint_path) / "BOOK_KEEPING_VARS.json"
    if not bookkeeping_path.exists():
        return 0

    import json

    with bookkeeping_path.open("r") as handle:
        bookkeeping = json.load(handle)
    return int(bookkeeping.get("cumulative_timesteps", 0))


def _install_chunk_timer(chunk_seconds):
    if chunk_seconds is None or chunk_seconds <= 0:
        return None

    from rlgym_ppo.util.kbhit import KBHit

    original_kbhit = KBHit.kbhit
    original_getch = KBHit.getch
    chunk_deadline = time.monotonic() + chunk_seconds

    def timed_kbhit(self):
        if getattr(self, "_auto_pause_ready", False):
            return True
        if time.monotonic() >= chunk_deadline:
            self._auto_pause_ready = True
            return True
        return original_kbhit(self)

    def timed_getch(self):
        if getattr(self, "_auto_pause_ready", False):
            self._auto_pause_ready = False
            return "q"
        return original_getch(self)

    KBHit.kbhit = timed_kbhit
    KBHit.getch = timed_getch

    def restore():
        KBHit.kbhit = original_kbhit
        KBHit.getch = original_getch

    return restore


def _install_safe_kbhit():
    from rlgym_ppo.util.kbhit import KBHit

    if getattr(KBHit, "_rocket_league_safe_patch", False):
        return

    original_init = KBHit.__init__
    original_set_normal_term = KBHit.set_normal_term
    original_kbhit = KBHit.kbhit
    original_getch = KBHit.getch

    def safe_init(self):
        try:
            original_init(self)
            self._no_tty = False
        except Exception:
            self._no_tty = True
            self.fd = None
            self.new_term = None
            self.old_term = None

    def safe_set_normal_term(self):
        if getattr(self, "_no_tty", False):
            return
        return original_set_normal_term(self)

    def safe_kbhit(self):
        if getattr(self, "_no_tty", False):
            return False
        return original_kbhit(self)

    def safe_getch(self):
        if getattr(self, "_no_tty", False):
            return ""
        return original_getch(self)

    KBHit.__init__ = safe_init
    KBHit.set_normal_term = safe_set_normal_term
    KBHit.kbhit = safe_kbhit
    KBHit.getch = safe_getch
    KBHit._rocket_league_safe_patch = True


def _build_learner(
    env_create_func,
    *,
    checkpoint_load_folder,
    timestep_limit,
    save_every_ts,
    n_proc,
    log_to_wandb,
    device,
    metrics_logger,
    models_dir: Path = MODELS_DIR,
):
    from rlgym_ppo import Learner

    _install_safe_kbhit()

    return Learner(
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
        metrics_logger=metrics_logger,
        checkpoints_save_folder=str(models_dir),
        add_unix_timestamp=False,
        checkpoint_load_folder=checkpoint_load_folder,
        save_every_ts=save_every_ts,
        timestep_limit=timestep_limit,
        log_to_wandb=log_to_wandb,
        device=device,
    )


def run_learner(
    env_create_func,
    *,
    checkpoint_load_folder="latest",
    timestep_limit=DEFAULT_TIMESTEP_LIMIT,
    save_every_ts=DEFAULT_SAVE_EVERY_TS,
    n_proc=DEFAULT_N_PROC,
    log_to_wandb=False,
    device="auto",
    train_segment_seconds=DEFAULT_TRAIN_SEGMENT_SECONDS,
    cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
    metrics_logger=None,
    checkpoint_migrate_fn=None,
    models_dir: Path = MODELS_DIR,
):
    prepare_runtime_locale()

    models_dir.mkdir(exist_ok=True)

    checkpoint_path = resolve_checkpoint_folder(checkpoint_load_folder, models_dir=models_dir)
    if checkpoint_path is not None and checkpoint_migrate_fn is not None:
        checkpoint_migrate_fn(checkpoint_path)
    effective_save_every_ts = save_every_ts
    if timestep_limit is not None:
        effective_save_every_ts = min(save_every_ts, timestep_limit)

    cooldown_enabled = (
        train_segment_seconds is not None
        and train_segment_seconds > 0
        and cooldown_seconds is not None
        and cooldown_seconds > 0
    )

    while True:
        learner = _build_learner(
            env_create_func,
            checkpoint_load_folder=checkpoint_path,
            timestep_limit=timestep_limit,
            save_every_ts=effective_save_every_ts,
            n_proc=n_proc,
            log_to_wandb=log_to_wandb,
            device=device,
            metrics_logger=metrics_logger,
            models_dir=models_dir,
        )

        restore_timer = _install_chunk_timer(train_segment_seconds) if cooldown_enabled else None
        try:
            learner.learn()
        finally:
            if restore_timer is not None:
                restore_timer()

        current_timesteps = learner.agent.cumulative_timesteps
        if timestep_limit is not None and current_timesteps >= timestep_limit:
            break

        if not cooldown_enabled:
            break

        print(
            f"Cooling down for {cooldown_seconds / 60:.0f} minutes "
            f"after {train_segment_seconds / 3600:.1f} hours of training..."
        )
        time.sleep(cooldown_seconds)
        checkpoint_path = resolve_checkpoint_folder("latest", models_dir=models_dir)
