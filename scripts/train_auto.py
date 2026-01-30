"""Training script for local datasets that don't require CLI config overrides.

This script is specifically designed for configs that use local data directories
(like LeRobotH1LocalDataConfig) where you don't need to override config values via CLI.

Usage:
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    uv run scripts/train_auto.py pi05_h1_auto --exp-name=my_experiment --overwrite
"""

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), chunked_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, chunked_loss), grads = nnx.value_and_grad(loss_fn, has_aux=True, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "chunked_loss": chunked_loss,  # Keep per-sample losses for post-processing
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(
    config_name: str,
    exp_name: str,
    overwrite: bool = False,
    resume: bool = False,
    wandb_enabled: bool = True,
    data_dir: str | None = None,
    max_epochs: int | None = None,
    save_interval: int | None = None,
    keep_period: int | None = None,
):
    """Main training function for local datasets.
    
    Args:
        config_name: Name of the config to load from _config.get_config()
        exp_name: Experiment name for saving checkpoints and logging
        overwrite: Whether to overwrite existing checkpoints
        resume: Whether to resume from the latest checkpoint
        wandb_enabled: Whether to enable Weights & Biases logging
        data_dir: Optional path to override the data_dir in the config (for LeRobotH1LocalDataConfig or LeRobotG1LocalDataConfig)
        max_epochs: Optional number of training steps (epochs) to override num_train_steps
        save_interval: Optional save interval (in steps/epochs) to override save_interval
        keep_period: Optional keep period (in steps/epochs) to override keep_period
    """
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    
    # Load the config by name
    config = _config.get_config(config_name)
    
    # If data_dir is provided, override in the config
    if data_dir is not None:
        if isinstance(config.data, _config.LeRobotH1LocalDataConfig) or isinstance(config.data, _config.LeRobotG1LocalDataConfig):
            data_updates = {}
            if data_dir is not None:
                data_updates['data_dir'] = data_dir
                logging.info(f"Overriding data_dir to: {data_dir}")
            
            config = dataclasses.replace(
                config,
                data=dataclasses.replace(config.data, **data_updates)
            )
    
    # Override training parameters if provided
    if max_epochs is not None:
        config = dataclasses.replace(config, num_train_steps=max_epochs)
        logging.info(f"Overriding num_train_steps (max_epochs) to: {max_epochs}")
    
    if save_interval is not None:
        config = dataclasses.replace(config, save_interval=save_interval)
        logging.info(f"Overriding save_interval to: {save_interval}")
    
    if keep_period is not None:
        config = dataclasses.replace(config, keep_period=keep_period)
        logging.info(f"Overriding keep_period to: {keep_period}")
    
    # Override exp_name and flags
    config = dataclasses.replace(
        config,
        exp_name=exp_name,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=wandb_enabled,
    )
    
    logging.info(f"Loaded config: {config.name}")
    logging.info(f"Experiment name: {config.exp_name}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch_tuple = next(data_iter)
    # Handle both 2-tuple (observation, actions) and 3-tuple (observation, actions, metadata)
    if len(batch_tuple) == 3:
        batch = (batch_tuple[0], batch_tuple[1])  # (observation, actions)
    else:
        batch = batch_tuple
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    # batch is (observation, actions) tuple
    observation_for_logging = batch[0]
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in observation_for_logging.images.values()], axis=1))
        for i in range(min(5, len(next(iter(observation_for_logging.images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    logged_debug = False
    for step in pbar:
        # Get next batch and extract metadata
        batch_tuple = next(data_iter)
        if len(batch_tuple) == 3:
            observation, actions, metadata = batch_tuple
            batch = (observation, actions)
            advantage_labels_raw = metadata.get("advantage_label", None)
        else:
            observation, actions = batch_tuple
            batch = batch_tuple
            advantage_labels_raw = None
        
        # Debug logging on first step
        if not logged_debug:
            logging.info(f"Debug: advantage_labels_raw = {advantage_labels_raw is not None}")
            if advantage_labels_raw is not None:
                logging.info(f"Debug: advantage_labels shape/len: {advantage_labels_raw.shape if hasattr(advantage_labels_raw, 'shape') else len(advantage_labels_raw) if hasattr(advantage_labels_raw, '__len__') else 'scalar'}")
            logged_debug = True
        
        if advantage_labels_raw is not None:
            advantage_labels_raw = np.asarray(advantage_labels_raw)
            batch_size = len(advantage_labels_raw) if hasattr(advantage_labels_raw, '__len__') else 1
            
            # Handle None values (dropped samples) - treat them as neither True nor False
            advantage_true_mask = np.zeros(batch_size, dtype=bool)
            advantage_false_mask = np.zeros(batch_size, dtype=bool)
            
            for i, label in enumerate(advantage_labels_raw):
                if label is None:
                    # Dropped sample - don't include in either mask
                    continue
                elif label:
                    advantage_true_mask[i] = True
                else:
                    advantage_false_mask[i] = True
        else:
            # No advantage labels available
            batch_size = next(iter(observation.images.values())).shape[0]
            advantage_true_mask = np.zeros(batch_size, dtype=bool)
            advantage_false_mask = np.zeros(batch_size, dtype=bool)
        
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        
        # Post-process info to compute advantage-specific losses
        chunked_loss = jax.device_get(info["chunked_loss"])
        
        # Compute separate losses for advantage=True and advantage=False
        if advantage_true_mask.any():
            loss_adv_true = np.mean(chunked_loss[advantage_true_mask])
            count_adv_true = int(advantage_true_mask.sum())
        else:
            loss_adv_true = np.nan
            count_adv_true = 0
        
        if advantage_false_mask.any():
            loss_adv_false = np.mean(chunked_loss[advantage_false_mask])
            count_adv_false = int(advantage_false_mask.sum())
        else:
            loss_adv_false = np.nan
            count_adv_false = 0
        
        # Add advantage-specific metrics to info
        info["loss_adv_true"] = loss_adv_true
        info["loss_adv_false"] = loss_adv_false
        info["count_adv_true"] = count_adv_true
        info["count_adv_false"] = count_adv_false
        
        # Remove chunked_loss from info (too large to log)
        del info["chunked_loss"]
        
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            # Use nanmean for advantage-specific losses (they may have NaN values)
            reduced_info = jax.device_get(jax.tree.map(
                lambda x: jnp.nanmean(x) if x.dtype in (jnp.float32, jnp.float64) else jnp.mean(x), 
                stacked_infos
            ))
            
            # Create info string with advantage-specific losses
            info_str_parts = []
            for k, v in reduced_info.items():
                if k.startswith("count_"):
                    info_str_parts.append(f"{k}={v:.0f}")
                else:
                    info_str_parts.append(f"{k}={v:.4f}")
            info_str = ", ".join(info_str_parts)
            
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    tyro.cli(main)

