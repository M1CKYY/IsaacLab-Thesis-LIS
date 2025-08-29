# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.
This script has been modified to manually instantiate the agent (PPO or SAC)
to bypass the limitations of the generic skrl Runner.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
# MODIFIED: Add "SAC" to the list of supported algorithms
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "SAC", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

# MODIFIED: Import agents and trainer directly
if args_cli.ml_framework.startswith("torch"):
    from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
    from skrl.memories.torch import RandomMemory
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.sac import SAC
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils.model_instantiators import torch as model_instantiator
    from skrl.resources.schedulers.torch import KLAdaptiveLR
elif args_cli.ml_framework.startswith("jax"):
    # Jax imports would go here if needed
    raise NotImplementedError("This modified script currently only supports torch.")

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

# config shortcuts
algorithm = args_cli.algorithm.lower()
# MODIFIED: Treat "sac" like "ppo" for finding the generic config entry point
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo", "sac"] else f"skrl_{algorithm}_cfg_entry_point"



@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # MODIFIED: Make total timesteps calculation specific to on-policy algorithms
    # if args_cli.max_iterations and algorithm in ["ppo", "ippo", "mappo"]:
    #     agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # MODIFIED: convert to single-agent instance for SAC as well
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo", "sac"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # -- MODIFIED: Manually instantiate models, memory, and agent --

    device = env.device

    # instantiate models using the skrl model instantiator
    models = {}
    for model_name, model_cfg in agent_cfg["models"].items():
        if model_name == "policy" and agent_cfg["agent"]["class"] == "SAC":
            models["policy"] = model_instantiator.gaussian_model(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                **model_cfg)
        elif model_name in ["critic_1", "critic_2"] and agent_cfg["agent"]["class"] == "SAC":
            models[model_name] = model_instantiator.deterministic_model(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                **model_cfg)
        elif model_name == "policy" and agent_cfg["agent"]["class"] == "PPO":
            models["policy"] = model_instantiator.gaussian_model(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                **model_cfg)
        elif model_name == "value" and agent_cfg["agent"]["class"] == "PPO":
            models["value"] = model_instantiator.deterministic_model(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                **model_cfg)

    # instantiate memory
    memory_cfg = agent_cfg["memory"]
    memory = RandomMemory(memory_size=memory_cfg["memory_size"],
                          num_envs=env.num_envs,
                          device=device)

    # instantiate agent
    agent_class_name = agent_cfg["agent"].pop("class")  # pop class from config
    if agent_class_name == "PPO":
        agent = PPO(models=models,
                    memory=memory,
                    cfg=agent_cfg["agent"],
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)
    elif agent_class_name == "SAC":
        agent = SAC(models=models,
                    memory=memory,
                    cfg=agent_cfg["agent"],
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)
    else:
        raise ValueError(f"Unknown agent class: {agent_class_name}")

    if isinstance(agent_cfg.get("learning_rate_scheduler"), str):
        if agent_cfg["learning_rate_scheduler"] == "KLAdaptiveLR":
            # Replace the string with the actual imported class
            agent_cfg["learning_rate_scheduler"] = KLAdaptiveLR
        else:
            # Handle other schedulers if needed, or raise an error
            raise ValueError(f"Unknown scheduler: {agent_cfg['learning_rate_scheduler']}")

    # -- MODIFIED: Instantiate and run the SequentialTrainer directly --

    # configure and instantiate the trainer
    trainer_cfg = agent_cfg["trainer"]
    trainer_cfg["close_environment_at_exit"] = False
    trainer = SequentialTrainer(env=env,
                                agents=agent,
                                cfg=trainer_cfg)
    # run training
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()