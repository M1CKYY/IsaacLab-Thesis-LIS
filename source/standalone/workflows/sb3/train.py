# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Flexible script to train RL agent with Stable-Baselines3.
Supports both PPO and SAC by choosing the algorithm via the command line.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import torch.nn as nn  # Required for evaluating policy_kwargs

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# MODIFIED: Add an argument to choose the algorithm
parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="RL Algorithm to use.")

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
from datetime import datetime

# MODIFIED: Import both PPO and SAC
from stable_baselines3 import PPO, SAC

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper

# MODIFIED: Logic to find the correct agent config file based on the chosen algorithm
agent_cfg_entry_point = f"sb3_{args_cli.algo}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    """Train with Stable-Baselines3 agent."""
    # override configurations with non-hydra CLI arguments
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for stable-baselines
    env = Sb3VecEnvWrapper(env)

    # -- MODIFIED: Separate training and agent hyperparameters ---

    # Pop the total timesteps from the agent config since it's a trainer argument, not an agent argument
    # Default to a large number if not specified
    total_timesteps = agent_cfg.pop("n_timesteps", 1_000_000)

    # If max_iterations is provided for PPO, it overrides the total_timesteps from the YAML
    if args_cli.max_iterations and "n_steps" in agent_cfg:
        total_timesteps = args_cli.max_iterations * agent_cfg["n_steps"] * env.num_envs

    # Handle policy_kwargs dictionary evaluation
    if "policy_kwargs" in agent_cfg and isinstance(agent_cfg["policy_kwargs"], str):
        agent_cfg["policy_kwargs"] = eval(agent_cfg["policy_kwargs"])

    # --- Create Agent ---

    # Now, agent_cfg only contains parameters for the agent's constructor
    if args_cli.algo == "ppo":
        agent = PPO(env=env, verbose=1, **agent_cfg)
        print("[INFO] Created PPO agent.")
    elif args_cli.algo == "sac":
        agent = SAC(env=env, verbose=1, **agent_cfg)
        print("[INFO] Created SAC agent.")
    else:
        raise ValueError(f"Unknown algorithm: {args_cli.algo}")

    # --- Train Agent ---

    # Create a directory for logging
    log_root_path = os.path.join("logs", "sb3", f"{args_cli.task}_{args_cli.algo}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_root_path, log_dir)
    os.makedirs(log_path, exist_ok=True)

    # Train the agent using the total_timesteps variable
    agent.learn(total_timesteps=total_timesteps)

    # Save the final model
    print(f"[INFO]: Saving model to: {log_path}")
    agent.save(os.path.join(log_path, "model"))
    dump_yaml(os.path.join(log_path, "params.yaml"), agent_cfg)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()