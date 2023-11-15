import os
import sys

os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"

import sumo_rl

if __name__ == "__main__":
    ray.init()

    env_name = "4x4grid"

    register_env(
        env_name,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file="nets/4x4-Lucas/4x4.net.xml",
                route_file="nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
                out_csv_name="~/ray_results/4x4grid/PPO",
                use_gui=False,
                num_seconds=10000,
            )
        ),
    )

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 10000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
