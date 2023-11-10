import os
import sys

import gymnasium as gym

os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from stable_baselines3.ppo.ppo import PPO

from sumo_rl import SumoEnvironment


env = SumoEnvironment(
    net_file="/Users/jakehession/Desktop/Ecotech/sumo-rl/nets/big-intersection/big-intersection.net.xml",
    route_file="/Users/jakehession/Desktop/Ecotech/sumo-rl/nets/big-intersection/routes.rou.xml",
    single_agent=True,
    out_csv_name="outputs/big-intersection/dqn",
    use_gui=False,
    num_seconds=5400,
    yellow_time=4,
    min_green=5,
    max_green=60,
)

model = PPO(
    env=env,
    policy="MlpPolicy",
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
)
model.learn(total_timesteps=10000)
