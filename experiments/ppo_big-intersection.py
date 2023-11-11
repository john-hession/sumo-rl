import os
import sys

import gymnasium as gym

os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo" # change to your SUMO_HOME path

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from stable_baselines3.ppo.ppo import PPO #changed the DQN to PPO, more resource intensive but policy converges in fewer iterations
# changed the reward_fn in the env file contained in sumo_rl folder
from sumo_rl import SumoEnvironment


env = SumoEnvironment(
    net_file="sumo-rl/nets/big-intersection/big-intersection.net.xml", # still need to add the version with john's net
    route_file="sumo-rl/nets/big-intersection/routes.rou.xml",
    single_agent=True,
    out_csv_name="outputs/big-intersection/ppo",
    use_gui=False,
    num_seconds=3000,
    yellow_time=4,
    min_green=20,
    max_green=120, # changed to max 120 per mick's rec
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
