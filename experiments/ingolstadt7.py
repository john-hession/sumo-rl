import os
import shutil
import subprocess

import numpy as np
import supersuit as ss
import traci
from pyvirtualdisplay.smartdisplay import SmartDisplay
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import trange

os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"


#os.environ["LIBSUMO_AS_TRACI"] = '1'

import sumo_rl
import pandas


RESOLUTION = (3200, 1800)

env = sumo_rl.ingolstadt7(use_gui=True, out_csv_name="outputs/ingolstadt7/ppo_test", virtual_display=RESOLUTION, render_mode = 'human')

max_time = env.unwrapped.env.sim_max_time
delta_time = env.unwrapped.env.delta_time

print("Environment created")


env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
env = VecMonitor(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=3,
    gamma=0.95,
    n_steps=256,
    ent_coef=0.0905168,
    learning_rate=0.002,
    vf_coef=0.042202,
    max_grad_norm=0.9,
    gae_lambda=0.99,
    n_epochs=3,
    clip_range=0.3,
    batch_size=512,
    tensorboard_log="./logs/ingolstadt7/ppo_test",
)

class TensorboardCallback(BaseCallback, EnvironmentError):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # how to get pressue, avg wait, etc.??
        # all in the traffic signal file, tried importing fns but can't access the env's TrafficSignal object. 
        # maybe use accumulate_rewards for multiagent??
        # gotta look in the PZ env, maybe write a method to get the signal states?
        # 
        pressure = 2
        self.logger.record("pressure", pressure)
        return True

print("Starting training")
model.learn(total_timesteps=5000, callback=TensorboardCallback(env))

print("Training finished. Starting evaluation")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)

print(mean_reward)
print(std_reward)

# Maximum number of steps before reset, +1 because I'm scared of OBOE
print("Starting rendering")
num_steps = (max_time // delta_time) + 1

obs = env.reset()

if os.path.exists("temp"):
    shutil.rmtree("temp")

os.mkdir("temp")
# img = disp.grab()
# img.save(f"temp/img0.jpg")
obs_lst, reward_lst, done_lst, info_lst = [], [], [], []

img = env.render()
for t in trange(num_steps):
    actions, _ = model.predict(obs, state=None, deterministic=False)
    obs, reward, done, info = env.step(actions)
    obs_lst.append(obs)
    reward_lst.append(reward)
    done_lst.append(done)
    info_lst.append(info)
    # print(obs, reward, done, info)
    # img = env.render()
    # img.save(f"temp/img{t}.jpg")

#subprocess.run(["ffmpeg", "-y", "-framerate", "5", "-i", "temp/img%d.jpg", "output.mp4"])
model.save("model.zip")
#df = pandas.DataFrame(data={"obs": obs_lst, "reward": reward_lst, "done": done_lst, "info": info_lst})
#df.to_csv('output.csv')
print("All done, cleaning up")
shutil.rmtree("temp")
env.close()