import random
import pandas as pd
import numpy as np
import gym
from gym import spaces
from EnvSimpy import *
import matplotlib.pyplot as plt
import os
import time

from stable_baselines3 import PPO, PPO2
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

model = SAC("MlpPolicy", "Pendulum-v0", tensorboard_log="/tmp/sac/", verbose=1)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.locals['writer'].add_summary(self.env.reward, self.num_timesteps)
        return True
    
class EnvGym(gym.Env) : 
    
    def __init__(self, paramSimu: dict, nb_actions: int, state_len: int, state_min: float, state_max: float, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.paramSimu = paramSimu
        self.action_space = spaces.Discrete(nb_actions)
        self.low = np.array([state_min for _ in range(state_len)], dtype=np.float32)
        self.high = np.array([state_max for _ in range(state_len)], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        
    def generate_demand(self, obj_fin_simu: int):
        return self.env.now/self.paramSimu["DureeSimulation"]*obj_fin_simu
       
    def _update_observation(self) -> None:
        
        self.state = self.env.getState() #Méthode de la simulation, state normalisé
    
    def _update_reward(self) -> None:
        
        self.respect_inv = sum(x**2 for x in self.env.getRespectInventaire())
        self.reward = self.respect_inv
                
    def reset(self) -> np.array: 
        
        self.env = EnvSimpy(self.paramSimu) # Nouvelle simulation simpy       
        self.done = False
        self.info = {}
        self.indicateurs = []                
        return self._update_observation()    
    
    def step (self, action, log_inds: bool=False) -> tuple: 

        self.done = self.env.stepSimpy(action)
        
        self._update_reward()
        self._update_observation()
        
        # logger les indicateurs pertinents 
        if log_inds:       
            self.indicateurs.append([self.env.now, self.reward])
        
        return self.state, self.reward, self.done, self.info
    
    def boucle(self) : 
        
        state = self.reset()  
        done = False
        while not done: 
            action = ChoixLoader(self.env)
            state, _, done, _ = self.step(action)
            
    def train_model(self, nb_timestep: int, nb_episode: int, log: bool=False, save: bool=False):
        if log:
            logdir = f"logs/{int(time.time())}/"
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            model = PPO('MlpPolicy', self, verbose=1, tensorboard_log=logdir)
        else:
            model = PPO('MlpPolicy', self)
        if save:
            models_dir = f"models/{int(time.time())}/"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
             
        self.reset()
        for i in range(nb_episode):
            model.learn(total_timesteps=nb_timestep, reset_num_timesteps=False, tb_log_name="PPO")
            if save:
                model.save(f"{models_dir}/{nb_timestep*i}")
                
        print("Début de l'évaluation du modèle final (pas nécessairement le meilleur)...")
        self.evaluate_model(model)
            
    def evaluate_model(self, model):
        obs = self.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = self.step(action, log_inds=True)
        df = pd.DataFrame(self.indicateurs, columns=["time", "reward"])
        df.plot(x="time", y=["reward"])
        plt.show()
        print(f"Meilleur reward : {df['reward'].max():.2f}")
        print(f"Reward moyen : {df['reward'].mean():.2f}")
        print(f"Reward final : {df['reward'].iloc[-1]:.2f}")
