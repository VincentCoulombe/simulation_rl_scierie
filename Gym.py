import random
import math
import pandas as pd
import numpy as np
import gym
from gym import spaces
from EnvSimpy import *
import matplotlib.pyplot as plt
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from utils import *

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
        
        self.respect_inv = -sum(x**2 for x in self.env.getRespectInventaire())
        self.reward = self.respect_inv
                
    def reset(self) -> np.array: 
        
        self.env = EnvSimpy(self.paramSimu) # Nouvelle simulation simpy       
        self.done = False
        self.info = {}
        self.indicateurs = []  
        self._update_observation()         
        return self.state   
    
    def step (self, action, log_inds: bool=False) -> tuple: 

        self.done = self.env.stepSimpy(action)
        
        self._update_reward()
        self._update_observation()
        
        # logger les indicateurs pertinents 
        if log_inds: 
            production_voulue, production_reelle = self.env.getProportions()      
            self.indicateurs.append([self.env.now, self.reward, *production_voulue, *production_reelle])
        
        return self.state, self.reward, self.done, self.info
    
    def boucle(self) : 
        
        state = self.reset()  
        done = False
        while not done: 
            action = ChoixLoader(self.env)
            state, _, done, _ = self.step(action)
            
    def train_model(self, nb_timestep: int, nb_episode: int, log: bool=True, save: bool=False):
        if log:
            logdir = f"logs/{int(time.time())}/"
            os.makedirs(logdir, exist_ok=True)
            model = PPO('MlpPolicy', self, verbose=1, tensorboard_log=logdir)
        else:
            model = PPO('MlpPolicy', self)
        if save:
            models_dir = f"models/{int(time.time())}/"
            os.makedirs(models_dir, exist_ok=True)
             
        self.reset()
        for i in range(nb_episode):
            if log:
                model.learn(total_timesteps=nb_timestep, reset_num_timesteps=False, tb_log_name="PPO")
            else:
                model.learn(total_timesteps=nb_timestep, reset_num_timesteps=False)
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
            
        nb_type_produit = get_action_space(self.env.paramSimu)
        columns_names_prod_voulue = [f"production_voulue_produit{x}" for x in range(nb_type_produit)]
        columns_names_prod_reelle = [f"production_reelle_produit{x}" for x in range(nb_type_produit)]
        df = pd.DataFrame(self.indicateurs, columns=["time", "reward", *columns_names_prod_voulue, *columns_names_prod_reelle])
        
        plt.plot(df["time"], df["reward"], label="reward", color="red")
        plt.title("Reward en fonction du Temps")        

        fig, axs = plt.subplots(math.ceil(nb_type_produit/2), 2, sharex=True, sharey=True)
        counter = 0
        for i in range(math.ceil(nb_type_produit/2)):
            for j in range(2):
                axs[i,j].plot(df[f"production_voulue_produit{counter}"], label="production_voulue", color="blue")
                axs[i,j].plot(df[f"production_reelle_produit{counter}"], label="production_reelle", color="green")
                axs[i,j].legend()
                axs[i,j].set_title(f"Produit {counter+1}")
                counter += 1
        plt.show()
        print(f"total production voulue : {df[columns_names_prod_voulue].iloc[-1].sum():.2f}")
        print(f"total production réelle : {df[columns_names_prod_reelle].max().sum():.2f}")
        print(f"Meilleur reward : {df['reward'].max():.2f}")
        print(f"Reward moyen : {df['reward'].mean():.2f}")
        print(f"Reward final : {df['reward'].iloc[-1]:.2f}")
