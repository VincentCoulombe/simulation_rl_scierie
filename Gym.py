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
    
    def __init__(self, paramSimu: dict, nb_actions: int, state_len: int, state_min: float, state_max: float, hyperparams: dict, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.paramSimu = paramSimu
        self.action_space = spaces.Discrete(nb_actions)
        self.low = np.array([state_min for _ in range(state_len)], dtype=np.float32)
        self.high = np.array([state_max for _ in range(state_len)], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        self.simu_counter = 0  
        self.rewards_moyens = []
        self.hyperparams = hyperparams        
    def _update_observation(self) -> None:
        
        self.state = self.env.getState() #Méthode de la simulation, state normalisé
    
    def _update_reward(self) -> None:
        
        self.respect_inv = -sum(x**2 for x in self.env.getRespectInventaire())
        self.reward = 10*self.respect_inv
        
    def get_avg_reward(self) -> float:
        return np.array(self.indicateurs)[:, 1].mean()
                
    def reset(self) -> np.array: 
        
        self.env = EnvSimpy(self.paramSimu) # Nouvelle simulation simpy       
        self.done = False
        self.info = {}
        self.indicateurs = []
        self._update_observation() 
        return self.state   
    
    def step (self, action) -> tuple: 
        self.done = self.env.stepSimpy(action)
        self._update_reward()
        self._update_observation()
        
        if self.done:
            self.simu_counter += 1
            print(f"Épisode {self.simu_counter} terminée")
            self.rewards_moyens.append(self.get_avg_reward())
            print(f"Reward moyen : {self.get_avg_reward():.2f}")
            
        production_voulue, production_reelle = self.env.getProportions()      
        self.indicateurs.append([self.env.now, self.reward, *production_voulue, *production_reelle])
        
        return self.state, self.reward, self.done, self.info
    
    def boucle(self) : 
        
        state = self.reset()  
        done = False
        while not done: 
            action = ChoixLoader(self.env)
            state, _, done, _ = self.step(action)
            
    def train_model(self, model: PPO, nb_episode: int, save: bool=False):
        logs_dir = f"logs/logs_{int(time.time())}/"
        os.makedirs(logs_dir, exist_ok=True)
        if save:
            models_dir = f"models/training_{int(time.time())}/"
            os.makedirs(models_dir, exist_ok=True)
        
        for i in range(nb_episode):
            self.reset()
            model.learn(total_timesteps=self.hyperparams["total_timesteps"], reset_num_timesteps=False)
            print(f"Indicateurs du modèle après 'épisode : {i}")
            self.evaluate_model(model)
            if save:
                model.save(f"{models_dir}/episode{i}_reward_moyen{self.get_avg_reward():.2f}")
                
        
        plt.plot(list(range(nb_episode)), self.rewards_moyens, label="reward moyen", color="green")
        plt.title(f"Reward moyen des {nb_episode} épisodes")  
        plt.show()
               
    def evaluate_model(self, model: PPO):
        obs = self.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = self.step(action)
            
        nb_type_produit = get_action_space(self.env.paramSimu)
        columns_names_prod_voulue = [f"production_voulue_produit{x}" for x in range(nb_type_produit)]
        columns_names_prod_reelle = [f"production_reelle_produit{x}" for x in range(nb_type_produit)]
        df = pd.DataFrame(self.indicateurs, columns=["time", "reward", *columns_names_prod_voulue, *columns_names_prod_reelle])
        
        plt.plot(df["time"], df["reward"], label="reward", color="red")
        plt.title("Reward en fonction du Temps")        

        fig, axs = plt.subplots(math.ceil(nb_type_produit/2), 2, sharex=True, sharey=True, figsize=(50, 50))
        counter = 0
        for i in range(math.ceil(nb_type_produit/2)):
            for j in range(2):
                axs[i,j].plot(df[f"production_voulue_produit{counter}"], label="production_voulue", color="blue")
                axs[i,j].plot(df[f"production_reelle_produit{counter}"], label="production_reelle", color="green")
                axs[i,j].legend()
                axs[i,j].set_title(f"Produit {counter+1}")
                counter += 1
        plt.show()
        print(f"Meilleur reward : {self.get_avg_reward():.2f}")

