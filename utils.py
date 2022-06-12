# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:52:21 2022

@author: ti_re
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from Gym import *
from Heuristiques import *

def train_model(env, model: PPO, nb_episode: int, save: bool=False, evaluate_every: int = 20, logs_dir: str = "") -> None:
    if logs_dir != "":
        os.makedirs(logs_dir, exist_ok=True)
    if save:
        models_dir = f"models/training_{int(time.time())}/"
        os.makedirs(models_dir, exist_ok=True)

    for i in range(nb_episode):
        env.reset()
        model.learn(total_timesteps=env.hyperparams["total_timesteps"], reset_num_timesteps=False)
        if nb_episode % evaluate_every == 0:
            print(f"Indicateurs du modèle après l'épisode : {i}")
            evaluate_model(env, model)
        if save:
            model.save(f"{models_dir}/episode{i}_reward_moyen{env.get_avg_reward():.2f}")


    plt.plot(list(range(env.simu_counter)), env.rewards_moyens, label="reward moyen", color="green")
    plt.title(f"Reward moyen des {env.simu_counter} épisodes")
    plt.show()
            
def evaluate_model(env, model: PPO) -> None:
    obs = env.reset(test=True)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        
    env.plot_inds_inventaires()
    env.plot_taux_utilisations()
    env.plot_progression_reward()
    print(f"Moyenne Reward : {env.get_avg_reward():.2f}")
        
def solve_w_heuristique(env, heuristique: str = "Aléatoire") -> None:
    _ = env.reset(test=True)  
    done = False  
    while not done: 
        if heuristique == "pile_la_plus_elevee":
            action = pile_la_plus_elevee(env.env)
        _, _, done, _ = env.step(action)      
    env.plot_inds_inventaires()
    env.plot_taux_utilisations()

def min_max_scaling(X,minPossible,maxPossible):
    
    data = np.array(X)
    data = np.where(data < minPossible,minPossible,data)
    data = np.where(data > maxPossible,maxPossible,data)
    return (data - minPossible) / (maxPossible - minPossible)
    
def get_action_space(paramSimu) : 
   
    return len(paramSimu["df_regles"])

def get_state_space(paramSimu) :
    
    #NbEspaceStateParProduits = 3 * len(paramSimu["df_produits"])
    NbDemandeVsProduit = len(paramSimu["df_regles"])
    NbParamPropEpinettesSortieSciage = 1
    return NbDemandeVsProduit + NbParamPropEpinettesSortieSciage
    
if __name__ == '__main__': 
    
    X = [-1,0,1,2,3,4,5]
    
    print(min_max_scaling(X,0,4))
    
    Y = [1,2,3,4,5,6,7]
    
    Z = X + Y
    
    print(Z)