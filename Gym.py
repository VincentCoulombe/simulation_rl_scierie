import random
import math
import pandas as pd
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from Heuristiques import *
from utils import *
from EnvSimpy import EnvSimpy
from stable_baselines3.common.callbacks import BaseCallback


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
        self.taux_utilisations.append([self.env.now, 
                                       self.env.getTauxUtilisationLoader(), 
                                       self.env.getTauxUtilisationScierie(), 
                                       self.env.getTauxUtilisationSechoirs(),
                                       self.env.getTauxRemplissageCours(),
                                       1 - self.env.getTauxCoursBonEtat()])
    
    def _update_reward(self) -> None:
        
        qte_dans_cours, obj_qte_total, obj_proportion_inf, obj_proportion_sup, _ = self.env.getIndicateursInventaire()
        self.inds_inventaires.append([self.env.now, *qte_dans_cours, *obj_qte_total, *obj_proportion_inf, *obj_proportion_sup])
        
        diff_proportion_qte = obj_proportion_inf-obj_qte_total # Si différence positive, on a trop de container de ce type dans la cours
        respect_obj_qte_total = -sum(np.sqrt(abs(ecart)) for ecart in diff_proportion_qte if ecart>0) # Punis si les containers de trop dans la cours
        
        outside_prop_range_prenalty = [] # Garder la proportion réelle dans le range de proportion voulu
        inside_prop_range_bonus = []
        for qte_reelle, obj_prop_inf, obj_prop_sup, obj_qte_total in zip(qte_dans_cours, obj_proportion_inf, obj_proportion_sup, obj_qte_total):
            if qte_reelle > obj_prop_sup:
                outside_prop_range_prenalty.append(qte_reelle-obj_prop_sup)
            elif qte_reelle < obj_prop_inf:
                outside_prop_range_prenalty.append(obj_prop_inf-qte_reelle)
            else:
                inside_prop_range_bonus.append(5)                
        respect_obj_proportion = -sum(abs(ecart) for ecart in outside_prop_range_prenalty) + sum(inside_prop_range_bonus) # Punis si la proportion sort du range voulu et récompense sinon
        
        self.reward = respect_obj_qte_total + respect_obj_proportion + 0.25*(self.action == gestion_horaire_et_pile(self.env))
        
    def get_avg_reward(self) -> float:
        return np.array(self.rewards)[:, 1].mean()
                
    def reset(self, test:bool =False) -> np.array: 
        self.env = EnvSimpy(self.paramSimu) # Nouvelle simulation simpy       
        self.done = False
        self.info = {}
        self.rewards = []
        self.inds_inventaires = []
        self.taux_utilisations = []
        self._update_observation() 
        self.step_counter = 0
        return self.state   
    
    def step (self, action, verbose=True) -> tuple: 
        
        self.action = action
        self.done = self.env.stepSimpy(action)
        self._update_reward()
        self._update_observation()
        self.step_counter += 1
        total_steps = self.env.paramSimu["NbStepSimulation"]
        if verbose and self.step_counter in [round(0.25*total_steps), round(0.5*total_steps), round(0.75*total_steps)]:
            env_name = self.env.paramSimu["RatioSapinEpinette"]
            print(f"Environnement : {env_name} | Step : {self.step_counter}/{total_steps} | Reward : {self.reward:.2f}")
        if self.done:
            self.simu_counter += 1
            print(f"Simulation {self.simu_counter} terminée.")
            self.rewards_moyens.append(self.get_avg_reward())
            print(f"Reward moyen : {self.rewards_moyens[-1]:.2f}")
            
        self.rewards.append([self.env.now, self.reward])
        
        return self.state, self.reward, self.done, self.info
    
    def plot_progression_reward(self) -> None:            
        # Afficher la progression du reward dans la simulation
        df_rewards = pd.DataFrame(self.rewards, columns=["time", "reward"])    
        plt.plot(df_rewards["time"], df_rewards["reward"], label="reward", color="red")
        plt.title("Reward en fonction du Temps") 
        plt.xlabel("Temps")
        plt.ylabel("Reward")   
        plt.legend()    
        plt.show()   
        
    def plot_inds_inventaires(self) -> None:
        # Afficher les indicateurs d'inventaire 
        nb_type_produit = get_action_space(self.env.paramSimu)
        qte_dans_cours = [f"quantite_reelle{x}" for x in range(nb_type_produit)]
        obj_qte_total = [f"quantite_voulue{x}" for x in range(nb_type_produit)]
        obj_proportion_inf = [f"proportion_voulue_min{x}" for x in range(nb_type_produit)]
        obj_proportion_sup = [f"proportion_voulue_max{x}" for x in range(nb_type_produit)]
        df_inds_inv = pd.DataFrame(self.inds_inventaires, columns=["time", *qte_dans_cours, *obj_qte_total, *obj_proportion_inf, *obj_proportion_sup])  
        for i in range(nb_type_produit):
            plt.plot(df_inds_inv[f"quantite_reelle{i}"], label="quantitée réelle", color="blue")
            plt.plot(df_inds_inv[f"quantite_voulue{i}"], label="quantitée voulue", color="red")
            plt.plot(df_inds_inv[f"proportion_voulue_min{i}"], label="proportion voulue min", color="green")
            plt.plot(df_inds_inv[f"proportion_voulue_max{i}"], label="proportion voulue max", color="green")
            plt.title(f"Chargement de type : {i}")
            plt.xlabel("Temps")
            plt.ylabel("Nombre de chargements dans la cours")
            plt.legend()
            plt.show()
            
    def plot_taux_utilisations(self) -> None:
        # Afficher les indicateurs de taux d'utilisation
        df_taux_utilisation = pd.DataFrame(self.taux_utilisations, columns=["time", "taux_utilisation_loader", "taux_utilisation_scierie",
                                                                            "taux_utilisation_séchoir", "taux_remplissage_cours", "taux_stock_pourris"])
        plt.plot(df_taux_utilisation["time"], df_taux_utilisation["taux_utilisation_loader"], label="taux utilisation loader", color="blue")
        plt.plot(df_taux_utilisation["time"], df_taux_utilisation["taux_utilisation_scierie"], label="taux utilisation scierie", color="green")
        plt.plot(df_taux_utilisation["time"], df_taux_utilisation["taux_utilisation_séchoir"], label="taux utilisation séchoir", color="red")
        plt.plot(df_taux_utilisation["time"], df_taux_utilisation["taux_remplissage_cours"], label="utilisation de la cours au temps t", color="yellow")
        plt.plot(df_taux_utilisation["time"], df_taux_utilisation["taux_stock_pourris"], label="taux du stock pourris dans la cours", color="purple")
        plt.title("Taux d'utilisation en fonction du temps")
        plt.xlabel("Temps")
        plt.ylabel("Taux d'utilisation")
        plt.legend()
        plt.show()
    
    def train_model(self, model: PPO, nb_episode: int, save: bool=False, evaluate_every: int = 20, save_dir: str = "") -> None:
        
        os.makedirs(save_dir, exist_ok=True)
        for i in range(nb_episode):
            model.learn(total_timesteps=self.hyperparams["total_timesteps"], reset_num_timesteps=False)
            # if nb_episode % evaluate_every == 0:
            #     print(f"Indicateurs du modèle après l'épisode : {i}")
            #     # self.evaluate_model(model)
            if save:
                model.save(f"{save_dir}/episode{i}_reward_moyen{self.rewards_moyens[-1]:.2f}")
            plt.plot(list(range(self.simu_counter)), self.rewards_moyens, label="reward moyen", color="green")
            plt.title(f"Reward moyen des {self.simu_counter} épisodes")
            plt.show()
        return max(self.rewards_moyens)
               
    def evaluate_model(self, model: PPO):
        
        obs = self.reset(test=True)
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = self.step(action)
            
        self.plot_inds_inventaires()
        self.plot_taux_utilisations()
        self.plot_progression_reward()
        print(f"Moyenne Reward : {self.rewards_moyens[-1]:.2f}")
            
    def solve_w_heuristique(self, heuristique: str = "aleatoire"):
        _ = self.reset(test=True)  
        done = False  
        while not done: 
            if heuristique == "pile_la_plus_elevee":
                action = pile_la_plus_elevee(self.env)
            elif heuristique == "gestion_horaire_et_pile":
                action = gestion_horaire_et_pile(self.env)
            else:
                action = aleatoire(self.env)
            obs, _, done, _ = self.step(action) 
        self.plot_inds_inventaires()
        self.plot_taux_utilisations()
        return self.rewards_moyens[-1]
