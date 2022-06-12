# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 08:29:43 2022

@author: ti_re
"""


import random
import pandas as pd
import numpy as np
import time
from Gym import *
from utils import *
from Temps import *
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == '__main__': 
        
    regles = pd.read_csv("DATA/regle.csv")
        
    paramSimu = {"df_regles": regles,
             "df_HoraireLoader" : work_schedule(),
             "df_HoraireScierie" : work_schedule(),                
             "NbStepSimulation": 64*1,
             "NbStepSimulationTest": 64*10,
             "nbLoader": 1,
             "nbSechoir1": 4,
             "CapaciteSortieSciage": 100, # En nombre de chargements avant de bloquer la scierie
             "CapaciteSechageAirLibre": 0,
             "CapaciteCours": 0,
             "TempsAttenteLoader": 1,
             "TempsAttenteActionInvalide": 10,
             "TempsSechageAirLibre": 7 * 24,
             "RatioSechageAirLibre": 0.1 * 12 / 52,
             "HresProdScieriesParSem": 168, #44 + 44,
             "VariationProdScierie": 0.1,  # Pourcentage de variation de la demande par rapport à la production de la scierie
             "VariationTempsSechage": 0.1,
             "VariationTempsDeplLoader": 0.1,
             "FacteurSortieScierie" : 0.7, # Permet de sortir plus ou moins de la scierie (1 correspond à sortir exactement ce qui est prévu)
             "ObjectifStableEnPMP" : 215000 * 4 * 2.5,
             "RatioSapinEpinette" : "50/50"
             }
    paramSimu7525 = paramSimu.copy()
    paramSimu7525["RatioSapinEpinette"] = "75/25"
    paramSimu2575 = paramSimu.copy()
    paramSimu2575["RatioSapinEpinette"] = "25/75"
    
    hyperparams = {"n_steps": 16,
                   "batch_size": 16,
                   "total_timesteps": paramSimu["NbStepSimulation"],
                   "n_epochs": 10,
                   "lr": 0.0003}
    random.seed(1)
    
    timer_avant = time.time()
    vec_env = DummyVecEnv([lambda: EnvGym(paramSimu, get_action_space(paramSimu), get_state_space(paramSimu), state_min = 0, state_max = 1, hyperparams=hyperparams),
                           lambda: EnvGym(paramSimu7525, get_action_space(paramSimu7525), get_state_space(paramSimu7525), state_min = 0, state_max = 1, hyperparams=hyperparams),
                           lambda: EnvGym(paramSimu2575, get_action_space(paramSimu2575), get_state_space(paramSimu2575), state_min = 0, state_max = 1, hyperparams=hyperparams)])
    
    model = PPO('MlpPolicy', vec_env, n_steps=hyperparams["n_steps"], batch_size=hyperparams["batch_size"], n_epochs=hyperparams["n_epochs"], learning_rate=hyperparams["lr"], verbose=0)
    
    #Tests
    # envRL5050.solve_w_heuristique("pile_la_plus_elevee")
    
    print(f"Temps d'exécution : {time.time()-timer_avant:.2f}")
    
      