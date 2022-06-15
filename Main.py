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
from Temps import *
from utils import *
from stable_baselines3.common.vec_env import DummyVecEnv

if __name__ == '__main__': 
        
    regles = pd.read_csv("DATA/regle.csv")
    df_HoraireScierie = work_schedule()
    df_HoraireLoader = work_schedule()

        
    paramSimu = {"df_regles": regles,
             "df_HoraireLoader" : df_HoraireLoader,
             "df_HoraireScierie" : df_HoraireScierie,
             "DetailsEvenements": True,
             "NbStepSimulation": 64*10,
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
             "MaxSechageAirLibre" : 30/100,
             "DureeDeteriorationEpinette" : 30 * 24, 
             "DureeDeteriorationSapin" : 4 * 30 * 24, 
             "HresProdScieriesParSem": sum(df_HoraireScierie[:168]["work_time"]),
             "VariationProdScierie": 0.1,  # Pourcentage de variation de la demande par rapport à la production de la scierie
             "VariationTempsSechage": 0.1,
             "VariationTempsDeplLoader": 0.1,
             "FacteurSortieScierie" : 0.35, # Permet de sortir plus ou moins de la scierie (1 correspond à sortir exactement ce qui est prévu)
             "FacteurTempsChargement" : 0.85, 
             "ObjectifStableEnPMP" : 215000 * 4 * 2.5,
             "RatioSapinEpinette" : "50/50"
             }
    
    hyperparams = {"n_steps": 128,
                   "batch_size": 128,
                   "total_timesteps": paramSimu["NbStepSimulation"],
                   "n_epochs": 10,
                   "lr": 0.0003}
    random.seed(1)
    
    timer_avant = time.time()
    env = EnvGym(paramSimu, get_action_space(paramSimu), get_state_space(paramSimu), state_min = 0, state_max = 1, hyperparams=hyperparams)
    # model = PPO('MlpPolicy', env, n_steps=hyperparams["n_steps"], batch_size=hyperparams["batch_size"], n_epochs=hyperparams["n_epochs"], learning_rate=hyperparams["lr"],
    #             verbose=0)
    # model = PPO.load(r"./models/training_wheels")
    
    #Tests
    env.solve_w_heuristique("gestion_horaire_et_pile")
    # best_avg_reward = env.evaluate_model(model)
    
    print(f"Temps d'exécution : {time.time()-timer_avant:.2f}")
    
      