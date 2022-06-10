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

if __name__ == '__main__': 
        
    regles = pd.read_csv("DATA/regle.csv")
        
    paramSimu = {"df_regles": regles,
             "NbStepSimulation": 64*5,
             "NbStepSimulationTest": 64*2,
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
             "FacteurSortieScierie" : 1, # Permet de sortir plus ou moins de la scierie (1 correspond à sortir exactement ce qui est prévu)
             "ObjectifStableEnPMP" : 215000 * 4 * 2.5
             }

    hyperparams = {"n_steps": 16,
                   "batch_size": 16,
                   "total_timesteps": paramSimu["NbStepSimulation"],
                   "n_epochs": 10,
                   "lr": 0.0003}

    # Pour faciliter le développement, on s'assure d'avoir toujours le mêmes
    # nombres aléatoires d'une exécution à l'autre
    random.seed(1)
    timer_avant = time.time()
    envRL = EnvGym(paramSimu, get_action_space(paramSimu), get_state_space(paramSimu), state_min = 0, state_max = 1, hyperparams=hyperparams)
    model = PPO('MlpPolicy', envRL, n_steps=hyperparams["n_steps"], batch_size=hyperparams["batch_size"], n_epochs=hyperparams["n_epochs"], learning_rate=hyperparams["lr"], verbose=0)
    # envRL.evaluate_model(model)
    model.learn(total_timesteps=hyperparams["total_timesteps"], reset_num_timesteps=False) # CTRL Z!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TOUJOURS PAS COMPRIS.............
    
    timer_après = time.time()

    # juste pour faciliter débuggage...
    npLots = envRL.env.npLots
    Evenement = envRL.env.Evenements
    df_rulesDetails = envRL.env.df_rulesDetails

    print(f"Temps d'exécution : {timer_après-timer_avant:.2f}")
    
      