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
        
    df_produits = pd.read_csv("DATA/df.csv")
    df_rulesDetails = pd.read_csv("DATA/rulesDetails.csv")
    df_produits = pd.concat([df_produits.iloc[:5],df_produits.iloc[75:80]],ignore_index = True) # limiter à un sous-ensemble de produits

    paramSimu = {"df_produits": df_produits,
                 "df_rulesDetails": df_rulesDetails,
                 "SimulationParContainer": False,
                 "NbStepSimulation": 500,
                 "nbLoader": 1,
                 "nbSechoir": 2,
                 "ConserverListeEvenements": True,
                 "CapaciteSortieSciage": 10,
                 "CapaciteSechageAirLibre": 0,
                 "CapaciteCours": 0,
                 "CapaciteSechoir": 1,
                 "TempsAttenteLoader": 1,
                 "TempsDeplacementLoader": 5,
                 "TempsSechageAirLibre": 7 * 24,
                 "RatioSechageAirLibre": 0.1 * 12 / 52,
                 "HresProdScieriesParSem": 44 + 44,
                 "VariationProdScierie": 0.1,
                 "VariationTempsSechage": 0.1,
                 "VariationDemandeVSProd" : 0.25
                 } # Pourcentage de variation de la demande par rapport à la production de la scierie}


    # Pour faciliter le développement, on s'assure d'avoir toujours le mêmes
    # nombres aléatoires d'une exécution à l'autre
    random.seed(1)

    timer_avant = time.time()
    envRL = EnvGym(paramSimu, get_action_space(paramSimu), get_state_space(paramSimu), state_min = 0, state_max = 1)
    model = PPO('MlpPolicy', envRL)
    envRL.evaluate_model()
    #envRL.train_model(nb_timestep=paramSimu["DureeSimulation"], nb_episode=10, log=True, save=False)

    timer_après = time.time()

    # juste pour faciliter débuggage...
    npLots = envRL.env.npLots
    Evenement = envRL.env.Evenements
    df_rulesDetails = envRL.env.df_rulesDetails

    print(f"Temps d'exécution : {timer_après-timer_avant:.2f}")
    
      