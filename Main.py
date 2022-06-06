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
        
    paramSimu = {}
    
    paramSimu["df_produits"] = df_produits
    paramSimu["df_rulesDetails"] = df_rulesDetails    

    paramSimu["SimulationParContainer"] = False
    paramSimu["DureeSimulation"] = 1000
    paramSimu["nbLoader"] = 2
    paramSimu["nbSechoir"] = 2
    paramSimu["ConserverListeEvenements"] = True # Si retire + rapide à l'exécution, mais perd le liste détaillée des choses qui se sont produites

    paramSimu["CapaciteSortieSciage"] = 10
    paramSimu["CapaciteSechageAirLibre"] = 10
    paramSimu["CapaciteCours"] = 5000
    paramSimu["CapaciteSechoir"] = 50
    paramSimu["TempsInterArriveeSciage"] = 1
    paramSimu["TempsAttenteLoader"] = 1 #0.05
    paramSimu["TempsDeplacementLoader"] = 5 #0.05
    paramSimu["TempsSechageAirLibre"] = 10
    
    paramSimu["HresProdScieriesParSem"] = 44+44
    paramSimu["VariationProdScierie"] = 0.1 # Pourcentage de variation de la production de la scierie par rapport aux chiffres généraux fournis

       
    # Pour faciliter le développement, on s'assure d'avoir toujours le mêmes
    # nombres aléatoires d'une exécution à l'autre
    random.seed(1)
    
    timer_avant = time.time()
    
    envRL = EnvGym(paramSimu, utils.get_action_space(), utils.get_state_space(), 0, 1)
    
    timer_après = time.time()
    
    # juste pour faciliter débuggage...
    pdLots = envRL.env.pdLots
    Evenement = envRL.env.Evenements
    df_rulesDetails = envRL.env.df_rulesDetails

    print("Temps d'exécution : ", timer_après-timer_avant)
    
    if paramSimu["ConserverListeEvenements"] : 
        print("Nb de déplacements de loader : ", len(Evenement[Evenement["Événement"] == "Début déplacement"]))
        print("Nb de déplacement par minutes : ", len(Evenement[Evenement["Événement"] == "Début déplacement"]) / (timer_après-timer_avant) * 60)
        print("Nb de déplacement par heures : ", len(Evenement[Evenement["Événement"] == "Début déplacement"]) / (timer_après-timer_avant) * 60 * 60)
          