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
from scipy.stats import t


def min_max_scaling(X,minPossible,maxPossible):
    
    data = np.array(X)
    data = np.where(data < minPossible,minPossible,data)
    data = np.where(data > maxPossible,maxPossible,data)
    return (data - minPossible) / (maxPossible - minPossible)
    
def get_action_space(paramSimu) : 
   
    return len(paramSimu["df_regles"])

def get_state_space(paramSimu) :
    
    NbDemandeVsProduit = len(paramSimu["df_regles"])
    NbParamPropEpinettesSortieSciage = 1
    NbParamTemps = 2
    NbParamTempsRestantSechoirs = 4
    return NbDemandeVsProduit + NbParamPropEpinettesSortieSciage + NbParamTemps + NbParamTempsRestantSechoirs
    
def IntervalleConfiance(lstIndicateur,alpha=0.05) :
    
    # Pour éviter des erreurs lors d'une seule réplication (on ne peut pas vraiment calculé
    # d'intervalles de confiances alors on retourne le seul résultat disponible)
    if len(lstIndicateur) == 1 :
        return []
    
    interval = []
    
    n = len(lstIndicateur)
    moyenne = sum(lstIndicateur)/n
    student = t.ppf(1-alpha/2, df=n-1) # divise par 2 car la fonction scipy gère séparément lower_bound et upper_bound
    variance = sum((np.array(lstIndicateur) - moyenne)**2) / (n-1)
    
    interval.append(moyenne - student * (variance/n)**(1/2))
    interval.append(moyenne + student * (variance/n)**(1/2))
    
    return interval

if __name__ == '__main__': 
    
    X = [-1,0,1,2,3,4,5]
    
    print(min_max_scaling(X,0,4))
    
    Y = [1,2,3,4,5,6,7]
    
    Z = X + Y
    
    print(Z)