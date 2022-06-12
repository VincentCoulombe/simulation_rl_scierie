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
    NbParamTemps = 2
    return NbDemandeVsProduit + NbParamPropEpinettesSortieSciage + NbParamTemps
    
if __name__ == '__main__': 
    
    X = [-1,0,1,2,3,4,5]
    
    print(min_max_scaling(X,0,4))
    
    Y = [1,2,3,4,5,6,7]
    
    Z = X + Y
    
    print(Z)