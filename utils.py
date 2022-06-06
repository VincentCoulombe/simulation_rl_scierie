# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:52:21 2022

@author: ti_re
"""

import numpy as np

def min_max_scaling(X,minPossible,maxPossible) :
    
    return (np.array(X) - minPossible) / (maxPossible - minPossible)
    
def get_action_space(paramSimu) : 
   
    return len(paramSimu["df_produits"])

def get_state_space(paramSimu) :
    
    NbEspaceStateParProduits = 3
    return NbEspaceStateParProduits * len(paramSimu["df_produits"])
    
