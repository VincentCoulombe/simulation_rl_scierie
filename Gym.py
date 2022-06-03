# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:39:10 2022

@author: ti_re
"""

import random
import pandas as pd
import numpy as np
import gym
from EnvSimpy import *


class EnvGym(gym.Env) : 
    def __init__(self,paramSimu, **kwargs):
        super().__init__(**kwargs)
        self.paramSimu = paramSimu
        self.boucle()     # Juste une patch pour tester reset et state... doit êre remplacé par vrai RL    
                
    def reset (self) : 

        # Nouvelle simulation simpy
        self.env = EnvSimpy(self.paramSimu)        
                
        state = self.returnState()
        
        return state     
    
    # Une action est définie comme (lot, source, destination)
    def step (self,action) : 
        
        self.env.stepSimpy(action)
                
        # Retourner l'information nécessaire pour le RL
        return self.returnState(), self.returnReward()
    
    def returnReward(self) : 
        
        return 0
    
    def returnState(self) : 
        
        return self.env        
        
    # Juste une patch pour tester reset et state... doit êre remplacé par vrai RL
    def boucle(self) : 
        
        state = self.reset()  
        
        while self.env.now < self.env.paramSimu["DureeSimulation"] : 
            action = ChoixLoader(state)
            state, reward = self.step(action)

if __name__ == '__main__': 
    

    pass