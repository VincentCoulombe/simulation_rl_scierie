import random
import pandas as pd
import numpy as np
import gym
from gym import spaces
from EnvSimpy import *


class EnvGym(gym.Env) : 
    
    def __init__(self, paramSimu: dict, nb_actions: int, state_len: int, state_min: float, state_max: float, **kwargs):
        
        super().__init__(**kwargs)
        self.paramSimu = paramSimu
        self.action_space = spaces.Discrete(nb_actions)
        self.low = np.array([state_min for _ in range(state_len)], dtype=np.float32)
        self.high = np.array([state_max for _ in range(state_len)], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        
    def _update_observation(self) -> np.array:

        return self.env.getState() #Méthode de la simulation, state normalisé
    
    def _update_reward(self) -> float:
        
        self.respect_demande = self.env.getRespectDemande() #Pas normalisé
        return 0.5*self.respect_demande + 0.1*0. #Calculer ici
                
    def reset(self) -> np.array: 
        
        self.env = EnvSimpy(self.paramSimu) # Nouvelle simulation simpy       
        self.done = False
        self.info = {}
                        
        return self._update_observation()    
    
    def step (self, action, log_inds: bool=False) -> tuple: 

        # ----------------------------------------------------------------------------------
        # 1) faire l'action et rouler simu
        
        # 2) calculer indicateurs (attributs)
        
        # 3) normaliser state (attributs)
        
        # 3) return done ou pas
        # ----------------------------------------------------------------------------------    
        self.done = self.env.stepSimpy(action)
        
        reward = self._update_reward()
        state = self._update_observation()
        
        # 4) logger les indicateurs pertinents        
        # self.indicateurs.append([self.respect_demande])
        
        return state, reward, self.done, self.info
    
    def returnReward(self) : 
        
        return 0
    
    def returnState(self) : 
        
        return self.env        
        
    # Juste une patch pour tester reset et state... doit êre remplacé par vrai RL
    def boucle(self) : 
        
        state = self.reset()  
        done = False
        while not done: 
            action = ChoixLoader(state)
            state, _, done, _ = self.step(action)
            

if __name__ == '__main__': 
    pass