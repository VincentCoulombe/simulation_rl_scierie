# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:39:10 2022

@author: ti_re
"""

import simpy
import random
import pandas as pd
import numpy as np
from Loader import *


################### MANQUE CONTRAINTE SECHAGE AIR LIBRE TERMINER #########################
################### MANQUE CONTRAINTES TOUS SECHAGE AIR LIBRE OU AUCUN SUR WAGON #########################
################### MANQUE CONTRAINTES WAGON EN GÉNÉRAL #########################


class EnvSimpy(simpy.Environment):
    def __init__(self,paramSimu,**kwargs):
        super().__init__(**kwargs)
        
        self.paramSimu = paramSimu
        self.Evenements = pd.DataFrame()
        self.EnrEven("Début simulation")
        self.DernierLot = 0
        self.pdLots = pd.DataFrame(columns=["Temps","Lot","Essence","Longueur", "Hauteur", "Largeur","Emplacement"])
        
        self.lesEmplacements = {}
        self.lesEmplacements["Sortie sciage"] = Emplacements(Nom = "Sortie sciage", env=self,capacity=self.paramSimu["CapaciteSortieSciage"])
        self.lesEmplacements["Cours"] = Emplacements(Nom = "Cours", env=self,capacity=self.paramSimu["CapaciteCours"])
        self.lesEmplacements["Séchage à l'air libre"] = Emplacements(Nom = "Séchage à l'air libre", env=self,capacity=self.paramSimu["CapaciteSechageAirLibre"])
        
        for i in range(self.paramSimu["nbSechoir"]) : 
            self.lesEmplacements["Préparation séchoir " + str(i+1)] = Emplacements(Nom = "Préparation séchoir " + str(i+1), env=self,capacity=self.paramSimu["CapaciteSechoir"])
        
        self.process(Sciage(self))
        
        self.lesLoader = {}
        for i in range(self.paramSimu["nbSechoir"]) : 
            self.lesLoader["Loader " + str(i+1)] = Loader(NomLoader = "Loader " + str(i+1), env = self)
            
    def EnrEven(self,Evenement,NomLoader=None, Lot = None, Source = None, Destination = None) : 

        if self.paramSimu["ConserverListeEvenements"] : 

            if Lot == None : 
                essence = None
                longueur = None
                hauteur = None
                largeur = None
            else : 
                pdlot_temp = self.pdLots[self.pdLots["Lot"] == Lot][["Essence","Longueur","Hauteur","Largeur"]].to_numpy()
                essence = pdlot_temp[0,0]
                longueur = pdlot_temp[0,1]
                hauteur = pdlot_temp[0,2]
                largeur = pdlot_temp[0,3]
        
            nouveau = pd.DataFrame([[self.now,Evenement,NomLoader, Source, Destination, Lot,essence,longueur,hauteur,largeur]],columns=["Temps","Événement","Loader","Source", "Destination","Lot","Essence","Longeur","Hauteur","Largeur"])
            self.Evenements = pd.concat([self.Evenements,nouveau],axis=0)
        
    def LogCapacite(self,Emplacement) :
        
        if Emplacement.count == Emplacement.capacity and len(Emplacement.queue)==0: 
            self.EnrEven("Capacité maximale atteinte", Destination=Emplacement.Nom)
        
    def AjoutSortieSciage(self) :
        self.DernierLot += 1
        essence = "Épinette" if random.randint(1,2) == 1 else "Sapin"
        longueur = 8 if random.randint(1,2) == 1 else 10
        hauteur = 2 if random.randint(1,2) == 1 else 4
        largeur = max(hauteur,random.choice([2,3,4,6,8,10]))
        emplacement = "Sortie sciage"
        nouveau = pd.DataFrame([[self.now,self.DernierLot,essence,longueur,hauteur,largeur,emplacement,False]],columns=["Temps","Lot","Essence","Longueur", "Hauteur", "Largeur","Emplacement","Air libre terminé ?"])
        self.pdLots = pd.concat([self.pdLots,nouveau],axis=0)
        self.EnrEven("Sortie sciage",Lot = self.DernierLot)

    def RetLoaderCourant(self) : 
        
        minTemps = 9999999999999999999        
        minNom = ""
        for key in self.lesLoader.keys():
            if self.lesLoader[key].ProchainTemps < minTemps : 
                minTemps = self.lesLoader[key].ProchainTemps
                minNom = key
        
        return self.lesLoader[minNom], minTemps

    def stepSimpy(self,action) : 
        
        # Effectuer l'action déterminée par le RL
        loader, _ = self.RetLoaderCourant()
        loader.DeplacerLoader(action)
        _, minTemps = self.RetLoaderCourant()
        
        # Avancé le temps jusqu'à la prochaine décision à prendre
        if minTemps > self.now : 
            self.run(until=minTemps)
        
        # Finaliser les déplacements au nouveau temps t pour être prêt à prendre la meilleure décision possible
        for key in self.lesLoader.keys() : 
            if self.lesLoader[key].ProchainTemps == self.now:
                self.lesLoader[key].FinDeplacementLoader()               
                
        if self.now >= self.paramSimu["DureeSimulation"] :            
            self.EnrEven("Fin de la simulation")

class Emplacements(simpy.Resource) : 
    def __init__(self,Nom,env, **kwargs):
        super().__init__(env, **kwargs)
        self.lstRequest = []
        self.Nom = Nom
        self.env = env
       
    def request(self,**kwargs) :
        request = super().request(**kwargs)
        self.lstRequest.append(request)
        return request
        
    def release(self,env) : 
        if self.count >= self.capacity : 
            env.EnrEven("Place à nouveau disponible", Destination=self.Nom)
        
        request = self.lstRequest.pop(0)
        super().release(request)
        
    def EstPlein(self) : 
        if self.count >= self.capacity : 
            return True
        else :
            return False

def Sciage(env) :
    
    while True :

        yield env.lesEmplacements["Sortie sciage"].request() 

        yield env.timeout(random.expovariate(1/env.paramSimu["TempsInterArriveeSciage"])) # Temps inter-sortie du sciage

        env.AjoutSortieSciage()

        env.LogCapacite(env.lesEmplacements["Sortie sciage"]) 

def SechageAirLibre(env,lot,destination): 
    
    env.EnrEven("Début séchage à l'air libre",Lot = lot, Destination = destination)
    yield env.timeout(env.paramSimu["TempsSechageAirLibre"])
    env.pdLots.loc[env.pdLots["Lot"] == lot,"Air libre terminé ?"] = True
    env.EnrEven("Fin séchage à l'air libre",Lot = lot, Destination = destination)        
    


if __name__ == '__main__': 
    
      pass