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
        
        # Définition de l'information sur les règles
        self.df_rulesDetails = paramSimu["df_rulesDetails"]
        nouveau = pd.DataFrame([["AUTRES",100,215000,245000]],columns=self.df_rulesDetails.columns)
        self.df_rulesDetails = pd.concat([self.df_rulesDetails,nouveau],axis=0)
        self.df_rulesDetails["Volume courant sciage"] = 0
        self.df_rulesDetails["Temps déplacement courant"] = 0
        
        self.df_produits = paramSimu["df_produits"]
        self.paramSimu = paramSimu
        self.Evenements = pd.DataFrame()
        self.EnrEven("Début simulation")
        self.DernierLot = 0
        self.pdLots = pd.DataFrame(columns=["Temps","Lot","produit","description","Emplacement"])
        
        self.lesEmplacements = {}
        self.lesEmplacements["Sortie sciage"] = Emplacements(Nom = "Sortie sciage", env=self,capacity=self.paramSimu["CapaciteSortieSciage"])
        
        if self.paramSimu["CapaciteCours"] > 0  : 
            self.lesEmplacements["Cours"] = Emplacements(Nom = "Cours", env=self,capacity=self.paramSimu["CapaciteCours"])
        
        if self.paramSimu["CapaciteSechageAirLibre"] > 0  : 
            self.lesEmplacements["Séchage à l'air libre"] = Emplacements(Nom = "Séchage à l'air libre", env=self,capacity=self.paramSimu["CapaciteSechageAirLibre"])
        
        for i in range(self.paramSimu["nbSechoir"]) : 
            self.lesEmplacements["Préparation séchoir " + str(i+1)] = Emplacements(Nom = "Préparation séchoir " + str(i+1), env=self,capacity=self.paramSimu["CapaciteSechoir"])
        
        for i in self.df_produits.index :
            self.process(Sciage(self,i))        
        
        self.lesLoader = {}
        for i in range(self.paramSimu["nbLoader"]) : 
            self.lesLoader["Loader " + str(i+1)] = Loader(NomLoader = "Loader " + str(i+1), env = self)
                    
    def EnrEven(self,Evenement,NomLoader=None, Lot = None, Source = None, Destination = None) : 

        if self.paramSimu["ConserverListeEvenements"] : 

            if Lot == None : 
                description = None
            else : 
                pdlot_temp = self.pdLots[self.pdLots["Lot"] == Lot][["description"]].to_numpy()
                description = pdlot_temp[0,0]
        
            nouveau = pd.DataFrame([[self.now,Evenement,NomLoader, Source, Destination, Lot,description]],columns=["Temps","Événement","Loader","Source", "Destination","Lot","description"])
            self.Evenements = pd.concat([self.Evenements,nouveau],axis=0)
        
    def LogCapacite(self,Emplacement) :
        
        if Emplacement.count == Emplacement.capacity and len(Emplacement.queue)==0: 
            self.EnrEven("Capacité maximale atteinte", Destination=Emplacement.Nom)
        
    def AjoutSortieSciage(self,indexProduit,produit,description,volumePaquet,epaisseur) :
        self.DernierLot += 1
        emplacement = "Sortie sciage"
        nouveau = pd.DataFrame([[self.now,self.DernierLot,produit,description,emplacement,False]],columns=["Temps","Lot","produit","description","Emplacement","Air libre terminé ?"])
        
        if self.paramSimu["SimulationParContainer"] : 
            pass
        
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

def Sciage(env,indexProduit) :
    
    prodMoyPMPSem = (max(0,env.df_produits.iloc[indexProduit]["production epinette"]) + max(0,env.df_produits.iloc[indexProduit]["production sapin"])) / 2
    prodMoyPMPHr = prodMoyPMPSem / env.paramSimu["HresProdScieriesParSem"]
    dureeMoy1Paquet = env.df_produits.iloc[indexProduit]["volume paquet"] / prodMoyPMPHr
    dureeMin1Paquet = dureeMoy1Paquet * (1-env.paramSimu["VariationProdScierie"])
    dureeMax1Paquet = dureeMoy1Paquet * (1+env.paramSimu["VariationProdScierie"])
    
    produit = env.df_produits.iloc[indexProduit]["produit"]
    description =  env.df_produits.iloc[indexProduit]["description"]
    volumePaquet = env.df_produits.iloc[indexProduit]["volume paquet"]
    epaisseur = env.df_produits.iloc[indexProduit]["epaisseur"]
    
    if dureeMoy1Paquet == 0 :
        print("Calcul de la vitesse de production impossible pour le produit",indexProduit)
        return 
                
    while True :

        yield env.timeout(random.triangular(dureeMin1Paquet,dureeMax1Paquet,dureeMoy1Paquet)) # Temps inter-sortie du sciage

        yield env.lesEmplacements["Sortie sciage"].request()         

        env.AjoutSortieSciage(indexProduit,produit,description,volumePaquet,epaisseur)

        env.LogCapacite(env.lesEmplacements["Sortie sciage"]) 

def Sechage() : 
    pass

def SechageAirLibre(env,lot,destination): 
    
    env.EnrEven("Début séchage à l'air libre",Lot = lot, Destination = destination)
    yield env.timeout(env.paramSimu["TempsSechageAirLibre"])
    env.pdLots.loc[env.pdLots["Lot"] == lot,"Air libre terminé ?"] = True
    env.EnrEven("Fin séchage à l'air libre",Lot = lot, Destination = destination)        
    


if __name__ == '__main__': 
       
    df_produits = pd.read_csv("DATA/df.csv")
    df_rulesDetails = pd.read_csv("DATA/rulesDetails.csv")
        
    paramSimu = {}
    
    paramSimu["df_produits"] = df_produits
    paramSimu["df_rulesDetails"] = df_rulesDetails    
    
    paramSimu["SimulationParContainer"] = False
    paramSimu["DureeSimulation"] = 300 # 1 an = 8760
    paramSimu["nbLoader"] = 2
    paramSimu["nbSechoir"] = 2
    paramSimu["ConserverListeEvenements"] = True # Si retire + rapide à l'exécution, mais perd le liste détaillée des choses qui se sont produites

    paramSimu["CapaciteSortieSciage"] = 10
    paramSimu["CapaciteSechageAirLibre"] = 0
    paramSimu["CapaciteCours"] = 0
    paramSimu["CapaciteSechoir"] = 1
    #paramSimu["TempsInterArriveeSciage"] = 1
    paramSimu["TempsAttenteLoader"] = 1 #0.05
    paramSimu["TempsDeplacementLoader"] = 5 #0.05
    paramSimu["TempsSechageAirLibre"] = 10
    
    paramSimu["HresProdScieriesParSem"] = 44+44
    paramSimu["VariationProdScierie"] = 0.1 # Pourcentage de variation de la production de la scierie par rapport aux chiffres généraux fournis
    
    env = EnvSimpy(paramSimu)
                
    while env.now < paramSimu["DureeSimulation"] : 
        action = ChoixLoader(env)
        env.stepSimpy(action)
    
    pdLots = env.pdLots
    df_rulesDetails = env.df_rulesDetails
    Evenement = env.Evenements
