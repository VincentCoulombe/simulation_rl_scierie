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
from utils import *


################### MANQUE CONTRAINTE SECHAGE AIR LIBRE TERMINER #########################
################### MANQUE CONTRAINTES TOUS SECHAGE AIR LIBRE OU AUCUN SUR WAGON #########################
################### MANQUE CONTRAINTES WAGON EN GÉNÉRAL #########################
################### MANQUE SIMPY POUR WAGON VS SECHAGE #########################
################### gérer temps attente loader comme du monde #########################
################### attention pour ne pas qu'il sèche complètement dans la cours #########################
# Pour l'instant envoi dans le premier séchoir qui a de la place si plusieurs libres
# pourquoi le temps de séchage semble diminuer presqu'uniformément par produit alors que les produits ne sont pas faits en même temps

class EnvSimpy(simpy.Environment):
    def __init__(self,paramSimu,**kwargs):
        super().__init__(**kwargs)
        
        self.RewardActionInvalide = False
        
        # Définition de l'information sur les règles
        self.df_rulesDetails = paramSimu["df_rulesDetails"]
        nouveau = pd.DataFrame([["AUTRES",100,215000,245000]],columns=self.df_rulesDetails.columns)
        self.df_rulesDetails = pd.concat([self.df_rulesDetails,nouveau],axis=0,ignore_index = True)
        self.df_rulesDetails["Volume courant sciage"] = 0
        self.df_rulesDetails["Temps déplacement courant"] = 0
        
        self.df_produits = paramSimu["df_produits"]
        self.paramSimu = paramSimu
        self.Evenements = pd.DataFrame()
        self.EnrEven("Début simulation")
        self.DernierLot = 0
        self.pdLots = pd.DataFrame(columns=["Temps","Lot","produit","description","Emplacement","temps sechage"])
        
        # Ajouter le temps de séchage aux produits à partir de la règle (avec une valeur par défaut au cas ou la règle n'est pas trouvée)
        # Ajout par le fait même d'une demande par produits
        self.df_produits["temps sechage"] = 100
        self.df_produits["demande"] = 0
        self.df_produits["Quantité produite"] = 0
        for i in self.df_produits["produit"] :
            
            variation = 1 + random.random() * 2 * paramSimu["VariationDemandeVSProd"] - paramSimu["VariationDemandeVSProd"]
            self.df_produits.loc[self.df_produits["produit"] == i,"demande"] =  (max(0,self.df_produits[self.df_produits["produit"] == i]["production epinette"].values[0])+max(0,self.df_produits[self.df_produits["produit"] == i]["production sapin"].values[0]))/2 /7/24 * paramSimu["DureeSimulation"] * variation
            
            regle = self.df_produits[self.df_produits["produit"] == i]["regle"].values[0]
            regle = self.df_rulesDetails[self.df_rulesDetails["regle"] == regle]
            if len(regle) == 0 :
                print("Incapable de trouver la règle pour le produit", i)
            else :               
                self.df_produits.loc[self.df_produits["produit"] == i,"temps sechage"] = regle["temps sechage"].values[0]
        
        
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

    def generate_demand(self):
        return self.now/self.paramSimu["DureeSimulation"]*self.df_produits["demande"]
       
    def getState(self) : 
        
        # Temps de séchage des produits qui sont disponibles au déplacement
        lstProduits = []
        self.LienActionLot = []
        for produit in self.df_produits["produit"] :
            
            lstTemps = [temps for temps in self.pdLots[(self.pdLots["produit"] == produit) & (self.pdLots["Emplacement"] == "Sortie sciage")]["temps sechage"]]
            lstlot = [temps for temps in self.pdLots[(self.pdLots["produit"] == produit) & (self.pdLots["Emplacement"] == "Sortie sciage")]["Lot"]]
            
            if len(lstTemps) == 0 : 
                lstTemps.append(250)
                lstlot.append(-1)
            
            argSort = np.argsort(np.array(lstTemps))
            lstlot = list(np.array(lstlot)[argSort])
            lstTemps.sort()
            
            lstProduits.append(lstTemps[0]) # Min   
            self.LienActionLot.append(lstlot[0])
            lstProduits.append(lstTemps[int((len(lstTemps)-1)/2)]) # médiane            
            self.LienActionLot.append(lstlot[int((len(lstTemps)-1)/2)])
            lstProduits.append(lstTemps[-1]) # Max        
            self.LienActionLot.append(lstlot[-1])
        lstProduits = min_max_scaling(lstProduits,0,250)
        
        # Où on se situe par rapport à la demande
        #print(self.generate_demand())
        self.lstProdVsDemande = self.generate_demand() - self.df_produits["Quantité produite"]
        print(self.lstProdVsDemande)
        #print(self.generate_demand())

        
        return lstProduits
    
    def getRespectDemande(self) : 
        return 0
    
    def EnrEven(self,Evenement,NomLoader=None, Lot = None, Source = None, Destination = None) : 

        if self.paramSimu["ConserverListeEvenements"] : 

            if Lot == None : 
                description = None
            else : 
                pdlot_temp = self.pdLots[self.pdLots["Lot"] == Lot][["description"]].to_numpy()
                description = pdlot_temp[0,0]
        
            nouveau = pd.DataFrame([[self.now,Evenement,NomLoader, Source, Destination, Lot,description]],columns=["Temps","Événement","Loader","Source", "Destination","Lot","description"])
            self.Evenements = pd.concat([self.Evenements,nouveau],axis=0,ignore_index = True)
        
    def LogCapacite(self,Emplacement) :
        
        if Emplacement.count == Emplacement.capacity and len(Emplacement.queue)==0: 
            self.EnrEven("Capacité maximale atteinte", Destination=Emplacement.Nom)
        
    def AjoutSortieSciage(self,indexProduit,produit,description,volumePaquet,epaisseur,tempsSechage) :
        self.DernierLot += 1
        emplacement = "Sortie sciage"
        nouveau = pd.DataFrame([[self.now,self.DernierLot,produit,description,emplacement,tempsSechage,False]],columns=["Temps","Lot","produit","description","Emplacement","temps sechage","Air libre terminé ?"])
        
        if self.paramSimu["SimulationParContainer"] : 
            pass
        
        self.pdLots = pd.concat([self.pdLots,nouveau],axis=0,ignore_index = True)
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
            return True
        
        return False

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
    tempsSechage = env.df_produits.iloc[indexProduit]["temps sechage"]
    
    if dureeMoy1Paquet == 0 :
        print("Calcul de la vitesse de production impossible pour le produit",indexProduit)
        return 
                
    while True :

        yield env.timeout(random.triangular(dureeMin1Paquet,dureeMax1Paquet,dureeMoy1Paquet)) # Temps inter-sortie du sciage

        yield env.lesEmplacements["Sortie sciage"].request()         

        env.AjoutSortieSciage(indexProduit,produit,description,volumePaquet,epaisseur,tempsSechage)

        env.LogCapacite(env.lesEmplacements["Sortie sciage"]) 

def Sechage(env,lot,destination) : 
    produit = env.pdLots[env.pdLots["Lot"] == lot]["produit"].values[0]
    tempsSechage = env.pdLots[env.pdLots["Lot"] == lot]["temps sechage"].values[0]
    tempsMin = tempsSechage * (1-env.paramSimu["VariationTempsSechage"])
    tempsMax = tempsSechage * (1+env.paramSimu["VariationTempsSechage"])
    
    env.EnrEven("Début séchage",Lot = lot, Destination = destination)
    
    yield env.timeout(random.triangular(tempsMin,tempsMax,tempsSechage))
    
    env.EnrEven("Fin séchage",Lot = lot, Destination = destination)
    env.pdLots.loc[env.pdLots["Lot"] == lot,"Emplacement"] = "Sortie séchoir"
    env.df_produits.loc[env.df_produits["produit"] == produit,"Quantité produite"] += env.df_produits.loc[env.df_produits["produit"] == produit,"volume paquet"]
    
    env.lesEmplacements[destination].release(env)
       
    
def SechageAirLibre(env,lot,destination): 
    
    #env.EnrEven("Début séchage à l'air libre",Lot = lot, Destination = destination)
    yield env.timeout(env.paramSimu["TempsSechageAirLibre"])
    
    TempsSechage = env.pdLots[env.pdLots["Lot"] == lot]["temps sechage"]
    
    env.pdLots.loc[env.pdLots["Lot"] == lot,"temps sechage"] = TempsSechage * (1- env.paramSimu["RatioSechageAirLibre"])


if __name__ == '__main__': 
       
    import time
    
    df_produits = pd.read_csv("DATA/df.csv")
    df_rulesDetails = pd.read_csv("DATA/rulesDetails.csv")
    df_produits = pd.concat([df_produits.iloc[0:5],df_produits.iloc[75:80]],ignore_index = True) # limiter à un sous-ensemble de produits
        
    paramSimu = {}
    
    paramSimu["df_produits"] = df_produits
    paramSimu["df_rulesDetails"] = df_rulesDetails    
    
    paramSimu["SimulationParContainer"] = False
    paramSimu["DureeSimulation"] = 50 # 1 an = 8760
    paramSimu["nbLoader"] = 1
    paramSimu["nbSechoir"] = 4
    paramSimu["ConserverListeEvenements"] = True # Si retire + rapide à l'exécution, mais perd le liste détaillée des choses qui se sont produites

    paramSimu["CapaciteSortieSciage"] = 10
    paramSimu["CapaciteSechageAirLibre"] = 0
    paramSimu["CapaciteCours"] = 0
    paramSimu["CapaciteSechoir"] = 1
    #paramSimu["TempsInterArriveeSciage"] = 1
    paramSimu["TempsAttenteLoader"] = 1 #0.05
    paramSimu["TempsDeplacementLoader"] = 5 #0.05
    paramSimu["TempsSechageAirLibre"] = 7*24
    paramSimu["RatioSechageAirLibre"] = 0.1*12/52
    
    paramSimu["HresProdScieriesParSem"] = 44+44
    paramSimu["VariationProdScierie"] = 0.1 # Pourcentage de variation de la production de la scierie par rapport aux chiffres généraux fournis
    paramSimu["VariationTempsSechage"] = 0.1 # Pourcentage de variation du temps de séchage par rapport à la prévision
    paramSimu["VariationDemandeVSProd"] = 0.25 # Pourcentage de variation de la demande par rapport à la production de la scierie
  
    # Pour faciliter le développement, on s'assure d'avoir toujours le mêmes
    # nombres aléatoires d'une exécution à l'autre
    random.seed(1)
    
    timer_avant = time.time()
    
    env = EnvSimpy(paramSimu)
               
    done = False
    state = env.getState()
    while not done: 
        action = ChoixLoader(env)
        done = env.stepSimpy(action)
        _ = env.getState()
    
    
    timer_après = time.time()
    
    # juste pour faciliter débuggage...
    pdLots = env.pdLots
    df_rulesDetails = env.df_rulesDetails
    Evenement = env.Evenements

    print("Temps d'exécution : ", timer_après-timer_avant)
    
    if paramSimu["ConserverListeEvenements"] : 
        print("Nb de déplacements de loader : ", len(Evenement[Evenement["Événement"] == "Début déplacement"]))
        print("Nb de déplacement par minutes : ", len(Evenement[Evenement["Événement"] == "Début déplacement"]) / (timer_après-timer_avant) * 60)
        print("Nb de déplacement par heures : ", len(Evenement[Evenement["Événement"] == "Début déplacement"]) / (timer_après-timer_avant) * 60 * 60)