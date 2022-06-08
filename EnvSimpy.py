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
# séchage ne part pas au bon moment
# demande en paquet en PMP ?
# conversion comme il faut en règles

class EnvSimpy(simpy.Environment):
    def __init__(self,paramSimu,**kwargs):
        super().__init__(**kwargs)
        
        self.cLots = {"Temps" : 0,
                      "Lot" : 1,
                      "produit" : 2,
                      "description" : 3,
                      "Emplacement" : 4,
                      "temps sechage" : 5}
        
        self.RewardActionInvalide = False
        
        self.PropEpinettesSortieSciage = 0.5
        
        # Définition de l'information sur les règles
        self.df_rulesDetails = paramSimu["df_rulesDetails"]
        nouveau = pd.DataFrame([["AUTRES",100,215000,245000]],columns=self.df_rulesDetails.columns)
        self.df_rulesDetails = pd.concat([self.df_rulesDetails,nouveau],axis=0,ignore_index = True)
        self.df_rulesDetails["Volume courant sciage"] = 0
        self.df_rulesDetails["Temps déplacement courant"] = 0
        
        self.np_produits = paramSimu["df_produits"]
        self.paramSimu = paramSimu
        self.Evenements = np.asarray([["Temps","Événement","Loader","Source", "Destination","Lot","description"]],dtype='U50')
        self.EnrEven("Début simulation")
        self.DernierLot = 0
        self.npLots = np.array([["Temps","Lot","produit","description","Emplacement","temps sechage"]],dtype='U50')
        
        # Ajouter le temps de séchage aux produits à partir de la règle (avec une valeur par défaut au cas ou la règle n'est pas trouvée)
        # Ajout par le fait même d'une demande par produits
        self.np_produits["temps sechage"] = 100
        self.np_produits["demande"] = 0
        self.np_produits["Quantité produite"] = 0
        for j,i in enumerate(self.np_produits["produit"]) :
            
            variation = 1 + random.random() * 2 * paramSimu["VariationDemandeVSProd"] - paramSimu["VariationDemandeVSProd"]
            self.np_produits.loc[self.np_produits["produit"] == i,"demande"] =  (j+1)/55 #(max(0,self.np_produits[self.np_produits["produit"] == i]["production epinette"].values[0])+max(0,self.np_produits[self.np_produits["produit"] == i]["production sapin"].values[0]))/2 /7/24 * paramSimu["DureeSimulation"] * variation
                    
            regle = self.np_produits[self.np_produits["produit"] == i]["regle"].values[0]
            regle = self.df_rulesDetails[self.df_rulesDetails["regle"] == regle]
            if len(regle) == 0 :
                print("Incapable de trouver la règle pour le produit", i)
            else :               
                self.np_produits.loc[self.np_produits["produit"] == i,"temps sechage"] = regle["temps sechage"].values[0]
        self.cProd = {}
        for indice,colonne in enumerate(self.np_produits.columns) : 
            self.cProd[colonne] = indice
        self.np_produits = np.vstack((np.asarray(self.np_produits.columns,dtype='U50'),self.np_produits.to_numpy(dtype='U50')))
                        
        self.lesEmplacements = {}
        self.lesEmplacements["Sortie sciage"] = Emplacements(Nom = "Sortie sciage", env=self,capacity=self.paramSimu["CapaciteSortieSciage"])
        
        if self.paramSimu["CapaciteCours"] > 0  : 
            self.lesEmplacements["Cours"] = Emplacements(Nom = "Cours", env=self,capacity=self.paramSimu["CapaciteCours"])
        
        if self.paramSimu["CapaciteSechageAirLibre"] > 0  : 
            self.lesEmplacements["Séchage à l'air libre"] = Emplacements(Nom = "Séchage à l'air libre", env=self,capacity=self.paramSimu["CapaciteSechageAirLibre"])
        
        for i in range(self.paramSimu["nbSechoir"]) : 
            self.lesEmplacements["Préparation séchoir " + str(i+1)] = Emplacements(Nom = "Préparation séchoir " + str(i+1), env=self,capacity=self.paramSimu["CapaciteSechoir"])
        
        for i in range(1,len(self.np_produits)) :
            self.process(Sciage(self,self.np_produits[i]))        
        
        self.lesLoader = {}
        for i in range(self.paramSimu["nbLoader"]) : 
            self.lesLoader["Loader " + str(i+1)] = Loader(NomLoader = "Loader " + str(i+1), env = self)

        self.updateApresStep()

    def getState(self) : 
        
        # Temps de séchage des produits qui sont disponibles au déplacement
        lstProduits = []
        self.LienActionLot = []
        for produit in self.np_produits[:,self.cProd["produit"]] :            
            
            # ne pas considérer l'entête
            if produit == "produit" : 
                continue
            
            lstTemps = []
            lstlot = []
            for temps in self.npLots[(self.npLots[:,self.cLots["produit"]] == produit) & (self.npLots[:,self.cLots["Emplacement"]] == "Sortie sciage")] : 
                lstTemps.append(float(temps[self.cLots["temps sechage"]]))
                lstlot.append(int(temps[self.cLots["Lot"]]))
                                                  
            if len(lstTemps) == 0 : 
                lstTemps.append(250)
                lstlot.append(-1)
            
            argSort = np.argsort(np.array(lstTemps))
            lstlot = list(np.array(lstlot)[argSort])
            lstTemps.sort()
            
            lstProduits.append(lstTemps[0]) # Min   
            self.LienActionLot.append(lstlot[0])
            #lstProduits.append(lstTemps[int((len(lstTemps)-1)/2)]) # médiane            
            #self.LienActionLot.append(lstlot[int((len(lstTemps)-1)/2)])
            #lstProduits.append(lstTemps[-1]) # Max        
            #self.LienActionLot.append(lstlot[-1])
        
        #lstProduits = min_max_scaling(lstProduits,0,250)
        
        # Où on se situe par rapport à la demande
        #lstProdVsDemandeMinMax = min_max_scaling(self.lstProdVsDemande,-1000,1000)
        
        # Est-ce que les stocks sont stables dans la cours
        lstProdVsDemandeMinMax = min_max_scaling(self.lstProdVsDemande.astype(float),-25000,25000)
        
        return np.concatenate(([self.PropEpinettesSortieSciage],lstProdVsDemandeMinMax))
    
    def updateApresStep(self) :
        self.updateRespectInventaire()
    
    def getProportions(self) : 
        
        return self.proportionVoulu, self.proportionReelle
        
    def updateRespectInventaire (self) :
        
        # Quantité dans la cours de chaque produit en s'assurant d'avoir le produit dans la liste même si la quantité est à 0
        volume = self.np_produits[1:,self.cProd["volume paquet"]].astype(int)
        Lots = np.concatenate((self.np_produits[1:,self.cProd["produit"]],self.npLots[self.npLots[:,self.cLots["Emplacement"]] == "Sortie sciage",self.cLots["produit"]]))
        unique,count = np.unique(Lots,return_counts = True)        
        count = (count-1) * volume
        count = count / max(1,sum(count))
        
        # Proportions qu'on veut avoir dans la cours
        self.proportionVoulu = self.np_produits[1:,self.cProd["demande"]].astype(float)
        self.proportionReelle = count
        
        # La différence est en PMP à l'heure pour rester dans des ranges similaires avec le temps pour une longue simulation
        self.lstProdVsDemande = (self.proportionVoulu - self.proportionReelle)
    
    def getRespectInventaire(self) : 
        return self.lstProdVsDemande
    
    def EnrEven(self,Evenement,NomLoader=None, Lot = None, Source = None, Destination = None) : 

        if self.paramSimu["ConserverListeEvenements"] : 

            if Lot == None : 
                description = None
            else : 
                description = self.npLots[Lot][self.cLots["description"]]
        
            nouveau = np.asarray([[self.now,Evenement,NomLoader, Source, Destination, Lot,description]],dtype='U50')
            self.Evenements = np.vstack((self.Evenements,nouveau))
            
        
    def LogCapacite(self,Emplacement) :
        
        if Emplacement.count == Emplacement.capacity and len(Emplacement.queue)==0: 
            self.EnrEven("Capacité maximale atteinte", Destination=Emplacement.Nom)
        
    def AjoutSortieSciage(self,produit,description,volumePaquet,epaisseur,tempsSechage) :
        self.DernierLot += 1
        emplacement = "Sortie sciage"
        nouveaunp = np.asarray([[self.now,self.DernierLot,produit,description,emplacement,tempsSechage]],dtype='U50')
        
        if self.paramSimu["SimulationParContainer"] : 
            pass
        
        self.npLots = np.vstack((self.npLots,nouveaunp))
        self.EnrEven("Sortie sciage",Lot = self.DernierLot)
        
        # Procédé au séchage à l'air libre
        self.process(SechageAirLibre(self,self.DernierLot))


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
                
        # Aucune action possible, le loader est tombé en attente, on attend qu'il se passe quelque chose avant de finalisé le step...
        if max(self.LienActionLot) == -1 and self.now < self.paramSimu["DureeSimulation"]:
            _ = self.getState() # Pour mettre à jour LienActionLot
            return self.stepSimpy(-1)
                
        self.updateApresStep()
        
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

def Sciage(env,np_produit) :
    
    produit = int(np_produit[env.cProd["produit"]])
    description = np_produit[env.cProd["description"]]
    volumePaquet = float(np_produit[env.cProd["volume paquet"]])
    epaisseur = float(np_produit[env.cProd["epaisseur"]])
    tempsSechage = float(np_produit[env.cProd["temps sechage"]])
    
    prodMoyPMPSem = (max(0,float(np_produit[env.cProd["production epinette"]])) + max(0,float(np_produit[env.cProd["production sapin"]]))) / 2
    prodMoyPMPHr = prodMoyPMPSem / env.paramSimu["HresProdScieriesParSem"]
    dureeMoy1Paquet = volumePaquet / prodMoyPMPHr
    dureeMin1Paquet = dureeMoy1Paquet * (1-env.paramSimu["VariationProdScierie"])
    dureeMax1Paquet = dureeMoy1Paquet * (1+env.paramSimu["VariationProdScierie"])   
    
    if dureeMoy1Paquet == 0 :
        print("Calcul de la vitesse de production impossible pour le produit",indexProduit)
        return 
                
    while True :

        yield env.timeout(random.triangular(dureeMin1Paquet,dureeMax1Paquet,dureeMoy1Paquet)) # Temps inter-sortie du sciage

        yield env.lesEmplacements["Sortie sciage"].request()         

        env.AjoutSortieSciage(produit,description,volumePaquet,epaisseur,tempsSechage)

        env.LogCapacite(env.lesEmplacements["Sortie sciage"]) 

def Sechage(env,lot,destination) : 
    produit = int(env.npLots[lot][env.cLots["produit"]])
    tempsSechage = float(env.npLots[lot][env.cLots["temps sechage"]])
    tempsMin = tempsSechage * (1-env.paramSimu["VariationTempsSechage"])
    tempsMax = tempsSechage * (1+env.paramSimu["VariationTempsSechage"])
    
    env.EnrEven("Début séchage",Lot = lot, Destination = destination)
    
    yield env.timeout(random.triangular(tempsMin,tempsMax,tempsSechage))
    
    env.EnrEven("Fin séchage",Lot = lot, Destination = destination)
    env.npLots[lot][env.cLots["Emplacement"]] = "Sortie séchoir"
    env.np_produits[env.np_produits[:,env.cProd["produit"]] == str(produit),env.cProd["Quantité produite"]] = str(int(env.np_produits[env.np_produits[:,env.cProd["produit"]] == str(produit),env.cProd["Quantité produite"]]) + int(env.np_produits[env.np_produits[:,env.cProd["produit"]] == str(produit),env.cProd["volume paquet"]]))
    
    env.lesEmplacements[destination].release(env)
       
    
def SechageAirLibre(env,lot): 
    
    while env.npLots[lot][env.cLots["Emplacement"]] != "Sortie séchoir" :
        
        yield env.timeout(env.paramSimu["TempsSechageAirLibre"])
        
        env.npLots[lot][env.cLots["temps sechage"]] = str(float(env.npLots[lot][env.cLots["temps sechage"]]) * (1- env.paramSimu["RatioSechageAirLibre"]))

if __name__ == '__main__': 
       
    import time
    
    df_produits = pd.read_csv("DATA/df.csv")
    df_rulesDetails = pd.read_csv("DATA/rulesDetails.csv")
    df_produits = pd.concat([df_produits.iloc[0:5],df_produits.iloc[75:80]],ignore_index = True) # limiter à un sous-ensemble de produits
        
    paramSimu = {"df_produits": df_produits,
             "df_rulesDetails": df_rulesDetails,
             "SimulationParContainer": False,
             "DureeSimulation": 50,
             "nbLoader": 1,
             "nbSechoir": 4,
             "ConserverListeEvenements": True,
             "CapaciteSortieSciage": 100,
             "CapaciteSechageAirLibre": 0,
             "CapaciteCours": 0,
             "CapaciteSechoir": 1,
             "TempsAttenteLoader": 1,
             "TempsDeplacementLoader": 5,
             "TempsSechageAirLibre": 7 * 24,
             "RatioSechageAirLibre": 0.1 * 12 / 52,
             "HresProdScieriesParSem": 44 + 44,
             "VariationProdScierie": 0.1,  # Pourcentage de variation de la demande par rapport à la production de la scierie
             "VariationTempsSechage": 0.1,
             "VariationDemandeVSProd" : 0.25
             }

    # Pour faciliter le développement, on s'assure d'avoir toujours le mêmes
    # nombres aléatoires d'une exécution à l'autre
    random.seed(1)
    
    timer_avant = time.time()
    
    env = EnvSimpy(paramSimu)
    
    timer_avant = time.time()
               
    done = False
    _ = env.getState()
    while not done: 
        action = ChoixLoader(env)
        done = env.stepSimpy(action)
        _ = env.getState()
        
        voulu, relle = env.getProportions()
    
    
    timer_après = time.time()
    
    # juste pour faciliter débuggage...
    npLots = env.npLots
    df_rulesDetails = env.df_rulesDetails
    Evenement = env.Evenements
    df_produits = env.np_produits

    print("Temps d'exécution : ", timer_après-timer_avant)
    
    if paramSimu["ConserverListeEvenements"] : 
        print("Nb de déplacements de loader : ", len(Evenement[Evenement[:,1] == "Début déplacement"]))
        print("Nb de déplacement par minutes : ", len(Evenement[Evenement[:,1] == "Début déplacement"]) / (timer_après-timer_avant) * 60)
        print("Nb de déplacement par heures : ", len(Evenement[Evenement[:,1] == "Début déplacement"]) / (timer_après-timer_avant) * 60 * 60)