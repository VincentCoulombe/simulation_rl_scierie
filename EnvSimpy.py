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
from Emplacements import *
from utils import *


# Inclure du séchage à l'air libre (voir procédure SechageAirLibre déjà commencée)
# Pour l'instant envoi dans le premier séchoir qui a de la place si plusieurs libres
# prendre en compte les différentes plannification de production (mixte, épinette, sapin)
# temps de déplacement des loader

################### MANQUE CONTRAINTES WAGON EN GÉNÉRAL #########################
################### MANQUE SIMPY POUR WAGON VS SECHAGE #########################
################### gérer temps attente loader comme du monde #########################
# séchage ne part pas au bon moment
# indicateurs taux d'utilisation (loader, scierie, séchage), quantitée sécher par règles
# Générer une cours initiale pour ne pas commencer à vide

class EnvSimpy(simpy.Environment):
    
    def __init__(self,paramSimu,**kwargs):
        super().__init__(**kwargs)
        
        # Initialisation des attributs avec valeur hardcoder
        self.nbStep = 0
        self.RewardActionInvalide = False
        self.PropEpinettesSortieSciage = 0.5
        self.DernierCharg = 0
        
        # Initialisation des attributs provenant de paramètres
        self.np_regles = paramSimu["df_regles"]        
        self.paramSimu = paramSimu
        
        # Création des numpy pour conserver tout le calendrier d'événements ainsi que chaque chargement produit
        self.Evenements = np.asarray([["Temps","Événement","Loader","Source", "Destination","Charg","description"]],dtype='U50')
        self.npCharg = np.array([["Temps","Charg","regle","description","Emplacement","temps sechage"]],dtype='U50')
                
        
        # Dictionnaires permettant de se rendre indépendant des numéros de colonnes malgré qu'on soit 
        # dans numpy.  ce dictionnaire est lié à l'attribut npCharg
        self.cCharg = {"Temps" : 0,
                      "Charg" : 1,
                      "regle" : 2,
                      "description" : 3,
                      "Emplacement" : 4,
                      "temps sechage" : 5}
        
        # Transférer le pandas dans un numpy en conservant le nom des colonnes comme première ligne
        # pour faciliter le débuggage.  Un dictionnaire permettant de se rendre indépendant des numéros 
        # de colonnes malgré qu'on soit dans numpy est créer en même temps.
        self.cProd = {}
        for indice,colonne in enumerate(self.np_regles.columns) : 
            self.cProd[colonne] = indice
        self.np_regles = np.vstack((np.asarray(self.np_regles.columns,dtype='U50'),self.np_regles.to_numpy(dtype='U50')))
        
        ####### Création des différents emplacements ou le bois peut se trouver.  
        self.lesEmplacements = {}
        self.lesEmplacements["Sortie sciage"] = Emplacements(Nom = "Sortie sciage", env=self,capacity=self.paramSimu["CapaciteSortieSciage"])
        
        if self.paramSimu["CapaciteCours"] > 0  : 
            self.lesEmplacements["Cours"] = Emplacements(Nom = "Cours", env=self,capacity=self.paramSimu["CapaciteCours"])
        
        if self.paramSimu["CapaciteSechageAirLibre"] > 0  : 
            self.lesEmplacements["Séchage à l'air libre"] = Emplacements(Nom = "Séchage à l'air libre", env=self,capacity=self.paramSimu["CapaciteSechageAirLibre"])
        
        for i in range(self.paramSimu["nbSechoir1"]) : 
            self.lesEmplacements["Préparation séchoir " + str(i+1)] = Emplacements(Nom = "Préparation séchoir " + str(i+1), env=self,capacity=self.paramSimu["CapaciteSechoir"])
        #######
        
        # Création des différents loaders nécessaires pour la simulation
        self.lesLoader = {}
        for i in range(self.paramSimu["nbLoader"]) : 
            self.lesLoader["Loader " + str(i+1)] = Loader(NomLoader = "Loader " + str(i+1), env = self)

        # Décoller des process de sciage pour toutes les règles
        for i in range(1,len(self.np_regles)) :
            self.process(Sciage(self,self.np_regles[i]))        

        # Utilisé pour calculer les attributs comme ils seraient après un step afin d'être prêt 
        # pour l'appel de step par l'agent
        self.updateApresStep()
        
        self.EnrEven("Début simulation")
        print("Début de la simulation")

    
    # Enregistre simplement un événement dans la liste des événements passés
    def EnrEven(self,Evenement,NomLoader=None, Charg = None, Source = None, Destination = None) : 

        if Charg == None : 
            description = None
        else : 
            description = self.npCharg[Charg][self.cCharg["description"]]
    
        nouveau = np.asarray([[self.now,Evenement,NomLoader, Source, Destination, Charg,description]],dtype='U50')
        self.Evenements = np.vstack((self.Evenements,nouveau))
            
    # Conserve un attribut LienActionCharg qui permet de faire le lien entre le numéro de l'action
    # retourné par l'agent et le chargement correspondant qu'on doit déplacer (dans npCharg)
    def updateLienActionChargement(self) : 

        self.LienActionCharg = []
        for produit in self.np_regles[1:,self.cProd["regle"]] :            

            charg = self.npCharg[(self.npCharg[:,self.cCharg["regle"]] == produit) & (self.npCharg[:,self.cCharg["Emplacement"]] == "Sortie sciage"),self.cCharg["Charg"]].astype(int)
            if len(charg) == 0 :
                charg = -1
            else:
                charg = np.min(charg)
            
            self.LienActionCharg.append(charg)  

    # Update des attributs qui conservent les différentes informations pour les métriques concernant
    # les quantitées en inventaire    
    def updateRespectInventaire (self) :
        
        # Quantité dans la cours de chaque produit en s'assurant d'avoir le produit dans la liste même si la quantité est à 0
        Charg = np.concatenate((self.np_regles[1:,self.cProd["regle"]],self.npCharg[self.npCharg[:,self.cCharg["Emplacement"]] == "Sortie sciage",self.cCharg["regle"]]))
        unique,count = np.unique(Charg.astype(int),return_counts = True)        
        count = (count-1)
        count = count / max(1,sum(count))
        
        # Proportions qu'on veut avoir dans la cours
        self.proportionVoulu = self.np_regles[1:,self.cProd["proportion 50/50"]].astype(float)
        self.proportionReelle = count
        
        # La différence est en PMP à l'heure pour rester dans des ranges similaires avec le temps pour une longue simulation
        self.lstProdVsDemande = (self.proportionVoulu - self.proportionReelle)

    # Lancée au début complètement et après chaque step pour calculer et mettre à jour les attributs 
    # une seule fois s'ils sont utiles à plusieurs endroits
    def updateApresStep(self) :

        self.updateLienActionChargement()
        self.updateRespectInventaire()

    # Retourne le Loader qui devrait être entrain de faire l'action courante ou la prochaine action
    # ainsi que l'heure à laquelle il prévoit faire cette dite action
    def RetLoaderCourant(self) : 
        
        minTemps = 9999999999999999999        
        minNom = ""
        for key in self.lesLoader.keys():
            if self.lesLoader[key].ProchainTemps < minTemps : 
                minTemps = self.lesLoader[key].ProchainTemps
                minNom = key
        
        return self.lesLoader[minNom], minTemps

    # Retourne si au moins une destination est disponible ou non
    def destinationDisponible(self) : 

        for key in self.lesEmplacements.keys() : 
            if "Préparation séchoir" in key :
                if not self.lesEmplacements[key].EstPlein() :
                    return True
                
        return False

    # Retourne si au moins une source est disponible ou non
    def sourceDisponible(self) : 
        
        return not (max(self.LienActionCharg) == -1)

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
        # Présentement, on fait attendre le loader 1h avant de vérifier le step à nouveau
        if (not self.sourceDisponible() or not self.destinationDisponible()) and self.nbStep < self.paramSimu["NbStepSimulation"] :
            self.updateLienActionChargement()
            return self.stepSimpy(-1)
                
        
        self.updateApresStep()
        self.nbStep += 1
        
        if self.nbStep >= self.paramSimu["NbStepSimulation"] :            
            self.EnrEven("Fin de la simulation")
            print("Simulation terminée")
            return True
        
        return False

    # Retourne le State à l'agent.  Celui-ci devrait être déjà prêt par les mises à jours lancées dans updateApresStep
    def getState(self) : 
        
        # Est-ce que les stocks sont stables dans la cours
        lstProdVsDemandeMinMax = min_max_scaling(self.lstProdVsDemande.astype(float),-25000,25000)
        
        return np.concatenate(([self.PropEpinettesSortieSciage],lstProdVsDemandeMinMax))
    
    # Retourne les proportions voulu et réelle du stock dans la cours
    def getProportions(self) : 
        
        return self.proportionVoulu, self.proportionReelle

    # Retourne la différence entre la proportion voulue dans la cours et la proportion réelle qu'on a    
    def getRespectInventaire(self) : 
        return self.lstProdVsDemande


# Lance les différents sciages selon le rythme prédéterminé dans les règles
def Sciage(env,npUneRegle) :
    
    # Récupération des informations de bases
    produit = int(npUneRegle[env.cProd["regle"]])
    description = npUneRegle[env.cProd["description"]]
    volumeSechoir1 = float(npUneRegle[env.cProd["sechoir1"]])
    tempsSechage = float(npUneRegle[env.cProd["temps sechage"]])
    
    # Ramener le temps nécessaire pour le produit en heure pour utilisation par le yield plus tard
    prodMoyPMPSem = max(0,float(npUneRegle[env.cProd["production mixte"]]))
    prodMoyPMPHr = prodMoyPMPSem / env.paramSimu["HresProdScieriesParSem"]
    dureeMoyCharg = volumeSechoir1 / prodMoyPMPHr
    dureeMinCharg = dureeMoyCharg * (1-env.paramSimu["VariationProdScierie"])
    dureeMaxCharg = dureeMoyCharg * (1+env.paramSimu["VariationProdScierie"])   
                    
    while True :

        # Temps inter-sortie du sciage
        yield env.timeout(random.triangular(dureeMinCharg,dureeMaxCharg,dureeMoyCharg)) 

        # On bloque la scierie s'il n'y a pas d'espace pour sortir le chargement
        yield env.lesEmplacements["Sortie sciage"].request()         

        # On sort la quantité de la scierie       
        env.DernierCharg += 1
        emplacement = "Sortie sciage"
        nouveaunp = np.asarray([[env.now,env.DernierCharg,produit,description,emplacement,tempsSechage]],dtype='U50')
        env.npCharg = np.vstack((env.npCharg,nouveaunp))
        env.EnrEven("Sortie sciage",Charg = env.DernierCharg)
        
        # Procédé au séchage à l'air libre
        env.process(SechageAirLibre(env,env.DernierCharg))
                
        env.lesEmplacements["Sortie sciage"].LogCapacite()

# Effectue le séchage selon les temps de séchage des chargements.  Le temps de séchage ici
# est affecté si on active le séchage à l'air libre.
def Sechage(env,charg,destination) : 
    
    produit = int(env.npCharg[charg][env.cCharg["regle"]])
    tempsSechage = float(env.npCharg[charg][env.cCharg["temps sechage"]])
    tempsMin = tempsSechage * (1-env.paramSimu["VariationTempsSechage"])
    tempsMax = tempsSechage * (1+env.paramSimu["VariationTempsSechage"])
    
    env.EnrEven("Début séchage",Charg = charg, Destination = destination)
    
    yield env.timeout(random.triangular(tempsMin,tempsMax,tempsSechage))
    
    env.EnrEven("Fin séchage",Charg = charg, Destination = destination)
    env.npCharg[charg][env.cCharg["Emplacement"]] = "Sortie séchoir"
    
    env.lesEmplacements[destination].release(env)
       

# Code débuté, mais non complété pour que le bois sèche dans la cours.
# Manque contraintes si on veut décidé de laissé X temps dans la cours pour forcer vraiment un séchage
# à l'air libre.  Manque aussi contrainte pour ne pas qu'il sèche complètement dans la cours.
# J'ai laissé l'appel parce qu'il est fonctionnel et on pourrait vouloir le réactiver, mais la ligne
# qui fait la mise à jour est en commentaire pour s'assurer que si on en tiens compte, on le fait comme 
# il faut.
def SechageAirLibre(env,charg): 
    
    while env.npCharg[charg][env.cCharg["Emplacement"]] != "Sortie séchoir" :
        
        yield env.timeout(env.paramSimu["TempsSechageAirLibre"])
        
        #env.npCharg[charg][env.cCharg["temps sechage"]] = str(float(env.npCharg[charg][env.cCharg["temps sechage"]]) * (1- env.paramSimu["RatioSechageAirLibre"]))

if __name__ == '__main__': 
       
    import time
    
    regles = pd.read_csv("DATA/regle.csv")
        
    paramSimu = {"df_regles": regles,
             "NbStepSimulation": 64*5,
             "NbStepSimulationTest": 64*2,
             "nbLoader": 1,
             "nbSechoir1": 4,
             "CapaciteSortieSciage": 100,
             "CapaciteSechageAirLibre": 0,
             "CapaciteCours": 0,
             "CapaciteSechoir": 1,
             "TempsAttenteLoader": 1,
             "TempsDeplacementLoader": 10,
             "TempsSechageAirLibre": 7 * 24,
             "RatioSechageAirLibre": 0.1 * 12 / 52,
             "HresProdScieriesParSem": 44 + 44,
             "VariationProdScierie": 0.1,  # Pourcentage de variation de la demande par rapport à la production de la scierie
             "VariationTempsSechage": 0.1,
             "VariationTempsDeplLoader": 0.1,
             "VariationDemandeVSProd" : 0.25
             }

    # Pour faciliter le développement, on s'assure d'avoir toujours le mêmes
    # nombres aléatoires d'une exécution à l'autre
    random.seed(1)
    
    timer_avant = time.time()
    
    env = EnvSimpy(paramSimu)
    
    timer_avant = time.time()
               
    done = False
    _, reelle = env.getProportions()
    propReelle = reelle
    while not done:         
        done = env.stepSimpy(ActionValideAleatoire(env))
        _, reelle = env.getProportions()
        propReelle += reelle
    
    
    timer_après = time.time()
    
    # juste pour faciliter débuggage...
    npCharg = env.npCharg
    Evenement = env.Evenements
    regles = env.np_regles

    #print(propVoulu)
    print(propReelle)
    print("Temps d'exécution : ", timer_après-timer_avant)
    #print("Durée en h/j/an de la simulation : ", env.now, env.now/ 24, env.now/24/365)
    
    print("Nb de déplacements de loader : ", len(Evenement[Evenement[:,1] == "Début déplacement"]))
    print("Nb de déplacement par minutes : ", len(Evenement[Evenement[:,1] == "Début déplacement"]) / (timer_après-timer_avant) * 60)
    print("Nb de déplacement par heures : ", len(Evenement[Evenement[:,1] == "Début déplacement"]) / (timer_après-timer_avant) * 60 * 60)