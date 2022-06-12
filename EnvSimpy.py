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
from Temps import *

############### Done
# prendre en compte dernier fichier Francis pour temsp chargement
# get taux utilisation loader
# get taux utilisation scierie avec commentaire que sort même si bloqué...
# get taux utilisation séchoir
# get quantités totales sciées et séchées
# attente au lieu d'action aléatoire sur action invalide
# getIndicateursInventaire
# nouveau state
# paramètre pour nuancer la sortie de la scierie 
# gestion tannante numpy (1ère ligne et string)
# récupération du main (les paramSimu)
# prendre en compte les différentes planification de production (mixte, épinette, sapin)
# horaire loader et scierie

############### Idées, mais à voir si on va le faire
# Inclure du séchage à l'air libre (voir procédure SechageAirLibre déjà commencée)
# Pour l'instant envoi dans le premier séchoir qui a de la place si plusieurs libres... pourrait remplir le rail devant mauvais séchoir...
# passage par la cours
# gérer temps attente loader pour prochain événement au lieu d'attendre 1 heure s'il n'a rien à faire...
# aléatoire selon les quantités de chaque produits dans la cours au lieu de juste choisir une règle (éviterait de toujours vider les règles)

############### à faire
# Générer une cours initiale pour ne pas commencer à vide (attention pour ne pas brisé les indicateurs qui compte les qtés qui sortent direct sur npCharg)
# gestion des planifications (dans simulation ou RL ?)



class EnvSimpy(simpy.Environment):
    
    def __init__(self,paramSimu,**kwargs):
        super().__init__(**kwargs)
        
        # Initialisation des attributs avec valeur hardcoder
        self.nbStep = 0
        self.RewardActionInvalide = False
        self.DernierCharg = 0
        self.DebutRegimePermanent = 999999999999
        
        # Gestion de la proportion Sapin/Épinette
        if paramSimu["RatioSapinEpinette"] == "50/50" : 
            self.PropEpinettesSortieSciage = 0.5
        elif paramSimu["RatioSapinEpinette"] == "75/25" : 
            self.PropEpinettesSortieSciage = 0.25
        elif paramSimu["RatioSapinEpinette"] == "25/75" : 
            self.PropEpinettesSortieSciage = 0.75
        else :
            print("Paramètre RatioSapinEpinette invalide.  Attend 25/75, 50/50 ou 75/25.")
            exit
                
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
        self.cRegle = {}
        for indice,colonne in enumerate(self.np_regles.columns) : 
            self.cRegle[colonne] = indice
        self.np_regles = np.vstack((np.asarray(self.np_regles.columns,dtype='U50'),self.np_regles.to_numpy(dtype='U50')))
        
        ####### Création des différents emplacements ou le bois peut se trouver.  
        self.lesEmplacements = {}
        self.lesEmplacements["Sortie sciage"] = Emplacements(Nom = "Sortie sciage", env=self,df_horaire=paramSimu["df_HoraireScierie"],capacity=self.paramSimu["CapaciteSortieSciage"])
        
        if self.paramSimu["CapaciteCours"] > 0  : 
            self.lesEmplacements["Cours"] = Emplacements(Nom = "Cours", env=self,capacity=self.paramSimu["CapaciteCours"])
        
        if self.paramSimu["CapaciteSechageAirLibre"] > 0  : 
            self.lesEmplacements["Séchage à l'air libre"] = Emplacements(Nom = "Séchage à l'air libre", env=self,capacity=self.paramSimu["CapaciteSechageAirLibre"])
        
        for i in range(self.paramSimu["nbSechoir1"]) : 
            self.lesEmplacements["Préparation séchoir " + str(i+1)] = Emplacements(Nom = "Préparation séchoir " + str(i+1), env=self,capacity=2)
            self.lesEmplacements["Séchoir " + str(i+1)] = Emplacements(Nom = "Séchoir " + str(i+1), env=self,capacity=1)
        #######
        
        # Création des différents loaders nécessaires pour la simulation
        self.df_HoraireLoader = paramSimu["df_HoraireLoader"]
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
        
        self.InitEnvInitialAleatoire()


    # Initialise une cours aléatoirement en roulant la simulation pendant un 
    # certains temps (régime transitoire).  On considère une cours intéressante
    # quand nous avons maximum 1 séchoir de libre et que notre cours contient
    # au moins le nombre de chargement de notre objectif stable - 2
    def InitEnvInitialAleatoire(self) : 
        
        while sum(self.QteDansCours - self.ObjectifStable) < -2 or NbSechoirsPleins < self.paramSimu["nbSechoir1"] -1 : 
            self.stepSimpy(pile_la_plus_elevee(self))
            
            NbSechoirsPleins = 0
            for i in range(self.paramSimu["nbSechoir1"]) : 
                NbSechoirsPleins += self.lesEmplacements["Séchoir " + str(i+1)].EstPlein()

            
        self.nbStep = 0
        self.DebutRegimePermanent = self.now
        
        self.EnrEven("Début régime permanent")
        print("Début régime permanent")
                
    # Enregistre simplement un événement dans la liste des événements passés
    def EnrEven(self,Evenement,NomLoader=None, Charg = None, Source = None, Destination = None) : 

        if Charg == None : 
            description = None
        else : 
            description = self.npCharg[Charg][self.cCharg["description"]]
    
        nouveau = np.asarray([[self.now,Evenement,NomLoader, Source, Destination, Charg,description]],dtype='U50')
        self.Evenements = np.vstack((self.Evenements,nouveau))
 
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

           
    # Conserve un attribut LienActionCharg qui permet de faire le lien entre le numéro de l'action
    # retourné par l'agent et le chargement correspondant qu'on doit déplacer (dans npCharg)
    def updateLienActionChargement(self) : 

        self.LienActionCharg = []
        for produit in self.np_regles[1:,self.cRegle["regle"]] :            

            charg = self.npCharg[(self.npCharg[:,self.cCharg["regle"]] == produit) & (self.npCharg[:,self.cCharg["Emplacement"]] == "Sortie sciage"),self.cCharg["Charg"]].astype(int)
            if len(charg) == 0 :
                charg = -1
            else:
                charg = np.min(charg)
            
            self.LienActionCharg.append(charg)  

    # Update des attributs qui conservent les différentes informations pour les métriques concernant
    # les quantitées en inventaire    
    def updateInventaire (self) :
        
        # Quantité sciée par règles en s'assurant d'avoir le produit dans la liste même si la quantité est à 0
        Charg = np.concatenate((self.np_regles[1:,self.cRegle["regle"]],self.npCharg[1:,self.cCharg["regle"]]))
        unique,count = np.unique(Charg.astype(int),return_counts = True)        
        count = (count-1)
        self.QteTotaleSciee = count
        
        # Quantité séchée par règles en s'assurant d'avoir le produit dans la liste même si la quantité est à 0
        Charg = np.concatenate((self.np_regles[1:,self.cRegle["regle"]],self.npCharg[self.npCharg[:,self.cCharg["Emplacement"]] == "Sortie séchoir",self.cCharg["regle"]]))
        unique,count = np.unique(Charg.astype(int),return_counts = True)        
        count = (count-1)
        self.QteTotaleSecher = count
        
        # Quantité dans la cours de chaque produit en s'assurant d'avoir le produit dans la liste même si la quantité est à 0
        QteRegle = self.np_regles[1:,self.cRegle["sechoir1"]].astype(int)
        Charg = np.concatenate((self.np_regles[1:,self.cRegle["regle"]],self.npCharg[self.npCharg[:,self.cCharg["Emplacement"]] == "Sortie sciage",self.cCharg["regle"]]))
        unique,count = np.unique(Charg.astype(int),return_counts = True)        
        count = (count-1)
        self.QteDansCours = count
        self.proportionReelle = count / max(1,sum(count))
                
        # Proportions qu'on veut avoir dans la cours
        self.proportionVoulu = self.np_regles[1:,self.cRegle["proportion 50/50"]].astype(float)
        
        # Ce que ça représente comme objectif stable et dynamiques
        self.ObjectifStable = self.proportionVoulu * self.paramSimu["ObjectifStableEnPMP"] / QteRegle
        ObjectifDynamique = self.proportionVoulu * sum(self.QteDansCours * QteRegle) / QteRegle
        self.ObjectifDynamiqueInf = ObjectifDynamique.astype(float) -1
        self.ObjectifDynamiqueSup = ObjectifDynamique.astype(float) +1 
        
        # La différence est en PMP à l'heure pour rester dans des ranges similaires avec le temps pour une longue simulation
        self.lstProdVsDemande = (self.proportionVoulu - self.proportionReelle)

    # Lancée au début complètement et après chaque step pour calculer et mettre à jour les attributs 
    # une seule fois s'ils sont utiles à plusieurs endroits
    def updateApresStep(self) :

        self.updateLienActionChargement()
        self.updateInventaire()

    def getIndicateursInventaire(self) : 
        
        return self.QteDansCours, self.ObjectifStable, self.ObjectifDynamiqueInf, self.ObjectifDynamiqueSup

    # Retourne le State à l'agent.  Celui-ci devrait être déjà prêt par les mises à jours lancées dans updateApresStep
    def getState(self) : 
        
        # Pour ancien state sur les proportion...
        # Est-ce que les stocks sont stables dans la cours
        #lstProdVsDemandeMinMax = min_max_scaling(self.lstProdVsDemande.astype(float),-25000,25000)
        
        # Quantités dans la cours (liste par règles)
        QteDansCours = min_max_scaling(self.QteDansCours,0,self.paramSimu["CapaciteSortieSciage"])
        
        # Information sur le moment
        day_of_week, hour = GetInfosTemps(self.now)
        day_of_week = min_max_scaling(day_of_week,1,7)
        hour = min_max_scaling(hour,0,24)

        return np.concatenate(([self.PropEpinettesSortieSciage],QteDansCours,[day_of_week,hour]))
    
    # Retourne la quantité totale séchée et sciée pour chaque produits depuis le début de la simulation
    def getQteTotale(self) :     
        return self.QteTotaleSecher,self.QteTotaleSciee
    
    # Retourne les proportions voulu et réelle du stock dans la cours
    def getProportions(self) : 
        
        return self.proportionVoulu, self.proportionReelle

    # Retourne la différence entre la proportion voulue dans la cours et la proportion réelle qu'on a    
    def getRespectInventaire(self) : 
        return self.lstProdVsDemande
    
    # Retourne le taux d'utilisation des loader.  Le calcul tient compte de toutes les attentes
    # terminées ainsi que l'attente en cours s'il y a lieu
    def getTauxUtilisationLoader(self) : 

        AttenteTotale = 0
        for key in self.lesLoader.keys() : 
            AttenteTotale += self.lesLoader[key].AttenteTotale            
            if self.lesLoader[key].bAttente :
                
                AttenteTotale += HeuresProductives(self.df_HoraireLoader,max(self.DebutRegimePermanent,self.lesLoader[key].debutAttente),self.now)
                
        #print(self.now,self.DebutRegimePermanent,AttenteTotale,HeuresProductives(self.df_HoraireLoader,self.DebutRegimePermanent,self.now))
        #exit
                
        return 1 - (AttenteTotale / self.paramSimu["nbLoader"] / HeuresProductives(self.df_HoraireLoader,self.DebutRegimePermanent,self.now)) if self.now>0 else 0

    # Retourne le taux d'utilisation de la scierie.  Le calcul tient compte de toutes les attentes
    # terminées ainsi que l'attente en cours s'il y a lieu.  L'attente est comptée juste
    # une fois car on a une seule scierie et ce même si plusieurs chargement tombent en attente
    def getTauxUtilisationScierie(self) : 
        
        return 1 - self.lesEmplacements["Sortie sciage"].getTauxUtilisationComplet()

    # Retourne le taux d'utilisation des séchoirs.  Le calcul tient compte de toutes les attentes
    # terminées ainsi que l'attente en cours s'il y a lieu.  
    def getTauxUtilisationSechoirs(self) : 
        
        TempsComplet = 0
        NbSechoirs = 0
        for key in self.lesEmplacements.keys() :
            if "Séchoir" in key :
                TempsComplet += self.lesEmplacements[key].getTauxUtilisationComplet()
                NbSechoirs += 1
                
        return TempsComplet / NbSechoirs
    
    def getTauxRemplissageCours(self):
        return self.lesEmplacements["Sortie sciage"].count / self.paramSimu["CapaciteSortieSciage"]

# Lance les différents sciages selon le rythme prédéterminé dans les règles
def Sciage(env,npUneRegle) :
    
    bPremierSciage = True
    
    # Récupération des informations de bases
    produit = int(npUneRegle[env.cRegle["regle"]])
    description = npUneRegle[env.cRegle["description"]]
    volumeSechoir1 = float(npUneRegle[env.cRegle["sechoir1"]])
    tempsSechage = float(npUneRegle[env.cRegle["temps sechage"]])
    
    # Ramener le temps nécessaire pour le produit en heure pour utilisation par le yield plus tard
    if env.paramSimu["RatioSapinEpinette"] == "50/50" : 
        prodMoyPMPSem = max(0,float(npUneRegle[env.cRegle["production mixte"]])) * env.paramSimu["FacteurSortieScierie"]
    elif env.paramSimu["RatioSapinEpinette"] == "75/25" : 
        prodMoyPMPSem = max(0,float(npUneRegle[env.cRegle["production epinette"]])) * env.paramSimu["FacteurSortieScierie"]
    elif env.paramSimu["RatioSapinEpinette"] == "25/75" : 
        prodMoyPMPSem = max(0,float(npUneRegle[env.cRegle["production sapin"]])) * env.paramSimu["FacteurSortieScierie"]
    
    prodMoyPMPHr = prodMoyPMPSem / env.paramSimu["HresProdScieriesParSem"]
    
    if prodMoyPMPHr > 0 : 
        dureeMoyCharg = volumeSechoir1 / prodMoyPMPHr
        dureeMinCharg = dureeMoyCharg * (1-env.paramSimu["VariationProdScierie"])
        dureeMaxCharg = dureeMoyCharg * (1+env.paramSimu["VariationProdScierie"])   
    
    while prodMoyPMPHr > 0 :

        # Première itération de la boucle, on met un délai pouvant être plus court pour ne pas avoir une
        # période au début de la simulation qu'il n'y a rien qui sort de la scierie
        if bPremierSciage : 
            duree = max(1,random.random() * dureeMoyCharg)
            yield env.timeout(task_total_length(env.lesEmplacements["Sortie sciage"].df_horaire,env.now,duree)) 
        
        # Temps inter-sortie du sciage standard pour les itérations autre que la première
        else : 
            duree = random.triangular(dureeMinCharg,dureeMaxCharg,dureeMoyCharg)
            yield env.timeout(task_total_length(env.lesEmplacements["Sortie sciage"].df_horaire,env.now,duree)) 

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
def Sechage(env,charg,NomPrepSechage) : 
    
    NomSechoir = "Séchoir" + NomPrepSechage[len("Préparation séchoir"):]   
    produit = int(env.npCharg[charg][env.cCharg["regle"]])
    tempsSechage = float(env.npCharg[charg][env.cCharg["temps sechage"]])
    tempsMin = tempsSechage * (1-env.paramSimu["VariationTempsSechage"])
    tempsMax = tempsSechage * (1+env.paramSimu["VariationTempsSechage"])
    
    # Transférer le wagon dans le séchoir et commencer le séchage
    yield env.lesEmplacements[NomSechoir].request()   
    env.EnrEven("Début séchage",Charg = charg, Destination = NomSechoir)
    env.lesEmplacements[NomSechoir].LogCapacite()
    yield env.timeout(random.triangular(tempsMin,tempsMax,tempsSechage))
    
    # Le séchage est terminé, libérer le séchoir pour pouvoir faire entrer un autre wagon
    env.EnrEven("Fin séchage",Charg = charg, Destination = NomSechoir)
    env.lesEmplacements[NomSechoir].release(env)   
    env.npCharg[charg][env.cCharg["Emplacement"]] = "Sortie séchoir"
    
    # Décharger le wagon qui vient de sortir du séchoir et le libérer pour qu'il 
    # puisse être à nouveau rempli (on prend pour acquis qu'on a un loader de disponible)
    regle = env.npCharg[charg][env.cCharg["regle"]]
    TempsDeChargement = float(env.np_regles[env.np_regles[:,env.cRegle["regle"]] == regle,env.cRegle["temps chargement"]])
    dureemin = TempsDeChargement * (1-env.paramSimu["VariationTempsDeplLoader"])
    dureemax = TempsDeChargement * (1+env.paramSimu["VariationTempsDeplLoader"]) 
    duree = random.triangular(dureemin,dureemax,TempsDeChargement)
    yield env.timeout(task_total_length(env.df_HoraireLoader,env.now,duree)) 
    env.lesEmplacements[NomPrepSechage].release(env)
       

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
    import matplotlib.pyplot as plt
    
    regles = pd.read_csv("DATA/regle.csv")
    df_HoraireScierie = work_schedule()
    df_HoraireLoader = work_schedule()
        
    paramSimu = {"df_regles": regles,
             "df_HoraireLoader" : df_HoraireLoader,
             "df_HoraireScierie" : df_HoraireScierie,
             "NbStepSimulation": 64*5,
             "NbStepSimulationTest": 64*2,
             "nbLoader": 1,
             "nbSechoir1": 4,
             "CapaciteSortieSciage": 250, # En nombre de chargements avant de bloquer la scierie
             "CapaciteSechageAirLibre": 0,
             "CapaciteCours": 0,
             "TempsAttenteLoader": 1,
             "TempsAttenteActionInvalide": 10,
             "TempsSechageAirLibre": 7 * 24,
             "RatioSechageAirLibre": 0.1 * 12 / 52,
             "HresProdScieriesParSem": sum(df_HoraireScierie[:168]["work_time"]),
             "VariationProdScierie": 0.1,  # Pourcentage de variation de la demande par rapport à la production de la scierie
             "VariationTempsSechage": 0.1,
             "VariationTempsDeplLoader": 0.1,
             "FacteurSortieScierie" : 1, # Permet de sortir plus ou moins de la scierie (1 correspond à sortir exactement ce qui est prévu)
             "ObjectifStableEnPMP" : 215000 * 4 * 2.5,
             "RatioSapinEpinette" : "50/50"
             }

    
    # Pour faciliter le développement, on s'assure d'avoir toujours le mêmes
    # nombres aléatoires d'une exécution à l'autre
    random.seed(1)
    
    timer_avant = time.time()
    
    env = EnvSimpy(paramSimu)
    
    timer_avant = time.time()
               
    done = False
    _, reelle = env.getProportions()
    lstQteDansCours = []
    lstQteStable = []
    lstinf = []
    lstsup = []
    lstUtilLoader = []
    lstUtilSechoir = []
    lstUtilScierie = []
    propReelle = reelle
    while not done:         
        done = env.stepSimpy(pile_la_plus_elevee(env)) #ActionValideAleatoire(env))
        _, reelle = env.getProportions()
        propReelle += reelle
        QteDansCours, QteStable, inf, sup = env.getIndicateursInventaire()
        lstQteDansCours.append(QteDansCours[4])
        lstQteStable.append(QteStable[4])
        lstinf.append(inf[4])
        lstsup.append(sup[4])
        lstUtilScierie.append(env.getTauxUtilisationScierie())
        lstUtilSechoir.append(env.getTauxUtilisationSechoirs())
        lstUtilLoader.append(env.getTauxUtilisationLoader())
    
    plt.plot(lstQteDansCours)
    plt.plot(lstQteStable)
    plt.plot(lstinf)
    plt.plot(lstsup)
    plt.show()
    
    plt.plot(lstUtilLoader,label="loader")
    plt.plot(lstUtilSechoir,label="Sechoir")
    plt.plot(lstUtilScierie,label="Scierie")
    plt.legend()  
    plt.show()
    
    timer_après = time.time()
    
    # juste pour faciliter débuggage...
    npCharg = env.npCharg
    Evenement = env.Evenements
    regles = env.np_regles

    #print(propVoulu)
    #print(propReelle)
    print("Temps d'exécution : ", timer_après-timer_avant)
    #print("Durée en h/j/an de la simulation : ", env.now, env.now/ 24, env.now/24/365)
    
    print("Nb de déplacements de loader : ", len(Evenement[Evenement[:,1] == "Début déplacement"]))
    print("Nb de déplacement par minutes : ", len(Evenement[Evenement[:,1] == "Début déplacement"]) / (timer_après-timer_avant) * 60)
    print("Nb de déplacement par heures : ", len(Evenement[Evenement[:,1] == "Début déplacement"]) / (timer_après-timer_avant) * 60 * 60)