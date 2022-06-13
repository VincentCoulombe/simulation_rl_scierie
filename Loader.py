# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 08:31:30 2022

@author: ti_re
"""

import simpy
import random
import pandas as pd
import numpy as np
import EnvSimpy
from Temps import *

        
class Loader() : 
    def __init__(self,NomLoader,env):
        self.NomLoader = NomLoader 
        self.bAttente = False
        self.env = env
        self.ProchainTemps = 0
        self.AttenteTotale = 0
        self.charg = None
        
    def DeplacerLoader(self, action) : 
       
        self.env.RewardActionInvalide = False
       
        source = "Sortie sciage"
              
        # Si on demande explicitement d'attendre (action = -1) alors c'est ce qu'on fait.  
        # On attend aussi si aucune action n'est possible.
        if action == -1 or not self.env.sourceDisponible() or not self.env.destinationDisponible() :
            charg = -1
            source = "Attente"
            duree = self.env.paramSimu["TempsAttenteLoader"]
            
        # On a demandé une action en particulier.  Si l'action est invalide, le loader attend
        # simplement un temps X en attendant une nouvelle action
        else :
            charg = self.env.LienActionCharg[action]
            if charg == -1 :
                source = "Attente"
                self.env.EnrEven("Action invalide demandée.",NomLoader = self.NomLoader,)
                self.env.RewardActionInvalide = True
                duree = self.env.paramSimu["TempsAttenteActionInvalide"]
                        
        # Trouver un séchoir de libre
        destination = self.env.GetDestinationCourante()
                                
        # On doit attendre, effectuer cette attente.  Mettre un message dans les événements seulement la
        # première fois s'il y a des attentes successives
        if destination == "Attente" or source == "Attente": 
            if not self.bAttente :
                self.bAttente = True 
                self.debutAttente = self.env.now
                self.env.EnrEven("Mise en attente d'un loader",NomLoader = self.NomLoader)
            self.ProchainTemps += task_total_length(self.env.df_HoraireLoader,self.env.now,duree)
           
        # Effectuer réellement l'action demandée car elle est valide
        else : 
            
            # Calcul de la durée prévue du déplacement du loader
            regle = self.env.npCharg[charg][self.env.cCharg["regle"]]
            TempsChargement = float(self.env.np_regles[self.env.np_regles[:,self.env.cRegle["regle"]] == regle,self.env.cRegle["temps chargement"]])
            dureemin = TempsChargement * (1-self.env.paramSimu["VariationTempsDeplLoader"])
            dureemax = TempsChargement * (1+self.env.paramSimu["VariationTempsDeplLoader"]) 
            duree = random.triangular(dureemin,dureemax,TempsChargement)
            
            if self.env.lesEmplacements[destination].EstPlein() : 
                raise "L'action choisie mène à une destination qui est pleine ce qui entraîne une attente infinie."
                
            self.env.lesEmplacements[destination].request()
                
            # s'il était précédemment en attente, terminé l'attente et conserver la durée pour nos indicateurs
            if self.bAttente : 
                self.bAttente = False
                self.AttenteTotale += HeuresProductives(self.env.df_HoraireLoader,max(self.env.DebutRegimePermanent,self.debutAttente),self.env.now)
            
            self.env.EnrEven("Début déplacement",NomLoader = self.NomLoader,Charg = charg, Source = source, Destination = destination)
            self.env.npCharg[charg][self.env.cCharg["Emplacement"]] = self.NomLoader
            self.env.lesEmplacements[source].release(self.env)
            
            self.source = source
            self.destination = destination
            self.charg = charg
            
            self.ProchainTemps += task_total_length(self.env.df_HoraireLoader,self.env.now,duree)
            
    def FinDeplacementLoader(self) : 
        
        # Si le loader est bien entrain de faire un déplacement, terminé ce dernier et lancer
        # les process nécessaires pour la suite de la vie du chargement
        if self.charg != None : 
        
            self.env.EnrEven("Fin déplacement",NomLoader = self.NomLoader,Charg = self.charg,Source = self.source, Destination = self.destination)
            self.env.npCharg[self.charg][self.env.cCharg["Emplacement"]] = self.destination
            self.env.lesEmplacements[self.destination].LogCapacite()
            
            # Procédé au séchage
            if "Préparation séchoir" in self.destination : 
                self.env.process(EnvSimpy.Sechage(self.env,self.charg,self.destination))
            
            self.charg = None


if __name__ == '__main__': 
    
      pass