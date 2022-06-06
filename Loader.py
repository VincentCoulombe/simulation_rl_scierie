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


def ActionValideAleatoire(env) :
    if max(env.LienActionLot) == -1 :
        return -1, -1
    lot = -1
    while lot == -1 :
        action = random.randint(0,3*(len(env.paramSimu["df_produits"])-1))        
        lot = env.LienActionLot[action]
            
    return action, lot

# retourne lot, source, destination selon un state passé en paramètre
def ChoixLoader(env) : 
           
    action,lot = ActionValideAleatoire(env)
    return action

    lstChoixSource = []
    lstChoixDestination = []
    
    if len(env.pdLots[env.pdLots["Emplacement"] == "Sortie sciage"]) != 0 :
        lstChoixSource.append("Sortie sciage")
        
    if len(env.pdLots[env.pdLots["Emplacement"] == "Cours"]) != 0 :
        lstChoixSource.append("Cours")
    
    if len(env.pdLots[env.pdLots["Emplacement"] == "Séchage à l'air libre"]) != 0 :
        lstChoixSource.append("Séchage à l'air libre")    
    ################### MANQUE CONTRAINTE SECHAGE AIR LIBRE TERMINER #########################
    
    
    if len(lstChoixSource) == 0 : 
        return np.nan, "Attente","Attente"
    
    else : 
        
        source = random.choice(lstChoixSource)
        
        if env.paramSimu["CapaciteCours"] > 0 : 
            if env.lesEmplacements["Cours"].EstPlein() == False and source == "Sortie sciage" :
                lstChoixDestination.append("Cours")
                
        if env.paramSimu["CapaciteSechageAirLibre"] > 0 : 
            if env.lesEmplacements["Séchage à l'air libre"].EstPlein() == False and source == "Sortie sciage" :
                lstChoixDestination.append("Séchage à l'air libre")
            
        for i in range(env.paramSimu["nbSechoir"]) : 
            if env.lesEmplacements["Préparation séchoir " + str(i+1)].EstPlein() == False :
                lstChoixDestination.append("Préparation séchoir " + str(i+1))
        
        if len(lstChoixDestination) == 0 : 
            return np.nan, "Attente", "Attente"    
        else :
            destination = random.choice(lstChoixDestination)
                        
            lot = env.pdLots[env.pdLots["Emplacement"] == source]["Lot"].to_numpy()
            lot = random.choice(lot)                    
            return lot, source, destination
        
class Loader() : 
    def __init__(self,NomLoader,env):
        self.NomLoader = NomLoader 
        self.bAttente = False
        self.env = env
        self.ProchainTemps = 0
        #self.LoaderEmplacement = Emplacements(Nom = NomLoader, env=env,capacity=1)
        self.lot = None
        
    def DeplacerLoader(self, action) : 
       
        self.env.RewardActionInvalide = False
       
        source = "Sortie sciage"
        lot = self.env.LienActionLot[action]
        if lot == -1 :
            action,lot = ActionValideAleatoire(self.env)
            if lot == -1 : 
                source = "Attente"
            else : 
                print("Action invalide demandée.  Remplacement par une action aléatoire.")
                self.env.RewardActionInvalide = True
                    
        destination = "Attente"
        for key in self.env.lesEmplacements.keys() : 
            if "Préparation séchoir" in key :
                if not self.env.lesEmplacements[key].EstPlein() :
                    destination = self.env.lesEmplacements[key].Nom
            
                    
        if destination == "Attente" or source == "Attente": 
            if not self.bAttente :
                self.bAttente = True 
                self.env.EnrEven("Mise en attente d'un loader",NomLoader = self.NomLoader)
            self.ProchainTemps += self.env.paramSimu["TempsAttenteLoader"]
            
        else : 
            
            if self.env.lesEmplacements[destination].EstPlein() : 
                raise "L'action choisie mène à une destination qui est pleine ce qui entraîne une attente infinie."
                
            self.env.lesEmplacements[destination].request()
                
            self.bAttente = False
            self.env.EnrEven("Début déplacement",NomLoader = self.NomLoader,Lot = lot, Source = source, Destination = destination)
            self.env.pdLots.loc[self.env.pdLots["Lot"] == lot,"Emplacement"] = self.NomLoader
            self.env.lesEmplacements[source].release(self.env)
            
            duree = random.expovariate(1/self.env.paramSimu["TempsDeplacementLoader"])
            
            self.source = source
            self.destination = destination
            self.lot = lot
            
            self.ProchainTemps += duree
            
    def FinDeplacementLoader(self) : #,duree,lot,source,destination) : 
        
        
        if self.lot != None : 
        
            self.env.EnrEven("Fin déplacement",NomLoader = self.NomLoader,Lot = self.lot,Source = self.source, Destination = self.destination)
            self.env.pdLots.loc[self.env.pdLots["Lot"] == self.lot,"Emplacement"] = self.destination
            self.env.LogCapacite(self.env.lesEmplacements[self.destination]) 
            
            # Procédé au séchage à l'air libre
            self.env.process(EnvSimpy.SechageAirLibre(self.env,self.lot,self.destination))
            
            # Procédé au séchage
            if "Préparation séchoir" in self.destination : 
                self.env.process(EnvSimpy.Sechage(self.env,self.lot,self.destination))
            
            self.lot = None


if __name__ == '__main__': 
    
      pass