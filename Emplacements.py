# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:17:14 2022

@author: ti_re
"""

import simpy
from Temps import *

class Emplacements(simpy.Resource) : 
    def __init__(self,Nom,env,df_horaire = None, **kwargs):
        super().__init__(env, **kwargs)
        self.lstRequest = []
        self.Nom = Nom
        self.env = env
        self.PleinTotale = 0
        self.DebutPlein = -1
        self.HeureFinPrevue = 0
        
        if df_horaire is None :
            self.df_horaire = work_schedule(day_start = 0, day_end = 24, work_on_weekend = True)
        else : 
            self.df_horaire = df_horaire       
               
    def request(self,**kwargs) :
        
        request = super().request(**kwargs)
        self.lstRequest.append(request)
        return request
        
    def release(self,env) : 
                        
        self.HeureFinPrevue = 0
        if self.count >= self.capacity : 

            env.EnrEven("Place à nouveau disponible", Destination=self.Nom)
            
            if self.DebutPlein != -1 : 
                self.PleinTotale += HeuresProductives(self.df_horaire,max(self.env.DebutRegimePermanent,self.DebutPlein),self.env.now)
                self.DebutPlein = -1
        
        request = self.lstRequest.pop(0)
        super().release(request)

    # L'événement est affiché séparément du request dans l'événement puisqu'il y a un gros décalage de temps
    # entre le moment ou on s'assure de la disponibilité et le moment ou on voit vraiment la quantité arrivée
    # C'est donc plus clair ainsi quand on debug en regardant la liste d'événements... 
    def LogCapacite(self) :
        
        if self.count == self.capacity : 
            self.env.EnrEven("Capacité maximale atteinte", Destination=self.Nom)
            self.DebutPlein = self.env.now
        
    def EstPlein(self): 
        return self.count >= self.capacity
        
    # Retourne la proportion du temps que l'emplacement était plein
    def getTauxUtilisationComplet(self) : 
        
        PleinTotal = self.PleinTotale
        if self.DebutPlein != -1 : 
            PleinTotal += HeuresProductives(self.df_horaire,max(self.env.DebutRegimePermanent,self.DebutPlein),self.env.now)
            
        return PleinTotal / HeuresProductives(self.df_horaire,self.env.DebutRegimePermanent,self.env.now) if HeuresProductives(self.df_horaire,self.env.DebutRegimePermanent,self.env.now) > 0 else 0
        
        