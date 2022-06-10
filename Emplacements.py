# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:17:14 2022

@author: ti_re
"""

import simpy

class Emplacements(simpy.Resource) : 
    def __init__(self,Nom,env, **kwargs):
        super().__init__(env, **kwargs)
        self.lstRequest = []
        self.Nom = Nom
        self.env = env
        self.PleinTotale = 0
        self.DebutPlein = -1
       
    def request(self,**kwargs) :
        
        request = super().request(**kwargs)
        self.lstRequest.append(request)
        return request
        
    def release(self,env) : 
        if self.count >= self.capacity : 

            env.EnrEven("Place à nouveau disponible", Destination=self.Nom)
            self.PleinTotale += self.env.now - self.DebutPlein
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
        
    def EstPlein(self) : 
        if self.count >= self.capacity : 
            return True
        else :
            return False
        
    # Retourne la proportion du temps que l'emplacement était plein
    def getTauxUtilisationComplet(self) : 
        
        PleinTotal = self.PleinTotale
        if self.DebutPlein != -1 : 
            PleinTotal += self.env.now - self.DebutPlein
        
        return PleinTotal / self.env.now
        
        