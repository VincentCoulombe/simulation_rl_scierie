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
       
    def request(self,**kwargs) :
        request = super().request(**kwargs)
        self.lstRequest.append(request)
        return request
        
    def release(self,env) : 
        if self.count >= self.capacity : 
            env.EnrEven("Place à nouveau disponible", Destination=self.Nom)
        
        request = self.lstRequest.pop(0)
        super().release(request)

    # L'événement est affiché séparément du request dans l'événement puisqu'il y a un gros décalage de temps
    # entre le moment ou on s'assure de la disponibilité et le moment ou on voit vraiment la quantité arrivée
    # C'est donc plus clair ainsi quand on debug en regardant la liste d'événements... 
    def LogCapacite(self) :
        
        if self.count == self.capacity and len(self.queue)==0: 
            self.env.EnrEven("Capacité maximale atteinte", Destination=self.Nom)
        
    def EstPlein(self) : 
        if self.count >= self.capacity : 
            return True
        else :
            return False