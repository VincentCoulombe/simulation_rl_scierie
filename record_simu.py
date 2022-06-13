from typing import Any
from Heuristiques import *
from utils import *
from EnvSimpy import EnvSimpy

class SimuRecorder():
    def __init__(self, env: EnvSimpy) -> None:
        self.env = env
        self.records: list[Any] = []
        
    def record(self, heuristique: str) -> pd.DataFrame:
        done = False  
        while not done: 
            if heuristique == "pile_la_plus_elevee":
                liste_action = pile_la_plus_elevee_liste(self.env)
            elif heuristique == "gestion_horaire_et_pile":
                liste_action = gestion_horaire_et_pile(self.env)
            else:
                liste_action = aleatoire(self.env)            
            done = self.env.stepSimpy(liste_action[-1]) 
            obs = self.env.getState()
            self.records.append([obs.tolist(), liste_action])

        return pd.DataFrame(self.records, columns=["obs", "action"])
    
if __name__ == "__main__":
    
    regles = pd.read_csv("DATA/regle.csv")
    df_HoraireScierie = work_schedule()
    df_HoraireLoader = work_schedule()

        
    paramSimu = {"df_regles": regles,
             "df_HoraireLoader" : work_schedule(),
             "df_HoraireScierie" : work_schedule(),                
             "NbStepSimulation": 64*1,
             "NbStepSimulationTest": 64*10,
             "nbLoader": 1,
             "nbSechoir1": 4,
             "CapaciteSortieSciage": 100, # En nombre de chargements avant de bloquer la scierie
             "CapaciteSechageAirLibre": 0,
             "CapaciteCours": 0,
             "TempsAttenteLoader": 1,
             "TempsAttenteActionInvalide": 10,
             "TempsSechageAirLibre": 7 * 24,
             "RatioSechageAirLibre": 0.1 * 12 / 52,
             "MaxSechageAirLibre" : 30/100,
             "DureeDeteriorationEpinette" : 30 * 24, 
             "DureeDeteriorationSapin" : 4 * 30 * 24, 
             "HresProdScieriesParSem": sum(df_HoraireScierie[:168]["work_time"]),
             "VariationProdScierie": 0.1,  # Pourcentage de variation de la demande par rapport à la production de la scierie
             "VariationTempsSechage": 0.1,
             "VariationTempsDeplLoader": 0.1,
             "FacteurSortieScierie" : 0.5, # Permet de sortir plus ou moins de la scierie (1 correspond à sortir exactement ce qui est prévu)
             "ObjectifStableEnPMP" : 215000 * 4 * 2.5,
             "RatioSapinEpinette" : "50/50",
             "DetailsEvenements" : False
             }
    
    env = EnvSimpy(paramSimu)
    recorder = SimuRecorder(env)
    df = recorder.record("pile_la_plus_elevee")
    df.to_csv("pile_la_plus_elevee.csv")

    