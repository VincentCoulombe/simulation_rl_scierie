from EnvSimpy import *
import matplotlib.pyplot as plt
import numpy as np


def pile_la_plus_elevee(env: EnvSimpy):
    return np.argmax(env.getState()[1:])
    
def aleatoire(env: EnvSimpy):
    #Mettre une action aléatoire valide
    pass

def gestion_horaire_et_pile(env: EnvSimpy):
    # Sécher pile la plus haute + gérer le vendredi
    pass
    
if __name__ == '__main__':
    regles = pd.read_csv("DATA/regle.csv")

    paramSimu = {"df_regles": regles,
            "NbStepSimulation": 64*5,
            "NbStepSimulationTest": 64*2,
            "nbLoader": 1,
            "nbSechoir1": 4,
            "CapaciteSortieSciage": 100, # En nombre de chargements avant de bloquer la scierie
            "CapaciteSechageAirLibre": 0,
            "CapaciteCours": 0,
            "TempsAttenteLoader": 1,
            "TempsAttenteActionInvalide": 10,
            "TempsSechageAirLibre": 7 * 24,
            "RatioSechageAirLibre": 0.1 * 12 / 52,
            "HresProdScieriesParSem": 168, #44 + 44,
            "VariationProdScierie": 0.1,  # Pourcentage de variation de la demande par rapport à la production de la scierie
            "VariationTempsSechage": 0.1,
            "VariationTempsDeplLoader": 0.1,
            "FacteurSortieScierie" : 1, # Permet de sortir plus ou moins de la scierie (1 correspond à sortir exactement ce qui est prévu)
            "ObjectifStableEnPMP" : 215000 * 4 * 2.5
            }
    done = False
    env = EnvSimpy(paramSimu) 
    _, reelle = env.getProportions()
    lstQteDansCours = []
    lstQteStable = []
    lstinf = []
    lstsup = []
    propReelle = reelle
    while not done:         
        done = env.stepSimpy(pile_la_plus_elevee(env))
        _, reelle = env.getProportions()
        propReelle += reelle
        QteDansCours, QteStable, inf, sup = env.getIndicateursInventaire()
        lstQteDansCours.append(QteDansCours[4])
        lstQteStable.append(QteStable[4])
        lstinf.append(inf[4])
        lstsup.append(sup[4])
        env.getState()
        
    plt.plot(lstQteDansCours)
    plt.plot(lstQteStable)
    plt.plot(lstinf)
    plt.plot(lstsup)
    plt.show()
    
            