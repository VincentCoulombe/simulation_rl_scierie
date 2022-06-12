import pandas as pd
import numpy as np


def work_schedule(nb_days = 365, day_start = 8, day_end = 24, month_start = 1,work_on_weekend = False):
    """
    Paramètres:
        nb_days:
            Nombre de jour de la simulation.
            (int: 1 à 365)
        day_start:
            Heure de début de la journée de travail.
            (int: 0 à 24)
        day_end:
            Heure de fin de la journée de travail.
            (int: 0 à 24)
        month_start:
            Mois de début de simulation.
            (int: 1 à 12)
        
    Sortie:
        Dataframes sur les heures de travail (0/1), journées de travail (0/1) et mois (1 à 12).
    """
    
    columns = ["work_time", "work_day", "month"]
    df= pd.DataFrame(columns = columns, index = range(nb_days * 24))
    
    # Heures de travail
    df.work_time = list(range(1,25)) * nb_days
    df.loc[(df.work_time < day_start+1) | (df.work_time >= day_end+1), "work_time"] = False
    df.loc[df.work_time != False, "work_time"] = True
    df = df.replace({True: 1, False: 0})
    
    # Semaine et fin de semaine
    df.work_day = np.ceil((df.index+1)/24)
    if work_on_weekend :
        df["work_day"] = 1
    else : 
        df.work_day = np.where((df.work_day % 7 == 0) | ((df.work_day + 1) % 7 == 0), 0, 1)
    df.loc[df.work_day == 0, "work_time"] = 0
    
    # Mois
    month_order = list(range(month_start,13)) + list(range(1, month_start))
    month_dict = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    month_length = np.array([month_dict[m] for m in month_order])*24
    for i in range(len(month_order)):
        idx = df.loc[df.month.isna(), "month"].index[:month_length[i]]
        df.loc[idx, "month"] = month_order[i]
    df.month = df.month.astype(int)
    
    return df

def task_total_length(df, task_start, task_time):
    """
    Paramètres:
        df:
            Dataframe sur les horaires de travail.
        task_start:
            Heure de début de la tâche.
            (float: 0 à len(df))
        task_time:
            Durée de la tâche.
            (float)
        
    Sortie:
        Durée finale de la tâche avec les arrêts de travail.
    """
    
    # Ramener le calcul à partir de la première semaine pour être certain
    # de ne pas défoncé l'année... Permet de réutiliser directement le code déjà 
    # fonctionnel pour 1 an, mais pourrait être à revoir si la gestion de semaines/mois
    # apporteraient des nuances...
    if task_start > 7 * 24 : 
        return task_total_length(df,task_start - int(task_start / (7*24))*(7*24),task_time)
    
    
    length = pd.Index(df.loc[int(task_start):, "work_time"].cumsum()).get_loc(int(task_time))
    
    # length est un slice plutôt qu'un int quand la dernière heure de travail est avant un arrêt de travail
    # Si task_start est une valeur sans décimale, la tâche se termine avant l'arrêt
    # Sinon, elle se termine au retour
    if isinstance(length, slice):
        if (task_start == int(task_start)) and (task_time == int(task_time)):
            length = length.start + 1
        else:
            length = length.stop
    else:
        length += 1
    
    # Ajustement de décimale si la tâche débute quand il n'y a pas de travail
    if df.loc[int(task_start), "work_time"] == 0:
        length = length - (task_start - int(task_start))
    
    # Ajustement de décimale si la durée de la tâche a une décimale
    length = length + (task_time - int(task_time))
    return length

def GetInfosTemps(now) : 
    """
    Paramètres:
        now:
            Temps en heure (réel)
        
    Sortie:
        day_of_week : jour de la semaine (1 à 7)
        hour : heure de la journée (réel)
    """
    
    day_of_week = ((int(now / 24)) % 7) + 1 
    hour = now % 24 
    
    return day_of_week, hour


def HeuresProductives(df,debut,fin):
    """
    Paramètres:
        df:
            Dataframe sur les horaires de travail.
        debut:
            Heure de début du calcul
        fin:
            Fin de la tâche.
        
    Sortie:
        Durée en heures (réel) de travail productive entre debut et fin
    """

    # Si l'heure de début excéde l'heure de fin, il n'y a pas d'heures productives
    # (Cette situation pourrait survenir si on ne gère pas bien certaines transitions entre le 
    # régime transitoire et le permanent.  Les indicateurs comptes à partir du régime permanent
    # et on se compare au now)
    if debut >= fin : 
        return 0

    # Ramener le calcul à partir de la première semaine pour être certain
    # de ne pas défoncé l'année... Permet de réutiliser directement le code déjà 
    # fonctionnel pour 1 an, mais pourrait être à revoir si la gestion de semaines/mois
    # apporteraient des nuances...
    if debut > 7 * 24 : 
        return HeuresProductives(df,debut - int(debut / (7*24))*(7*24),fin - int(debut / (7*24))*(7*24))

    # Retirer les semaines complètes pour ne pas défoncer l'année de calcul
    if (fin - debut) > 24 * 7 : 
        NbHeuresUneSemaine = sum(df[:168]["work_time"])
        NbSemainesComplètes = int((fin-debut) / (24*7))
        NbHeuresSemainesComplètes = NbSemainesComplètes * NbHeuresUneSemaine
        DureeSemIncomplete = HeuresProductives(df,debut,fin - NbSemainesComplètes*(7*24))
        return NbHeuresSemainesComplètes + DureeSemIncomplete
    
    nbHeures = sum(df[int(debut):int(fin)]["work_time"])
    MinutesDebut = (debut - int(debut)) * df.iloc[int(debut)]["work_time"]
    MinutesFin = (fin - int(fin)) * df.iloc[int(fin)]["work_time"]
    
    return nbHeures - MinutesDebut + MinutesFin 
    

if __name__ == "__main__":
    
    
    df = work_schedule(nb_days = 365, day_start = 0, day_end = 24, month_start = 1,work_on_weekend=True)
    task_length = task_total_length(df, task_start = 360*24, task_time = 23)
    #print(task_length)
    
    if 1 ==0 : 
        # Pour faciliter le développement, on s'assure d'avoir toujours le mêmes
        # nombres aléatoires d'une exécution à l'autre
        import random
        random.seed(1)
        
        now = 0
        for jour in range(1,5000) : 
            print(jour)
            for heure in range(1) : 
                for minutes in range(1) : 
                    task_start = jour*24 + heure + minutes/60
                    task_time = random.random() * 1500 + 1
                    
                    #bk = task_total_length_bk(df, task_start = jour*24 + heure + minutes/60, task_time = 50)
                    new = task_total_length(df, task_start = task_start, task_time = task_time)
                    end = new + task_start
                    
                    prod = HeuresProductives(df,task_start,end)
                    
                    
                    if round(prod,4) != round(task_time,4) : 
                        #print("PROBLÈME", task_start, task_time, new,prod)
                        #   task_start = jour*24 + heure + minutes/60
                        
                      #  task_time = 50
                        
                       # print(jour,heure,minutes,"PROBLÈME")
                       # print(bk)
                       # print(new)
                        print("task start", task_start, "task time", task_time)
                        print("new",new)
                        print("prod",prod)
                        print("end",end)
                       # print(task_start - int(task_start / (7*24))*(7*24))
                        exit
                              
        print("FINI")                    
            #    print(now, jour, heure)
            #print("now : ", now,"jour : ",  jour)
                    
            #month, day_of_week, hour = GetInfosTemps(now)
                    
            #print(day_of_week)
            #print()
                    
            #now += 1 * 24
    debut = 189.17756324661573 
    fin = debut-5
    print("Calculer",HeuresProductives(df,debut,fin))
    print("Bon 24/7", fin-debut)
    print(GetInfosTemps(debut))