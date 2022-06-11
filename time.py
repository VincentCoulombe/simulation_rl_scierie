import pandas as pd
import numpy as np


def work_schedule(nb_days = 365, day_start = 8, day_end = 24, month_start = 1):
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


if __name__ == "__main__":
    df = work_schedule(nb_days = 365, day_start = 8, day_end = 24, month_start = 1)
    task_length = task_total_length(df, task_start = 12.5, task_time = 12.5)