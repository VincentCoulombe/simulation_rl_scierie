import pandas as pd


sapin = pd.read_csv("planif sapin.csv")
epinette = pd.read_csv("planif épinette.csv")
tout = pd.concat([sapin, epinette], axis = 0)
tout = tout[~tout["Nom du produit"].duplicated(keep = "last")]

df = pd.DataFrame(columns = ["description", "longueur", "largeur", "epaisseur", "essence",
                             "nb_planches", "volume planche", "volume paquet",
                             "production epinette", "production sapin", "regle"])

# Produits
df.description = pd.concat([sapin["Nom du produit"], epinette["Nom du produit"]], axis = 0).unique()
df = df.sort_values("description").reset_index(drop = True)
df.index.name = "produit"

# Production
df["production epinette"] = df.description.map(epinette[["Nom du produit", "PMP/sem"]].set_index("Nom du produit")["PMP/sem"])
df["production sapin"] = df.description.map(sapin[["Nom du produit", "PMP/sem"]].set_index("Nom du produit")["PMP/sem"])

# Essence
df["essence"] = df.description.map(tout[["Nom du produit", "Essence"]].set_index("Nom du produit")["Essence"])

# Lecture str de description
txt = df.description.str.split(r"[\_\ \-]")
txt = pd.concat([txt, pd.Series([len(i) for i in txt], name = "n")], axis = 1)

txt["taille"] = [i[1] for i in txt.description]
txt["longueur"] = [i[-2] for i in txt.description]
txt["nb_planches"] = [i[-1] for i in txt.description]

txt["taille"] = txt["taille"].str.split(r"X")
txt["longueur"] = txt["longueur"].str.replace("'", "").astype(int)
txt["nb_planches"] = txt["nb_planches"].astype(int)

# Nombre de planches
df["nb_planches"] = df.index.map(txt["nb_planches"])

# Longueur
df["longueur"] = df.index.map(txt["longueur"])

# Largeur
df["largeur"] = pd.Series([i[1] for i in txt["taille"]]).astype(int)

# Épaisseur
df["epaisseur"] = pd.Series([i[0] for i in txt["taille"]])
df["epaisseur"] = df["epaisseur"].str.replace("P", "")
df["epaisseur"] = df["epaisseur"].str.replace("5/4", "1.25")
df["epaisseur"] = df["epaisseur"].astype(float)

# Volume planche et paquet
df["volume planche"] = (df["longueur"] * df["largeur"] * df["epaisseur"]) / 12
df["volume paquet"] = (df["volume planche"] * df["nb_planches"]).astype(int)

# Règles
rules = pd.read_csv("rules.csv")
df["regle"] = df.description.map(tout[["Nom du produit", "Code"]].set_index("Nom du produit")["Code"])
for r in rules.columns:
    df.loc[df["regle"].isin(rules[r].tolist()), "regle"] = r
    
df.to_csv("df.csv")
