"""Variables tarifaires.

Pour le GLM, on regroupe les variables continues en classes, comme dans un tarif réel :
cela rend les effets non linéaires (âge du conducteur, âge du véhicule) et donne des
coefficients lisibles sous forme de relativités. XGBoost reçoit les variables brutes.
"""

import numpy as np
import pandas as pd

DRIV_AGE_BINS = [17, 20, 25, 30, 40, 50, 70, 120]
DRIV_AGE_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
VEH_AGE_BINS = [-1, 0, 10, 200]
VEH_AGE_LABELS = ["0", "1-10", "11+"]

CAT_GLM = ["Area", "VehPowerGrp", "VehAgeGrp", "DrivAgeGrp", "VehBrand", "VehGas", "Region"]
NUM_GLM = ["BonusMalusCap", "LogDensity"]

# Coût moyen : peu de sinistres et une forte variabilité. Un modèle parcimonieux, choisi par
# validation croisée à 5 plis sur le jeu d'entraînement, généralise mieux que le modèle complet.
CAT_GLM_SEV = ["VehPowerGrp", "VehAgeGrp", "DrivAgeGrp", "VehGas"]
NUM_GLM_SEV = ["BonusMalusCap"]

CAT_GBM = ["Area", "VehBrand", "VehGas", "Region"]
NUM_GBM = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]

# Niveaux de référence des variables catégorielles du GLM (classe la plus exposée)
REFERENCE = {
    "Area": "C",
    "VehPowerGrp": "6",
    "VehAgeGrp": "1-10",
    "DrivAgeGrp": "51-70",
    "VehBrand": "B1",
    "VehGas": "Regular",
    "Region": "R24",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["VehPowerGrp"] = out["VehPower"].clip(upper=9).astype(str)
    out["VehAgeGrp"] = pd.cut(out["VehAge"], VEH_AGE_BINS, labels=VEH_AGE_LABELS).astype(str)
    out["DrivAgeGrp"] = pd.cut(out["DrivAge"], DRIV_AGE_BINS, labels=DRIV_AGE_LABELS).astype(str)
    out["BonusMalusCap"] = out["BonusMalus"].clip(upper=150)
    out["LogDensity"] = np.log(out["Density"])
    out["Frequency"] = out["ClaimNb"] / out["Exposure"]
    return out


def gbm_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matrice d'entrée de XGBoost : numériques brutes + catégorielles natives."""
    X = df[NUM_GBM + CAT_GBM].copy()
    for col in CAT_GBM:
        X[col] = X[col].astype("category")
    return X


# GLM enrichi : interactions et non-linéarité du bonus-malus, choisies sur une validation
# interne au jeu d'entraînement (gain de déviance, AIC), en restant lisibles.
GLM_INTERACTIONS = (
    " + C(DrivAgeGrp, Treatment(reference='51-70')):BonusMalusCap"
    " + C(VehAgeGrp, Treatment(reference='1-10')):BonusMalusCap"
    " + I((BonusMalusCap - 50) ** 2) + I((BonusMalusCap - 50) ** 3)"
)


def glm_formula(target: str, cats=CAT_GLM, nums=NUM_GLM) -> str:
    terms = [f"C({c}, Treatment(reference='{REFERENCE[c]}'))" for c in cats] + list(nums)
    return f"{target} ~ " + " + ".join(terms)
