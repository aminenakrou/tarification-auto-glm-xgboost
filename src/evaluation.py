"""Métriques de tarification."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance


def poisson_deviance(df: pd.DataFrame, pred_freq: np.ndarray) -> float:
    """Déviance de Poisson moyenne sur la fréquence, pondérée par l'exposition."""
    return mean_poisson_deviance(df["Frequency"], pred_freq, sample_weight=df["Exposure"])


def gamma_deviance(df: pd.DataFrame, pred_sev: np.ndarray) -> float:
    """Déviance Gamma moyenne sur le coût moyen des contrats sinistrés, pondérée par le nombre de sinistres."""
    mask = df["ClaimNb"].to_numpy() > 0
    avg_cost = (df["ClaimAmount"] / df["ClaimNb"]).to_numpy()[mask]
    return mean_gamma_deviance(avg_cost, pred_sev[mask], sample_weight=df["ClaimNb"].to_numpy()[mask])


def lorenz_curve(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray):
    """Courbe de Lorenz ordonnée : contrats triés du moins risqué au plus risqué selon la prédiction.

    Abscisse : part cumulée de l'exposition ; ordonnée : part cumulée de la charge sinistre observée.
    """
    order = np.argsort(y_pred)
    exp_sorted = exposure[order]
    claims_sorted = (y_true * exposure)[order]
    x = np.concatenate([[0], np.cumsum(exp_sorted) / exp_sorted.sum()])
    y = np.concatenate([[0], np.cumsum(claims_sorted) / claims_sorted.sum()])
    return x, y


def gini(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> float:
    """Indice de Gini = 1 - 2 × aire sous la courbe de Lorenz (plus il est élevé, mieux le modèle classe les risques)."""
    x, y = lorenz_curve(y_true, y_pred, exposure)
    return 1 - 2 * np.trapezoid(y, x)


def calibration_by_decile(df: pd.DataFrame, pred_pp: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Prime pure moyenne observée et prédite par décile de prime prédite (pondéré par l'exposition)."""
    tmp = pd.DataFrame({
        "pred": pred_pp,
        "exposure": df["Exposure"].to_numpy(),
        "charge": df["ClaimAmount"].to_numpy(),
    }).sort_values("pred")
    cum = tmp["exposure"].cumsum() / tmp["exposure"].sum()
    tmp["decile"] = np.minimum((cum * n).astype(int) + 1, n)
    g = tmp.groupby("decile")
    return pd.DataFrame({
        "prime_pure_predite": g.apply(lambda d: np.average(d["pred"], weights=d["exposure"]), include_groups=False),
        "prime_pure_observee": g["charge"].sum() / g["exposure"].sum(),
        "exposition": g["exposure"].sum(),
    })
