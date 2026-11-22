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


def bootstrap_gini(y_true: np.ndarray, preds: dict, exposure: np.ndarray, n_boot: int = 200, seed: int = 2026) -> dict:
    """Intervalles de confiance à 95 % du Gini de chaque modèle et des écarts entre modèles (bootstrap sur les contrats)."""
    rng = np.random.default_rng(seed)
    names = list(preds)
    draws = {n: [] for n in names}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for name in names:
            draws[name].append(gini(y_true[idx], preds[name][idx], exposure[idx]))
    out = {}
    for name in names:
        d = np.array(draws[name])
        out[name] = {"gini": float(gini(y_true, preds[name], exposure)),
                     "ic95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = np.array(draws[b]) - np.array(draws[a])
            out[f"{b} - {a}"] = {"ecart": float(np.mean(diff)),
                                 "ic95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    return out


def double_lift(df: pd.DataFrame, pred_a: np.ndarray, pred_b: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Double lift : contrats triés selon le rapport B / A, puis prime moyenne observée, A et B par décile.

    Là où les deux tarifs sont en désaccord, on regarde lequel suit le mieux la charge observée.
    Les trois séries sont normalisées par leur moyenne globale.
    """
    expo = df["Exposure"].to_numpy()
    tmp = pd.DataFrame({"ratio": pred_b / pred_a, "a": pred_a, "b": pred_b, "e": expo,
                        "charge": df["ClaimAmount"].to_numpy()}).sort_values("ratio")
    cum = tmp["e"].cumsum() / tmp["e"].sum()
    tmp["decile"] = np.minimum((cum * n).astype(int) + 1, n)
    g = tmp.groupby("decile")
    obs = g["charge"].sum() / g["e"].sum()
    a = g.apply(lambda d: np.average(d["a"], weights=d["e"]), include_groups=False)
    b = g.apply(lambda d: np.average(d["b"], weights=d["e"]), include_groups=False)
    mean_obs = tmp["charge"].sum() / tmp["e"].sum()
    return pd.DataFrame({
        "observee": obs / mean_obs,
        "modele_a": a / np.average(tmp["a"], weights=tmp["e"]),
        "modele_b": b / np.average(tmp["b"], weights=tmp["e"]),
    })
