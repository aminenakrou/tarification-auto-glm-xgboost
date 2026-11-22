"""Chargement et préparation des données freMTPL2 (assurance automobile française).

Deux tables publiques, hébergées sur OpenML :
- freMTPL2freq (id 41214) : un contrat par ligne, avec son exposition et son nombre de sinistres ;
- freMTPL2sev  (id 41215) : un sinistre par ligne, avec son montant.

Les fichiers sont téléchargés une seule fois puis mis en cache dans data/raw/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# Plafonds classiques sur ce jeu de données
MAX_CLAIMS = 4          # quelques contrats déclarent jusqu'à 16 sinistres : valeurs aberrantes
MAX_EXPOSURE = 1.0      # l'exposition est une fraction d'année
LARGE_CLAIM = 200_000   # au-delà, sinistre « grave » écrêté pour stabiliser le modèle de coût


def _fetch(data_id: int, name: str) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = fetch_openml(data_id=data_id, as_frame=True, parser="pandas").frame
    df.to_parquet(cache, index=False)
    return df


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = _fetch(41214, "freMTPL2freq")
    sev = _fetch(41215, "freMTPL2sev")
    return freq, sev


def build_dataset() -> pd.DataFrame:
    """Une ligne par contrat : variables tarifaires, exposition, nombre et charge des sinistres."""
    freq, sev = load_raw()
    freq = freq.copy()
    freq["IDpol"] = freq["IDpol"].astype(int)
    sev = sev.copy()
    sev["IDpol"] = sev["IDpol"].astype(int)

    for col in ["Area", "VehBrand", "VehGas", "Region"]:
        freq[col] = freq[col].astype(str).str.strip("'")

    # Charge sinistre par contrat, avec écrêtement des sinistres graves
    sev["ClaimAmountCapped"] = sev["ClaimAmount"].clip(upper=LARGE_CLAIM)
    agg = sev.groupby("IDpol").agg(
        ClaimAmount=("ClaimAmountCapped", "sum"),
        ClaimAmountRaw=("ClaimAmount", "sum"),
        NbAmounts=("ClaimAmount", "size"),
    )
    df = freq.merge(agg, left_on="IDpol", right_index=True, how="left")
    df[["ClaimAmount", "ClaimAmountRaw", "NbAmounts"]] = df[["ClaimAmount", "ClaimAmountRaw", "NbAmounts"]].fillna(0)

    # Cohérence fréquence / coût : on retient le nombre de sinistres ayant un montant connu
    # (certains contrats ont des sinistres déclarés sans montant dans la table des coûts).
    df["ClaimNb"] = df["NbAmounts"].astype(int).clip(upper=MAX_CLAIMS)
    df["Exposure"] = df["Exposure"].clip(upper=MAX_EXPOSURE)
    df = df[df["Exposure"] > 0].reset_index(drop=True)
    df = df.drop(columns=["NbAmounts"])
    return df


def train_test_split_policies(df: pd.DataFrame, test_size: float = 0.2, seed: int = 2026):
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(df)) < test_size
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
