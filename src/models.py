"""Modèles de fréquence et de coût moyen : GLM (statsmodels) et XGBoost.

Prime pure = fréquence attendue × coût moyen attendu (par année d'exposition).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb

from .features import CAT_GLM_SEV, GLM_INTERACTIONS, NUM_GLM_SEV, gbm_matrix, glm_formula

SEED = 2026


# ------------------------------------------------------------------ GLM
def fit_glm_frequency(train: pd.DataFrame, interactions: bool = False):
    """GLM de Poisson, lien log, avec log(exposition) en offset (avec ou sans interactions)."""
    formula = glm_formula("ClaimNb") + (GLM_INTERACTIONS if interactions else "")
    model = smf.glm(
        formula,
        data=train,
        family=sm.families.Poisson(),
        offset=np.log(train["Exposure"]),
    )
    return model.fit()


def fit_glm_severity(train: pd.DataFrame):
    """GLM Gamma, lien log, sur le coût moyen des contrats sinistrés, pondéré par le nombre de sinistres."""
    claims = train[train["ClaimNb"] > 0].copy()
    claims["AvgCost"] = claims["ClaimAmount"] / claims["ClaimNb"]
    model = smf.glm(
        glm_formula("AvgCost", CAT_GLM_SEV, NUM_GLM_SEV),
        data=claims,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=claims["ClaimNb"],
    )
    return model.fit()


def predict_glm_frequency(res, df: pd.DataFrame) -> np.ndarray:
    """Fréquence annuelle prédite (exposition = 1)."""
    return np.asarray(res.predict(df, offset=np.zeros(len(df))))


def predict_glm_severity(res, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(res.predict(df))


def relativities(res) -> pd.DataFrame:
    """Coefficients du GLM convertis en relativités (exp(coef)), avec intervalle de confiance à 95 %."""
    ci = res.conf_int()
    out = pd.DataFrame({
        "terme": res.params.index,
        "relativite": np.exp(res.params.values),
        "ic95_bas": np.exp(ci[0].values),
        "ic95_haut": np.exp(ci[1].values),
        "p_value": res.pvalues.values,
    })
    out["variable"] = out["terme"].str.extract(r"C\((\w+),")[0].fillna(out["terme"])
    out["niveau"] = out["terme"].str.extract(r"\[T\.(.+)\]")[0].fillna("")
    return out[["variable", "niveau", "relativite", "ic95_bas", "ic95_haut", "p_value"]]


# ------------------------------------------------------------------ XGBoost
def _split_validation(df: pd.DataFrame, frac: float = 0.1):
    rng = np.random.default_rng(SEED)
    mask = rng.random(len(df)) < frac
    return df[~mask], df[mask]


def fit_gbm_frequency(train: pd.DataFrame):
    """XGBoost objectif Poisson, log(exposition) en base_margin, arrêt précoce sur 10 % du train."""
    fit, val = _split_validation(train)
    dtrain = xgb.DMatrix(gbm_matrix(fit), label=fit["ClaimNb"], base_margin=np.log(fit["Exposure"]), enable_categorical=True)
    dval = xgb.DMatrix(gbm_matrix(val), label=val["ClaimNb"], base_margin=np.log(val["Exposure"]), enable_categorical=True)
    params = {
        "objective": "count:poisson",
        "eta": 0.05,
        "max_depth": 5,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_cat_to_onehot": 1,
        "tree_method": "hist",
        "seed": SEED,
    }
    return xgb.train(params, dtrain, num_boost_round=2000, evals=[(dval, "val")],
                     early_stopping_rounds=50, verbose_eval=False)


def fit_gbm_severity(train: pd.DataFrame):
    """XGBoost objectif Gamma sur le coût moyen, pondéré par le nombre de sinistres."""
    claims = train[train["ClaimNb"] > 0].copy()
    claims["AvgCost"] = claims["ClaimAmount"] / claims["ClaimNb"]
    fit, val = _split_validation(claims)
    dtrain = xgb.DMatrix(gbm_matrix(fit), label=fit["AvgCost"], weight=fit["ClaimNb"], enable_categorical=True)
    dval = xgb.DMatrix(gbm_matrix(val), label=val["AvgCost"], weight=val["ClaimNb"], enable_categorical=True)
    params = {
        "objective": "reg:gamma",
        "eta": 0.03,
        "max_depth": 3,
        "min_child_weight": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_cat_to_onehot": 1,
        "tree_method": "hist",
        "seed": SEED,
    }
    return xgb.train(params, dtrain, num_boost_round=2000, evals=[(dval, "val")],
                     early_stopping_rounds=50, verbose_eval=False)


def predict_gbm_frequency(booster, df: pd.DataFrame) -> np.ndarray:
    """Fréquence annuelle prédite.

    base_margin = 0 (= log 1 an) : le modèle a été entraîné avec log(exposition) en base_margin ;
    sans cela, XGBoost ajouterait son intercept par défaut.
    """
    d = xgb.DMatrix(gbm_matrix(df), base_margin=np.zeros(len(df)), enable_categorical=True)
    return booster.predict(d, iteration_range=(0, booster.best_iteration + 1))


def predict_gbm_severity(booster, df: pd.DataFrame) -> np.ndarray:
    """Coût moyen prédit (modèle entraîné sans base_margin)."""
    d = xgb.DMatrix(gbm_matrix(df), enable_categorical=True)
    return booster.predict(d, iteration_range=(0, booster.best_iteration + 1))


def shap_values(booster, df: pd.DataFrame) -> pd.DataFrame:
    """Contributions SHAP natives de XGBoost (dernière colonne = biais, retirée)."""
    d = xgb.DMatrix(gbm_matrix(df), enable_categorical=True)
    contrib = booster.predict(d, pred_contribs=True, iteration_range=(0, booster.best_iteration + 1))
    return pd.DataFrame(contrib[:, :-1], columns=gbm_matrix(df).columns)
