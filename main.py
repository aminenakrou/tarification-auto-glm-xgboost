"""Pipeline complet : données -> GLM, GLM avec interactions et XGBoost (fréquence × coût moyen) -> comparaison.

Utilisation :
    python main.py

Sorties dans outputs/ : figures, metrics.json, relativités du GLM de fréquence.
"""

import json
import sys
from pathlib import Path

import numpy as np

from src import plots
from src.data import build_dataset, train_test_split_policies
from src.evaluation import (
    bootstrap_gini, calibration_by_decile, double_lift, gamma_deviance, gini, poisson_deviance,
)
from src.features import add_features
from src.models import (
    fit_gbm_frequency, fit_gbm_severity, fit_glm_frequency, fit_glm_severity,
    predict_gbm_frequency, predict_gbm_severity, predict_glm_frequency, predict_glm_severity,
    relativities, shap_values,
)

OUT = Path(__file__).resolve().parent / "outputs"
FIG = OUT / "figures"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)

    df = add_features(build_dataset())
    train, test = train_test_split_policies(df)
    print(f"Contrats : {len(df):,} (train {len(train):,}, test {len(test):,})".replace(",", " "))
    print(f"Fréquence observée : {df['ClaimNb'].sum() / df['Exposure'].sum():.4f} | "
          f"coût moyen : {df['ClaimAmount'].sum() / df['ClaimNb'].sum():.0f} €")

    plots.frequency_by_variable(df, FIG / "frequence_par_variable.png")

    print("GLM...")
    glm_f = fit_glm_frequency(train)
    glm_fi = fit_glm_frequency(train, interactions=True)
    glm_s = fit_glm_severity(train)
    print("XGBoost...")
    gbm_f, gbm_s = fit_gbm_frequency(train), fit_gbm_severity(train)

    predictors = {
        "GLM": lambda d: (predict_glm_frequency(glm_f, d), predict_glm_severity(glm_s, d)),
        "GLM + interactions": lambda d: (predict_glm_frequency(glm_fi, d), predict_glm_severity(glm_s, d)),
        "XGBoost": lambda d: (predict_gbm_frequency(gbm_f, d), predict_gbm_severity(gbm_s, d)),
    }

    # Correction d'équilibre : un coefficient global, calculé sur le jeu d'entraînement,
    # pour que la charge prédite égale la charge observée (le tarif doit couvrir les sinistres).
    balance = {}
    for name, predict in predictors.items():
        f_tr, s_tr = predict(train)
        balance[name] = train["ClaimAmount"].sum() / (f_tr * s_tr * train["Exposure"]).sum()
    preds = {name: predict(test) for name, predict in predictors.items()}

    # Sinistres graves : les modèles sont entraînés sur des montants écrêtés à 200 000 €.
    # La part au-delà du seuil est mutualisée par une surcharge proportionnelle, estimée sur le train.
    surcharge_graves = train["ClaimAmountRaw"].sum() / train["ClaimAmount"].sum()

    null_freq = np.full(len(test), train["ClaimNb"].sum() / train["Exposure"].sum())
    null_sev = np.full(len(test), train["ClaimAmount"].sum() / train["ClaimNb"].sum())

    y_pp = (test["ClaimAmount"] / test["Exposure"]).to_numpy()
    expo = test["Exposure"].to_numpy()
    observed_total = test["ClaimAmount"].sum()

    metrics = {"null": {
        "deviance_poisson_frequence": poisson_deviance(test, null_freq),
        "deviance_gamma_cout_moyen": gamma_deviance(test, null_sev),
    }}
    pp, cals = {}, {}
    for name, (f, s) in preds.items():
        pp[name] = f * s * balance[name]
        metrics[name] = {
            "deviance_poisson_frequence": poisson_deviance(test, f),
            "deviance_gamma_cout_moyen": gamma_deviance(test, s),
            "gini_prime_pure": gini(y_pp, pp[name], expo),
            "charge_predite_sur_observee": float((pp[name] * expo).sum() / observed_total),
            "coefficient_equilibre": float(balance[name]),
        }
        m = metrics[name]
        m["gain_deviance_frequence_vs_null_pct"] = 100 * (1 - m["deviance_poisson_frequence"] / metrics["null"]["deviance_poisson_frequence"])
        m["gain_deviance_cout_vs_null_pct"] = 100 * (1 - m["deviance_gamma_cout_moyen"] / metrics["null"]["deviance_gamma_cout_moyen"])
        cals[name] = calibration_by_decile(test, pp[name])

    print("Bootstrap du Gini...")
    metrics["bootstrap_gini"] = bootstrap_gini(y_pp, pp, expo, n_boot=200)
    metrics["sinistres_graves"] = {
        "seuil_ecretement": 200_000,
        "surcharge_mutualisee": float(surcharge_graves),
        "charge_brute_test_couverte_xgboost": float((pp["XGBoost"] * surcharge_graves * expo).sum() / test["ClaimAmountRaw"].sum()),
    }
    metrics["glm_frequence_aic"] = {"sans_interactions": float(glm_f.aic), "avec_interactions": float(glm_fi.aic)}
    metrics["xgboost_nb_arbres"] = {"frequence": gbm_f.best_iteration + 1, "cout_moyen": gbm_s.best_iteration + 1}

    rel = relativities(glm_f)
    rel.to_csv(OUT / "relativites_glm_frequence.csv", index=False, float_format="%.4f")

    plots.glm_relativities(rel, FIG / "relativites_glm.png")
    plots.lorenz(test, pp, FIG / "lorenz_prime_pure.png", {k: metrics[k]["gini_prime_pure"] for k in pp})
    plots.calibration(cals, FIG / "calibration_deciles.png")
    plots.double_lift(double_lift(test, pp["GLM + interactions"], pp["XGBoost"]),
                      "GLM + interactions", "XGBoost", FIG / "double_lift.png")
    sample = test.sample(20_000, random_state=2026)
    plots.shap_importance(shap_values(gbm_f, sample), FIG / "shap_frequence.png")

    with open(OUT / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
