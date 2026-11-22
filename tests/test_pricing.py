import numpy as np
import pandas as pd
import pytest

from src.evaluation import calibration_by_decile, gini, lorenz_curve
from src.features import add_features, glm_formula


def _toy():
    return pd.DataFrame({
        "ClaimNb": [0, 1, 0, 2],
        "Exposure": [1.0, 0.5, 0.25, 1.0],
        "ClaimAmount": [0.0, 1000.0, 0.0, 3000.0],
        "VehPower": [4, 12, 6, 9],
        "VehAge": [0, 5, 15, 10],
        "DrivAge": [19, 35, 80, 50],
        "BonusMalus": [100, 50, 50, 230],
        "Density": [1, 100, 1000, 27000],
    })


def test_features_bins_and_caps():
    out = add_features(_toy())
    assert list(out["DrivAgeGrp"]) == ["18-20", "31-40", "71+", "41-50"]
    assert list(out["VehAgeGrp"]) == ["0", "1-10", "11+", "1-10"]
    assert out["VehPowerGrp"].tolist() == ["4", "9", "6", "9"]
    assert out["BonusMalusCap"].max() == 150
    assert out["Frequency"].tolist() == [0.0, 2.0, 0.0, 2.0]


def test_glm_formula_uses_reference_levels():
    f = glm_formula("ClaimNb")
    assert f.startswith("ClaimNb ~ ")
    assert "Treatment(reference='51-70')" in f


def test_lorenz_and_gini_bounds():
    rng = np.random.default_rng(0)
    exposure = rng.uniform(0.1, 1, 1000)
    y = rng.gamma(0.3, 500, 1000)
    x, c = lorenz_curve(y, y, exposure)
    assert x[0] == 0 and c[0] == 0
    assert x[-1] == pytest.approx(1) and c[-1] == pytest.approx(1)
    # Trier selon la vraie valeur donne un Gini positif ; un ordre aléatoire, un Gini proche de 0
    assert gini(y, y, exposure) > 0.3
    assert abs(gini(y, rng.random(1000), exposure)) < 0.1


def test_calibration_by_decile_preserves_totals():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"Exposure": rng.uniform(0.1, 1, 500), "ClaimAmount": rng.gamma(0.2, 1000, 500)})
    pred = rng.uniform(50, 200, 500)
    cal = calibration_by_decile(df, pred, n=10)
    assert len(cal) == 10
    assert cal["exposition"].sum() == pytest.approx(df["Exposure"].sum())


def test_bootstrap_gini_returns_intervals_and_differences():
    from src.evaluation import bootstrap_gini
    rng = np.random.default_rng(2)
    exposure = rng.uniform(0.1, 1, 800)
    y = rng.gamma(0.3, 500, 800)
    preds = {"bon": y + rng.normal(0, 50, 800), "aleatoire": rng.random(800)}
    res = bootstrap_gini(y, preds, exposure, n_boot=50)
    lo, hi = res["bon"]["ic95"]
    assert lo <= res["bon"]["gini"] <= hi
    # le modèle aléatoire classe moins bien : l'écart « aleatoire - bon » est négatif
    assert res["aleatoire - bon"]["ic95"][1] < 0


def test_double_lift_is_normalised():
    from src.evaluation import double_lift
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"Exposure": rng.uniform(0.1, 1, 1000), "ClaimAmount": rng.gamma(0.2, 1000, 1000)})
    a, b = rng.uniform(50, 200, 1000), rng.uniform(50, 200, 1000)
    dl = double_lift(df, a, b)
    assert len(dl) == 10
    assert set(dl.columns) == {"observee", "modele_a", "modele_b"}


def test_interaction_formula_is_valid():
    from src.features import GLM_INTERACTIONS
    f = glm_formula("ClaimNb") + GLM_INTERACTIONS
    assert f.count(":BonusMalusCap") == 2
