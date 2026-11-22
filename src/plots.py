"""Figures du projet (enregistrées dans outputs/figures/)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluation import lorenz_curve  # noqa: E402
from .features import DRIV_AGE_LABELS  # noqa: E402

BLUE, ORANGE, GREY = "#1f5f8b", "#d9822b", "#9aa5b1"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def frequency_by_variable(df: pd.DataFrame, path: Path):
    """Fréquence observée (points) et exposition (barres) pour trois variables tarifaires."""
    specs = [
        ("DrivAgeGrp", "Âge du conducteur", DRIV_AGE_LABELS),
        ("VehAgeGrp", "Âge du véhicule", ["0", "1-10", "11+"]),
        ("BonusMalusGrp", "Bonus-malus", None),
    ]
    d = df.copy()
    d["BonusMalusGrp"] = pd.cut(d["BonusMalus"], [49, 50, 60, 80, 100, 400],
                                labels=["50", "51-60", "61-80", "81-100", "> 100"]).astype(str)
    specs[2] = ("BonusMalusGrp", "Bonus-malus", ["50", "51-60", "61-80", "81-100", "> 100"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, title, order) in zip(axes, specs):
        g = d.groupby(col).agg(exp=("Exposure", "sum"), n=("ClaimNb", "sum")).reindex(order)
        ax2 = ax.twinx()
        ax2.bar(range(len(g)), g["exp"] / 1000, color=GREY, alpha=0.35)
        ax2.set_ylabel("Exposition (milliers d'années)", color="#5b6670")
        ax.plot(range(len(g)), g["n"] / g["exp"], marker="o", color=BLUE, lw=2.5, zorder=3)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xticks(range(len(g)), g.index)
        ax.set_title(title)
        ax.set_ylabel("Fréquence de sinistres (par an)")
        ax.set_ylim(0, None)
    fig.suptitle("Fréquence observée selon les principales variables", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def glm_relativities(rel: pd.DataFrame, path: Path):
    """Relativités du GLM de fréquence pour l'âge du conducteur et l'âge du véhicule."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (var, title, order, ref) in zip(axes, [
        ("DrivAgeGrp", "Âge du conducteur", DRIV_AGE_LABELS, "51-70"),
        ("VehAgeGrp", "Âge du véhicule", ["0", "1-10", "11+"], "1-10"),
    ]):
        r = rel[rel["variable"] == var].set_index("niveau")
        vals, lo, hi = [], [], []
        for lvl in order:
            if lvl == ref:
                vals.append(1.0); lo.append(1.0); hi.append(1.0)
            else:
                vals.append(r.loc[lvl, "relativite"]); lo.append(r.loc[lvl, "ic95_bas"]); hi.append(r.loc[lvl, "ic95_haut"])
        vals, lo, hi = map(np.array, (vals, lo, hi))
        colors = [GREY if l == ref else BLUE for l in order]
        ax.bar(order, vals, color=colors)
        ax.errorbar(order, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="black", capsize=4, lw=1)
        ax.axhline(1, color="black", lw=0.8, ls="--")
        for i, (v, h) in enumerate(zip(vals, hi)):
            ax.text(i, h + 0.03, f"{v:.2f}", ha="center", fontsize=10)
        ax.set_ylim(0, hi.max() + 0.15)
        ax.set_title(f"{title} (référence : {ref}, en gris)")
        ax.set_ylabel("Relativité de fréquence")
    fig.suptitle("GLM de Poisson : effet multiplicatif sur la fréquence, avec IC à 95 %", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def lorenz(df: pd.DataFrame, preds: dict, path: Path, ginis: dict):
    y = (df["ClaimAmount"] / df["Exposure"]).to_numpy()
    e = df["Exposure"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for (name, p), color in zip(preds.items(), [BLUE, ORANGE]):
        x, c = lorenz_curve(y, p, e)
        ax.plot(x, c, color=color, lw=2.2, label=f"{name} (Gini = {ginis[name]:.3f})")
    x, c = lorenz_curve(y, y, e)
    ax.plot(x, c, color="black", ls=":", lw=1.5, label="Oracle (charge observée)")
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.2, label="Tarif uniforme")
    ax.set_xlabel("Part cumulée de l'exposition (contrats triés du moins au plus risqué)")
    ax.set_ylabel("Part cumulée de la charge sinistre")
    ax.set_title("Courbes de Lorenz de la prime pure (jeu de test)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, path)


def calibration(cals: dict, path: Path):
    fig, axes = plt.subplots(1, len(cals), figsize=(6 * len(cals), 4.2), sharey=True)
    for ax, (name, cal) in zip(np.atleast_1d(axes), cals.items()):
        x = cal.index.to_numpy()
        ax.bar(x - 0.2, cal["prime_pure_observee"], width=0.4, color=GREY, label="Observée")
        ax.bar(x + 0.2, cal["prime_pure_predite"], width=0.4, color=BLUE, label="Prédite")
        ax.set_xticks(x)
        ax.set_xlabel("Décile de prime pure prédite")
        ax.set_title(name)
        ax.legend(fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Prime pure (€ par an)")
    fig.suptitle("Calibration par décile (jeu de test)", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def shap_importance(shap_df: pd.DataFrame, path: Path):
    imp = shap_df.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("Moyenne des |valeurs SHAP| (échelle log de la fréquence)")
    ax.set_title("XGBoost fréquence : importance des variables (SHAP)", fontweight="bold")
    _save(fig, path)
