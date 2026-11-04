"""SHAP explanations for the gradient-boosting RUL model (exact TreeSHAP, additive)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

SENSOR_INFO = {  # C-MAPSS sensor descriptions (Saxena et al., 2008, Table 2)
    "s1": "Fan inlet temperature (T2)", "s2": "LPC outlet temperature (T24)", "s3": "HPC outlet temperature (T30)",
    "s4": "LPT outlet temperature (T50)", "s5": "Fan inlet pressure (P2)", "s6": "Bypass-duct pressure (P15)",
    "s7": "HPC outlet pressure (P30)", "s8": "Physical fan speed (Nf)", "s9": "Physical core speed (Nc)",
    "s10": "Engine pressure ratio (epr)", "s11": "HPC outlet static pressure (Ps30)", "s12": "Fuel flow / Ps30 (phi)",
    "s13": "Corrected fan speed (NRf)", "s14": "Corrected core speed (NRc)", "s15": "Bypass ratio (BPR)",
    "s16": "Burner fuel-air ratio (farB)", "s17": "Bleed enthalpy (htBleed)", "s18": "Demanded fan speed (Nf_dmd)",
    "s19": "Demanded corrected fan speed (PCNfR_dmd)", "s20": "HPT coolant bleed (W31)", "s21": "LPT coolant bleed (W32)",
}


def describe(feature: str) -> str:
    base = feature.split("_")[0]
    if base == "cycle":
        return "Engine age (cycles)"
    kind = feature[len(base):]
    label = SENSOR_INFO.get(base, base)
    if kind.startswith("_mean"):
        return f"{label}, {kind[5:]}-cycle mean"
    if kind.startswith("_std"):
        return f"{label}, {kind[4:]}-cycle variability"
    if kind.startswith("_slope"):
        return f"{label}, trend over {kind[6:]} cycles"
    if kind == "_ewm":
        return f"{label}, smoothed level"
    return label


class Explainer:
    def __init__(self, model, feature_names: list[str]):
        self.explainer = shap.TreeExplainer(model)
        self.features = feature_names
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])

    def values(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.explainer.shap_values(X[self.features]))

    def top_factors(self, X_row: pd.DataFrame, k: int = 5) -> list[dict]:
        """Top-|SHAP| drivers for one prediction. Negative contribution = lowers predicted RUL."""
        sv = self.values(X_row)[0]
        order = np.argsort(-np.abs(sv))[:k]
        return [{"feature": self.features[i], "description": describe(self.features[i]),
                 "value": float(X_row[self.features[i]].iloc[0]), "shap": float(sv[i])} for i in order]

    def global_importance(self, X: pd.DataFrame, k: int = 10) -> list[dict]:
        mean_abs = np.abs(self.values(X)).mean(0)
        order = np.argsort(-mean_abs)[:k]
        return [{"feature": self.features[i], "description": describe(self.features[i]), "mean_abs_shap": float(mean_abs[i])}
                for i in order]
