"""Evaluation helpers for plots and per-segment analysis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def price_band_analysis(y_true, y_pred, prices):
    bins = [0, 100_000, 200_000, 300_000, 450_000, 700_000, 2_000_000]
    labels = ["<100k", "100-200k", "200-300k", "300-450k", "450-700k", "700k+"]
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["band"] = pd.cut(df["y_true"], bins=bins, labels=labels)
    grouped = df.groupby("band", observed=False).apply(
        lambda g: pd.Series({
            "n": len(g),
            "rmse": np.sqrt(((g["y_true"] - g["y_pred"])**2).mean()) if len(g)>0 else np.nan,
            "mae": (g["y_true"] - g["y_pred"]).abs().mean() if len(g)>0 else np.nan,
            "median_pct_err": ( (g["y_true"]-g["y_pred"]).abs()/g["y_true"] ).median() if len(g)>0 else np.nan
        })
    ).reset_index()
    return grouped

def zip_analysis(df_test, y_true, y_pred, zip_col="zip_code"):
    df = pd.DataFrame({"zip": df_test[zip_col].values if zip_col in df_test.columns else ["all"]*len(y_true), "y_true": y_true, "y_pred": y_pred})
    grouped = df.groupby("zip").apply(
        lambda g: pd.Series({
            "n": len(g),
            "rmse": np.sqrt(((g["y_true"] - g["y_pred"])**2).mean()),
            "mae": (g["y_true"] - g["y_pred"]).abs().mean(),
        })
    ).reset_index().sort_values("n", ascending=False)
    return grouped

def residual_plot(y_true, y_pred, save_path=None):
    resid = y_true - y_pred
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].scatter(y_pred, resid, alpha=0.4, s=10)
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].set_xlabel("Predicted Price (USD)")
    ax[0].set_ylabel("Residual (True - Pred)")
    ax[0].set_title("Residuals vs Predicted")
    ax[1].hist(resid, bins=50, edgecolor="white")
    ax[1].set_xlabel("Residual (USD)")
    ax[1].set_title("Residual Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
