"""SHAP explanations — global and local."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def compute_shap(model_pipeline, X_train_sample, X_test_sample, max_samples=200):
    """Compute SHAP values for tree model. Falls back to linear coefficients."""
    # Extract model step
    try:
        model = model_pipeline.named_steps["model"]
        preproc = model_pipeline.named_steps["preprocess"]
        X_train_t = preproc.transform(X_train_sample)
        X_test_t = preproc.transform(X_test_sample)

        feature_names = None
        try:
            feature_names = preproc.get_feature_names_out().tolist()
        except:
            feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

        import shap
        # Use TreeExplainer for XGB, LinearExplainer for Ridge
        if hasattr(model, "get_booster") or "XGB" in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_t[:max_samples])
        else:
            explainer = shap.LinearExplainer(model, X_train_t[:max_samples])
            shap_values = explainer.shap_values(X_test_t[:max_samples])

        # Global importance
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)

        return shap_values, feature_names, importance, explainer
    except Exception as e:
        print(f"SHAP failed: {e}")
        return None, None, None, None

def plot_shap_bar(importance, save_path=None, top_n=12):
    if importance is None or importance.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(top_n).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2C4A5A")
    ax.set_xlabel("Mean |SHAP| (impact on predicted price)")
    ax.set_title("Top features driving price estimates")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def local_shap_waterfall_data(shap_values, feature_names, instance_idx=0):
    if shap_values is None:
        return None
    vals = shap_values[instance_idx]
    df = pd.DataFrame({"feature": feature_names, "shap": vals})
    df = df.reindex(df["shap"].abs().sort_values(ascending=False).index)
    return df.head(8)
