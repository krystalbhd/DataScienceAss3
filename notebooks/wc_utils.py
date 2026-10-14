"""
wc_utils.py  -  shared helpers for Objective 2 (FIFA World Cup 2026 regressions)
Sydney Group 19  |  Foundations of Data Science, Assessment 3

Only numpy / pandas / scipy / scikit-learn / matplotlib are needed, so the
notebooks run on a standard Anaconda install (no statsmodels dependency).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------------------------------------------------------
# 1. Geography
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------
# 2. Ordinary least squares with full inference (replaces statsmodels.OLS)
# --------------------------------------------------------------------------
class OLSInference:
    """OLS fitted by QR/lstsq, reporting coefficients, SE, t, p, 95% CI,
    R2, adjusted R2, F-test, AIC/BIC and residual diagnostics."""

    def fit(self, X: pd.DataFrame, y):
        self.names = ["const"] + list(X.columns)
        Xm = np.column_stack([np.ones(len(X)), X.values.astype(float)])
        y = np.asarray(y, dtype=float)
        n, k = Xm.shape
        beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        resid = y - Xm @ beta
        sse = resid @ resid
        sst = ((y - y.mean()) ** 2).sum()
        df_res = n - k
        sigma2 = sse / df_res
        cov = sigma2 * np.linalg.inv(Xm.T @ Xm)
        se = np.sqrt(np.diag(cov))
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), df_res)
        tcrit = stats.t.ppf(0.975, df_res)
        self.table = pd.DataFrame({
            "coef": beta, "std_err": se, "t": t, "p_value": p,
            "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se},
            index=self.names)
        self.r2 = 1 - sse / sst
        self.adj_r2 = 1 - (1 - self.r2) * (n - 1) / df_res
        self.f_stat = ((sst - sse) / (k - 1)) / sigma2
        self.f_p = stats.f.sf(self.f_stat, k - 1, df_res)
        ll = -n / 2 * (np.log(2 * np.pi) + np.log(sse / n) + 1)
        self.aic = 2 * k - 2 * ll
        self.bic = k * np.log(n) - 2 * ll
        self.resid, self.fitted, self.Xm, self.y = resid, Xm @ beta, Xm, y
        self.n, self.k = n, k
        return self

    def summary(self):
        print(f"n = {self.n}, predictors = {self.k - 1}")
        print(f"R2 = {self.r2:.3f}   adj R2 = {self.adj_r2:.3f}   "
              f"F = {self.f_stat:.2f} (p = {self.f_p:.2e})   "
              f"AIC = {self.aic:.1f}   BIC = {self.bic:.1f}")
        return self.table.round(4)

    def cluster_robust(self, groups):
        """Cluster-robust (CR1) standard errors - needed when rows are not
        independent, e.g. the two team-rows of the same match."""
        groups = np.asarray(groups)
        XtX_inv = np.linalg.inv(self.Xm.T @ self.Xm)
        meat = np.zeros((self.k, self.k))
        G = 0
        for g in np.unique(groups):
            idx = groups == g
            s = self.Xm[idx].T @ self.resid[idx]
            meat += np.outer(s, s); G += 1
        corr = G / (G - 1) * (self.n - 1) / (self.n - self.k)
        cov = corr * XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.diag(cov))
        beta = self.table["coef"].values
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), G - 1)
        tcrit = stats.t.ppf(0.975, G - 1)
        return pd.DataFrame({"coef": beta, "cluster_se": se, "t": t, "p_value": p,
                             "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se},
                            index=self.names)

    # ---- diagnostics ------------------------------------------------------
    def diagnostics(self):
        e = self.resid
        # Breusch-Pagan (studentised / Koenker version): n * R2 of e^2 on X
        e2 = e ** 2
        b, *_ = np.linalg.lstsq(self.Xm, e2, rcond=None)
        r2_aux = 1 - ((e2 - self.Xm @ b) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
        bp = self.n * r2_aux
        bp_p = stats.chi2.sf(bp, self.k - 1)
        sw, sw_p = stats.shapiro(e)
        dw = np.sum(np.diff(e) ** 2) / np.sum(e ** 2)
        return pd.Series({"Breusch-Pagan LM": bp, "BP p-value": bp_p,
                          "Shapiro-Wilk W": sw, "SW p-value": sw_p,
                          "Durbin-Watson": dw,
                          "Skew": stats.skew(e), "Kurtosis": stats.kurtosis(e)})


def vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor for every column of X."""
    out = {}
    for c in X.columns:
        others = X.drop(columns=c)
        A = np.column_stack([np.ones(len(X)), others.values])
        b, *_ = np.linalg.lstsq(A, X[c].values, rcond=None)
        r2 = 1 - ((X[c].values - A @ b) ** 2).sum() / ((X[c] - X[c].mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return pd.Series(out, name="VIF")


# --------------------------------------------------------------------------
# 3. Cross-validation that respects grouping (match_id) and repetition
# --------------------------------------------------------------------------
def repeated_group_folds(groups, n_splits=10, n_repeats=10, seed=42):
    """Yield (repeat, train_idx, test_idx). All rows that share a group
    (e.g. both team-rows of one match) always land in the same fold,
    which prevents information leakage between the two halves of a match."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    for r in range(n_repeats):
        perm = rng.permutation(uniq)
        fold_of = {g: i % n_splits for i, g in enumerate(perm)}
        f = np.array([fold_of[g] for g in groups])
        for k in range(n_splits):
            yield r, np.where(f != k)[0], np.where(f == k)[0]


def cross_validate_models(models: dict, X, y, groups, n_splits=10,
                          n_repeats=10, seed=42, extra_metrics=None):
    """Evaluate every model on IDENTICAL folds. Returns long DataFrame of
    per-fold scores (one row per model x repeat x fold)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    rows = []
    for r, tr, te in repeated_group_folds(groups, n_splits, n_repeats, seed):
        for name, mdl in models.items():
            m = clone(mdl).fit(X[tr], y[tr])
            pred = m.predict(X[te])
            row = {"model": name, "repeat": r,
                   "RMSE": np.sqrt(mean_squared_error(y[te], pred)),
                   "MAE": mean_absolute_error(y[te], pred),
                   "R2": r2_score(y[te], pred) if len(te) > 1 else np.nan,
                   "n_test": len(te)}
            if extra_metrics:
                for mname, fn in extra_metrics.items():
                    row[mname] = fn(y[te], pred)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_cv(scores: pd.DataFrame, metrics=("RMSE", "MAE", "R2")):
    g = scores.groupby("model")
    out = g[list(metrics)].mean().round(3)
    sd = g[list(metrics)].std().round(3)
    for m in metrics:
        out[m + "_sd"] = sd[m]
    return out.sort_values(metrics[0])


def corrected_paired_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for comparing two
    models evaluated on the same CV folds (accounts for overlapping training
    sets, which make the naive paired t-test over-confident)."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), J - 1)
    return d.mean(), t, p


# --------------------------------------------------------------------------
# 4. Football-specific metric
# --------------------------------------------------------------------------
def outcome_accuracy(y_true_gd, y_pred_gd, draw_band=0.5):
    """Share of matches whose W/D/L outcome is called correctly when a
    predicted |GD| < draw_band is read as a draw."""
    t = np.sign(np.round(y_true_gd))
    p = np.where(np.abs(y_pred_gd) < draw_band, 0, np.sign(y_pred_gd))
    return float(np.mean(t == p))
