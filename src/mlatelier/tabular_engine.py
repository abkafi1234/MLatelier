from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import chi2, skew
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    PassiveAggressiveClassifier, PassiveAggressiveRegressor,
    Ridge, RidgeClassifier, SGDClassifier, SGDRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    auc, classification_report, confusion_matrix,
    f1_score, hamming_loss, make_scorer,
    mean_absolute_error, mean_squared_error,
    multilabel_confusion_matrix,
    r2_score, roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    Binarizer, LabelEncoder, MinMaxScaler,
    OneHotEncoder, RobustScaler, StandardScaler, label_binarize,
)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    import dice_ml
    _HAS_DICE = True
except ImportError:
    _HAS_DICE = False


# ── Progress helpers ──────────────────────────────────────────────────────────

def _update_progress(progress_bar=None, value: float = 0.0):
    if progress_bar is not None:
        progress_bar.progress(max(0.0, min(1.0, float(value))))


def _update_status(status_text=None, message: str = ""):
    if status_text is not None and message:
        status_text.info(message)


# ── PreparedData ──────────────────────────────────────────────────────────────

@dataclass
class PreparedData:
    """All artefacts produced by _prepare_data.

    X_train / X_test are raw DataFrames — transforms live in per-model
    ImbPipelines so each CV fold fits its own preprocessor.
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: Optional[list]
    task_type: str
    feature_names: list
    scaler_used: str
    cv: object
    metadata: dict


# ── Utility helpers ───────────────────────────────────────────────────────────

def _is_regression(y: pd.Series, cardinality_ratio_threshold: float = 0.05) -> bool:
    if not pd.api.types.is_numeric_dtype(y):
        return False
    return (y.nunique() / len(y)) > cardinality_ratio_threshold


def _fingerprint_dataframe(df: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _score_predictions(task_type: str, y_true: np.ndarray, preds: np.ndarray) -> float:
    if task_type == "regression":
        return r2_score(y_true, preds)
    if task_type == "multilabel":
        return f1_score(y_true, preds, average="micro", zero_division=0)
    return f1_score(y_true, preds, average="macro", zero_division=0)


def _cv_strategy(task_type: str, y: np.ndarray, random_state: int, n_folds: Optional[int] = None):
    if task_type == "classification":
        y_1d = np.asarray(y).ravel()
        min_class_count = int(pd.Series(y_1d).value_counts().min())
        n_splits = max(2, min(n_folds or 3, min_class_count))
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # regression and multilabel both use plain KFold
    n_splits = max(2, min(n_folds or 3, len(y)))
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _class_distribution(y: np.ndarray, class_names: Optional[list]) -> dict:
    counts = pd.Series(y).value_counts().sort_index()
    if class_names is None:
        return {str(k): int(v) for k, v in counts.items()}
    return {
        str(class_names[int(k)]) if int(k) < len(class_names) else str(k): int(v)
        for k, v in counts.items()
    }


# ── Model registries ──────────────────────────────────────────────────────────

def get_classification_models(rs: int = 42) -> dict:
    models = {
        "Random Forest":              RandomForestClassifier(random_state=rs, n_jobs=-1),
        "XGBoost":                    xgb.XGBClassifier(random_state=rs, eval_metric="logloss", n_jobs=1),
        "Gradient Boosting":          GradientBoostingClassifier(random_state=rs),
        "Extra Trees":                ExtraTreesClassifier(random_state=rs, n_jobs=-1),
        "Hist Gradient Boosting":     HistGradientBoostingClassifier(random_state=rs),
        "Support Vector Machine":     SVC(probability=True, random_state=rs),
        "Logistic Regression":        LogisticRegression(random_state=rs, max_iter=1000, n_jobs=-1),
        "Ridge Classifier":           RidgeClassifier(random_state=rs),
        "SGD Classifier":             SGDClassifier(random_state=rs, n_jobs=-1),
        "Passive Aggressive":         PassiveAggressiveClassifier(random_state=rs),
        "Linear SVC":                 LinearSVC(random_state=rs, dual=False, max_iter=2000),
        "Decision Tree":              DecisionTreeClassifier(random_state=rs),
        "KNN":                        KNeighborsClassifier(n_jobs=-1),
        "Gaussian NB":                GaussianNB(),
        "Multinomial NB":             MultinomialNB(),
        "Bernoulli NB":               BernoulliNB(),
        "MLP Classifier":             MLPClassifier(random_state=rs, max_iter=500),
        "QDA":                        QuadraticDiscriminantAnalysis(),
        "LDA":                        LinearDiscriminantAnalysis(),
        "AdaBoost":                   AdaBoostClassifier(random_state=rs),
    }
    if _HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMClassifier(random_state=rs, verbose=-1, n_jobs=-1)
    if _HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(random_seed=rs, verbose=0, thread_count=-1)
    return models


def get_regression_models(rs: int = 42) -> dict:
    models = {
        "Random Forest Regressor":          RandomForestRegressor(random_state=rs, n_jobs=-1),
        "XGBoost Regressor":                xgb.XGBRegressor(random_state=rs, n_jobs=1),
        "Gradient Boosting Regressor":      GradientBoostingRegressor(random_state=rs),
        "Extra Trees Regressor":            ExtraTreesRegressor(random_state=rs, n_jobs=-1),
        "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(random_state=rs),
        "Support Vector Regressor":         SVR(),
        "Linear Regression":                LinearRegression(n_jobs=-1),
        "Ridge Regressor":                  Ridge(),
        "SGD Regressor":                    SGDRegressor(random_state=rs),
        "Passive Aggressive Regressor":     PassiveAggressiveRegressor(random_state=rs),
        "Linear SVR":                       LinearSVR(random_state=rs, dual=False, max_iter=2000),
        "Decision Tree Regressor":          DecisionTreeRegressor(random_state=rs),
        "KNN Regressor":                    KNeighborsRegressor(n_jobs=-1),
        "MLP Regressor":                    MLPRegressor(random_state=rs, max_iter=500),
        "AdaBoost Regressor":               AdaBoostRegressor(random_state=rs),
    }
    if _HAS_LIGHTGBM:
        models["LightGBM Regressor"] = lgb.LGBMRegressor(random_state=rs, verbose=-1, n_jobs=-1)
    if _HAS_CATBOOST:
        models["CatBoost Regressor"] = CatBoostRegressor(random_seed=rs, verbose=0, thread_count=-1)
    return models


# ── Bayesian search spaces ────────────────────────────────────────────────────

def get_search_spaces() -> dict:
    spaces: dict = {
        "Random Forest": {
            "n_estimators":    Integer(50, 400),
            "max_depth":       Integer(3, 25),
            "min_samples_split": Integer(2, 12),
            "min_samples_leaf":  Integer(1, 8),
            "max_features":    Categorical(["sqrt", "log2", None]),
        },
        "Random Forest Regressor": {
            "n_estimators":    Integer(50, 400),
            "max_depth":       Integer(3, 25),
            "min_samples_split": Integer(2, 12),
            "min_samples_leaf":  Integer(1, 8),
        },
        "XGBoost": {
            "learning_rate":   Real(0.005, 0.3, prior="log-uniform"),
            "max_depth":       Integer(3, 12),
            "n_estimators":    Integer(50, 400),
            "subsample":       Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "gamma":           Real(0.0, 5.0),
        },
        "XGBoost Regressor": {
            "learning_rate":   Real(0.005, 0.3, prior="log-uniform"),
            "max_depth":       Integer(3, 12),
            "n_estimators":    Integer(50, 400),
            "subsample":       Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
        },
        "Gradient Boosting": {
            "learning_rate":   Real(0.005, 0.3, prior="log-uniform"),
            "max_depth":       Integer(2, 10),
            "n_estimators":    Integer(50, 400),
            "subsample":       Real(0.5, 1.0),
            "min_samples_leaf": Integer(1, 10),
        },
        "Gradient Boosting Regressor": {
            "learning_rate":   Real(0.005, 0.3, prior="log-uniform"),
            "max_depth":       Integer(2, 10),
            "n_estimators":    Integer(50, 400),
            "subsample":       Real(0.5, 1.0),
        },
        "Extra Trees": {
            "n_estimators":    Integer(50, 400),
            "max_depth":       Integer(3, 25),
            "min_samples_split": Integer(2, 12),
        },
        "Extra Trees Regressor": {
            "n_estimators":    Integer(50, 400),
            "max_depth":       Integer(3, 25),
        },
        "Hist Gradient Boosting": {
            "learning_rate":   Real(0.005, 0.3, prior="log-uniform"),
            "max_iter":        Integer(50, 400),
            "max_depth":       Integer(3, 20),
            "min_samples_leaf": Integer(5, 50),
            "l2_regularization": Real(0.0, 10.0),
        },
        "Hist Gradient Boosting Regressor": {
            "learning_rate":   Real(0.005, 0.3, prior="log-uniform"),
            "max_iter":        Integer(50, 400),
            "max_depth":       Integer(3, 20),
        },
        "Support Vector Machine": {
            "C":     Real(0.01, 200.0, prior="log-uniform"),
            "gamma": Real(1e-4, 1.0, prior="log-uniform"),
        },
        "Support Vector Regressor": {
            "C":       Real(0.01, 200.0, prior="log-uniform"),
            "epsilon": Real(0.001, 1.0, prior="log-uniform"),
            "gamma":   Real(1e-4, 1.0, prior="log-uniform"),
        },
        "Logistic Regression": {
            "C": Real(0.001, 100.0, prior="log-uniform"),
        },
        "Ridge Classifier": {"alpha": Real(0.001, 100.0, prior="log-uniform")},
        "Ridge Regressor":  {"alpha": Real(0.001, 100.0, prior="log-uniform")},
        "MLP Classifier": {
            "alpha":               Real(1e-5, 1e-1, prior="log-uniform"),
            "learning_rate_init":  Real(1e-4, 1e-2, prior="log-uniform"),
            "hidden_layer_sizes":  Categorical([(64,), (128,), (256,), (64, 64), (128, 64), (256, 128)]),
        },
        "MLP Regressor": {
            "alpha":              Real(1e-5, 1e-1, prior="log-uniform"),
            "learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
            "hidden_layer_sizes": Categorical([(64,), (128,), (256,), (64, 64), (128, 64)]),
        },
        "Decision Tree": {
            "max_depth":       Integer(2, 25),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf":  Integer(1, 10),
            "criterion":       Categorical(["gini", "entropy"]),
        },
        "Decision Tree Regressor": {
            "max_depth":       Integer(2, 25),
            "min_samples_split": Integer(2, 20),
            "criterion":       Categorical(["squared_error", "friedman_mse"]),
        },
        "KNN": {
            "n_neighbors": Integer(2, 25),
            "weights":     Categorical(["uniform", "distance"]),
            "p":           Integer(1, 2),
        },
        "KNN Regressor": {
            "n_neighbors": Integer(2, 25),
            "weights":     Categorical(["uniform", "distance"]),
        },
        "AdaBoost": {
            "n_estimators":  Integer(30, 300),
            "learning_rate": Real(0.01, 2.0, prior="log-uniform"),
        },
        "AdaBoost Regressor": {
            "n_estimators":  Integer(30, 300),
            "learning_rate": Real(0.01, 2.0, prior="log-uniform"),
        },
        "SGD Classifier": {
            "alpha":  Real(1e-6, 1e-1, prior="log-uniform"),
            "l1_ratio": Real(0.0, 1.0),
        },
        "SGD Regressor": {
            "alpha":  Real(1e-6, 1e-1, prior="log-uniform"),
            "l1_ratio": Real(0.0, 1.0),
        },
    }

    if _HAS_LIGHTGBM:
        lgb_common = {
            "n_estimators":    Integer(50, 400),
            "max_depth":       Integer(3, 15),
            "learning_rate":   Real(0.005, 0.3, prior="log-uniform"),
            "num_leaves":      Integer(15, 127),
            "subsample":       Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "reg_alpha":       Real(0.0, 5.0),
            "reg_lambda":      Real(0.0, 5.0),
        }
        spaces["LightGBM"] = lgb_common
        spaces["LightGBM Regressor"] = {k: v for k, v in lgb_common.items()
                                         if k not in ["subsample"]}

    if _HAS_CATBOOST:
        cb_common = {
            "iterations":     Integer(50, 400),
            "max_depth":      Integer(3, 10),
            "learning_rate":  Real(0.005, 0.3, prior="log-uniform"),
            "l2_leaf_reg":    Real(1.0, 20.0, prior="log-uniform"),
            "bagging_temperature": Real(0.0, 1.0),
        }
        spaces["CatBoost"] = cb_common
        spaces["CatBoost Regressor"] = {k: v for k, v in cb_common.items()
                                         if k != "bagging_temperature"}

    return spaces


# ── Statistical significance ──────────────────────────────────────────────────

def calculate_mcnemar_p_value(y_true, y_base_pred, y_opt_pred) -> float:
    """McNemar's test with continuity correction (Edwards, 1948)."""
    b = sum(1 for t, b, o in zip(y_true, y_base_pred, y_opt_pred) if o == t and b != t)
    c = sum(1 for t, b, o in zip(y_true, y_base_pred, y_opt_pred) if b == t and o != t)
    if b + c == 0:
        return 1.0
    return float(chi2.sf(((abs(b - c) - 1) ** 2) / (b + c), 1))


# ── Scaler registry and model family sets ────────────────────────────────────

_SCALER_MAP = {"minmax": MinMaxScaler, "robust": RobustScaler, "standard": StandardScaler}

_SCALED_MODELS = {
    "Support Vector Machine", "Support Vector Regressor",
    "Logistic Regression", "Ridge Classifier", "Ridge Regressor",
    "SGD Classifier", "SGD Regressor",
    "Passive Aggressive", "Passive Aggressive Regressor",
    "Linear SVC", "Linear SVR",
    "KNN", "KNN Regressor",
    "MLP Classifier", "MLP Regressor",
    "LDA", "QDA",
}

_UNSCALED_MODELS = {
    "Random Forest", "Random Forest Regressor",
    "XGBoost", "XGBoost Regressor",
    "Gradient Boosting", "Gradient Boosting Regressor",
    "Extra Trees", "Extra Trees Regressor",
    "Hist Gradient Boosting", "Hist Gradient Boosting Regressor",
    "Decision Tree", "Decision Tree Regressor",
    "AdaBoost", "AdaBoost Regressor",
    "LightGBM", "LightGBM Regressor",
    "CatBoost", "CatBoost Regressor",
    "Gaussian NB",
}


def auto_select_scaler(df: pd.DataFrame, features_x: list) -> str:
    """Choose scaler from outlier ratio and mean skewness of numeric features."""
    numeric_df = df[features_x].select_dtypes(include=[np.number])
    if numeric_df.empty:
        return "minmax"
    outlier_count = total_values = 0
    skew_values = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_count += int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        total_values += len(s)
        try:
            skew_values.append(abs(float(skew(s))))
        except Exception:
            pass
    if total_values == 0:
        return "minmax"
    if outlier_count / total_values > 0.05:
        return "robust"
    if skew_values and float(np.mean(skew_values)) < 1.0:
        return "standard"
    return "minmax"


# ── Pipeline factory ──────────────────────────────────────────────────────────

def _make_one_hot_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def _is_text_column(series: pd.Series,
                    min_avg_words: float = 3.0,
                    high_cardinality_ratio: float = 0.5) -> bool:
    """True when a column looks like free-text rather than a low-cardinality label.

    A column is considered text if either:
    - its average token count per entry is ≥ min_avg_words, OR
    - ≥ 50 % of its values are unique (high cardinality typical of text).
    """
    sample = series.dropna().astype(str)
    if len(sample) == 0:
        return False
    avg_words = float(sample.str.split().str.len().mean())
    cardinality_ratio = sample.nunique() / len(sample)
    return avg_words >= min_avg_words or cardinality_ratio >= high_cardinality_ratio


def _make_text_vectorizer(model_name: str, max_features: int = 500) -> TfidfVectorizer:
    """Return a TF-IDF vectorizer tuned for the model family."""
    if model_name == "Bernoulli NB":
        # Bernoulli NB expects binary features; binary=True binarises TF weights.
        return TfidfVectorizer(max_features=max_features, binary=True,
                               strip_accents="unicode", sublinear_tf=False, min_df=2)
    # MultinomialNB and all others work fine with sublinear TF + bigrams.
    return TfidfVectorizer(max_features=max_features, sublinear_tf=True,
                           strip_accents="unicode", ngram_range=(1, 2), min_df=2)


def _build_preprocessor(X: pd.DataFrame, model_name: str, scaler_key: str) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric  = [c for c in X.columns if c not in numeric_cols]
    # Split non-numeric columns: free-text gets TF-IDF, low-cardinality gets OHE.
    text_cols = [c for c in non_numeric if _is_text_column(X[c])]
    cat_cols  = [c for c in non_numeric if c not in text_cols]

    if model_name == "Bernoulli NB":
        num_steps = [("imputer", SimpleImputer(strategy="median")),
                     ("binarizer", Binarizer(threshold=0.0))]
    elif model_name == "Multinomial NB":
        num_steps = [("imputer", SimpleImputer(strategy="median")),
                     ("scaler", MinMaxScaler())]
    elif model_name in _SCALED_MODELS:
        scaler_cls = _SCALER_MAP.get(scaler_key, StandardScaler)
        num_steps = [("imputer", SimpleImputer(strategy="median")),
                     ("scaler", scaler_cls())]
    else:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]

    transformers = []
    if numeric_cols:
        transformers.append(("num", ImbPipeline(num_steps), numeric_cols))
    if cat_cols:
        transformers.append(("cat",
            ImbPipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", _make_one_hot_encoder())]),
            cat_cols))
    # One TF-IDF transformer per text column so each column gets its own vocabulary.
    for col in text_cols:
        transformers.append((f"text_{col}", _make_text_vectorizer(model_name), col))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _build_model_pipeline(model_name: str, model, data: PreparedData,
                           handle_imbalance: bool) -> ImbPipeline:
    steps = [("preprocess", _build_preprocessor(data.X_train, model_name, data.scaler_used))]

    if data.task_type == "classification" and handle_imbalance:
        min_count = int(pd.Series(data.y_train).value_counts().min())
        n_splits   = max(2, min(3, min_count))
        cv_min     = int(min_count * ((n_splits - 1) / n_splits))
        if model_name == "Bernoulli NB" or cv_min < 2:
            sampler = RandomOverSampler(random_state=data.metadata["random_seed"])
        else:
            sampler = SMOTE(k_neighbors=min(5, cv_min - 1),
                            random_state=data.metadata["random_seed"])
        steps.append(("sampler", sampler))

    if data.task_type == "multilabel":
        # Wrap in MultiOutputClassifier so every label gets its own binary estimator.
        # SMOTE is skipped for multilabel (multi-column y not supported by imblearn).
        import copy
        steps.append(("model", MultiOutputClassifier(copy.deepcopy(model), n_jobs=1)))
    else:
        steps.append(("model", model))
    return ImbPipeline(steps)


def _prefix_search_space(search_space: dict) -> dict:
    return {f"model__{k}": v for k, v in search_space.items()}


def _strip_model_prefix(params: dict) -> dict:
    return {k.replace("model__", ""): v for k, v in params.items()}


def _feature_names_from_pipeline(pipeline: ImbPipeline, fallback: list) -> list:
    try:
        return list(pipeline.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        return fallback


# ── SHAP explainability ───────────────────────────────────────────────────────

def compute_shap_values(pipeline: ImbPipeline, X_test: pd.DataFrame,
                         model_name: str, feature_names: list) -> dict:
    """Return mean |SHAP| per feature (top 20), or {} on failure."""
    if not _HAS_SHAP:
        return {}
    try:
        actual_model  = pipeline.named_steps.get("model")
        preprocessor  = pipeline.named_steps.get("preprocess")

        if preprocessor is not None:
            X_t = preprocessor.transform(X_test)
            if hasattr(X_t, "toarray"):
                X_t = X_t.toarray()
            try:
                feat_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feat_names = [f"f{i}" for i in range(X_t.shape[1])]
        else:
            X_t = X_test.values if hasattr(X_test, "values") else np.array(X_test)
            feat_names = feature_names

        n = min(300, len(X_t))
        X_s = X_t[:n]

        _TREE_MODELS = {
            "Random Forest", "XGBoost", "Gradient Boosting", "Extra Trees",
            "Hist Gradient Boosting", "Decision Tree", "LightGBM", "CatBoost",
            "AdaBoost", "Random Forest Regressor", "XGBoost Regressor",
            "Gradient Boosting Regressor", "Extra Trees Regressor",
            "Hist Gradient Boosting Regressor", "Decision Tree Regressor",
            "LightGBM Regressor", "CatBoost Regressor", "AdaBoost Regressor",
        }
        _LINEAR_MODELS = {
            "Logistic Regression", "Ridge Classifier", "Ridge Regressor",
            "Linear Regression", "SGD Classifier", "SGD Regressor", "LDA",
        }

        def _shap_mean_abs(est, X_arr, name):
            """Return mean |SHAP| array for a single estimator."""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name in _TREE_MODELS:
                    exp = shap.TreeExplainer(est)
                    sv  = exp.shap_values(X_arr)
                elif name in _LINEAR_MODELS:
                    exp = shap.LinearExplainer(est, X_arr)
                    sv  = exp.shap_values(X_arr)
                else:
                    bg  = shap.kmeans(X_arr, min(15, len(X_arr)))
                    exp = shap.KernelExplainer(est.predict, bg)
                    sv  = exp.shap_values(X_arr[:min(50, len(X_arr))])
            abs_v = np.mean([np.abs(a) for a in sv], axis=0) if isinstance(sv, list) else np.abs(sv)
            if abs_v.ndim == 1:
                abs_v = abs_v.reshape(1, -1)
            return np.mean(abs_v, axis=0)

        # Multi-label: per-label TreeExplainer, then average importances across labels.
        if isinstance(actual_model, MultiOutputClassifier):
            per_label = []
            for est in actual_model.estimators_:
                try:
                    per_label.append(_shap_mean_abs(est, X_s, model_name))
                except Exception:
                    continue
            if not per_label:
                return {}
            mean_abs = np.mean(per_label, axis=0)
        else:
            mean_abs = _shap_mean_abs(actual_model, X_s, model_name)

        raw = {name: float(val) for name, val in zip(feat_names, mean_abs)}
        return dict(sorted(raw.items(), key=lambda x: x[1], reverse=True)[:20])

    except Exception as e:
        warnings.warn(f"SHAP failed for '{model_name}': {e}", UserWarning)
        return {}


# ── ROC curve data ────────────────────────────────────────────────────────────

def compute_roc_data(y_test: np.ndarray, pipeline: ImbPipeline,
                      X_test: pd.DataFrame, class_names: Optional[list]) -> Optional[dict]:
    """Compute ROC data for binary or multi-class classification."""
    try:
        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)
        elif hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X_test)
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
        else:
            return None

        n_classes = len(class_names) if class_names else len(np.unique(y_test))
        names     = class_names or [str(i) for i in range(n_classes)]

        if n_classes == 2:
            y_score = scores[:, 1] if scores.ndim > 1 else scores.ravel()
            fpr, tpr, _ = roc_curve(y_test, y_score)
            return {
                "type": "binary",
                "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "auc": round(float(auc(fpr, tpr)), 4),
                "class_names": names,
            }

        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        result = {"type": "multiclass", "classes": {}, "class_names": names}
        for i, cn in enumerate(names):
            if i < scores.shape[1]:
                fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
                result["classes"][cn] = {
                    "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                    "auc": round(float(auc(fpr, tpr)), 4),
                }
        return result

    except Exception as e:
        warnings.warn(f"ROC computation failed: {e}", UserWarning)
        return None


# ── Model export ──────────────────────────────────────────────────────────────

def export_tabular_model(pipeline: ImbPipeline, export_dir: str,
                          model_name: str, metadata: Optional[dict] = None) -> str:
    """Persist the fitted ImbPipeline as a .joblib file."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in model_name)
    path = os.path.join(export_dir, f"MLatelier_{safe}.joblib")
    joblib.dump(pipeline, path)
    if metadata:
        with open(os.path.join(export_dir, f"MLatelier_{safe}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4, default=str)
    return path


def export_tabular_model_onnx(
    pipeline: ImbPipeline,
    export_dir: str,
    model_name: str,
    n_features: int,
) -> Optional[str]:
    """Export the inference-only portion of the pipeline to ONNX (requires skl2onnx).

    Returns the .onnx file path on success, or None if skl2onnx is not installed
    or the model family is not supported by the converter.

    The SMOTE/oversampler step is intentionally excluded — it is a training-only
    component and has no role in inference.
    """
    try:
        from skl2onnx import convert_sklearn, update_registered_converter
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.pipeline import Pipeline as _SkPipeline
    except ImportError:
        warnings.warn(
            "skl2onnx is not installed; ONNX export skipped. "
            "Install with: pip install skl2onnx onnxmltools", UserWarning)
        return None

    try:
        # Build an inference-only sklearn Pipeline (drop the SMOTE sampler step).
        inference_steps = [
            (k, v) for k, v in pipeline.named_steps.items()
            if k not in ("sampler",)
        ]
        inference_pipe = _SkPipeline(inference_steps)

        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model   = convert_sklearn(inference_pipe, initial_types=initial_type,
                                       target_opset=17)

        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in model_name)
        onnx_path = os.path.join(export_dir, f"MLatelier_{safe}.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return onnx_path

    except Exception as e:
        warnings.warn(f"ONNX export failed for '{model_name}': {e}", UserWarning)
        return None


# ── Data preparation ──────────────────────────────────────────────────────────

def _prepare_data(
    df: pd.DataFrame,
    features_x: list,
    target_y,           # str (single label) or list[str] (multi-label)
    handle_imbalance: bool,
    random_state: int = 42,
    scaler: str = "auto",
    test_size: float = 0.2,
    cv_folds: Optional[int] = None,
) -> PreparedData:
    label_cols   = [target_y] if isinstance(target_y, str) else list(target_y)
    is_multilabel = len(label_cols) > 1

    df = df.dropna(subset=label_cols).copy()
    if df.shape[0] < 20:
        raise ValueError(f"Dataset has only {df.shape[0]} rows; at least 20 are required.")
    for col in label_cols:
        if col in features_x:
            raise ValueError(f"Target '{col}' must not appear in features_x (target leakage).")

    X = df[features_x].copy()

    if is_multilabel:
        task_type   = "multilabel"
        y           = df[label_cols].values.astype(float)
        class_names = label_cols
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    else:
        target_col = label_cols[0]
        task_type  = "regression" if _is_regression(df[target_col]) else "classification"
        y = df[target_col].copy()

        class_names = None
        if task_type == "classification":
            le  = LabelEncoder()
            y   = le.fit_transform(y)
            class_names = [str(c) for c in le.classes_]

        strat = y if task_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=strat)
        except ValueError:
            warnings.warn("Stratified split failed; using random split.", UserWarning)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)

    actual_scaler = scaler if scaler != "auto" else auto_select_scaler(X_train, X_train.columns.tolist())
    cv = _cv_strategy(task_type, y_train, random_state, cv_folds)

    detected_text_cols = [c for c in features_x
                          if X[c].dtype == object and _is_text_column(X[c])]
    detected_cat_cols  = [c for c in features_x
                          if X[c].dtype == object and c not in detected_text_cols]

    return PreparedData(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        class_names=class_names,
        task_type=task_type,
        feature_names=list(X_train.columns),
        scaler_used=actual_scaler,
        cv=cv,
        metadata={
            "random_seed": random_state,
            "dataset_fingerprint_sha256": _fingerprint_dataframe(df),
            "target_column": label_cols[0] if not is_multilabel else label_cols,
            "feature_columns": list(features_x),
            "task_type": task_type,
            "preprocessing": {
                "strategy": "model-specific ImbPipelines; all transforms fit inside CV folds",
                "text_columns_tfidf": detected_text_cols,
                "categorical_columns_ohe": detected_cat_cols,
                "text_vectorizer": "TfidfVectorizer(max_features=500, sublinear_tf=True, ngram_range=(1,2))",
                "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
                "numeric_imputer": "SimpleImputer(strategy='median')",
                "user_scaler_choice": scaler,
                "resolved_scaler": actual_scaler,
                "sampler_inside_cv": bool(task_type == "classification" and handle_imbalance),
            },
            "split": {
                "test_size": test_size,
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "cv_folds": cv.get_n_splits(),
                "train_class_dist": _class_distribution(y_train, class_names)
                    if task_type == "classification" else None,
                "test_class_dist": _class_distribution(y_test, class_names)
                    if task_type == "classification" else None,
            },
        },
    )


# ── Baseline phase ────────────────────────────────────────────────────────────

def run_tabular_baseline(
    df: pd.DataFrame,
    features_x: list,
    target_y: str,
    selected_models: list,
    handle_imbalance: bool = True,
    random_state: int = 42,
    scaler: str = "auto",
    test_size: float = 0.2,
    cv_folds: Optional[int] = None,
    progress_bar=None,
    status_text=None,
    model_callback=None,
) -> tuple:
    """Cross-validate every selected model.
    Returns ({model: {mean, std, fold_scores, time_s}}, task_type).
    model_callback(name, mean, std, time_s, fold_scores) is called after each model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _prepare_data(df, features_x, target_y, handle_imbalance,
                             random_state=random_state, scaler=scaler,
                             test_size=test_size, cv_folds=cv_folds)

    registry = (get_regression_models(rs=random_state)
                if data.task_type == "regression"
                else get_classification_models(rs=random_state))
    if data.task_type == "regression":
        scoring = "r2"
    elif data.task_type == "multilabel":
        scoring = make_scorer(f1_score, average="micro", zero_division=0)
    else:
        scoring = "f1_macro"

    results: dict = {}
    runnable = [m for m in selected_models if m in registry]
    total    = max(len(runnable), 1)

    for idx, name in enumerate(runnable, 1):
        _update_status(status_text, f"Baseline {idx}/{total}: {name}")
        t0 = time.perf_counter()
        try:
            pipe = _build_model_pipeline(name, registry[name], data, handle_imbalance)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipe, data.X_train, data.y_train,
                                         cv=data.cv, scoring=scoring, n_jobs=1)
            mean_s  = float(np.nanmean(scores))
            std_s   = float(np.nanstd(scores))
            elapsed = round(time.perf_counter() - t0, 2)
            folds   = scores.tolist()
            results[name] = {"mean": mean_s, "std": std_s,
                              "fold_scores": folds, "time_s": elapsed}
            if model_callback is not None:
                try:
                    model_callback(name, mean_s, std_s, elapsed, folds)
                except Exception:
                    pass
        except Exception as e:
            warnings.warn(f"Baseline CV failed for '{name}': {e}", UserWarning)
            elapsed = round(time.perf_counter() - t0, 2)
            results[name] = {"mean": float("nan"), "std": 0.0,
                              "fold_scores": [], "time_s": elapsed}
            if model_callback is not None:
                try:
                    model_callback(name, float("nan"), 0.0, elapsed, [])
                except Exception:
                    pass
        _update_progress(progress_bar, idx / total)

    return results, data.task_type


# ── Bayesian optimisation ─────────────────────────────────────────────────────

def _compute_n_iter(search_space: dict, base: int) -> int:
    return base * min(len(search_space), 3)


def run_tabular_optimization(
    df: pd.DataFrame,
    features_x: list,
    target_y: str,
    selected_models: list,
    baseline_results: dict,
    handle_imbalance: bool = True,
    random_state: int = 42,
    scaler: str = "auto",
    test_size: float = 0.2,
    cv_folds: Optional[int] = None,
    n_iter: int = 10,
    export_model: bool = True,
    export_dir: Optional[str] = None,
    progress_bar=None,
    status_text=None,
    model_callback=None,
) -> tuple:
    """
    For each selected model:
      1. Build full ImbPipeline.
      2. Fit baseline on training split; score on holdout.
      3. BayesSearchCV on training split; winner chosen by CV score.
      4. Evaluate winner on holdout for final reporting.
      5. Compute SHAP, ROC curves, confusion matrix.
      6. Optionally export the winning pipeline.

    Returns 12-tuple:
      pipeline_results, winning_test_score, avg_base_score,
      improvement, p_val_str, winning_params, winning_report,
      class_names, task_type, winning_curves,
      actual_scaler_key, exported_model_path
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _prepare_data(df, features_x, target_y, handle_imbalance,
                             random_state=random_state, scaler=scaler,
                             test_size=test_size, cv_folds=cv_folds)

    actual_scaler = data.scaler_used
    registry      = (get_regression_models(rs=random_state)
                     if data.task_type == "regression"
                     else get_classification_models(rs=random_state))
    search_spaces = get_search_spaces()
    if data.task_type == "regression":
        scoring    = "r2"
        metric_col = "R² Score"
    elif data.task_type == "multilabel":
        scoring    = make_scorer(f1_score, average="micro", zero_division=0)
        metric_col = "Micro F1"
    else:
        scoring    = "f1_macro"
        metric_col = "Macro F1"

    pipeline_results     = []
    best_cv_score        = -float("inf")   # winner selected by inner CV score — test set not consulted
    winning_test_score   = winning_base_test_score = None
    winning_model_name   = None
    winning_base_preds   = winning_opt_preds = None
    winning_params       = winning_pipeline = None
    winning_bo_scores: list = []
    winning_bo_history: list = []
    all_exported_paths: dict = {}          # name → path for every fitted model

    runnable = [m for m in selected_models if m in registry]
    total    = max(len(runnable), 1)

    for idx, name in enumerate(runnable, 1):
        _update_status(status_text, f"Optimizing {idx}/{total}: {name}")
        model_t0 = time.perf_counter()

        bl            = baseline_results.get(name, {})
        base_cv_mean  = bl.get("mean", float("nan")) if isinstance(bl, dict) else float(bl or "nan")
        base_cv_std   = bl.get("std", 0.0) if isinstance(bl, dict) else 0.0

        base_test = opt_test = float("nan")
        base_preds = opt_preds = None
        best_params_run = {"optimization_applied": False, "reason": "No search space defined"}

        base_pipe = None
        if not np.isnan(base_cv_mean):
            try:
                base_pipe = _build_model_pipeline(name, registry[name], data, handle_imbalance)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    base_pipe.fit(data.X_train, data.y_train)
                base_preds  = base_pipe.predict(data.X_test)
                base_test   = _score_predictions(data.task_type, data.y_test, base_preds)
                opt_test    = base_test
            except Exception as e:
                warnings.warn(f"Baseline fit failed for '{name}': {e}", UserWarning)
                base_preds = np.zeros_like(data.y_test)
        else:
            base_preds = np.zeros_like(data.y_test)

        sel_score      = base_cv_mean
        current_pipe   = base_pipe
        cur_bo_scores: list = []
        cur_bo_history: list = []

        if name in search_spaces and not np.isnan(base_cv_mean):
            try:
                opt_pipe = _build_model_pipeline(name, registry[name], data, handle_imbalance)
                # For MultiOutputClassifier the param path goes through model__estimator__
                if data.task_type == "multilabel":
                    prefixed_sp = {f"model__estimator__{k}": v
                                   for k, v in search_spaces[name].items()}
                else:
                    prefixed_sp = _prefix_search_space(search_spaces[name])
                eff_n_iter    = _compute_n_iter(search_spaces[name], n_iter)

                opt = BayesSearchCV(
                    estimator=opt_pipe,
                    search_spaces=prefixed_sp,
                    n_iter=eff_n_iter,
                    cv=data.cv,
                    scoring=scoring,
                    random_state=random_state,
                    n_jobs=1,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    opt.fit(data.X_train, data.y_train)

                raw_params      = dict(opt.best_params_)
                best_params_run = {
                    k.replace("model__estimator__", "").replace("model__", ""): v
                    for k, v in raw_params.items()
                }
                sel_score       = float(opt.best_score_)
                opt_preds       = opt.predict(data.X_test)
                opt_test        = _score_predictions(data.task_type, data.y_test, opt_preds)
                current_pipe    = opt.best_estimator_

                cur_bo_scores  = opt.cv_results_["mean_test_score"].tolist()
                cur_bo_params  = opt.cv_results_["params"]
                cur_bo_history = [
                    {"trial": i + 1, "score": float(cur_bo_scores[i]),
                     **{k.replace("model__", ""): (v if isinstance(v, (int, float, str)) else str(v))
                        for k, v in cur_bo_params[i].items()}}
                    for i in range(len(cur_bo_scores))
                ]
            except Exception as e:
                warnings.warn(f"BO failed for '{name}': {e}", UserWarning)

        if opt_preds is None:
            opt_preds = base_preds

        model_elapsed = round(time.perf_counter() - model_t0, 2)
        pipeline_results.append({
            "Algorithm":                          name,
            f"CV {metric_col} (mean±std)":        f"{base_cv_mean:.4f} ± {base_cv_std:.4f}" if not np.isnan(base_cv_mean) else "—",
            f"Baseline Test {metric_col}":         base_test,
            f"Optimized Test {metric_col}":        opt_test,
            f"Selection CV {metric_col}":          sel_score,
            "Training Time (s)":                   model_elapsed,
        })
        if model_callback is not None:
            try:
                model_callback(name, opt_test, sel_score, model_elapsed)
            except Exception:
                pass

        # Winner = best inner CV score (BayesSearchCV best_score_, or baseline CV mean
        # when no search space exists). D_test is reserved for the single final
        # evaluation of the chosen winner only — it plays no role in selection.
        _compare = sel_score if not np.isnan(sel_score) else -float("inf")
        if _compare > best_cv_score:
            best_cv_score           = _compare
            winning_test_score      = opt_test
            winning_base_test_score = base_test
            winning_model_name      = name
            winning_base_preds      = base_preds
            winning_opt_preds       = opt_preds
            winning_params          = best_params_run
            winning_pipeline        = current_pipe
            winning_bo_scores       = cur_bo_scores
            winning_bo_history      = cur_bo_history

        # Export every successfully fitted model so users can download any of them.
        if export_model and export_dir and current_pipe is not None:
            try:
                os.makedirs(export_dir, exist_ok=True)
                _emeta = {
                    "model_name":   name,
                    "task_type":    data.task_type,
                    "feature_names": data.feature_names,
                    "class_names":  data.class_names,
                    "scaler_used":  actual_scaler,
                    "best_params":  best_params_run,
                    "test_score":   float(opt_test) if not np.isnan(opt_test) else None,
                    "cv_score":     float(sel_score) if not np.isnan(sel_score) else None,
                    "metric":       metric_col,
                }
                _epath = export_tabular_model(current_pipe, export_dir, name, _emeta)
                all_exported_paths[name] = _epath
                # Best-effort ONNX export alongside the .joblib file.
                export_tabular_model_onnx(
                    current_pipe, export_dir, name,
                    n_features=len(data.feature_names))
            except Exception as _ee:
                warnings.warn(f"Export failed for '{name}': {_ee}", UserWarning)

        _update_progress(progress_bar, idx / total)

    # ── Early exit ──
    if winning_model_name is None:
        return (None,) * 12

    # ── Final evaluation on holdout ──
    if data.task_type == "regression":
        _yt  = np.array(data.y_test).flatten()
        _yp  = np.array(winning_opt_preds).flatten()
        _mse = float(mean_squared_error(_yt, _yp))
        winning_report = {
            "R-Squared (R2)":          f"{winning_test_score:.4f}",
            "Mean Squared Error (MSE)": f"{_mse:.4f}",
            "Root MSE (RMSE)":          f"{float(np.sqrt(_mse)):.4f}",
            "Mean Absolute Error (MAE)": f"{float(mean_absolute_error(_yt, _yp)):.4f}",
            "y_true": _yt[:2000].tolist(),
            "y_pred": _yp[:2000].tolist(),
        }
        p_val_str = "N/A (Regression)"
        roc_data  = None
        cm_data   = None
    elif data.task_type == "multilabel":
        _yt = np.array(data.y_test)
        _yp = np.array(winning_opt_preds)
        per_label = classification_report(
            _yt, _yp, target_names=data.class_names, output_dict=True, zero_division=0)
        winning_report = {
            **per_label,
            "hamming_loss":      f"{hamming_loss(_yt, _yp):.4f}",
            "micro_f1":          f"{f1_score(_yt, _yp, average='micro', zero_division=0):.4f}",
            "macro_f1":          f"{f1_score(_yt, _yp, average='macro', zero_division=0):.4f}",
            "samples_f1":        f"{f1_score(_yt, _yp, average='samples', zero_division=0):.4f}",
            "label_names":       data.class_names,
            "_task":             "multilabel",
        }
        mcm = multilabel_confusion_matrix(_yt, _yp)
        cm_data = {"matrix": mcm.tolist(), "labels": data.class_names or [], "multilabel": True}
        winning_report["confusion_matrix"]        = mcm.tolist()
        winning_report["confusion_matrix_labels"] = data.class_names
        p_val_str = "N/A (Multi-label)"
        roc_data  = None
    else:
        winning_report = classification_report(
            data.y_test, winning_opt_preds, output_dict=True, zero_division=0)
        winning_report["accuracy_score"] = float(
            accuracy_score(data.y_test, winning_opt_preds))
        p_val_raw = calculate_mcnemar_p_value(data.y_test, winning_base_preds, winning_opt_preds)
        p_val_str = (f"{p_val_raw:.3f} (Significant)" if p_val_raw < 0.05
                     else f"{p_val_raw:.3f} (Not Sig.)")

        cm       = confusion_matrix(data.y_test, winning_opt_preds)
        cm_data  = {"matrix": cm.tolist(), "labels": data.class_names or []}
        winning_report["confusion_matrix"] = cm.tolist()
        winning_report["confusion_matrix_labels"] = data.class_names

        roc_data = compute_roc_data(data.y_test, winning_pipeline, data.X_test, data.class_names)

    avg_base   = float(np.nanmean([r[f"Baseline Test {metric_col}"] for r in pipeline_results]))
    improvement = (
        ((winning_test_score - winning_base_test_score) / abs(winning_base_test_score)) * 100
        if winning_base_test_score and winning_base_test_score != 0 else 0.0
    )

    # ── Feature importance ──
    feat_importances: dict = {}
    if winning_pipeline is not None:
        try:
            m      = winning_pipeline.named_steps.get("model")
            tnames = _feature_names_from_pipeline(winning_pipeline, data.feature_names)
            # For MultiOutputClassifier, average importances across per-label estimators
            if isinstance(m, MultiOutputClassifier):
                sub_imps = []
                for est in m.estimators_:
                    if hasattr(est, "feature_importances_"):
                        sub_imps.append(est.feature_importances_)
                    elif hasattr(est, "coef_"):
                        c = est.coef_
                        sub_imps.append(np.abs(c[0] if c.ndim > 1 else c))
                if sub_imps:
                    avg_imp = np.mean(sub_imps, axis=0)
                    feat_importances = {f: float(v) for f, v in zip(tnames, avg_imp)}
            elif m is not None and hasattr(m, "feature_importances_"):
                feat_importances = {f: float(i) for f, i in
                                    zip(tnames, m.feature_importances_)}
            elif m is not None and hasattr(m, "coef_"):
                imp = m.coef_
                if imp.ndim > 1:
                    imp = imp[0]
                feat_importances = {f: float(abs(i)) for f, i in zip(tnames, imp)}
        except Exception as e:
            warnings.warn(f"Feature importance extraction failed: {e}", UserWarning)

    sorted_feats = dict(
        sorted(feat_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:15])

    # ── SHAP ──
    shap_vals: dict = {}
    if winning_pipeline is not None:
        _update_status(status_text, "Computing SHAP explainability (this may take a moment)...")
        shap_vals = compute_shap_values(
            winning_pipeline, data.X_test, winning_model_name, data.feature_names)

    # All models were exported inside the loop; grab the winner's path.
    exported_path = all_exported_paths.get(winning_model_name)

    winning_curves = {
        "bo_scores":           winning_bo_scores,
        "bo_history":          winning_bo_history,
        "feature_importances": sorted_feats,
        "shap_values":         shap_vals,
        "roc_data":            roc_data,
        "confusion_matrix":    cm_data,
        "exported_model_path": exported_path,
        "all_exported_paths":  all_exported_paths,
        "reproducibility_metadata": {
            **data.metadata,
            "evaluation": {
                "winner_selected_by": "inner cross-validation score (BayesSearchCV best_score_)",
                "winning_test_score": float(winning_test_score) if winning_test_score is not None else None,
            },
            "bayesian_optimization": {
                "library":            "scikit-optimize BayesSearchCV",
                "base_n_iter":        n_iter,
                "cv_folds":           data.cv.get_n_splits(),
                "scoring":            scoring,
                "pipeline_aware":     True,
                "preprocessing_inside_cv": True,
                "sampler_inside_cv":  bool(data.task_type == "classification" and handle_imbalance),
            },
        },
    }

    return (
        pipeline_results,
        winning_test_score,
        avg_base,
        improvement,
        p_val_str,
        winning_params,
        winning_report,
        data.class_names,
        data.task_type,
        winning_curves,
        actual_scaler,
        exported_path,
    )


# ── Calibration curve ─────────────────────────────────────────────────────────

def compute_calibration_data(
    pipeline: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: Optional[list] = None,
    n_bins: int = 10,
) -> Optional[dict]:
    """Compute reliability-diagram data for binary or multiclass classification.
    Returns None for regression or if predict_proba is unavailable.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    try:
        if not hasattr(pipeline, "predict_proba"):
            return None

        y_prob   = pipeline.predict_proba(X_test)
        n_cls    = y_prob.shape[1]
        names    = class_names or [str(i) for i in range(n_cls)]

        if n_cls == 2:
            prob_pos = y_prob[:, 1]
            fop, mpv = calibration_curve(y_test, prob_pos, n_bins=n_bins,
                                          strategy="quantile")
            brier = float(brier_score_loss(y_test, prob_pos))
            return {
                "type":        "binary",
                "fop":         fop.tolist(),
                "mpv":         mpv.tolist(),
                "brier_score": round(brier, 4),
                "class_names": names,
            }

        # Multiclass one-vs-rest
        y_bin   = label_binarize(y_test, classes=list(range(n_cls)))
        classes_data: dict = {}
        for i, cn in enumerate(names):
            if i >= y_prob.shape[1]:
                break
            try:
                fop, mpv = calibration_curve(
                    y_bin[:, i], y_prob[:, i], n_bins=n_bins, strategy="quantile")
                brier = float(brier_score_loss(y_bin[:, i], y_prob[:, i]))
                classes_data[cn] = {"fop": fop.tolist(), "mpv": mpv.tolist(),
                                     "brier_score": round(brier, 4)}
            except Exception:
                pass
        return {"type": "multiclass", "classes": classes_data, "class_names": names}

    except Exception as e:
        warnings.warn(f"Calibration curve computation failed: {e}", UserWarning)
        return None


# ── Ablation study ────────────────────────────────────────────────────────────

def run_tabular_ablation(
    df: pd.DataFrame,
    features_x: list,
    target_y: str,
    model_name: str,
    ablation_configs: list,
    random_state: int = 42,
    progress_bar=None,
    status_text=None,
) -> list:
    """
    Run one model under multiple pipeline configurations for ablation analysis.

    ablation_configs: list of dicts, each with:
        "label": str                   — display name for this configuration
        "handle_imbalance": bool       — (optional, default True)
        "scaler": str                  — (optional, default "auto")
        "test_size": float             — (optional, default 0.2)
        "cv_folds": int | None         — (optional, default None)

    Returns: list of dicts: Config, CV Score, Std(±), Time(s), Delta(%)
    """
    results = []
    baseline_score: Optional[float] = None
    total = max(len(ablation_configs), 1)

    for i, cfg in enumerate(ablation_configs):
        label = cfg.get("label", f"Config {i + 1}")
        hi    = cfg.get("handle_imbalance", True)
        sc    = cfg.get("scaler", "auto")
        ts    = cfg.get("test_size", 0.2)
        cf    = cfg.get("cv_folds", None)

        _update_status(status_text, f"Ablation {i + 1}/{total}: {label}")
        t0 = time.perf_counter()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = _prepare_data(df, features_x, target_y, hi,
                                      random_state=random_state,
                                      scaler=sc, test_size=ts, cv_folds=cf)

            registry = (get_regression_models(rs=random_state)
                        if data.task_type == "regression"
                        else get_classification_models(rs=random_state))

            if model_name not in registry:
                results.append({"Config": label, "CV Score": float("nan"),
                                  "Std (±)": 0.0, "Time (s)": 0.0, "Delta (%)": "—"})
                continue

            scoring = "r2" if data.task_type == "regression" else "f1_macro"
            pipe    = _build_model_pipeline(model_name, registry[model_name], data, hi)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(pipe, data.X_train, data.y_train,
                                          cv=data.cv, scoring=scoring, n_jobs=1)

            elapsed    = round(time.perf_counter() - t0, 2)
            mean_score = float(np.nanmean(scores))
            std_score  = float(np.nanstd(scores))

            if baseline_score is None:
                baseline_score = mean_score

            if baseline_score and baseline_score != 0 and not np.isnan(mean_score):
                delta = (mean_score - baseline_score) / abs(baseline_score) * 100
                delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
            else:
                delta_str = "—"

            results.append({
                "Config":    label,
                "CV Score":  round(mean_score, 4),
                "Std (±)":   round(std_score, 4),
                "Time (s)":  elapsed,
                "Delta (%)": delta_str,
            })

        except Exception as e:
            warnings.warn(f"Ablation failed for config '{label}': {e}", UserWarning)
            elapsed = round(time.perf_counter() - t0, 2)
            results.append({"Config": label, "CV Score": float("nan"),
                              "Std (±)": 0.0, "Time (s)": elapsed, "Delta (%)": "—"})

        _update_progress(progress_bar, (i + 1) / total)

    return results


# ── Learning curve ─────────────────────────────────────────────────────────────

def compute_learning_curve(
    df: pd.DataFrame,
    features_x: list,
    target_y: str,
    model_name: str,
    handle_imbalance: bool = True,
    random_state: int = 42,
    scaler: str = "auto",
    cv_folds: Optional[int] = None,
    n_points: int = 6,
    progress_bar=None,
    status_text=None,
) -> dict:
    """
    Compute sklearn learning_curve for `model_name` with the full pipeline.
    Returns dict with train_sizes, train_means, train_stds, val_means, val_stds,
    metric_name.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _prepare_data(df, features_x, target_y, handle_imbalance,
                             random_state=random_state, scaler=scaler,
                             test_size=0.2, cv_folds=cv_folds)

    registry = (get_regression_models(rs=random_state)
                if data.task_type == "regression"
                else get_classification_models(rs=random_state))
    if model_name not in registry:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    pipe    = _build_model_pipeline(model_name, registry[model_name], data, handle_imbalance)
    scoring = "r2" if data.task_type == "regression" else "f1_macro"

    _update_status(status_text, f"Computing learning curve for {model_name}…")

    train_sizes_abs = np.linspace(0.10, 1.0, n_points)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.model_selection import learning_curve as _lc
        X_all = pd.concat([data.X_train, data.X_test], ignore_index=True)
        y_all = np.concatenate([data.y_train, data.y_test])
        ts, tr_scores, val_scores = _lc(
            pipe, X_all, y_all,
            train_sizes=train_sizes_abs,
            cv=data.cv,
            scoring=scoring,
            n_jobs=1,
            shuffle=True,
            random_state=random_state,
        )

    _update_progress(progress_bar, 1.0)
    return {
        "train_sizes":  ts.tolist(),
        "train_means":  np.nanmean(tr_scores,  axis=1).tolist(),
        "train_stds":   np.nanstd(tr_scores,   axis=1).tolist(),
        "val_means":    np.nanmean(val_scores,  axis=1).tolist(),
        "val_stds":     np.nanstd(val_scores,   axis=1).tolist(),
        "metric_name":  "R² Score" if data.task_type == "regression" else "Macro F1",
    }


# ── Feature ablation (drop-one) ───────────────────────────────────────────────

def run_feature_ablation(
    df: pd.DataFrame,
    features_x: list,
    target_y: str,
    model_name: str,
    handle_imbalance: bool = True,
    random_state: int = 42,
    scaler: str = "auto",
    cv_folds: Optional[int] = None,
    progress_bar=None,
    status_text=None,
) -> list:
    """
    Drop one feature at a time; return list of dicts with Feature Dropped,
    Baseline CV, CV Without, Delta (%), Time (s).
    """
    if len(features_x) < 2:
        raise ValueError("Feature ablation requires at least 2 features.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = _prepare_data(df, features_x, target_y, handle_imbalance,
                             random_state=random_state, scaler=scaler,
                             test_size=0.2, cv_folds=cv_folds)

    registry = (get_regression_models(rs=random_state)
                if data.task_type == "regression"
                else get_classification_models(rs=random_state))
    if model_name not in registry:
        raise ValueError(f"Model '{model_name}' not found.")

    scoring = "r2" if data.task_type == "regression" else "f1_macro"

    # Baseline score (all features)
    _update_status(status_text, f"Feature ablation baseline ({model_name})…")
    base_pipe = _build_model_pipeline(model_name, registry[model_name], data, handle_imbalance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_scores = cross_val_score(
            base_pipe, data.X_train, data.y_train,
            cv=data.cv, scoring=scoring, n_jobs=1)
    baseline_cv = float(np.nanmean(base_scores))

    results = []
    total   = len(features_x)
    for i, feat in enumerate(features_x, 1):
        _update_status(status_text, f"Ablating feature {i}/{total}: {feat}")
        t0       = time.perf_counter()
        remaining = [f for f in features_x if f != feat]
        try:
            sub_data = _prepare_data(
                df, remaining, target_y, handle_imbalance,
                random_state=random_state, scaler=scaler,
                test_size=0.2, cv_folds=cv_folds)
            sub_pipe = _build_model_pipeline(
                model_name, registry[model_name], sub_data, handle_imbalance)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    sub_pipe, sub_data.X_train, sub_data.y_train,
                    cv=sub_data.cv, scoring=scoring, n_jobs=1)
            cv_without = float(np.nanmean(scores))
            delta      = (cv_without - baseline_cv) / (abs(baseline_cv) or 1) * 100
            elapsed    = round(time.perf_counter() - t0, 2)
            results.append({
                "Feature Dropped": feat,
                "Baseline CV":     round(baseline_cv, 4),
                "CV Without":      round(cv_without, 4),
                "Delta (%)":       round(delta, 2),
                "Time (s)":        elapsed,
            })
        except Exception as e:
            results.append({
                "Feature Dropped": feat,
                "Baseline CV":     round(baseline_cv, 4),
                "CV Without":      float("nan"),
                "Delta (%)":       float("nan"),
                "Time (s)":        round(time.perf_counter() - t0, 2),
            })
            warnings.warn(f"Feature ablation failed for {feat}: {e}", UserWarning)

        _update_progress(progress_bar, i / total)

    return sorted(results, key=lambda r: r.get("Delta (%)", 0) or 0)


# ── Partial Dependence ────────────────────────────────────────────────────────

def compute_pdp(
    pipeline,
    X_sample: pd.DataFrame,
    feature_names: list,
    top_n: int = 6,
    grid_resolution: int = 40,
) -> dict:
    """
    Compute marginal (partial dependence) for top_n numeric features.
    Returns {feature: (xs_list, ys_list)} using brute-force ICE averaging.
    """
    results: dict = {}
    numeric_feats = [f for f in feature_names
                     if f in X_sample.columns
                     and pd.api.types.is_numeric_dtype(X_sample[f])][:top_n]
    if not numeric_feats or X_sample.empty:
        return results

    X_ref = X_sample.copy().reset_index(drop=True)
    for feat in numeric_feats:
        try:
            col    = X_ref[feat].dropna()
            xs     = np.linspace(col.quantile(0.02), col.quantile(0.98), grid_resolution)
            ys_acc = []
            for xv in xs:
                X_mod = X_ref.copy()
                X_mod[feat] = xv
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if hasattr(pipeline, "predict_proba"):
                        p = pipeline.predict_proba(X_mod)
                        ys_acc.append(float(np.mean(p[:, -1])))
                    else:
                        p = pipeline.predict(X_mod)
                        ys_acc.append(float(np.mean(p)))
            results[feat] = (xs.tolist(), ys_acc)
        except Exception:
            continue
    return results


# ── SHAP waterfall for one sample ─────────────────────────────────────────────

def compute_shap_waterfall(
    pipeline,
    X_sample: pd.DataFrame,
    feature_names: list,
    sample_idx: int = 0,
) -> dict:
    """
    Compute SHAP values for one row and return waterfall-ready dict.
    Returns {"features": [...], "shap_values": [...], "base_value": float}.
    """
    try:
        import shap
    except ImportError:
        return {}

    if X_sample.empty or sample_idx >= len(X_sample):
        return {}

    X_ref = X_sample.reindex(columns=feature_names)
    try:
        model_step = pipeline.named_steps.get("model")
        preproc    = [v for k, v in pipeline.named_steps.items()
                      if k != "model" and k != "sampler"]
        if preproc:
            import functools
            X_transformed = functools.reduce(
                lambda acc, tr: tr.transform(acc), preproc, X_ref)
        else:
            X_transformed = X_ref

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Multi-label: average SHAP across per-label estimators.
            if isinstance(model_step, MultiOutputClassifier):
                per_label_rows = []
                base_values_list = []
                for est in model_step.estimators_:
                    try:
                        exp = shap.TreeExplainer(est)
                        sv  = exp.shap_values(X_transformed)
                        sv_row = sv[sample_idx] if not isinstance(sv, list) else \
                                 np.mean(np.abs([s[sample_idx] for s in sv]), axis=0)
                        per_label_rows.append(np.abs(sv_row) if sv_row.ndim == 1 else
                                              np.mean(np.abs(sv_row), axis=-1))
                        ev = exp.expected_value
                        base_values_list.append(float(np.mean(ev) if hasattr(ev, "__len__") else ev))
                    except Exception:
                        continue
                if not per_label_rows:
                    return {}
                row_shap   = np.mean(per_label_rows, axis=0)
                base_value = float(np.mean(base_values_list))
                cols = _feature_names_from_pipeline(pipeline, feature_names)
                if len(cols) != len(row_shap):
                    cols = feature_names[:len(row_shap)]
                return {
                    "features":    cols,
                    "shap_values": [float(v) for v in row_shap],
                    "base_value":  base_value,
                }
            try:
                explainer   = shap.TreeExplainer(model_step)
                shap_vals   = explainer.shap_values(X_transformed)
                base_value  = float(np.mean(explainer.expected_value)
                                    if hasattr(explainer.expected_value, "__len__")
                                    else explainer.expected_value)
            except Exception:
                explainer   = shap.KernelExplainer(
                    model_step.predict, X_transformed[:50])
                shap_vals   = explainer.shap_values(X_transformed[sample_idx:sample_idx+1])
                base_value  = float(explainer.expected_value)

        if isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim == 3:
                # Shape may be (n_classes, n_samples, n_features) or
                # (n_samples, n_features, n_classes). Detect by comparing axis-0 to n_samples.
                n_samples = len(X_transformed)
                if shap_vals.shape[0] == n_samples:
                    # (n_samples, n_features, n_classes) — pick sample, then mean-abs across classes
                    row_shap = np.mean(np.abs(shap_vals[sample_idx]), axis=-1)
                else:
                    # (n_classes, n_samples, n_features) — slice sample from each class
                    row_shap = np.mean(np.abs(shap_vals[:, sample_idx, :]), axis=0)
            else:
                row_shap = shap_vals[sample_idx]
        elif isinstance(shap_vals, list):
            # List of (n_samples, n_features) arrays, one per class → mean-abs across classes
            per_class = np.array([sv[sample_idx] for sv in shap_vals])  # (n_classes, n_features)
            row_shap  = np.mean(np.abs(per_class), axis=0)
        else:
            row_shap = shap_vals.values[sample_idx]

        # Final safety: collapse any remaining 2-D shape
        row_shap = np.array(row_shap)
        if row_shap.ndim > 1:
            row_shap = np.mean(np.abs(row_shap), axis=0)

        cols = _feature_names_from_pipeline(pipeline, feature_names)
        if len(cols) != len(row_shap):
            cols = feature_names[:len(row_shap)]

        return {
            "features":    cols,
            "shap_values": [float(v) for v in row_shap],
            "base_value":  base_value,
        }
    except Exception:
        return {}


# ── LIME for one sample ───────────────────────────────────────────────────────

def compute_lime_explanation(
    pipeline,
    X_train: pd.DataFrame,
    X_sample: pd.DataFrame,
    feature_names: list,
    task_type: str = "classification",
    sample_idx: int = 0,
) -> dict:
    """
    Compute LIME explanation for one sample.
    Returns {"features": [...], "weights": [...], "label": str}.
    """
    try:
        import lime.lime_tabular as _lime_tab
    except ImportError:
        return {}

    if X_sample.empty or X_train.empty:
        return {}

    try:
        X_tr = X_train.reindex(columns=feature_names).fillna(0).values
        X_sp = X_sample.reindex(columns=feature_names).fillna(0).values

        mode = "classification" if task_type == "classification" else "regression"
        explainer = _lime_tab.LimeTabularExplainer(
            X_tr,
            feature_names=feature_names,
            mode=mode,
            random_state=42,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task_type == "classification":
                exp = explainer.explain_instance(
                    X_sp[sample_idx],
                    pipeline.predict_proba,
                    num_features=min(15, len(feature_names)),
                )
                pred_class = str(pipeline.predict(X_sample.iloc[[sample_idx]])[0])
            else:
                exp = explainer.explain_instance(
                    X_sp[sample_idx],
                    pipeline.predict,
                    num_features=min(15, len(feature_names)),
                )
                pred_class = f"{pipeline.predict(X_sample.iloc[[sample_idx]])[0]:.4f}"

        feat_weights = exp.as_list()
        return {
            "features": [fw[0] for fw in feat_weights],
            "weights":  [fw[1] for fw in feat_weights],
            "label":    pred_class,
        }
    except Exception:
        return {}


# ── Counterfactual explanations (DiCE) ───────────────────────────────────────

def compute_counterfactuals(
    pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_sample: pd.DataFrame,
    feature_names: list,
    target_col: str,
    class_names: Optional[list] = None,
    sample_idx: int = 0,
    n_cfs: int = 5,
    desired_class: str = "opposite",
) -> dict:
    """Generate counterfactual explanations for one sample using DiCE.

    Returns a dict with keys:
        "original"       : {feature: value} for the query instance
        "original_pred"  : predicted class label (str)
        "counterfactuals": list of {feature: value, "_pred": label} dicts
        "n_found"        : number of valid CFs generated

    Returns {} if dice_ml is not installed, the task is not classification,
    or CF generation fails for any reason.
    """
    if not _HAS_DICE:
        warnings.warn(
            "dice_ml is not installed; counterfactual generation skipped. "
            "Install with: pip install dice-ml", UserWarning)
        return {}
    if X_sample.empty or X_train.empty:
        return {}

    try:
        import dice_ml

        # Reconstruct a labelled training DataFrame DiCE expects.
        y_labels = (
            [class_names[int(v)] for v in y_train]
            if class_names is not None
            else [str(int(v)) for v in y_train]
        )
        train_df = X_train.reindex(columns=feature_names).copy()
        train_df[target_col] = y_labels

        numeric_feats     = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_feats = [c for c in feature_names if c not in numeric_feats]

        dice_data = dice_ml.Data(
            dataframe=train_df,
            continuous_features=numeric_feats,
            outcome_name=target_col,
        )
        dice_model = dice_ml.Model(model=pipeline, backend="sklearn")
        explainer  = dice_ml.Dice(dice_data, dice_model, method="random")

        query = X_sample.reindex(columns=feature_names).iloc[[sample_idx]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = explainer.generate_counterfactuals(
                query,
                total_CFs=n_cfs,
                desired_class=desired_class,
                random_seed=42,
            )

        cf_df   = result.cf_examples_list[0].final_cfs_df
        orig_pred = str(pipeline.predict(query)[0])
        if class_names:
            try:
                orig_pred = class_names[int(orig_pred)]
            except (ValueError, IndexError):
                pass

        cfs = []
        for _, row in cf_df.iterrows():
            entry = {f: row[f] for f in feature_names if f in row}
            pred_val = str(row[target_col]) if target_col in row else "?"
            entry["_pred"] = pred_val
            cfs.append(entry)

        return {
            "original":        query.iloc[0].to_dict(),
            "original_pred":   orig_pred,
            "counterfactuals": cfs,
            "n_found":         len(cfs),
        }

    except Exception as e:
        warnings.warn(f"Counterfactual generation failed: {e}", UserWarning)
        return {}

