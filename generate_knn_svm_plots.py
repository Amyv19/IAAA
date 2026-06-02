from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


TARGET = "price"
FEATURES_17 = [
    "accommodates",
    "estimated_revenue_l365d",
    "bedrooms",
    "beds",
    "longitude",
    "calculated_host_listings_count",
    "host_total_listings_count",
    "review_scores_cleanliness",
    "review_scores_location",
    "host_acceptance_rate",
    "host_listings_count",
    "review_scores_accuracy",
    "instant_bookable",
    "host_response_time",
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
]
CAT_FEATURES = [
    "host_response_time",
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
]
NUM_FEATURES = [feature for feature in FEATURES_17 if feature not in CAT_FEATURES]


def project_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)


def clean_price_column(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(project_path("listings.csv"), low_memory=False)
    df[TARGET] = clean_price_column(df[TARGET])

    for col in ["host_response_rate", "host_acceptance_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%", "").str.strip(),
                errors="coerce",
            )

    for col in ["host_is_superhost", "has_availability", "instant_bookable"]:
        if col in df.columns:
            df[col] = df[col].map({"t": 1, "f": 0, True: 1, False: 0}).astype(float)

    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET] > 0].copy()

    q1, q3 = df[TARGET].quantile(0.25), df[TARGET].quantile(0.75)
    iqr = q3 - q1
    df = df[(df[TARGET] >= q1 - 1.5 * iqr) & (df[TARGET] <= q3 + 1.5 * iqr)].copy()

    available_features = [feature for feature in FEATURES_17 if feature in df.columns]
    if len(available_features) != len(FEATURES_17):
        missing = sorted(set(FEATURES_17) - set(available_features))
        raise ValueError(f"Faltan variables requeridas: {missing}")

    return df[FEATURES_17 + [TARGET]].copy()


def build_preprocessor() -> ColumnTransformer:
    num_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_transformer, NUM_FEATURES),
            ("cat", cat_transformer, CAT_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "k vecinos": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("reg", KNeighborsRegressor(n_neighbors=12, weights="distance")),
            ]
        ),
        "svm": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("reg", SVR(kernel="rbf", C=120, epsilon=35, gamma="scale")),
            ]
        ),
    }


def save_scatter_plot(y_true: pd.Series, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=140)
    ax.scatter(y_true, y_pred, s=14, alpha=0.55, color="#2d8fd5", edgecolors="none")
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.2, color="#ff6b6b")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Valores reales")
    ax.set_ylabel("Valores predichos")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_residual_plot(y_true: pd.Series, y_pred: np.ndarray, title: str, output_path: Path) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=140)
    ax.scatter(y_pred, residuals, s=14, alpha=0.55, color="#2d8fd5", edgecolors="none")
    ax.axhline(0, linestyle="--", linewidth=1.2, color="#ff6b6b")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicciones")
    ax.set_ylabel("Residuos")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_panel_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    scatter_title: str,
    residual_title: str,
    output_path: Path,
) -> None:
    residuals = y_true - y_pred
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=150)

    axes[0].scatter(y_true, y_pred, s=12, alpha=0.55, color="#2d8fd5", edgecolors="none")
    axes[0].plot([low, high], [low, high], linestyle="--", linewidth=1.2, color="#ff6b6b")
    axes[0].set_title(scatter_title, fontsize=10)
    axes[0].set_xlabel("Valores reales")
    axes[0].set_ylabel("Valores predichos")
    axes[0].grid(alpha=0.18)

    axes[1].scatter(y_pred, residuals, s=12, alpha=0.55, color="#2d8fd5", edgecolors="none")
    axes[1].axhline(0, linestyle="--", linewidth=1.2, color="#ff6b6b")
    axes[1].set_title(residual_title, fontsize=10)
    axes[1].set_xlabel("Predicciones")
    axes[1].set_ylabel("Residuos")
    axes[1].grid(alpha=0.18)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def export_to_targets(filename: str, source_path: Path) -> None:
    for folder in [
        project_path("static", "plots"),
        project_path("docs", "static", "plots"),
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, folder / filename)


def main() -> None:
    df = load_dataset()
    X = df[FEATURES_17]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    metrics_rows = []
    tmp_dir = project_path("_tmp_plot_exports")
    tmp_dir.mkdir(exist_ok=True)

    for model_name, pipeline in build_models().items():
        print(f"Entrenando {model_name}...")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = r2_score(y_test, predictions)
        metrics_rows.append({"Modelo": model_name, "RMSE": rmse, "R2": r2})

        if model_name == "k vecinos":
            scatter_name = "Regresion_k_vecinos_reales_vs_predichos.png"
            residual_name = "Residuos_Modelo_k_vecinos.png"
            panel_name = "Panel_Modelo_k_vecinos.png"
            scatter_title = "Regresion k vecinos: reales vs predichos"
            residual_title = "Residuos - Modelo k vecinos"
        else:
            scatter_name = "Regresion_svm_reales_vs_predichos.png"
            residual_name = "Residuos_Modelo_svm.png"
            panel_name = "Panel_Modelo_svm.png"
            scatter_title = "Regresion SVM: reales vs predichos"
            residual_title = "Residuos - Modelo SVM"

        scatter_tmp = tmp_dir / scatter_name
        residual_tmp = tmp_dir / residual_name
        panel_tmp = tmp_dir / panel_name
        save_scatter_plot(y_test, predictions, scatter_title, scatter_tmp)
        save_residual_plot(y_test, predictions, residual_title, residual_tmp)
        save_panel_plot(y_test, predictions, scatter_title, residual_title, panel_tmp)
        export_to_targets(scatter_name, scatter_tmp)
        export_to_targets(residual_name, residual_tmp)
        export_to_targets(panel_name, panel_tmp)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(project_path("knn_svm_metrics_17_variables.csv"), index=False)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
