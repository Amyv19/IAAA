import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

os.makedirs("static/plots", exist_ok=True)

TARGET = "price"
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CANDIDATES = [
    os.getenv("COIL_DATA_PATH"),
    r"C:\Users\AmyVa\Downloads\listings.csv.gz",
    str(ROOT / "listings.csv.gz"),
    str(ROOT / "listings.csv"),
]


def load_dataset(path):
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, low_memory=False, compression=compression)


def clean_price_column(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def resolve_training_dataset():
    attempted = []
    for candidate in DEFAULT_DATA_CANDIDATES:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        df_candidate = load_dataset(path)
        if TARGET not in df_candidate.columns:
            attempted.append(f"{path} (sin columna {TARGET})")
            continue
        valid_prices = clean_price_column(df_candidate[TARGET]).dropna()
        valid_prices = valid_prices[valid_prices > 0]
        if len(valid_prices) > 0:
            return path, df_candidate
        attempted.append(f"{path} (sin precios validos)")
    raise ValueError(
        "Ningun dataset disponible contiene precios validos. Revisados: "
        + "; ".join(attempted)
    )


def evaluar(nombre, modelo, Xtr, Xte, ytr, yte):
    ytr_p = modelo.predict(Xtr)
    yte_p = modelo.predict(Xte)
    return {
        "Modelo": nombre,
        "MAE_train": round(mean_absolute_error(ytr, ytr_p), 2),
        "RMSE_train": round(np.sqrt(mean_squared_error(ytr, ytr_p)), 2),
        "R2_train": round(r2_score(ytr, ytr_p), 4),
        "MAE_test": round(mean_absolute_error(yte, yte_p), 2),
        "RMSE_test": round(np.sqrt(mean_squared_error(yte, yte_p)), 2),
        "R2_test": round(r2_score(yte, yte_p), 4),
    }, yte_p


data_path, df = resolve_training_dataset()
print(f"Usando dataset: {data_path}")

cols_to_drop = [
    "id",
    "listing_url",
    "scrape_id",
    "last_scraped",
    "source",
    "name",
    "description",
    "neighborhood_overview",
    "picture_url",
    "host_id",
    "host_url",
    "host_profile_id",
    "host_profile_url",
    "host_name",
    "host_since",
    "host_location",
    "host_about",
    "host_thumbnail_url",
    "host_picture_url",
    "host_neighbourhood",
    "host_verifications",
    "host_has_profile_pic",
    "host_identity_verified",
    "neighbourhood",
    "neighbourhood_group_cleansed",
    "bathrooms_text",
    "calendar_updated",
    "last_review",
    "license",
    "amenities",
    "calendar_last_scraped",
    "first_review",
    "bathrooms",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

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

df.dropna(subset=[TARGET], inplace=True)
df = df[df[TARGET] > 0]

q1, q3 = df[TARGET].quantile(0.25), df[TARGET].quantile(0.75)
iqr = q3 - q1
df_limpio = df[(df[TARGET] >= q1 - 1.5 * iqr) & (df[TARGET] <= q3 + 1.5 * iqr)].copy()
print(f"Filas: {len(df)} -> {len(df_limpio)} (sin outliers)")

useful_cat = ["host_response_time", "neighbourhood_cleansed", "property_type", "room_type"]
X = df_limpio.drop(columns=[TARGET])
cat_valid = [c for c in useful_cat if c in X.columns]
num_valid = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X = X[num_valid + cat_valid]
y = df_limpio[TARGET]

numeric_features = num_valid
categorical_features = cat_valid
print(f"Features numericas: {len(numeric_features)}, categoricas: {len(categorical_features)}")

corr = df_limpio[numeric_features + [TARGET]].corr()[TARGET].drop(TARGET).abs()
top10_features = corr.nlargest(10).index.tolist()
print("Top 10 features por correlacion:", top10_features)

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
preprocessor = ColumnTransformer(
    [
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

Xn = df_limpio[top10_features].copy()
yn = df_limpio[TARGET]
Xn_train, Xn_test, yn_train, yn_test = train_test_split(
    Xn, yn, test_size=0.20, random_state=42
)

X_train_tree = X_train.sample(min(10000, len(X_train)), random_state=42)
y_train_tree = y_train.loc[X_train_tree.index]
Xn_train_poly = Xn_train.sample(min(8000, len(Xn_train)), random_state=42)
yn_train_poly = yn_train.loc[Xn_train_poly.index]

resultados = []

modelos_generales = [
    (
        "Regresion lineal multiple",
        Pipeline([("preprocessor", preprocessor), ("reg", LinearRegression())]),
    ),
    (
        "Ridge",
        Pipeline([("preprocessor", preprocessor), ("reg", Ridge(alpha=1.0))]),
    ),
    (
        "Elastic Net",
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("reg", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000)),
            ]
        ),
    ),
    (
        "Random Forest",
        Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "reg",
                    RandomForestRegressor(
                        n_estimators=60,
                        max_depth=12,
                        min_samples_leaf=4,
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    ),
    (
        "Gradient Boosting",
        Pipeline(
            [
                ("preprocessor", preprocessor),
                ("reg", GradientBoostingRegressor(random_state=42)),
            ]
        ),
    ),
]

artifacts = {}
for nombre, modelo in modelos_generales:
    print(f"Entrenando {nombre}...")
    if nombre == "Random Forest":
        modelo.fit(X_train_tree, y_train_tree)
    else:
        modelo.fit(X_train, y_train)
    res, pred = evaluar(nombre, modelo, X_train, X_test, y_train, y_test)
    resultados.append(res)
    artifacts[nombre] = {"model": modelo, "pred": pred}

modelos_polinomicos = [
    (
        "Regresion polinomica grado 2",
        Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("reg", LinearRegression()),
            ]
        ),
    ),
    (
        "Regresion polinomica grado 3",
        Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("reg", LinearRegression()),
            ]
        ),
    ),
]

for nombre, modelo in modelos_polinomicos:
    print(f"Entrenando {nombre}...")
    modelo.fit(Xn_train_poly, yn_train_poly)
    res, pred = evaluar(nombre, modelo, Xn_train, Xn_test, yn_train, yn_test)
    resultados.append(res)
    artifacts[nombre] = {"model": modelo, "pred": pred}

resultados_df = pd.DataFrame(resultados).sort_values(
    ["RMSE_test", "MAE_test"], ascending=[True, True]
)
print("\nTabla comparativa:")
print(resultados_df.to_string(index=False))

m_lineal = artifacts["Regresion lineal multiple"]["model"]
ohe = m_lineal.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
feat_names = [(f[:40] + "...") if len(f) > 40 else f for f in numeric_features + cat_names]
coefs = np.ravel(m_lineal.named_steps["reg"].coef_)
min_len = min(len(feat_names), len(coefs))
coef_df = pd.DataFrame(
    {
        "Variable": feat_names[:min_len],
        "Coeficiente": coefs[:min_len],
    }
).sort_values("Coeficiente", ascending=False)

best_model_name = resultados_df.iloc[0]["Modelo"]
best_model = artifacts[best_model_name]["model"]

resultados_df.to_csv("resultados_modelos.csv", index=False)
coef_df.to_csv("coeficientes_modelo_lineal.csv", index=False)
joblib.dump(m_lineal, "modelo_lineal.pkl")
joblib.dump(best_model, "modelo_principal.pkl")

metadata = pd.DataFrame(
    [{"dataset_path": str(data_path), "best_model": best_model_name, "rows_clean": len(df_limpio)}]
)
metadata.to_csv("metadata_modelo.csv", index=False)

X_train.iloc[[0]].to_csv("template_prediccion.csv", index=False)

scatter_lin = pd.DataFrame(
    {"real": y_test.values, "pred": artifacts["Regresion lineal multiple"]["pred"]}
)
scatter_lin.sample(min(2000, len(scatter_lin)), random_state=42).to_csv(
    "scatter_lineal.csv", index=False
)
scatter_p2 = pd.DataFrame(
    {"real": yn_test.values, "pred": artifacts["Regresion polinomica grado 2"]["pred"]}
)
scatter_p2.sample(min(2000, len(scatter_p2)), random_state=42).to_csv(
    "scatter_poly2.csv", index=False
)

corr_export = (
    df_limpio[numeric_features + [TARGET]]
    .corr()[TARGET]
    .drop(TARGET)
    .sort_values(ascending=False)
)
corr_export.reset_index().rename(columns={"index": "feature", TARGET: "corr"}).to_csv(
    "correlacion.csv", index=False
)

price_by_deleg = (
    df_limpio.groupby("neighbourhood_cleansed")[TARGET]
    .median()
    .sort_values(ascending=False)
    .reset_index()
)
price_by_deleg.columns = ["neighbourhood", "median_price"]
price_by_deleg.to_csv("price_by_neighbourhood.csv", index=False)

df_limpio[[TARGET]].sample(min(3000, len(df_limpio)), random_state=42).to_csv(
    "price_sample.csv", index=False
)

print("\nArchivos exportados correctamente.")
