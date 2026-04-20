# ============================================================
# PROYECTO: PREPROCESAMIENTO + MODELADO + EVALUACIÓN
# Dataset: listings.csv (Inside Airbnb – CDMX completo)
# Variable objetivo: price
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib

os.makedirs("static/plots", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. CARGA DE DATOS
# =========================
df = pd.read_csv("listings.csv", low_memory=False)

cols_to_drop = [
    'id','listing_url','scrape_id','last_scraped','source',
    'name','description','neighborhood_overview','picture_url',
    'host_id','host_url','host_name','host_since','host_location',
    'host_about','host_thumbnail_url','host_picture_url',
    'host_neighbourhood','host_verifications','host_has_profile_pic',
    'host_identity_verified','neighbourhood','neighbourhood_group_cleansed',
    'bathrooms_text','calendar_updated','last_review','license',
    'calculated_host_listings_count_entire_homes',
    'calculated_host_listings_count_private_rooms',
    'calculated_host_listings_count_shared_rooms',
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# Limpiar precio
target = "price"
df[target] = df[target].astype(str).str.replace(r'[\$,]', '', regex=True)
df[target] = pd.to_numeric(df[target], errors='coerce')

# Porcentajes → float
for col in ['host_response_rate', 'host_acceptance_rate']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('%','').str.strip(), errors='coerce')

# Booleanos
for col in ['host_is_superhost','has_availability','instant_bookable']:
    if col in df.columns:
        df[col] = df[col].map({'t':1,'f':0,True:1,False:0}).astype(float)

# =========================
# 2. LIMPIEZA Y OUTLIERS
# =========================
df.dropna(subset=[target], inplace=True)
df = df[df[target] > 0]

Q1, Q3 = df[target].quantile(0.25), df[target].quantile(0.75)
IQR = Q3 - Q1
df_limpio = df[(df[target] >= Q1-1.5*IQR) & (df[target] <= Q3+1.5*IQR)].copy()
print(f"Filas: {len(df)} -> {len(df_limpio)} (sin outliers)")

# =========================
# 3. FEATURES
# =========================
useful_cat = ['host_response_time','neighbourhood_cleansed','property_type','room_type']
X = df_limpio.drop(columns=[target])
# Quitar columnas de texto libre
cat_valid = [c for c in useful_cat if c in X.columns]
num_valid  = X.select_dtypes(include=['int64','float64']).columns.tolist()
X = X[num_valid + cat_valid]

numeric_features     = num_valid
categorical_features = cat_valid
print(f"Features numéricas: {len(numeric_features)}, categóricas: {len(categorical_features)}")

y = df_limpio[target]

# =========================
# 4. TOP-10 NUMÉRICAS (para modelos polinómicos — evitar explosión de memoria)
# =========================
corr = df_limpio[numeric_features + [target]].corr()[target].drop(target).abs()
top10_features = corr.nlargest(10).index.tolist()
print("Top 10 features por correlación:", top10_features)

# =========================
# 5. PREPROCESAMIENTO
# =========================
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_transformer, numeric_features),
    ("cat", cat_transformer, categorical_features),
])

def evaluar(nombre, modelo, Xtr, Xte, ytr, yte):
    ytr_p = modelo.predict(Xtr)
    yte_p = modelo.predict(Xte)
    return {
        "Modelo": nombre,
        "MAE_train":  round(mean_absolute_error(ytr, ytr_p),2),
        "RMSE_train": round(np.sqrt(mean_squared_error(ytr, ytr_p)),2),
        "R2_train":   round(r2_score(ytr, ytr_p),4),
        "MAE_test":   round(mean_absolute_error(yte, yte_p),2),
        "RMSE_test":  round(np.sqrt(mean_squared_error(yte, yte_p)),2),
        "R2_test":    round(r2_score(yte, yte_p),4),
    }, yte_p

# =========================
# 6. SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

Xn = df_limpio[top10_features].copy()
yn = df_limpio[target]
Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.20, random_state=42)

# =========================
# 7. MODELOS
# =========================
print("Entrenando modelo lineal...")
m_lineal = Pipeline([("preprocessor", preprocessor), ("reg", LinearRegression())])
m_lineal.fit(X_train, y_train)
res_lin, pred_lin = evaluar("Regresión lineal múltiple", m_lineal, X_train, X_test, y_train, y_test)

print("Entrenando polinómica grado 2...")
m_poly2 = Pipeline([
    ("imp",   SimpleImputer(strategy="median")),
    ("scl",   StandardScaler()),
    ("poly",  PolynomialFeatures(degree=2, include_bias=False)),
    ("reg",   LinearRegression()),
])
m_poly2.fit(Xn_train, yn_train)
res_p2, pred_p2 = evaluar("Regresión polinómica grado 2", m_poly2, Xn_train, Xn_test, yn_train, yn_test)

print("Entrenando polinómica grado 3 (top-10 features)...")
m_poly3 = Pipeline([
    ("imp",   SimpleImputer(strategy="median")),
    ("scl",   StandardScaler()),
    ("poly",  PolynomialFeatures(degree=3, include_bias=False)),
    ("reg",   LinearRegression()),
])
m_poly3.fit(Xn_train, yn_train)
res_p3, pred_p3 = evaluar("Regresión polinómica grado 3", m_poly3, Xn_train, Xn_test, yn_train, yn_test)

# =========================
# 8. RESULTADOS
# =========================
resultados = pd.DataFrame([res_lin, res_p2, res_p3])
print("\nTabla comparativa:")
print(resultados.to_string(index=False))

# =========================
# 9. COEFICIENTES (modelo lineal)
# =========================
ohe = m_lineal.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
feat_names = [(f[:40]+'...') if len(f)>40 else f for f in numeric_features + cat_names]
coefs = np.ravel(m_lineal.named_steps["reg"].coef_)
min_len = min(len(feat_names), len(coefs))
coef_df = pd.DataFrame({
    "Variable": feat_names[:min_len],
    "Coeficiente": coefs[:min_len]
}).sort_values("Coeficiente", ascending=False)

# =========================
# 10. GUARDAR
# =========================
resultados.to_csv("resultados_modelos.csv", index=False)
coef_df.to_csv("coeficientes_modelo_lineal.csv", index=False)
joblib.dump(m_lineal, "modelo_lineal.pkl")

# Template y datos para gráficas en Flask
X_train.iloc[[0]].to_csv("template_prediccion.csv", index=False)

# Guardar datos de scatter para gráficas interactivas
scatter_lin = pd.DataFrame({"real": y_test.values, "pred": pred_lin})
scatter_lin.sample(min(2000, len(scatter_lin)), random_state=42).to_csv("scatter_lineal.csv", index=False)
scatter_p2 = pd.DataFrame({"real": yn_test.values, "pred": pred_p2})
scatter_p2.sample(min(2000, len(scatter_p2)), random_state=42).to_csv("scatter_poly2.csv", index=False)

# Datos de correlación numérica limpios
corr_export = df_limpio[numeric_features + [target]].corr()[target].drop(target).sort_values(ascending=False)
corr_export.reset_index().rename(columns={"index":"feature", target:"corr"}).to_csv("correlacion.csv", index=False)

# Datos de precios por delegacion
price_by_deleg = df_limpio.groupby("neighbourhood_cleansed")[target].median().sort_values(ascending=False).reset_index()
price_by_deleg.columns = ["neighbourhood","median_price"]
price_by_deleg.to_csv("price_by_neighbourhood.csv", index=False)

# Precios para boxplot (muestra)
df_limpio[[target]].sample(min(3000, len(df_limpio)), random_state=42).to_csv("price_sample.csv", index=False)

print("\nTodos los archivos guardados con éxito.")
