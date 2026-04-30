# Airbnb CDMX - Price Predictor

Dashboard interactivo de Machine Learning para predicción de precios de Airbnb en Ciudad de México.

## Stack

- **Backend**: Python, Flask, Scikit-Learn
- **Frontend**: HTML5, CSS y JavaScript
- **Visualizaciones**: Plotly.js y Leaflet.js
- **Datos**: [Inside Airbnb - Mexico City](https://insideairbnb.com/mexico-city/)

## Funcionalidades

- 6 gráficas interactivas: correlación, boxplot, scatter y coeficientes.
- Mapa interactivo con 3,000 listings coloreados por precio.
- Ranking de las 16 alcaldías de CDMX por precio mediano.
- Calculadora de precio en tiempo real con Machine Learning.
- Comparativa de 3 modelos: lineal, polinómica grado 2 y polinómica grado 3.

## Flujo del modelo

```mermaid
flowchart TD
    A["Inicio: listings.csv"] --> B["Carga de datos con pandas"]
    B --> C["Limpieza inicial"]
    C --> C1["Eliminar URLs, IDs, texto libre y columnas administrativas"]
    C --> C2["Convertir price a número"]
    C --> C3["Convertir porcentajes y booleanos"]
    C1 --> D["Filtrar precios válidos"]
    C2 --> D
    C3 --> D
    D --> E["Eliminar outliers con regla IQR"]
    E --> F["Seleccionar variables predictoras"]
    F --> F1["Variables numéricas"]
    F --> F2["Variables categóricas"]
    F1 --> G["Pipeline numérico: mediana + escalado"]
    F2 --> H["Pipeline categórico: moda + OneHotEncoder"]
    G --> I["ColumnTransformer"]
    H --> I
    I --> J["Separación train/test 80/20"]
    J --> K1["Regresión lineal múltiple"]
    J --> K2["Regresión polinómica grado 2"]
    J --> K3["Regresión polinómica grado 3"]
    K1 --> L["Evaluar MAE, RMSE y R²"]
    K2 --> L
    K3 --> L
    L --> M["Seleccionar mejor modelo"]
    M --> N["Guardar modelo_lineal.pkl"]
    N --> O["Integrar en Flask"]
    O --> P["Dashboard y calculadora de precio"]
```

## Setup local

```bash
pip install -r requirements.txt
python modelo.py
python app.py
```

La app se levanta en `http://localhost:5000`.

## Modelo

| Modelo | RMSE test | R² test |
|--------|-----------|---------|
| Regresión lineal múltiple | 376.4 | **0.629** |
| Polinómica grado 2 | 457.1 | 0.453 |
| Polinómica grado 3 | 553.2 | 0.199 |

El mejor modelo es la regresión lineal múltiple con `R² = 0.63`.
