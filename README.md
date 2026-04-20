# Airbnb CDMX — Price Predictor

> Dashboard interactivo de Machine Learning para predicción de precios de Airbnb en Ciudad de México.

## Stack

- **Backend**: Python · Flask · Scikit-Learn
- **Frontend**: HTML5 · Vanilla CSS · Vanilla JS
- **Visualizaciones**: Plotly.js (interactivo) · Leaflet.js (mapa)
- **Datos**: [Inside Airbnb — Mexico City](https://insideairbnb.com/mexico-city/)

## Features

- 📊 6 gráficas interactivas (correlación, boxplot, scatter, coeficientes)
- 🗺️ Mapa interactivo con 3,000 listings coloreados por precio
- 🏙️ Ranking de las 16 alcaldías de CDMX por precio mediano
- 🧮 Calculadora de precio en tiempo real (ML)
- 📈 Comparativa de 3 modelos: Lineal, Polinómica G2 y G3

## Setup local

```bash
pip install -r requirements.txt
python modelo.py   # Entrena el modelo y genera archivos
python app.py      # Levanta el servidor en http://localhost:5000
```

## Modelo

| Modelo | RMSE (test) | R² (test) |
|--------|------------|-----------|
| Regresión Lineal Múltiple | 376.4 | **0.629** |
| Polinómica Grado 2 | 457.1 | 0.453 |
| Polinómica Grado 3 | 553.2 | 0.199 |

> El mejor modelo es la Regresión Lineal Múltiple con R² = 0.63.
