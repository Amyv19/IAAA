# Airbnb CDMX — Price Predictor

> Dashboard interactivo de Machine Learning para predicción de precios de Airbnb en Ciudad de México.

## Stack

- **Backend**: Python · Flask · Scikit-Learn
- **Frontend**: HTML5 · Vanilla CSS · Vanilla JS
- **Visualizaciones**: Plotly.js (interactivo) · Leaflet.js (mapa)
- **Datos**: [Inside Airbnb — Mexico City](https://insideairbnb.com/mexico-city/)


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
