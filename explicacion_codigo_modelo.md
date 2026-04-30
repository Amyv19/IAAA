# Explicación del código del modelo

Este documento explica el código en lenguaje sencillo para que puedas entenderlo y exponerlo.

## 1. Problema

Queremos predecir `price`, es decir, el precio por noche de un alojamiento Airbnb en CDMX.

`X` son las características que usamos para predecir. Por ejemplo: alcaldía, tipo de cuarto, tipo de propiedad, número de personas, camas, baños y disponibilidad.

`y` es la respuesta correcta que el modelo aprende: el precio real.

## 2. Carga de datos

```python
df = pd.read_csv("listings.csv", low_memory=False)
```

Lee el CSV y lo convierte en una tabla de pandas. Cada fila es un alojamiento y cada columna es una característica.

## 3. Limpieza

Se eliminan URLs, IDs, textos largos y columnas administrativas porque no son útiles para este modelo.

`price` se limpia porque puede venir como `$1,200`. El código quita `$` y comas, y después convierte el valor a número.

Los porcentajes como `95%` se convierten a `95`.

Los booleanos `t` y `f` se convierten a `1` y `0`.

## 4. Outliers

Se eliminan precios nulos, cero o negativos.

Después se usa IQR para quitar precios extremos. Esto evita que alojamientos demasiado caros o raros deformen el modelo.

## 5. Variables

Las numéricas se usan directamente después de imputar y escalar.

Las categóricas, como `room_type` o `neighbourhood_cleansed`, se convierten con OneHotEncoder a columnas de 0 y 1.

## 6. Pipeline

```python
m_lineal = Pipeline([
    ("preprocessor", preprocessor),
    ("reg", LinearRegression()),
])
```

Primero limpia y transforma los datos. Luego entrena la regresión lineal. La ventaja es que todo queda guardado junto en `modelo_lineal.pkl`.

## 7. Entrenamiento y prueba

El 80% de los datos se usa para entrenar y el 20% para probar.

Esto permite saber si el modelo funciona con datos que no había visto.

## 8. Modelos comparados

Se probaron regresión lineal múltiple, polinómica grado 2 y polinómica grado 3.

La regresión lineal múltiple fue la mejor porque obtuvo mayor `R2_test` y menor error en prueba.

## 9. Resultado

El modelo final tuvo:

```text
MAE_test = 268.00
RMSE_test = 376.42
R2_test = 0.6293
```

En palabras simples: el modelo explica aproximadamente 63% de la variación del precio y se equivoca en promedio alrededor de 268 MXN según MAE.

## 10. Conexión con Flask

La app carga `modelo_lineal.pkl`, toma los datos del formulario, arma una fila con `template_prediccion.csv` y llama:

```python
modelo.predict(df_input)
```

Eso devuelve el precio estimado.

## Explicación corta para decir en clase

Limpiamos los datos, quitamos valores extremos, separamos variables numéricas y categóricas, construimos un pipeline reproducible, comparamos tres modelos y elegimos la regresión lineal múltiple porque fue la más estable y explicable. Después guardamos el modelo y lo conectamos a una app Flask.
