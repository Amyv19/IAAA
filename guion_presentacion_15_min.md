# Guion para presentación de máximo 15 minutos

Duración sugerida: 12 a 15 minutos.

## Diapositiva 1 - Contexto y objetivo (0:00 a 2:00)

Presentar el problema: estimar el precio por noche de alojamientos Airbnb en CDMX usando datos históricos de listings.

Frase sugerida: "El objetivo fue construir un modelo que, con características de un alojamiento, estime un precio aproximado por noche. No buscamos solo entrenar un modelo, sino también integrarlo a una app para que se pueda usar."

## Diapositiva 2 - Datos y limpieza (2:00 a 5:00)

Mencionar que el dataset original tiene 27,051 registros y 79 columnas.

Explicar que se quitaron columnas como URLs, IDs y textos largos porque no eran útiles para este modelo. Luego explicar que `price` se convirtió de texto a número, las tasas con `%` se volvieron números y los booleanos `t/f` se volvieron `1/0`.

Cerrar con el filtro IQR: "También retiramos precios extremos para que el modelo aprendiera el comportamiento normal del mercado y no casos muy raros."

## Diapositiva 3 - Pipeline y comparación de modelos (5:00 a 9:00)

Explicar el preprocesamiento con palabras simples.

Imputación significa rellenar datos faltantes.

Escalado significa poner variables numéricas en escalas comparables.

OneHotEncoder significa convertir categorías de texto en columnas numéricas de 0 y 1.

Comparar los tres modelos entrenados. Resaltar que la regresión lineal múltiple obtuvo el mejor resultado: R2 test de 0.629 y RMSE de 376.42 MXN.

## Diapositiva 4 - Interpretación (9:00 a 12:00)

Explicar que las variables de capacidad del alojamiento, tipo de propiedad, tipo de cuarto y ubicación tienen alta relevancia.

Frase sugerida: "El modelo no solo da un número; también permite revisar qué variables empujan el precio hacia arriba o hacia abajo mediante los coeficientes."

Mencionar alcaldías con precios medianos más altos: Miguel Hidalgo, Cuajimalpa y Cuauhtémoc. Aclarar que los outliers premium no son el foco del modelo.

## Diapositiva 5 - Producto final y cierre (12:00 a 15:00)

Mostrar que el modelo se integró a un dashboard Flask con mapa, gráficas y calculadora de precio.

Cerrar con la idea principal: "El resultado más importante es que el modelo más simple fue el más estable, interpretable y fácil de desplegar."

Proponer mejoras futuras: amenities estructurados, validación temporal y modelos regularizados o ensembles.
