# Citridata · Dashboard de rendimiento

Requiere Python 3.10 o superior. Descomprime el ZIP, abre una terminal dentro de la carpeta y ejecuta:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La aplicación abre una dirección local en el navegador. Incluye el JSON de esta conversación; puedes sustituirlo desde el cargador lateral.

## Qué incluye

Filtros por campaña, finca, variedad y referencia de parcela. Evolución de rendimiento observado y predicho con desviación típica muestral; dispersión y residuos; ETo, precipitación y déficit ETo menos precipitación; GDD, días >35 °C y heladas con eje independiente; seis índices de vegetación y correlaciones; simulador con contribuciones; tabla exportable a CSV.

## Decisiones del modelo

Coeficientes fijos del modelo original, déficit ETo − precipitación con coeficiente −0.0070 y parcelArea. Los índices de vegetación se exploran, pero no intervienen en la ecuación.

Por defecto se utilizan previousSeasonVarietyYield, twoSeasonsAgoVarietyYield e historicalMeanVarietyYield del JSON, que son históricos de variedad, y se excluyen predicciones incompletas. Se puede elegir reconstruir históricos de parcela usando todos los datos ANTES de filtrar: mismo conjunto de referencias catastrales, años exactos anteriores y media de años previos. Las duplicidades de referencia/año se consideran ambiguas. Esto puede dar resultados distintos al Excel cuando faltan años o hay varias variedades en la misma parcela.

La opción de imputación usa las medianas fijas 35.664381315, 36.165105995 y 37.489863321. Solo sustituye históricos, nunca clima o área. Los ceros son datos válidos. Ninguna opción añade penalizaciones del modelo con NDVI.

Las medias son por registro, sin ponderación por superficie. Para comparar observado y predicho en idénticas filas usa la casilla de pares completos. La desviación típica (ddof=1) no es un intervalo de confianza ni de predicción; no se calcula para n=1.

MAE, RMSE y R² se calculan sobre pares completos seleccionados y son descriptivos, no una validación independiente. R² es 1 − SSE/SST, puede ser negativo y no es la correlación al cuadrado. No se importan cifras de ajuste del texto adjunto ni se entrena Random Forest. Para demostrar capacidad predictiva sería necesario reservar campañas futuras y ajustar el modelo solo con las anteriores.

El simulador permite modificar manualmente predictores sin cambiar el archivo. Sus valores ausentes comienzan en cero y deben revisarse. Las contribuciones no son importancias relativas porque las variables tienen unidades distintas.

Los datos permanecen en el proceso local; el código no envía datos a servicios externos. No publiques el JSON con nombres de productores sin revisar los datos que deseas compartir.
