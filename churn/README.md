# B. Abandono de clientes

Este caso estudia el riesgo asignado a un usuario a partir de sus sesiones y pedidos. La base proporciona una etiqueta por usuario y carece de marcas de tiempo. Las respuestas distinguen clasificación de riesgo y predicción de un cambio futuro.

## Ejecución

Desde la raíz del repositorio, con Python 3.11 y las dependencias de la práctica instaladas:

```powershell
py -3.11 churn/analisis.py --datos "C:\ruta\a\las\bases"
```

El directorio de entrada debe contener `ecommerce_churn_ltv.db`. Por omisión se busca en Descargas del usuario. El programa emplea un hilo de cálculo y usa disco temporal para SQL. La fuente se abre en modo de solo lectura. No se necesitan bases de datos ni registros individuales dentro del repositorio.

- `etl.py` retira duplicados completos, rechaza claves en conflicto y comprueba las relaciones. Normaliza los textos categóricos, detecta centinelas, calcula agregados de navegación y compra por separado y construye diez variables por usuario.
- `analisis.py` crea particiones estratificadas disjuntas, aplica ruido artificial a una copia de entrenamiento, aprende imputación y escalas allí y ajusta la regresión logística, XGBoost y la red neuronal. Compara además los dos primeros modelos en los primeros cinco componentes principales.
- `informe.py` dibuja las figuras y escribe los textos de la memoria a partir de `resultados/metricas.json`. Para regenerar solo esa parte, ejecutar `py -3.11 churn/informe.py`.
- `resultados/metricas.json` contiene la huella de la fuente, auditoría, cuantiles, prevalencias, soportes por partición, recuentos de ruido, coeficientes del PCA, puntuaciones y métricas por clase, historial de pérdida y tasas por canal. No contiene registros individuales.
- `informe.json` incluye las tres respuestas y los pies de las figuras utilizadas en la memoria. Las cinco gráficas PNG están en `resultados/`.

## Diseño del experimento

La caracterización usa los 800 000 usuarios distintos. Para acotar el cómputo y comparar alternativas en igualdad de condiciones, se seleccionaron 220 000 por muestreo estratificado con semilla fija. Se reservaron 80 000 para ajuste, 20 000 para selección y 120 000 para evaluación. Se excluyeron las claves, la etiqueta, el canal y los datos de perfil de los diez predictores de conducta. El canal solo se utiliza en el análisis de errores.

Las medias y tasas desconocidas se imputan con medianas aprendidas en entrenamiento. La proporción móvil se calcula solo donde se conoce el sistema operativo. Las tasas de devolución y rebote conservan las observaciones con cero eventos de devolución o rebote cuando existe denominador. El número de pedidos y el de sesiones permiten distinguir inactividad. La prueba conserva sus ausencias naturales y se transforma con los parámetros de entrenamiento.

Para B1 se fija en entrenamiento la dirección de cada marcador y se mide su AUC ROC de forma aislada entre las clases 1 y 2 de prueba. Para B2 se vuelven a entrenar los modelos con cinco componentes manteniendo las mismas particiones y el mismo protocolo de parada. Para B3 se elige por macro F1 de validación entre los tres modelos completos y se emplean intervalos binomiales de Wilson. No se optimizan reglas de canal ni umbrales con usuarios de prueba.

Con la opción `--cache` se puede guardar una caché local para acelerar repeticiones. Esta contiene valores agregados por usuario y debe residir fuera del repositorio. El script verifica que la huella de la base coincide antes de reutilizarla. La opción es innecesaria para reproducir los resultados partiendo del SQLite.
