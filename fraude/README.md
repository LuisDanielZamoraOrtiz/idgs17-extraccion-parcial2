# Caso A · Fraude financiero

Este módulo parte de `fintech_fraud.db` y reproduce la extracción, preparación, entrenamiento y evaluación del caso de fraude. Las salidas publicables contienen conteos, puntuaciones y curvas agregadas.

## Ejecución

Desde la raíz del repositorio, con Python 3.11 y las dependencias del proyecto ya instaladas:

```powershell
py -3.11 fraude/entrenar.py --datos "C:/datos/parcial2"
```

El directorio indicado debe contener la base SQLite. Se utiliza una hebra de CPU por operación. Por defecto se seleccionan aproximadamente 50 de cada 1 000 claves y se realizan 80 épocas en cada ajuste neuronal. El entrenamiento es supervisado con las etiquetas que ya contiene la base. No requiere GPU.

## Entradas y decisiones

- Se consulta el total de operaciones, el número de claves y la calidad del monto en toda la tabla. El análisis posterior toma claves que cumplen `((tx_id * 48271) % 2147483647) % 1000 < 50`. La fórmula no utiliza información de fraude, fecha o comercio. Es un submuestreo determinista y no constituye una muestra aleatoria certificada de toda la población.
- Se normalizan textos, se retiran repeticiones y se excluye una clave si quedan versiones contradictorias. Las llaves de cliente sirven para verificar correspondencia y las de comercio sirven para unir MCC y país. Ninguna clave se utiliza como predictor. Las columnas de catálogo sin fecha de vigencia requieren una hipótesis de disponibilidad al autorizar el cargo, que se declara en el reporte.
- Se conservan operaciones con monto vacío, negativo o con el pico repetido 999999,9. Para el ajuste se considera desconocido, se añade un indicador y se usa la mediana de entrenamiento por moneda. Las categorías ausentes tienen un nivel propio. Se transforman los importes mediante `log1p`; se ajustan escala y codificación únicamente con enero–agosto.
- Septiembre–octubre se usa para seleccionar el modelo por precisión promedio (AP), las mejores rondas o épocas y el umbral que maximiza la sensibilidad bajo 1,5 % de falsos positivos. El menor FPR resuelve empates en sensibilidad. Si persiste la igualdad se escoge el umbral mayor. Noviembre–diciembre queda reservado para la evaluación final.
- XGBoost de profundidad 4 se contrasta con árboles de una sola división, que producen una suma aditiva. Los perfiles de monto y comercio para las interacciones se fijan en entrenamiento. Las comparaciones de tasas en prueba son descriptivas.
- A2 utiliza las probabilidades de todas las operaciones de prueba. PR-AUC integra la curva precisión–sensibilidad por trapecios a todos sus umbrales distintos. AP usa la suma escalonada de precisión ponderada por el cambio de sensibilidad. Solo la figura se reduce a 251 puntos por modelo.
- El intervalo percentil del 95 % de cada diferencia de AP frente a logística utiliza 1 000 remuestreos pareados de bloques móviles circulares de siete días. En cada réplica se sortean inicios de bloque con reemplazo, se conservan 61 días en total y se usan las mismas operaciones y multiplicidades para todos los modelos. Las AP se recalculan con todos los umbrales, agrupando las puntuaciones empatadas. Los intervalos reflejan la variabilidad de ese periodo con los modelos fijos.
- Se entrenan dos redes idénticas con Adam y retropropagación. Una aprende las etiquetas originales y otra aprende una copia con un 1,5 % adicional de etiquetas invertidas solo en entrenamiento. Se selecciona la época de menor pérdida de validación y se busca la primera divergencia sostenida durante tres épocas. El criterio y su margen de 0,0005 están implementados en `red`. Se desconoce qué etiquetas se habían invertido al crear la base.

## Archivos

- `datos.py` implementa la auditoría, la selección de operaciones y la transformación.
- `entrenar.py` ajusta logística, XGBoost, red neuronal y controles, y conserva resultados agregados.
- `informe.py` construye las cuatro figuras y el texto del caso. Si ya existen los resultados, `py -3.11 fraude/informe.py` permite regenerarlos sin reentrenar.
- `resultados/metricas.json` documenta la huella SHA-256 de la base, conteos, variables, métricas, matriz de conteos a cada umbral, distribución y parámetros.
- `resultados/epocas_red.csv` contiene pérdidas por época. `resultados/frontera_validacion.csv` contiene la máxima sensibilidad observada bajo distintos presupuestos de falsos positivos.
- `informe.json` contiene las respuestas y rutas de las figuras utilizadas en el PDF.
