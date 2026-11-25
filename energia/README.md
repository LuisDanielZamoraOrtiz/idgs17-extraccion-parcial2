# C. Consumo y estabilidad de la red eléctrica

El caso estima la energía media por lectura de una subestación durante la hora siguiente y evalúa una alerta de consumo alto. La unidad del objetivo es kWh por lectura. La capacidad instalada está en MW, el voltaje en V, la corriente en A y la temperatura en °C. La base carece de tarifas y precios del kWh, por lo que el análisis expresa la demanda en unidades físicas.

## Ejecutar

Con las dependencias del proyecto instaladas, desde la raíz del repositorio:

```powershell
py -3.11 energia/energia.py --datos "C:\datos\parcial2"
```

La carpeta indicada debe contener `smart_grid_energy.db`. `preparacion.py` depura lecturas, clima y subestaciones, construye agregados y enlaza horas consecutivas. `energia.py` ajusta los modelos, mide su rendimiento y genera cinco figuras y `resultados/metricas.json`. `texto_energia.py` convierte estos resultados en el texto de `informe.json` que utiliza la memoria.

## Interpretar las clases y los resultados

Los umbrales provienen de las horas de entrenamiento de cada subestación. El consumo nominal (clase 0) queda por debajo de su percentil 70. El elevado (1) va del percentil 70 hasta antes del 90. El crítico relativo (2) comienza en el percentil 90. Las tres clases miden la posición respecto del historial propio. Verificar una sobrecarga física exige mediciones operativas y cobertura de medidores adicionales.

- **Pronóstico a una hora.** En los 261 068 pares de prueba, Ridge obtuvo un error absoluto medio de 29,02 kWh por lectura. Repetir el último valor dio 40,04.
- **Alertas industriales con más de 30 °C.** Con el umbral de probabilidad 0,1414 elegido en validación, la red detectó 2 321 de 2 637 horas críticas y emitió 11 101 avisos. Esto supone 88,0 % de sensibilidad y 20,9 % de precisión.
- **Sensores y antigüedad.** El componente que más distinguió consumo nominal y elevado estuvo relacionado sobre todo con temperatura y corriente. Las diferencias de error y de frecuencia crítica entre estaciones instaladas antes de 2005 y las posteriores fueron pequeñas; los intervalos obtenidos al remuestrear estaciones incluyeron cero.

Los tamaños de las muestras, la comparación de modelos, las referencias de persistencia y los intervalos por cohorte pueden consultarse en [las métricas agregadas](resultados/metricas.json).
