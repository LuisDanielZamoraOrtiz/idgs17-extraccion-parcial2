# Proyecto Parcial 2 · Extracción de conocimiento en bases de datos

Universidad Tecnológica de Querétaro · IDGS 17. Docente: M.C.C. Emmanuel Martinez Hernandez. Sesión de revisión y ejecución del 24 de noviembre de 2026.

La memoria responde nueve preguntas sobre detección de fraude, abandono de clientes y consumo eléctrico. Contiene el tratamiento de los datos, la comparación de modelos y la interpretación de cada gráfica. El índice y los marcadores del PDF permiten saltar a cada caso. El comando de composición genera la memoria de entrega en el equipo del lector. Cada gráfica también está disponible en la carpeta de resultados de su caso.

## Qué se encontró

- **A. Fraude financiero.** Entre 209 995 operaciones únicas de la muestra, se respetó el orden temporal de 2025 al entrenar, elegir el umbral y evaluar. Ningún modelo consiguió en validación capturar al menos 80 % del fraude con menos de 1,5 % de falsas alarmas. Con el umbral operativo de XGBoost (0,25986), en noviembre y diciembre se detectaron 284 de 2 178 fraudes (13,04 %) y se señalaron 468 de 32 913 operaciones legítimas (1,42 %). La curva precisión–sensibilidad y las pérdidas por época ayudan a explicar ese compromiso. [Método y resultados de fraude](fraude/)
- **B. Abandono de clientes.** Se resumieron tres millones de sesiones y un millón y medio de pedidos para 800 000 usuarios distintos. Entre los cuatro indicadores solicitados, el tiempo medio de envío separó mejor los riesgos medio y alto (AUC 0,726). La red obtuvo el mayor macro F1 de validación y alcanzó 0,8561 en prueba. El canal orgánico tuvo la mayor tasa puntual de riesgos altos omitidos (15,53 %), y Social Ads concentró el mayor número de omisiones (959). [Método y resultados de abandono](churn/)
- **C. Estabilidad de red.** Se estimó el consumo medio de la hora siguiente y se evaluó una alerta de consumo crítico relativo al historial de cada estación. En las horas industriales de prueba con más de 30 °C, la red con el umbral seleccionado en validación detectó 2 321 de 2 637 eventos críticos (88,0 % de sensibilidad). De 11 101 alertas, 2 321 correspondían a un evento crítico (20,9 % de precisión). La fuente carece de una etiqueta verificada de sobrecarga física. [Método y resultados de energía](energia/)

## Reproducir desde un clon

Se probó en Windows de 64 bits con Python 3.11, SQLite incluido en Python y XGBoost para CPU. Los modelos se ejecutan de forma secuencial. El caso de energía lee las fuentes por lotes y utiliza el directorio temporal del sistema.

1. Clona el repositorio y crea el entorno desde PowerShell.

   ```powershell
   git clone https://github.com/LuisDanielZamoraOrtiz/idgs17-extraccion-parcial2.git
   cd idgs17-extraccion-parcial2
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Para revisar los resultados, abre el PDF incluido. Si quieres volver a componerlo con los informes y figuras publicados, ejecuta este comando. Este modo funciona sin acceso a las bases.

   ```powershell
   .\.venv\Scripts\python.exe ejecutar.py --solo-reporte
   ```

3. Para repetir el análisis completo, reúne las tres bases entregadas para la actividad en una carpeta local, por ejemplo `C:\datos\parcial2`. Los nombres esperados son `fintech_fraud.db`, `ecommerce_churn_ltv.db` y `smart_grid_energy.db`. El programa ejecuta fraude, abandono y energía, contrasta el inventario con las métricas y compone la memoria en su ruta publicada.

   ```powershell
   .\.venv\Scripts\python.exe ejecutar.py --datos "C:\datos\parcial2"
   ```

   Sin `--datos`, el programa busca las bases en la carpeta `Downloads` del usuario. La reproducción completa requiere las copias originales. El modo de composición usa los resultados agregados ya versionados. Ambos modos generan el archivo local `memoria/Proyecto_Parcial_2_IDGS17.pdf`. Puedes personalizar la portada añadiendo `--alumno "Nombre del estudiante"` al comando.

## Archivos que sustentan la memoria

| Ruta | Contenido y uso |
| --- | --- |
| [ejecutar.py](ejecutar.py) | Punto de entrada para el flujo completo y la composición del PDF. |
| [requirements.txt](requirements.txt) | Versiones de las bibliotecas empleadas en la ejecución publicada. |
| [fraude/](fraude/) | ETL, entrenamiento, comparación de umbrales, precisión promedio e historial de la red con y sin perturbación adicional de etiquetas. |
| [churn/](churn/) | Agregación por usuario, comparación de cuatro indicadores, cinco componentes principales y auditoría de errores por canal. |
| [energia/](energia/) | Integración de lecturas, zona climática y subestación, pronóstico de la hora siguiente, cargas del PCA y comparación por antigüedad. |
| [metodologia/](metodologia/) | Inventario verificable, diccionario de campos y documentación del contraste con las métricas por caso. |
| [memoria/](memoria/) | Generador y recurso institucional para componer el PDF de entrega. |

Cada caso contiene su `informe.json`, los resultados numéricos en `resultados/metricas.json` y las figuras en `resultados/`. Se pueden consultar tamaños de muestra, denominadores y criterios de entrenamiento directamente en esos archivos. Los SQLite originales, registros individuales, cachés locales y modelos serializados se excluyen de la entrega pública. [Diccionario de variables](metodologia/diccionario.md) · [Inventario y huellas de las fuentes](metodologia/inventario.json)

## Cómo interpretar las comparaciones

- **Particiones.** Fraude y energía reservan un periodo posterior para la prueba. Abandono separa usuarios porque las tablas carecen de fechas; su prueba reúne 120 000 clientes distintos de los de ajuste y validación. Los pasos que aprenden parámetros de los datos se ajustan en entrenamiento. Los modelos, épocas y umbrales se seleccionan en validación y se evalúan posteriormente en prueba.
- **Métricas.** En fraude interesa la sensibilidad bajo un límite de falsas alarmas; la precisión promedio resume el ordenamiento de las transacciones. En abandono, el macro F1 da el mismo peso a las tres clases y las tasas de falsos negativos usan como denominador los altos reales de cada canal. En energía, la sensibilidad mide cuántos picos se detectan y la precisión qué fracción de las alertas acierta.
- **Alcance.** Las etiquetas de fraude disponibles pueden contener inversiones de origen sin una lista de casos identificados. En abandono, una etiqueta por cliente permite comparar perfiles, pero faltan observaciones longitudinales para medir la transición personal. En energía, un consumo alto respecto del propio historial requiere comprobación operativa antes de interpretarse como sobrecarga de infraestructura.

## Fuentes

El enunciado «Proyecto Parcial 2 EXTRACCION DE CON BD», compartido mediante Drive para la sesión de IDGS 17, dio origen a las fases y a las preguntas A1–C3. Las tres bases SQLite entregadas para la práctica son la fuente de las cifras. El inventario documenta las huellas de los archivos y las estructuras leídas durante la ejecución del 24 de noviembre de 2026. [Inventario de fuentes](metodologia/inventario.json)

Se consultaron las guías **scikit-learn** y **scientific-visualization** de K-Dense en la revisión `c21d13b3cc2894680229ba9959a53281acb9200e`. Los resultados numéricos proceden de los scripts de este proyecto. [K-Dense · Scientific Agent Skills, revisión consultada](https://github.com/K-Dense-AI/scientific-agent-skills/tree/c21d13b3cc2894680229ba9959a53281acb9200e)
