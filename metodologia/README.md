# Inventario y diccionario

`inventario.json` reúne el nombre, tamaño en bytes y SHA-256 completo de cada base SQLite. Por tabla, registra las filas brutas, el número de claves distintas, las claves nulas, las repeticiones excedentes, las declaraciones de columnas y el DDL. No contiene filas de clientes, usuarios, transacciones, sesiones, pedidos ni lecturas individuales.

`diccionario.md` explica los campos de las nueve tablas, sus unidades y su función en cada análisis. `catalogo.py` contiene la definición semántica y la clave lógica que usa cada caso. El generador coteja todos los nombres de tablas y campos con el esquema leído.

Desde la raíz del proyecto, en Python 3.11, se puede regenerar con la biblioteca estándar.

```powershell
py -3.11 metodologia/inventariar.py --datos "C:/ruta/a/las/bases"
```

Los tres archivos de entrada deben llamarse `fintech_fraud.db`, `ecommerce_churn_ltv.db` y `smart_grid_energy.db`. El proceso abre las fuentes en modo de solo lectura y contrasta conteos y claves con `resultados/metricas.json` de cada caso. Además compara las huellas que ya almacenan fraude y abandono. Para generar el inventario antes de entrenar, se puede añadir `--solo-fuentes`. La opción `--salida` escribe ambas salidas en otra carpeta.

La repetición excedente de una clave cuenta las filas posteriores a su primera aparición. Dos filas que comparten identificador pueden diferir en otros campos. Cada caso describe su regla para resolver esas diferencias. La unidad de observación usada en los modelos se determina después del ETL. Los conteos de este inventario corresponden a las fuentes antes del procesamiento.
