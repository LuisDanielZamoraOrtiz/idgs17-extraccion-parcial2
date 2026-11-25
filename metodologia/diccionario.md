# Diccionario de campos de origen

Los tipos corresponden al esquema SQLite de las bases entregadas. Los roles describen su uso efectivo en los scripts de la práctica.
Las relaciones se implementan en el ETL mediante claves lógicas. Los nueve esquemas no declaran restricciones de clave primaria o foránea.
Las unidades no documentadas se conservan en su escala de origen.

## fintech_fraud.db

### clientes

Filas de origen 502 500. Claves distintas no nulas 500 000.

| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |
| --- | --- | --- | --- |
| `client_id` | INTEGER | Clave lógica | Identifica al cliente para auditar correspondencias con transacciones. |
| `edad` | INTEGER | Años | Edad registrada en el catálogo. Excluida de los modelos de fraude. |
| `ingreso_anual` | REAL | Importe anual, moneda sin especificar | Ingreso registrado sin fecha de vigencia. Excluido. |
| `score_buro` | REAL | Puntos del sistema de origen | Puntuación crediticia sin escala oficial ni fecha de vigencia documentadas. Excluida. |
| `antiguedad_meses` | INTEGER | Meses | Antigüedad registrada en el catálogo. Excluida. |
| `segmento` | TEXT | Categoría nominal | Segmento comercial de la persona en la instantánea del catálogo. Excluido. |

### comercios

Filas de origen 50 250. Claves distintas no nulas 50 000.

| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |
| --- | --- | --- | --- |
| `merchant_id` | INTEGER | Clave lógica | Permite asociar MCC y país a una operación. |
| `categoria_mcc` | TEXT | Categoría nominal | Giro del comercio, representado con nombres de categorías. Predictor y estrato para análisis. |
| `pais` | TEXT | Categoría nominal | País del comercio. Predictor categórico. |
| `riesgo_historico` | REAL | Escala de origen no documentada | Riesgo histórico sin fecha de cálculo. Excluido para prevenir información posterior. |

### transacciones

Filas de origen 4 284 000. Claves distintas no nulas 4 200 000.

| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |
| --- | --- | --- | --- |
| `tx_id` | INTEGER | Clave lógica | Identifica el movimiento. Sirve para muestreo y deduplicación. |
| `client_id` | INTEGER | Referencia lógica | Relaciona el movimiento con clientes para auditoría. |
| `merchant_id` | INTEGER | Referencia lógica | Relaciona el movimiento con comercios para incorporar MCC y país. |
| `timestamp` | TEXT | Fecha y hora, huso no documentado | Determina el orden temporal, los periodos y los predictores de calendario. |
| `monto` | REAL | Unidades de la moneda de la operación | Importe nominal. Se imputa por moneda con entrenamiento y se aplica log(1 + monto). |
| `moneda` | TEXT | Código de moneda | Divisa nominal. Predictor, grupo de imputación y referencia del monto alto. |
| `dispositivo` | TEXT | Categoría nominal | Dispositivo o terminal de la operación. Predictor. |
| `intentos_pin` | INTEGER | Conteo de intentos | Intentos del PIN observables al autorizar, según el supuesto de la práctica. Predictor. |
| `is_fraud` | INTEGER | Indicador 0/1 | Etiqueta observada. 1 representa fraude y 0 representa operación legítima. |

## ecommerce_churn_ltv.db

### usuarios

Filas de origen 804 000. Claves distintas no nulas 800 000.

| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |
| --- | --- | --- | --- |
| `user_id` | INTEGER | Clave lógica | Unidad de agregación y partición por usuario. |
| `edad` | REAL | Años | Edad del usuario. Fuera de las diez entradas conductuales. |
| `canal_adquisicion` | TEXT | Categoría nominal | Canal de adquisición. Se conserva para auditar falsos negativos por canal. |
| `suscripcion_prime` | INTEGER | Indicador 0/1 | Estado de suscripción. Fuera de los predictores. |
| `riesgo_abandono` | INTEGER | Clases ordinales 0, 1 y 2 | Respuesta de riesgo bajo, medio y alto. La base contiene una etiqueta por usuario y carece de fechas de transición. |

### sesiones_web

Filas de origen 3 060 000. Claves distintas no nulas 3 000 000.

| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |
| --- | --- | --- | --- |
| `session_id` | INTEGER | Clave lógica | Identifica una sesión y permite retirar copias antes de contarla. |
| `user_id` | INTEGER | Referencia lógica | Agrupa sesiones de cada usuario. |
| `duracion_seg` | REAL | Segundos | Duración de la sesión. Se calcula la media de valores válidos por usuario. |
| `paginas_vistas` | REAL | Conteo de páginas | Número de páginas por sesión. Se calcula la media por usuario. |
| `dispositivo_os` | TEXT | Categoría nominal | Sistema operativo de la sesión. Se deriva la proporción móvil entre sesiones cuyo sistema se identifica. |
| `rebote` | INTEGER | Indicador 0/1 | Un valor de 1 señala rebote. Su media por usuario es la tasa de rebote. |

### ordenes_compra

Filas de origen 1 530 000. Claves distintas no nulas 1 500 000.

| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |
| --- | --- | --- | --- |
| `order_id` | INTEGER | Clave lógica | Identifica un pedido y permite retirar copias antes de contar pedidos. |
| `user_id` | INTEGER | Referencia lógica | Agrupa pedidos de cada usuario. |
| `total_pagado` | REAL | Unidades monetarias, divisa no documentada | Valor pagado por pedido. Se usa el pago promedio válido del usuario. |
| `dias_envio` | INTEGER | Días | Duración del envío. Se promedia por usuario. |
| `devuelto` | INTEGER | Indicador 0/1 | Un valor de 1 indica devolución. Se calcula la proporción de pedidos devueltos. |
| `calificacion_estrellas` | REAL | Estrellas, escala de 1 a 5 en los valores válidos | Valoración del pedido. Se promedia por usuario. |

## smart_grid_energy.db

### subestaciones

Filas de origen 200. Claves distintas no nulas 200.

| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |
| --- | --- | --- | --- |
| `substation_id` | INTEGER | Clave lógica | Identifica la instalación para unir sus lecturas, calcular series y umbrales por estación. |
| `zona` | TEXT | Categoría nominal | Zona de la estación. Permite recuperar la zona de lecturas y caracterizar la infraestructura. |
| `capacidad_mw` | REAL | Megavatios (MW) | Capacidad instalada. Predictor y referencia para una comprobación condicional de utilización. |
| `anio_instalacion` | INTEGER | Año calendario | Año de puesta en servicio. Predictor y estrato antes de 2005 frente a 2005 o posterior. |

### telemetria_clima

Filas de origen 808 000. Claves distintas no nulas 800 000.

| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |
| --- | --- | --- | --- |
| `clima_id` | INTEGER | Clave lógica | Identifica el registro meteorológico y permite retirar copias. |
| `zona` | TEXT | Categoría nominal | Parte de la clave de unión con las mediciones horarias. |
| `hora_idx` | INTEGER | Índice horario | Coordenada temporal relativa para el clima. Junto con zona define la unión horaria. |
| `temperatura_c` | REAL | Grados Celsius (°C) | Temperatura en la zona. Predictor en la hora t y filtro de calor para la evaluación. |
| `humedad_pct` | REAL | Porcentaje | Humedad registrada. Predictor meteorológico. |
| `velocidad_viento` | REAL | Unidad de origen sin especificar | Velocidad del viento. Se usa en la escala que contiene la base. |

### lecturas_medidores

Filas de origen 4 590 000. Claves distintas no nulas 4 500 000.

| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |
| --- | --- | --- | --- |
| `lectura_id` | INTEGER | Clave lógica | Identifica una lectura y permite retirar repeticiones. |
| `substation_id` | INTEGER | Referencia lógica | Relaciona la lectura con la estación. |
| `zona` | TEXT | Categoría nominal | Ubicación registrada. La zona del catálogo sirve de referencia para recuperar faltantes. |
| `hora_idx` | INTEGER | Índice horario | Permite construir pares de horas consecutivas, rezagos y cortes cronológicos. |
| `voltaje_v` | TEXT | Voltios (V) | Voltaje almacenado como texto. Se retira el sufijo V cuando aparece y se convierte a valor numérico. |
| `corriente_a` | REAL | Amperios (A) | Intensidad eléctrica medida. Predictor y variable física para PCA. |
| `factor_potencia` | REAL | Proporción adimensional | Factor de potencia medido. Predictor y variable física para PCA. |
| `consumo_kwh` | REAL | Kilovatios-hora (kWh) | Energía por lectura. Se agrega por estación y hora. La media observada en t+1 define el objetivo continuo y sus umbrales de clase relativos. |
