"""Unidades y uso de los campos en las tres prácticas, en el orden del DDL."""

TABLAS = {
    "fintech_fraud.db": {
        "clientes": ("client_id", [
            ("client_id", "Clave lógica", "Identifica al cliente para auditar correspondencias con transacciones."),
            ("edad", "Años", "Edad registrada en el catálogo. Excluida de los modelos de fraude."),
            ("ingreso_anual", "Importe anual, moneda sin especificar", "Ingreso registrado sin fecha de vigencia. Excluido."),
            ("score_buro", "Puntos del sistema de origen", "Puntuación crediticia sin escala oficial ni fecha de vigencia documentadas. Excluida."),
            ("antiguedad_meses", "Meses", "Antigüedad registrada en el catálogo. Excluida."),
            ("segmento", "Categoría nominal", "Segmento comercial de la persona en la instantánea del catálogo. Excluido.")]),
        "comercios": ("merchant_id", [
            ("merchant_id", "Clave lógica", "Permite asociar MCC y país a una operación."),
            ("categoria_mcc", "Categoría nominal", "Giro del comercio, representado con nombres de categorías. Predictor y estrato para análisis."),
            ("pais", "Categoría nominal", "País del comercio. Predictor categórico."),
            ("riesgo_historico", "Escala de origen no documentada", "Riesgo histórico sin fecha de cálculo. Excluido para prevenir información posterior.")]),
        "transacciones": ("tx_id", [
            ("tx_id", "Clave lógica", "Identifica el movimiento. Sirve para muestreo y deduplicación."),
            ("client_id", "Referencia lógica", "Relaciona el movimiento con clientes para auditoría."),
            ("merchant_id", "Referencia lógica", "Relaciona el movimiento con comercios para incorporar MCC y país."),
            ("timestamp", "Fecha y hora, huso no documentado", "Determina el orden temporal, los periodos y los predictores de calendario."),
            ("monto", "Unidades de la moneda de la operación", "Importe nominal. Se imputa por moneda con entrenamiento y se aplica log(1 + monto)."),
            ("moneda", "Código de moneda", "Divisa nominal. Predictor, grupo de imputación y referencia del monto alto."),
            ("dispositivo", "Categoría nominal", "Dispositivo o terminal de la operación. Predictor."),
            ("intentos_pin", "Conteo de intentos", "Intentos del PIN observables al autorizar, según el supuesto de la práctica. Predictor."),
            ("is_fraud", "Indicador 0/1", "Etiqueta observada. 1 representa fraude y 0 representa operación legítima.")])},
    "ecommerce_churn_ltv.db": {
        "usuarios": ("user_id", [
            ("user_id", "Clave lógica", "Unidad de agregación y partición por usuario."),
            ("edad", "Años", "Edad del usuario. Fuera de las diez entradas conductuales."),
            ("canal_adquisicion", "Categoría nominal", "Canal de adquisición. Se conserva para auditar falsos negativos por canal."),
            ("suscripcion_prime", "Indicador 0/1", "Estado de suscripción. Fuera de los predictores."),
            ("riesgo_abandono", "Clases ordinales 0, 1 y 2", "Respuesta de riesgo bajo, medio y alto. La base contiene una etiqueta por usuario y carece de fechas de transición.")]),
        "sesiones_web": ("session_id", [
            ("session_id", "Clave lógica", "Identifica una sesión y permite retirar copias antes de contarla."),
            ("user_id", "Referencia lógica", "Agrupa sesiones de cada usuario."),
            ("duracion_seg", "Segundos", "Duración de la sesión. Se calcula la media de valores válidos por usuario."),
            ("paginas_vistas", "Conteo de páginas", "Número de páginas por sesión. Se calcula la media por usuario."),
            ("dispositivo_os", "Categoría nominal", "Sistema operativo de la sesión. Se deriva la proporción móvil entre sesiones cuyo sistema se identifica."),
            ("rebote", "Indicador 0/1", "Un valor de 1 señala rebote. Su media por usuario es la tasa de rebote.")]),
        "ordenes_compra": ("order_id", [
            ("order_id", "Clave lógica", "Identifica un pedido y permite retirar copias antes de contar pedidos."),
            ("user_id", "Referencia lógica", "Agrupa pedidos de cada usuario."),
            ("total_pagado", "Unidades monetarias, divisa no documentada", "Valor pagado por pedido. Se usa el pago promedio válido del usuario."),
            ("dias_envio", "Días", "Duración del envío. Se promedia por usuario."),
            ("devuelto", "Indicador 0/1", "Un valor de 1 indica devolución. Se calcula la proporción de pedidos devueltos."),
            ("calificacion_estrellas", "Estrellas, escala de 1 a 5 en los valores válidos", "Valoración del pedido. Se promedia por usuario.")])},
    "smart_grid_energy.db": {
        "subestaciones": ("substation_id", [
            ("substation_id", "Clave lógica", "Identifica la instalación para unir sus lecturas, calcular series y umbrales por estación."),
            ("zona", "Categoría nominal", "Zona de la estación. Permite recuperar la zona de lecturas y caracterizar la infraestructura."),
            ("capacidad_mw", "Megavatios (MW)", "Capacidad instalada. Predictor y referencia para una comprobación condicional de utilización."),
            ("anio_instalacion", "Año calendario", "Año de puesta en servicio. Predictor y estrato antes de 2005 frente a 2005 o posterior.")]),
        "telemetria_clima": ("clima_id", [
            ("clima_id", "Clave lógica", "Identifica el registro meteorológico y permite retirar copias."),
            ("zona", "Categoría nominal", "Parte de la clave de unión con las mediciones horarias."),
            ("hora_idx", "Índice horario", "Coordenada temporal relativa para el clima. Junto con zona define la unión horaria."),
            ("temperatura_c", "Grados Celsius (°C)", "Temperatura en la zona. Predictor en la hora t y filtro de calor para la evaluación."),
            ("humedad_pct", "Porcentaje", "Humedad registrada. Predictor meteorológico."),
            ("velocidad_viento", "Unidad de origen sin especificar", "Velocidad del viento. Se usa en la escala que contiene la base.")]),
        "lecturas_medidores": ("lectura_id", [
            ("lectura_id", "Clave lógica", "Identifica una lectura y permite retirar repeticiones."),
            ("substation_id", "Referencia lógica", "Relaciona la lectura con la estación."),
            ("zona", "Categoría nominal", "Ubicación registrada. La zona del catálogo sirve de referencia para recuperar faltantes."),
            ("hora_idx", "Índice horario", "Permite construir pares de horas consecutivas, rezagos y cortes cronológicos."),
            ("voltaje_v", "Voltios (V)", "Voltaje almacenado como texto. Se retira el sufijo V cuando aparece y se convierte a valor numérico."),
            ("corriente_a", "Amperios (A)", "Intensidad eléctrica medida. Predictor y variable física para PCA."),
            ("factor_potencia", "Proporción adimensional", "Factor de potencia medido. Predictor y variable física para PCA."),
            ("consumo_kwh", "Kilovatios-hora (kWh)", "Energía por lectura. Se agrega por estación y hora. La media observada en t+1 define el objetivo continuo y sus umbrales de clase relativos.")])}}
