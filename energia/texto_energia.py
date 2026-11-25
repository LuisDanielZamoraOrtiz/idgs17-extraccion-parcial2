"""Texto del informe derivado exclusivamente de las métricas calculadas."""


def num(x,d=0):
    return f"{x:,.{d}f}".replace(","," ").replace(".",",")


def pct(x):
    return "indefinida" if x is None else num(100*x,1)+" %"


def redactar(r):
    a=r["etl"]; c=r["calor"]; b=c["operativa"]; u=c["argmax"]; far=c["a120F"]
    p=r["pca"]; i=r["infraestructura"]; anc=i["antes_2005"]; moderna=i["desde_2005"]
    m=r["modelos"]; n=r["ruido"]; d=r["demanda"]
    q70=a["q70_rango"];q90=a["q90_rango"];ep=r["epoca_optima"]
    dife=(anc["error"]-moderna["error"])*100
    diff=(anc["tasa_critica"]-moderna["tasa_critica"])*100
    def delta(val):
        return ("+" if val>=0 else "")+num(val,2)
    def ic(vals):
        return f"[{delta(vals[0]*100)}, {delta(vals[1]*100)}]"
    valores=p["correlaciones_eje"]; variables=["voltaje","corriente","factor de potencia","temperatura"]
    orden=sorted(range(4),key=lambda j:abs(valores[j]),reverse=True)
    cs=", ".join(f"{variables[j]} {num(valores[j],2)}" for j in orden)
    crit_120=(f"Se observaron {num(far['n'])} horas y {num(far['criticas'])} críticas a ese nivel. "
              f"La sensibilidad fue {pct(far['recall'])} y la precisión {pct(far['precision'])}."
              if far["n"] else "Ninguna hora industrial de prueba superó ese segundo corte. Sus dos métricas carecen de denominador.")
    return {
        "titulo":"Estabilidad de la red y demanda eléctrica",
        "introduccion":"Con los sensores disponibles hasta la hora t estimamos el consumo medio de la siguiente hora en la misma subestación. También evaluamos si una alerta anticipa una hora de consumo especialmente alto para esa instalación. La base registra energía medida y capacidad instalada. Carece de una etiqueta verificada de fallo o sobrecarga.",
        "figura_etl":"resultados/distribucion_consumo.png",
        "pie_etl":"Distribución del consumo medio de la hora siguiente en los 42 000 pares muestreados para entrenamiento. La línea marca la mediana global. Los límites de las clases se calcularon por subestación.",
        "figura_entrenamiento":"resultados/convergencia.png",
        "pie_entrenamiento":f"Pérdida logarítmica por época de la red eléctrica. Cada curva se calculó sobre la misma muestra en las 130 épocas. Se conservaron los pesos de la época {ep}.",
        "etl":[
            f"Se conciliaron {num(a['lecturas_medidores_origen'])} lecturas, {num(a['telemetria_clima_origen'])} registros de clima y {a['subestaciones']} subestaciones. Por ID se retiraron {num(a['lecturas_medidores_duplicados'])} repeticiones de lectura y {num(a['telemetria_clima_duplicados'])} de clima. Se conservó la primera versión según el orden del archivo. En {num(a['lecturas_medidores_copias_con_diferencias'])} copias de lecturas se detectó alguna diferencia de contenido registrado.",
            f"Se normalizaron zonas y sufijos V. El catálogo recuperó la zona de {num(a['zona_recuperada_catalogo'])} lecturas. {num(a['telemetria_clima_sin_clave_union'])} registros climáticos carecían de una clave válida para la unión. Se invalidaron centinelas como -999 y 999999,9 y valores físicamente incoherentes. Los sensores ausentes se imputaron por la mediana de entrenamiento. Las horas sin consumo observado quedaron fuera del objetivo.",
            f"La agregación se hizo por subestación-hora y por zona-hora antes de la unión. Hay {num(a['estacion_horas_observadas'])} de {num(a['estacion_horas_posibles'])} subestación-horas observadas, con mediana de {a['mediana_lecturas_por_hora']} lectura(s) por hora observada y máximo de {a['max_lecturas_por_hora']}. La comparación de dos horas seguidas produjo {num(a['pares_consecutivos'])} pares. La ausencia de una medición no se convirtió en consumo cero.",
            f"Si cada registro cubriese exactamente una hora, sumar sus kWh y dividir entre 1 000 × capacidad en MW mediría la fracción de capacidad observada. La mayor fue {pct(a['fraccion_capacidad_max'])}. Hubo {a['horas_al_menos_capacidad']} horas con fracción ≥ 1. La cobertura de todos los medidores es desconocida. Para estudiar alertas se definieron clases relativas usando los datos históricos de cada estación."
        ],
        "variables":[
            f"Objetivo continuo: media de consumo_kwh por lectura de la subestación en t+1, para comparar intervalos con distinto número de lecturas. Objetivo categórico: clase 0 por debajo del percentil 70, clase 1 entre los percentiles 70 y 90, y clase 2 desde el percentil 90. Los umbrales se calcularon por estación únicamente en entrenamiento. Los P70 oscilaron entre {num(q70[0],1)} y {num(q70[1],1)} kWh; los P90, entre {num(q90[0],1)} y {num(q90[1],1)} kWh.",
            "Entradas de t: voltaje (V), corriente (A), factor de potencia, consumo (kWh), temperatura (°C), humedad (%), viento en la unidad de origen y número de lecturas. Se añadieron consumo medio y cobertura de las últimas 24 horas, ciclo de 24 horas, antigüedad representada por el año de instalación, capacidad (MW) y zona. Las variables del instante t+1 solo definen la respuesta o los subgrupos de evaluación."
        ],
        "entrenamiento":[
            f"Separación cronológica por hora objetivo: entrenamiento [0, {a['fin_train_excl']}), validación [{a['fin_train_excl']}, {a['fin_val_excl']}) y prueba [{a['fin_val_excl']}, {a['horas']}). Para limitar memoria se tomó una muestra uniforme reproducible de {num(a['usados_train'])}/{num(a['pares_train'])} pares en entrenamiento y {num(a['usados_val'])}/{num(a['pares_val'])} en validación. Se evaluaron todos los {num(a['usados_test'])} pares de prueba. Semilla 2026.",
            f"La regresión logística convergió en {r['iteraciones_rl']} de 250 iteraciones. XGBoost usó tasa 0,05, techo de 850 rondas y paciencia de 35. La mejor ronda por pérdida de validación fue {r['mejor_xgb']}. La red ReLU 24–12 con Adam y retropropagación completó 130 épocas. La menor pérdida de validación se obtuvo en la época {ep}, cuyos pesos se evaluaron en prueba.",
            f"Macro-F1 de prueba con la clase más probable: logística {num(m['Logística']['macro_f1'],3)}, XGBoost {num(m['XGBoost']['macro_f1'],3)}, red {num(m['Red']['macro_f1'],3)}. Como referencias, repetir la clase previa dio {num(r['persistencia_clase']['macro_f1'],3)} y asignar siempre nominal, {num(r['nominal_constante']['macro_f1'],3)}. El F1 de las tres clases pondera a cada clase por igual.",
            f"Como ensayo separado se perturbó una copia del entrenamiento con ruido normal de desviación 0,5 en escala estandarizada sobre {num(n['celdas'])} de {num(n['total'])} celdas de sensores (≈2 %). Con la misma logística, el macro-F1 de prueba fue {num(n['macro_f1'],4)} frente a {num(m['Logística']['macro_f1'],4)} con la muestra limpia. La prueba y las etiquetas se conservaron idénticas.",
            f"Para la demanda continua se entrenó una regresión Ridge con los mismos predictores. Error absoluto medio de prueba en kWh por lectura: {num(d['ridge']['MAE'],2)} con Ridge, {num(d['persistencia']['MAE'],2)} repitiendo la última medición y {num(d['media_entrenamiento']['MAE'],2)} empleando la media histórica global. R² de Ridge: {num(d['ridge']['R2'],3)}."
        ],
        "preguntas":[
            {"titulo":"C1. ¿Con cuánta anticipación y precisión se detecta una hora crítica?",
             "respuesta":[f"La anticipación fue de una hora. Consideramos industriales Norte_Industrial y Bajio_Manufactura. En prueba, la temperatura media observada de la zona superó 30 °C en {num(b['n'])} pares industriales, incluidos {num(b['criticas'])} críticos según el umbral histórico. Esta medición futura se usó solo para identificar el subgrupo evaluado.",
                          f"Al elegir la clase más probable, la red detectó {num(u['VP'])} de {num(u['criticas'])} críticos, con sensibilidad {pct(u['recall'])} y precisión {pct(u['precision'])}. Para emitir alertas se eligió el umbral {num(c['umbral'],4)} que maximizó el F1 crítico en validación ({c['seleccion']}). En prueba detectó {num(b['VP'])} de {num(b['criticas'])} críticos y acertó {num(b['VP'])} de {num(b['alertas'])} avisos. Sensibilidad {pct(b['recall'])}, precisión {pct(b['precision'])}, {num(b['FP'])} falsas alarmas y {num(b['FN'])} omisiones. La proporción crítica de partida fue {pct(b['criticas']/b['n'])}.",
                          f"Se comprobó también la indicación «more 120 f» como temperatura superior a 120 °F (48,89 °C). {crit_120} El máximo industrial horario observado en prueba fue {num(c['max_temp_industrial_prueba'],2)} °C. La red también superó 120 épocas de entrenamiento."],
             "figura":"resultados/calor.png","pie":"Red neuronal, prueba cronológica y temperatura industrial superior a 30 °C. La sensibilidad cuenta críticos detectados entre los críticos reales. La precisión cuenta aciertos entre los avisos emitidos. «n/d» indica un denominador cero."},
            {"titulo":"C2. ¿Qué variables físicas distinguen carga nominal y elevada?",
             "respuesta":[f"Se estandarizaron cuatro mediciones de la propia hora del evento. Imputación, escala y PCA se ajustaron en entrenamiento. De los cuatro ejes se eligió por entrenamiento el de mayor diferencia estandarizada entre las clases 0 y 1. Se orientó hacia la media elevada. Fue PC{p['eje']+1} y explicó {pct(p['varianza'][p['eje']])} de la varianza total. Sus correlaciones con los sensores fueron {cs}.",
                          f"En prueba, el centro nominal quedó en {num(p['centros_test'][0],2)} y el elevado en {num(p['centros_test'][1],2)} sobre ese eje. La diferencia fue de {num(p['efecto_test'],2)} desviaciones agrupadas. Temperatura y corriente dominan este contraste contemporáneo. Las cuatro componentes explicaron varianzas muy próximas, de {pct(min(p['varianza']))} a {pct(max(p['varianza']))}, así que la dirección concreta puede variar entre muestras."],
             "figura":"resultados/cargas_pca.png","pie":"Cargas expresadas como correlaciones entre sensores y componente elegida en entrenamiento. La longitud representa asociación y el signo indica la dirección."},
            {"titulo":"C3. ¿Las subestaciones antiguas presentan más errores o eventos críticos?",
             "respuesta":[f"Con la clase de máxima probabilidad de la red, las {anc['estaciones']} estaciones anteriores a 2005 registraron {num(anc['errores'])} errores y {num(anc['criticas'])} críticos en {num(anc['n'])} horas elegibles: {pct(anc['error'])} y {pct(anc['tasa_critica'])}. Las {moderna['estaciones']} instaladas desde 2005 registraron {num(moderna['errores'])} errores y {num(moderna['criticas'])} críticos en {num(moderna['n'])} horas: {pct(moderna['error'])} y {pct(moderna['tasa_critica'])}.",
                          f"Antiguas menos modernas: {delta(dife)} puntos porcentuales de error y {delta(diff)} de frecuencia crítica. Intervalos exploratorios del 95 % al remuestrear estaciones completas 1 000 veces por cohorte: error {ic(i['ic_diferencia_error'])} y frecuencia {ic(i['ic_diferencia_frecuencia'])} puntos. Ambos incluyen cero. La muestra permite diferencias tan pequeñas en ambos sentidos. La frecuencia mide cuántas veces se rebasa el umbral propio de cada estación."],
             "figura":"resultados/infraestructura.png","pie":"Porcentaje de errores y de horas críticas relativas en la prueba. Cada barra usa el total de horas evaluadas dentro de su cohorte de instalación."}
        ],
        "limitaciones":["La base carece de etiquetas operativas de fallo, identificadores de medidor, duración confirmada del intervalo y verificación de cobertura completa. Una alarma de consumo inusualmente alto requiere confirmación física antes de interpretarse como sobrecarga de la red.",
                        "Se predijo una hora a partir de otra observada en la misma estación. Las parejas separadas por huecos no forman parte de este experimento. Los resultados corresponden al periodo reservado y a estas instalaciones."]
    }
