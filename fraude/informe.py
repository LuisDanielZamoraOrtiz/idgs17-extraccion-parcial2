"""Figuras y comentarios a partir de métricas agregadas, sin publicar transacciones."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


COLORES={"Logística":"#586776","XGBoost":"#007A70","Red neuronal":"#BE6328"}


def n(v): return f"{v:,}".replace(","," ")
def dec(v,d=3): return f"{v:.{d}f}".replace(".",",")
def pct(v,d=2): return dec(100*v,d)+" %"
def dif(v): return ("+" if v>=0 else "−")+dec(abs(v),4)


def graficas(m,out):
    plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.spines.top":False,
                         "axes.spines.right":False,"axes.labelcolor":"#253F50","text.color":"#253F50",
                         "xtick.color":"#364D5A","ytick.color":"#364D5A","axes.edgecolor":"#91A7AE",
                         "legend.frameon":False,"savefig.facecolor":"white"})
    def lienzo():
        fig,ax=plt.subplots(figsize=(8.1,4.45),layout="constrained")
        ax.set_axisbelow(True);ax.grid(axis="y",alpha=.15)
        return fig,ax
    def guardar(fig,name):
        fig.savefig(out/(name+".png"),dpi=175);plt.close(fig)
    dist=m["distribucion"]; fig,ax=lienzo()
    labels=["Ene–ago\nEntrenamiento","Sep–oct\nValidación","Nov–dic\nPrueba"]
    rates=[dist[key]["tasa"] for key in ["train","val","test"]]
    bars=ax.bar(labels,rates,color=["#689499","#30746F","#224959"],width=.5)
    ax.bar_label(bars,labels=[pct(x) for x in rates],padding=6)
    ax.set_ylim(0,max(rates)*1.35);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_ylabel("Operaciones con etiqueta de fraude")
    guardar(fig,"a0_distribucion")
    fig,ax=lienzo(); frontier=pd.read_csv(out/"frontera_validacion.csv")
    for name,sub in frontier.groupby("modelo",sort=False):
        ax.step(sub.limite_fpr,sub.recall,where="post",label=name,color=COLORES[name],lw=2)
    ax.axvline(.015,color="#343D45",ls=":",lw=1.6);ax.axhline(.8,color="#343D45",ls="--",lw=1.4)
    ax.text(.016,.85,"Límite de bloqueos 1,5 %",fontsize=9)
    ax.text(.049,.81,"Meta de captura 80 %",ha="right",fontsize=9)
    ax.set(xlabel="Tasa máxima de falsos positivos en validación",ylabel="Mayor sensibilidad alcanzada",xlim=(0,.05),ylim=(0,1))
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right",ncol=1);guardar(fig,"a1_frontera")
    fig,ax=lienzo()
    for name,r in m["modelos"].items():
        pts=r["curva_pr"]
        ax.plot([p["recall"] for p in pts],[p["precision"] for p in pts],lw=1.9,
                color=COLORES[name],label=f"{name}  ·  AP {dec(r['ap_prueba'],4)}  ·  PR-AUC {dec(r['pr_auc_trapezoidal_prueba'],4)}")
    ax.axhline(dist["test"]["tasa"],color="#70787E",ls="--",lw=1.1,label="Proporción de fraude")
    ax.set(xlabel="Sensibilidad",ylabel="Precisión de las alertas",xlim=(0,1),ylim=(0,1.025))
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper right",fontsize=9);guardar(fig,"a2_precision_recall")
    fig,ax=lienzo()
    for key,mark,lang in [("red","-","etiquetas observadas"),("red_ruido_extra","--","ruido adicional")]:
        vals=m[key]["historia"]
        for col,c,which in [("train_loss","#187E72","Entrenamiento"),("val_loss","#9C4A26","Validación")]:
            ax.plot([v["epoca"] for v in vals],[v[col] for v in vals],color=c,ls=mark,lw=1.8,label=f"{which}, {lang}")
    e=m["red"]["epoca_optima"]; ax.scatter([e],[m["red"]["min_val_loss"]],color="#712E14",s=42,zorder=5)
    ax.set(xlabel="Época",ylabel="Entropía cruzada binaria",xlim=(1,m["red"]["epocas"]))
    ax.legend(loc="best",fontsize=8.2,ncol=2);guardar(fig,"a3_perdida")


def divergencia(r):
    d=r["divergencia"]
    if d is None: return f"el criterio no detectó divergencia persistente durante las {r['epocas']} épocas"
    return (f"la primera divergencia persistente comenzó en la época {d['inicio']}, tomando como referencia la "
            f"{d['referencia']}, y quedó confirmada en la {d['confirmacion']}")


def elaborar(m,out):
    graficas(m,out)
    a=m["auditoria"];dist=m["distribucion"]; models=m["modelos"]; win=m["seleccion_validacion"]
    th=models[win]["umbrales"];v=th["operativo"]["val"];t=th["operativo"]["test"];e=th["recall80"]["val"]
    nn=m["red"];extra=m["red_ruido_extra"]
    feasible=[name for name,r in models.items() if r["umbrales"]["operativo"]["val"]["recall"]>=.8]
    if feasible:
        operacion="En validación cumplieron ambas condiciones "+", ".join(feasible)+"."
    else:
        operacion="Ninguno de los tres modelos alcanzó simultáneamente el 80 % de sensibilidad y menos del 1,5 % de falsos positivos en validación."
    ap={name:r["ap_prueba"] for name,r in models.items()}
    trap={name:r["pr_auc_trapezoidal_prueba"] for name,r in models.items()}
    bandas=m["remuestreo_AP"]["IC_delta_AP_vs_logistica"]
    tabla=[]
    for name,r in models.items():
        lo,hi=bandas.get(name,(None,None))
        tabla.append({"modelo":name,"operaciones_prueba":dist["test"]["n"],"fraudes_prueba":dist["test"]["positivos"],
                      "umbrales_distintos":r["umbrales_distintos_curva_pr"],"ap_validacion":r["ap_validacion"],
                      "pr_auc_trapezoidal_prueba":trap[name],"delta_pr_auc_vs_logistica":trap[name]-trap["Logística"],
                      "ap_prueba":ap[name],"delta_ap_vs_logistica":ap[name]-ap["Logística"],
                      "ic95_delta_ap_inferior":lo,"ic95_delta_ap_superior":hi})
    pd.DataFrame(tabla).to_csv(out/"a2_metricas.csv",index=False)
    def ic(nombre):
        lo,hi=bandas[nombre]
        return "["+dif(lo)+", "+dif(hi)+"]"
    abarcan=[nombre for nombre,(lo,hi) in bandas.items() if lo<=0<=hi]
    lectura_ic=("Ambos intervalos incluyen cero." if len(abarcan)==2 else
                "El intervalo de "+abarcan[0]+" incluye cero." if abarcan else
                "Ambos intervalos excluyen cero.")
    it=m["interacciones"]; rates=it["grupos_prueba"]; joint=rates["Tres condiciones"]
    days=[("train","enero a agosto"),("val","septiembre y octubre"),("test","noviembre y diciembre")]
    split=". ".join(f"De {label}, {n(dist[key]['n'])} operaciones y {n(dist[key]['positivos'])} fraudes" if key=="train" else f"En {label}, {n(dist[key]['n'])} operaciones y {n(dist[key]['positivos'])} fraudes" for key,label in days)
    cells=it["celdas_en_mcc_alto"]
    day=100*(cells["alto1_noche0"]["tasa"]-cells["alto0_noche0"]["tasa"])
    night=100*(cells["alto1_noche1"]["tasa"]-cells["alto0_noche1"]["tasa"])
    conclusion_capacidad=("El control aditivo superó al árbol más profundo en prueba. La ganancia sobre la logística es compatible con efectos no lineales de variables individuales y estos datos no la atribuyen a interacciones."
                          if m["tocones"]["ap_prueba"]>=ap["XGBoost"] else
                          "El modelo con varias divisiones mejoró frente al control aditivo. La capacidad para combinar variables es una explicación posible, aunque el contraste no identifica su contribución causal.")
    document={
      "titulo":"Detección de fraude al autorizar un pago",
      "introduccion":"Se estima la probabilidad de fraude de una transacción con tarjeta y se evalúa cuántos fraudes registrados pueden detectarse respetando un límite de alertas sobre operaciones legítimas. La decisión usa información disponible al autorizar el pago.",
      "etl":[
        f"Se auditaron {n(a['filas'])} filas de transacciones y {n(a['claves'])} identificadores distintos. Los catálogos contienen {n(a['clientes']['claves'])} claves de cliente y {n(a['comercios']['claves'])} de comercio. La selección determinista por clave, independiente de la etiqueta y de la fecha, retuvo aproximadamente {pct(m['por_mil']/1000,1)} de las operaciones para permitir la reproducción en CPU.",
        f"La muestra inicial contenía {n(a['muestra_filas'])} filas. Se quitaron {n(a['copias_identicas_muestra'])} copias repetidas tras normalizar las categorías. Se excluyeron {n(a['claves_conflictivas_muestra'])} claves con versiones contradictorias y {n(a['sin_fecha_o_etiqueta_valida_muestra'])} filas sin fecha o clase válida. Quedaron {n(a['filas_analizadas'])} operaciones únicas. La comprobación de llaves encontró {n(a['sin_cliente_maestro_muestra'])} clientes y {n(a['sin_comercio_maestro_util_muestra'])} comercios sin correspondencia utilizable en la muestra.",
        f"En la muestra depurada, {n(a['monto_nulo_original_muestra'])} importes estaban vacíos y {n(a['monto_invalido_muestra'])} eran negativos o iguales al marcador repetido 999999,9. Se conservó la operación, se indicó la ausencia y se imputó el importe con la mediana de su moneda aprendida en entrenamiento; en ausencia de una mediana por moneda, se usa la global de entrenamiento. Los códigos de moneda, dispositivo, categoría comercial y país se normalizaron; los vacíos tienen un nivel explícito."],
      "variables":[
        "La respuesta binaria is_fraud identifica el fraude observado. Las entradas son importe nominal transformado con log(1 + monto), intentos de PIN, indicadores de datos faltantes, hora y día de semana cíclicos, indicador de madrugada (00:00–05:59), moneda, dispositivo, categoría comercial (MCC) y país del comercio.",
        f"Se obtuvieron {m['preparacion']['columnas']} columnas numéricas mediante codificación por categoría y estandarización de entradas continuas e indicadores numéricos con parámetros del entrenamiento. Se excluyeron los identificadores, el riesgo histórico calculado sin fecha de disponibilidad y los atributos de cliente sin versión histórica. La etiqueta jamás entra como predictor. Se supone que MCC y país ya se conocen al autorizar."],
      "entrenamiento":[
        "Partición cronológica de 2025: "+split+". El conjunto intermedio eligió el modelo, el umbral y el mejor punto de entrenamiento. La prueba se consultó después de fijar esas decisiones.",
        f"Regresión logística, penalización L2, máximo de 250 iteraciones y {m['logistica_iteraciones']} efectivas. XGBoost, máximo de 850 árboles, tasa 0,05, profundidad 4 y paciencia de 40 rondas sin mejora en pérdida de validación. Se seleccionó la ronda {m['xgb_rondas']} de {m['xgb_rondas_entrenadas']} entrenadas. Para contrastar las interacciones se repitió XGBoost con árboles de una sola división.",
        f"Red con retropropagación, dos capas ocultas de 48 y 24 neuronas ReLU, Adam, lotes de 512 y {nn['epocas']} épocas. Se conservó el estado de menor entropía cruzada de validación. Para el control de ruido se repitió el entrenamiento con idénticos pesos iniciales y parámetros."],
      "figura_etl":"resultados/a0_distribucion.png","pie_etl":"Frecuencia de fraude observado en cada periodo. Las barras usan la proporción de la clase 1 sobre todas las operaciones de ese bloque.",
      "preguntas":[
        {"titulo":"A1. ¿Existe un umbral que capture el 80 % del fraude con menos del 1,5 % de falsos positivos?",
         "respuesta":[operacion+f" Se recorrieron todos los cortes sobre probabilidades de validación. {win} tuvo la mayor precisión promedio en validación ({dec(models[win]['ap_validacion'],4)}). Para este modelo se maximizó el fraude capturado dentro del presupuesto de alertas, usando el umbral {dec(v['umbral'],5)}. En validación alcanzó {pct(v['recall'])} de sensibilidad y {pct(v['fpr'])} de falsos positivos.",
           f"Al congelar ese umbral y aplicarlo a noviembre–diciembre, detectó {n(t['TP'])} de {n(t['TP']+t['FN'])} fraudes ({pct(t['recall'])}) y marcó {n(t['FP'])} de {n(t['FP']+t['TN'])} operaciones legítimas ({pct(t['fpr'])}). El primer corte de validación que sí alcanzó el 80 % de captura fue {dec(e['umbral'],5)} y requirió {pct(e['fpr'])} de falsos positivos. La tasa cuenta operaciones legítimas bloqueadas, no personas distintas.",
           "En el mismo modelo, con umbral 0,65 se obtuvieron en validación sensibilidad de "+pct(th["ejemplo_065"]["val"]["recall"])+" y falsos positivos de "+pct(th["ejemplo_065"]["val"]["fpr"])+"; con 0,82, "+pct(th["ejemplo_082"]["val"]["recall"])+" y "+pct(th["ejemplo_082"]["val"]["fpr"])+", respectivamente."],
         "figura":"resultados/a1_frontera.png","pie":"Cada trazo representa el mayor fraude capturado para el presupuesto de falsos positivos del eje horizontal, en validación. Las guías señalan las dos metas operativas."},
        {"titulo":"A2. ¿Cuánto cambia el área precisión–sensibilidad y qué patrones respaldan esa diferencia?",
         "respuesta":[f"En prueba, el área trapezoidal bajo la curva completa (PR-AUC) fue {dec(trap['Logística'],4)} para logística, {dec(trap['XGBoost'],4)} para XGBoost y {dec(trap['Red neuronal'],4)} para la red. Los cambios frente a logística fueron {dif(trap['XGBoost']-trap['Logística'])} y {dif(trap['Red neuronal']-trap['Logística'])}. La precisión promedio (AP) suma los incrementos de sensibilidad en escalones y dio {dec(ap['Logística'],4)}, {dec(ap['XGBoost'],4)} y {dec(ap['Red neuronal'],4)}, respectivamente.",
           f"XGBoost cambió la AP en {dif(ap['XGBoost']-ap['Logística'])} (IC 95 % {ic('XGBoost')}) y la red en {dif(ap['Red neuronal']-ap['Logística'])} (IC 95 % {ic('Red neuronal')}). Se remuestrearon los mismos bloques de siete días para los tres modelos en 1 000 réplicas del periodo de prueba, con los modelos fijos. "+lectura_ic,
           f"El XGBoost aditivo de profundidad 1 obtuvo AP {dec(m['tocones']['ap_prueba'],4)}. "+conclusion_capacidad,
           f"El monto alto comienza en el percentil 90 del entrenamiento para cada moneda conocida. Las categorías MCC de mayor tasa en entrenamiento fueron {', '.join(it['categorias_mayor_tasa_train'])}. En prueba, {n(joint['positivos'])} de {n(joint['n'])} pagos de monto alto en madrugada en estas categorías tenían etiqueta de fraude ({pct(joint['tasa'])}). Dentro de estas categorías, el cambio de tasa entre monto ordinario y alto fue {dec(night,1)} puntos porcentuales en madrugada y {dec(day,1)} el resto del día. La asociación observada describe esos pagos y depende del tamaño del subgrupo."],
         "figura":"resultados/a2_precision_recall.png","pie":"La sensibilidad mide la fracción de fraudes detectados. La precisión mide cuántas alertas corresponden a fraude. La línea horizontal marca la prevalencia. Ambas áreas se calcularon con todos los umbrales antes de resumir los trazos."},
        {"titulo":"A3. ¿Desde qué época divergen las pérdidas y qué ocurre al agregar ruido de etiqueta?",
         "respuesta":[f"Se fijó una regla antes de entrenar. Se toma la primera secuencia de tres épocas consecutivas en las que la pérdida de validación supera en más de 0,0005 su mínimo previo y la de entrenamiento queda por debajo de su valor en la época de ese mínimo. En el entrenamiento con etiquetas observadas, {divergencia(nn)}. El mínimo de validación fue {dec(nn['min_val_loss'],5)} en la época {nn['epoca_optima']}.",
           f"Como control se invirtió exactamente el 1,5 % redondeado de las etiquetas de entrenamiento ({n(extra['etiquetas_invertidas_adicionales'])} operaciones), por sorteo reproducible. Se mantuvieron intactas las etiquetas observadas de validación y prueba. Con ruido adicional, {divergencia(extra)}. El mínimo de validación fue {dec(extra['min_val_loss'],5)} en la época {extra['epoca_optima']}.",
           "El enunciado indica aproximadamente 1,5 % de inversiones previas. La tabla entrega una sola etiqueta por operación. El inicio detectado mide divergencia de pérdidas bajo la regla elegida; atribuir una época exacta a la memorización de las inversiones originales exigiría etiquetas verdaderas o la identidad de las operaciones alteradas."],
         "figura":"resultados/a3_perdida.png","pie":"Las líneas continuas corresponden a etiquetas observadas y las discontinuas al control con ruido adicional. Una menor pérdida indica mejor ajuste a las etiquetas del bloque respectivo. El punto identifica el mínimo de validación del ajuste original."}],
      "limitaciones":["Los resultados describen una muestra de operaciones de un año y las etiquetas disponibles. La identidad de los fraudes verdaderos invertidos en origen y las pérdidas económicas por decisión no están documentadas. El importe se conserva en unidades nominales; los percentiles para contrastes se calculan dentro de cada moneda. La presencia de clientes recurrentes en distintos periodos refleja el uso cronológico; el estudio no valida específicamente la generalización a personas nuevas."]}
    (out.parent/"informe.json").write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    out=Path(__file__).resolve().parent/"resultados"
    elaborar(json.loads((out/"metricas.json").read_text(encoding="utf-8")),out)
