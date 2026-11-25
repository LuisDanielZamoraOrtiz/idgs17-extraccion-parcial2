"""Create the classroom figures and interpretation from aggregate metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from etl import FEATURES

INK="#193245"
TEAL="#007C80"
BLUE="#3867A1"
GOLD="#A45113"


def number(value,digits=0):
    return f"{value:,.{digits}f}".replace(","," ").replace(".",",")


def pct(value,digits=1):
    return number(100*value,digits)+" %"


def make_report(d,root):
    dest=root/"resultados"
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelcolor":INK,"xtick.color":INK,
                         "ytick.color":INK,"text.color":INK,"axes.edgecolor":"#ADB8BE","svg.fonttype":"none"})
    def canvas(size=(7.6,3.8)):
        fig,ax=plt.subplots(figsize=size,layout="constrained")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_axisbelow(True)
        return fig,ax

    def save(fig,name):
        fig.savefig(dest/name,dpi=180,bbox_inches="tight",pad_inches=0.12,facecolor="white")
        plt.close(fig)

    audit=d["auditoria"]
    models=d["modelos"]
    a=d["b1"]
    fig,ax=canvas()
    positions=np.arange(len(a))
    values=[v["auc"] for v in a]
    ax.barh(positions,values,color=[TEAL]+[BLUE]*(len(a)-1),height=.55)
    ax.set_yticks(positions,[FEATURES[v["variable"]] for v in a])
    ax.invert_yaxis()
    ax.set_xlim(0,1)
    ax.axvline(.5,color="#624B33",ls="--",lw=1)
    ax.set_xlabel("AUC ROC individual · referencia al azar = 0,50")
    for i,v in enumerate(values):
        ax.text(v+.012,i,number(v,3),va="center",fontweight="bold")
    ax.grid(axis="x",alpha=.18)
    save(fig,"b1_navegacion_logistica.png")

    fig,ax=canvas()
    for i,(name,color,group) in enumerate([("Logística",BLUE,-.18),("XGBoost",TEAL,.18)]):
        vals=[models[name]["prueba"]["macro_f1"],models[name+" PCA"]["prueba"]["macro_f1"]]
        pos=np.arange(2)+group
        ax.bar(pos,vals,width=.32,color=color,label=name)
        for x,y in zip(pos,vals):
            ax.text(x,y+.018,number(y,3),ha="center",va="bottom",fontweight="bold",fontsize=10)
    ax.set_xticks([0,1],["10 variables de conducta","5 componentes principales"])
    ax.set_ylim(0,1)
    ax.set_ylabel("Macro F1 en prueba")
    ax.legend(loc="upper right",frameon=False,ncols=2)
    ax.grid(axis="y",alpha=.18)
    save(fig,"b2_macro_f1.png")

    channels=sorted((v for v in d["b3"] if v["tasa"] is not None),key=lambda v:v["tasa"],reverse=True)
    fig,ax=canvas((7.6,4.4))
    names={"Organico":"Orgánico","Social_Ads":"Social Ads","Paid_Search":"Paid Search"}
    y=np.arange(len(channels))
    for idx,v in enumerate(channels):
        color=TEAL if v["canal"] in ["Organico","Social_Ads","Paid_Search"] else "#677986"
        lower,upper=[100*e for e in v["IC95"]]
        rate=100*v["tasa"]
        ax.plot([lower,upper],[idx,idx],color=color,lw=2)
        ax.scatter(rate,idx,color=color,s=38,zorder=4)
    ax.set_yticks(y,[names.get(v["canal"],v["canal"]) for v in channels])
    ax.invert_yaxis()
    ax.set_ylim(len(channels)-.45,-.6)
    minp=min(v["IC95"][0] for v in channels)*100
    maxp=max(v["IC95"][1] for v in channels)*100
    span=max(4,maxp-minp)
    ax.set_xlim(max(0,minp-0.2*span), min(100,maxp+0.2*span))
    ax.set_xlabel("Riesgos altos omitidos entre los altos reales (%) · intervalo de Wilson al 95 %")
    ax.grid(axis="x",alpha=.25)
    save(fig,"b3_canales.png")

    net=models["Red neuronal"]["ajuste"]
    fig,ax=canvas()
    hist=net["historial"]
    epochs=[v["epoca"] for v in hist]
    for key,color,label in [("perdida_entrenamiento",BLUE,"Entrenamiento"),("perdida_validacion",TEAL,"Validación")]:
        ax.plot(epochs,[v[key] for v in hist],color=color,lw=1.8,label=label)
    ax.axvline(net["epoca_elegida"],ls="--",color=GOLD,lw=1.2,label=f'Época elegida ({net["epoca_elegida"]})')
    ax.set_xlabel("Época de retropropagación")
    ax.set_ylabel("Entropía cruzada")
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.2)
    save(fig,"entrenamiento_red.png")

    fig,ax=canvas()
    cols=["edad","duracion_seg","paginas_vistas","total_pagado","calificacion_estrellas"]
    captions=["Edad","Duración","Páginas","Pago","Estrellas"]
    denominators=[audit["tablas"][t]["unicas"] for t in ("usuarios","sesiones_web","sesiones_web","ordenes_compra","ordenes_compra")]
    initial=np.array([audit["numericas"][c]["nulos"]/n*100 for c,n in zip(cols,denominators)])
    bad=np.array([audit["numericas"][c]["invalidos"]/n*100 for c,n in zip(cols,denominators)])
    ypos=np.arange(5)
    ax.barh(ypos,initial,color=BLUE,label="Nulo original",height=.55)
    ax.barh(ypos,bad,left=initial,color=GOLD,label="Valor inválido",height=.55)
    ax.set_yticks(ypos,captions)
    ax.invert_yaxis()
    ax.set_xlabel("Porcentaje de su tabla después de quitar duplicados")
    ax.set_xlim(0, max(initial+bad)*1.2)
    ax.grid(axis="x",alpha=.2)
    ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1.17),ncols=2)
    save(fig,"etl_calidad.png")

    def checked(key):
        v=audit["numericas"][key]
        return number(v["invalidos"])
    noise=d["ruido_entrenamiento"]
    injected_missing=sum(v["nulos_inyectados"] for v in noise.values())
    injected_sentinel=sum(v["centinelas_inyectados"] for v in noise.values())
    prevalence=audit["clases"]
    parts=d["particiones"]
    nv=sum(parts["validacion"]); nt=sum(parts["prueba"]); ntr=sum(parts["entrenamiento"])
    spec=next(v for v in channels if v["canal"] in ("Social_Ads","Paid_Search","Organico"))
    focus=[v for v in channels if v["canal"] in ("Social_Ads","Paid_Search","Organico")]
    focus_string="; ".join(f'{names.get(v["canal"],v["canal"])}: {number(v["fn"])} de {number(v["positivos"])} ({pct(v["tasa"],2)}; IC 95 % {pct(v["IC95"][0],2)} a {pct(v["IC95"][1],2)})' for v in focus)
    alt_channel=channels[0]
    allmax=("También ocupa el primer lugar al incluir los demás canales." if alt_channel["canal"]==spec["canal"] else
            f'Entre todos los canales, {names.get(alt_channel["canal"],alt_channel["canal"])} alcanza {pct(alt_channel["tasa"],2)}.')
    most_fn=max(channels,key=lambda a:a["fn"])
    overlap=any(spec["IC95"][0] <= q["IC95"][1] and q["IC95"][0] <= spec["IC95"][1] for q in focus[1:])
    ranking="; ".join(f'{FEATURES[v["variable"]]} {number(v["auc"],3)}' for v in a)
    detail=a[0]
    top=FEATURES[detail["variable"]]
    exemplar_display=lambda v: pct(v) if detail["variable"] in ["tasa_rebote","tasa_devoluciones"] else number(v,2)
    test_macro={k:v["prueba"]["macro_f1"] for k,v in models.items()}
    test_description="; ".join(f'{k} {number(test_macro[k],4)}' for k in ["Logística","XGBoost","Red neuronal"])
    variances=d["pca_varianza"]
    b2=[]
    for k in ["Logística","XGBoost"]:
        base=test_macro[k]; reduced=test_macro[k+" PCA"]; diff=base-reduced
        description=f"cayó {number(abs(100*diff),2)}" if diff>=0 else f"aumentó {number(abs(100*diff),2)}"
        b2.append(f"{k} pasó de {number(base,4)} a {number(reduced,4)} de macro F1 en prueba. Su puntuación {description} puntos porcentuales.")
    train_winner=d["seleccion"]
    modelstr="; ".join(f'{k} {number(models[k]["validacion"]["macro_f1"],4)}' for k in ["Logística","XGBoost","Red neuronal"])
    xgb=models["XGBoost"]["ajuste"]
    report={
        "titulo":"Abandono de clientes de comercio electrónico",
        "introduccion":"Se evaluó si los patrones de navegación y compra identifican a usuarios con riesgo bajo (0), medio (1) o alto (2). Cada fila analítica representa un usuario; sus sesiones y pedidos se resumen antes del cruce.",
        "etl":[
            "Se abrieron las tres tablas en modo de solo lectura. Usuarios pasó de 804 000 a 800 000 filas, sesiones web de 3 060 000 a 3 000 000 y órdenes de 1 530 000 a 1 500 000. Las filas repetidas eran copias completas. Se verificaron llaves únicas y cero relaciones huérfanas.",
            f"Con los duplicados ya retirados, quedaron fuera del cálculo {checked('duracion_seg')} duraciones, {checked('paginas_vistas')} conteos de páginas y {checked('total_pagado')} pagos con valores imposibles o centinelas. Los códigos −999, −50 y 999999,9 aparecen en la base. Se respetaron los datos ausentes y se normalizaron espacios y mayúsculas en canal y sistema operativo.",
            f"Las sesiones y los pedidos se agregaron por separado para evitar multiplicar filas. Se conservaron los 800 000 usuarios, incluidos {number(audit['sin_sesiones'])} sin sesiones y {number(audit['sin_pedidos'])} sin pedidos. El número de eventos ausentes vale cero; las medias o tasas sin denominador se imputan con la mediana de entrenamiento.",
            f"En una copia del entrenamiento se alteró el 2 % de celdas observadas en cuatro predictores: {number(injected_missing)} ausentes y {number(injected_sentinel)} centinelas. Tras detectar estos últimos, ambos tipos se imputaron con medianas aprendidas en entrenamiento. Las etiquetas y las particiones de evaluación conservaron sus valores originales."],
        "variables":[
            f"Etiqueta ordinal de riesgo en toda la base: bajo {number(prevalence[0])}, medio {number(prevalence[1])} y alto {number(prevalence[2])}. La clase alta representa el {pct(prevalence[2]/sum(prevalence))}.",
            "Se usaron diez predictores: número de sesiones, tasa de rebote, duración media en segundos, páginas por sesión, fracción móvil, número de pedidos, pago medio, días medios de envío, tasa de devoluciones y valoración media. Las tasas están entre 0 y 1 y las estrellas observadas entre 1 y 5.",
            "Entre los usuarios con datos válidos, la mediana de duración media fue 233,5 segundos (cuartiles 172,7 y 302), la de rebote fue 25 % (0 % y 50 %) y la de envío fue 3 días (2 y 4,75). Al menos tres de cada cuatro usuarios con pedidos tenían una tasa de devolución de cero. En los 800 000 usuarios hubo una mediana de cuatro sesiones y dos pedidos.",
            "La fracción móvil usa sesiones con sistema identificado. Los demás promedios utilizan eventos con un valor válido. La edad, Prime, las claves y el canal se excluyeron de los predictores para estudiar comportamiento; el canal se conservó como variable nominal de auditoría."],
        "entrenamiento":[
            f"Se extrajo una muestra estratificada reproducible por usuario (semilla 2026). Se asignaron {number(ntr)} usuarios al entrenamiento, {number(nv)} a validación y {number(nt)} a prueba. Los demás solo participaron en la caracterización. Los tres conjuntos son disjuntos. Imputación, escalado y PCA se ajustaron únicamente en entrenamiento.",
            f"La logística tuvo un techo de 250 iteraciones y convergió en {models['Logística']['ajuste']['iteraciones']}. XGBoost usó profundidad 4, tasa 0,05 y techo de 800 rondas, con tres árboles por ronda (una clase por árbol). Parada temprana tras 25 rondas sin mejorar la pérdida de validación. Para las diez variables eligió la ronda {xgb['ronda_elegida']} de {xgb['rondas_ejecutadas']} ejecutadas.",
            f"La red con capas ocultas 24 y 12, ReLU y Adam entrenó {net['epocas_ejecutadas']} épocas por retropropagación. Se conservaron los pesos de la época {net['epoca_elegida']}, con la menor pérdida de validación observada. Se fijaron un mínimo de 50, un techo de 120 y paciencia de 12.",
            f"Macro F1 de validación: {modelstr}. Se seleccionó {train_winner} por ese criterio entre los tres modelos completos. Macro F1 en prueba: {test_description}. La regla de asignar siempre riesgo bajo obtuvo {number(d['base_mayoritaria_macro_f1'],4)}."],
        "preguntas":[
            {"titulo":"B1. Navegación y logística en riesgo medio y alto","respuesta":[
                f"Se compararon {number(a[0]['n_medio'])} usuarios de riesgo medio y {number(a[0]['n_alto'])} de riesgo alto de prueba. Para cada variable se fijó la dirección en entrenamiento. La AUC ROC mide con qué frecuencia ordena un usuario alto por encima de uno medio, con medio crédito a empates. Resultados: {ranking}.",
                f"El promedio de días de envío fue el indicador con mayor separación. Entre quienes tienen una observación válida, fue de {exemplar_display(detail['media_1'])} días en riesgo medio y {exemplar_display(detail['media_2'])} en alto. A mayor riesgo se observaron sesiones más cortas y más rebotes y devoluciones. Faltan la fecha del evento y una segunda etiqueta por usuario para comprobar una escalada posterior."],
             "figura":"resultados/b1_navegacion_logistica.png","pie":"Poder discriminante de cuatro marcadores sobre los mismos usuarios de prueba de clases 1 y 2. La línea vertical en 0,50 indica ordenación al azar."},
            {"titulo":"B2. Cinco componentes y rendimiento predictivo","respuesta":[
                f"El PCA de las diez variables estandarizadas conservó {pct(sum(variances),2)} de la varianza de entrenamiento en cinco componentes. Las contribuciones individuales, en orden, fueron {', '.join(pct(v,2) for v in variances)}.",
                *b2,
                "Cada modelo se entrenó de nuevo con cinco componentes; ambas versiones de cada algoritmo comparten usuarios y evaluación. El macro F1 pondera por igual bajo, medio y alto. Conservar variación general no implica conservar toda la información de la etiqueta."],
             "figura":"resultados/b2_macro_f1.png","pie":"Macro F1 de logística y XGBoost en la misma prueba, con diez variables o cinco componentes. Una puntuación mayor indica mejor balance entre precisión y sensibilidad por clase."},
            {"titulo":"B3. Falsos negativos por canal de adquisición","respuesta":[
                f"El modelo elegido en validación ({train_winner.lower()}) asigna la clase de mayor probabilidad. Una omisión es un riesgo alto clasificado como bajo o medio. La tasa divide las omisiones entre todos los altos reales de cada canal. {focus_string}.",
                f"Entre los tres canales solicitados, {names[spec['canal']]} presentó la tasa puntual más elevada. {allmax} En cantidad de casos, {names.get(most_fn['canal'],most_fn['canal'])} acumuló más omisiones ({number(most_fn['fn'])}). "+
                ("Los intervalos de los tres canales de interés se solapan al menos en parte. Conviene revisar omisiones de forma prioritaria y medir una regla piloto antes de justificar un tratamiento distinto por canal." if overlap else "La diferencia observada justifica revisar ese canal como prioridad; una regla de retención debe evaluarse en una cohorte posterior.")],
             "figura":"resultados/b3_canales.png","pie":"Tasa de falsos negativos de la clase alta e intervalos de Wilson al 95 %. El eje horizontal muestra el rango observado. El color verde identifica los tres canales de la pregunta."}],
        "limitaciones":["La base no incluye fechas, coste de retención, abandono observado posterior ni horizonte de anticipación. Se evalúa concordancia con la etiqueta disponible. Para estudiar transiciones habría que enlazar historiales fechados con una etiqueta posterior.",
                         "Los modelos comparan muestras estratificadas de usuarios de la misma base. Las métricas estiman rendimiento en ese contexto y requieren reevaluación con una cohorte temporal externa antes de orientar decisiones operativas."],
        "figura_etl":"resultados/etl_calidad.png",
        "pie_etl":"Fracción ausente y fracción inválida de cinco variables, calculadas en las tablas sin duplicados antes de imputar.",
        "figura_entrenamiento":"resultados/entrenamiento_red.png",
        "pie_entrenamiento":"Pérdida de red en entrenamiento y validación por época. La línea vertical señala los pesos restaurados."}
    (root/"informe.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__=="__main__":
    root=Path(__file__).resolve().parent
    make_report(json.loads((root/"resultados/metricas.json").read_text(encoding="utf-8")),root)
