"""Caso A. Ejecutar py -3.11 fraude/entrenar.py --datos C:/ruta/a/las/bases."""
import os
for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, log_loss,
                            precision_recall_curve, roc_curve)
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from datos import leer, preparar

BASE = Path(__file__).resolve().parent
SEED = 2026


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,allow_nan=False),encoding="utf-8")


def red(x,y,v,z,ruido=False,epochs=80):
    target=y.copy()
    if ruido:
        rng=np.random.default_rng(SEED+1)
        flipped=rng.choice(len(y),round(.015*len(y)),replace=False)
        target[flipped]=1-target[flipped]
    net=MLPClassifier(hidden_layer_sizes=(48,24),activation="relu",solver="adam",batch_size=512,
                      learning_rate_init=.001,alpha=.001,shuffle=True,
                      random_state=np.random.RandomState(SEED))
    history=[]; best=None; best_val=np.inf
    # Se calcula BCE al terminar la época con los mismos pesos, sin penalización L2.
    for epoch in range(1,epochs+1):
        net.partial_fit(x,target,classes=[0,1])
        loss=log_loss(target,net.predict_proba(x)[:,1],labels=[0,1])
        loss_val=log_loss(z,net.predict_proba(v)[:,1],labels=[0,1])
        history.append({"epoca":epoch,"train_loss":float(loss),"val_loss":float(loss_val)})
        if loss_val<best_val:
            best,best_val,best_epoch=deepcopy(net),loss_val,epoch
        if epoch%20==0:
            print(f"MLP ruido adicional={ruido}, época {epoch}/{epochs}",flush=True)
    # Primera secuencia de tres épocas: val supera en 0.0005 su mejor referencia
    # anterior al bloque; simultáneamente train queda por debajo de esa misma época.
    dv=None
    for i in range(1,len(history)-2):
        ref=min(history[:i],key=lambda r:r["val_loss"])
        if all(h["val_loss"]>ref["val_loss"]+.0005 and h["train_loss"]<ref["train_loss"]
               for h in history[i:i+3]):
            dv={"inicio":i+1,"referencia":ref["epoca"],"confirmacion":i+3}; break
    return best,{"epocas":epochs,"epoca_optima":best_epoch,"min_val_loss":float(best_val),"divergencia":dv,
                 "etiquetas_invertidas_adicionales":int(np.sum(target!=y)),"historia":history}


def score(y,p,t):
    tn,fp,fn,tp=map(int,confusion_matrix(y,p>=t,labels=[0,1]).ravel())
    return {"umbral":float(t),"TN":tn,"FP":fp,"FN":fn,"TP":tp,"recall":tp/(tp+fn),
            "fpr":fp/(fp+tn),"precision":tp/(tp+fp) if tp+fp else None}


def umbrales(y,p):
    fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False)
    eligible=np.flatnonzero((fpr<.015)&np.isfinite(thresholds))
    if len(eligible):
        maxrec=np.max(tpr[eligible])
        index=eligible[np.flatnonzero(tpr[eligible]==maxrec)[0]]
        target=float(thresholds[index])
    else:
        target=float(np.nextafter(1.,2.))
    eighty=float(thresholds[np.flatnonzero(tpr>=.80)[0]])
    return target,eighty


def compacto_pr(pre,rec):
    # Conservar puntos reales. Se recorre el eje de recall a intervalos regulares.
    ix=np.clip(np.searchsorted(-rec,-np.linspace(1,0,251)),0,len(rec)-1)
    return [{"recall":float(rec[i]),"precision":float(pre[i])} for i in ix]


def remuestreo_ap(y, probabilidades, fechas, repeticiones=1000, dias_bloque=7):
    """Bootstrap temporal pareado. Recalcula la AP completa en cada réplica."""
    fechas=fechas.to_numpy(dtype="datetime64[D]")
    indice_dia=(fechas-fechas.min()).astype(int)
    n_dias=int(indice_dia.max()+1)
    n_bloques=int(np.ceil(n_dias/dias_bloque))
    rankings={}
    def ap_ponderada(datos,peso_dia):
        dias,clases,cortes=datos
        w=peso_dia[dias]
        pos=np.add.reduceat(w*clases,cortes)
        total=np.cumsum(np.add.reduceat(w,cortes))
        pre=np.divide(np.cumsum(pos),total,out=np.zeros(len(total),dtype=float),where=total>0)
        return float(np.dot(pos,pre)/pos.sum())
    for name,p in probabilidades.items():
        ix=np.argsort(-p,kind="stable")
        scores=p[ix]
        bordes=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
        rankings[name]=(indice_dia[ix],y[ix],bordes)
        # Tratar empates como un único umbral reproduce sklearn exactamente.
        assert np.isclose(ap_ponderada(rankings[name],np.ones(n_dias,dtype=int)),
                          average_precision_score(y,p),atol=1e-12,rtol=0)
    rng=np.random.default_rng(SEED+201)
    differences={name:[] for name in rankings if name!="Logística"}
    for _ in range(repeticiones):
        inicios=rng.integers(0,n_dias,size=n_bloques)
        # Bootstrap móvil circular: todos los días tienen igual probabilidad
        # marginal, incluidos los extremos del periodo.
        extraidos=((inicios[:,None]+np.arange(dias_bloque))%n_dias).ravel()[:n_dias]
        pesos=np.bincount(extraidos,minlength=n_dias)
        values={k:ap_ponderada(rank,pesos) for k,rank in rankings.items()}
        for name,serie in differences.items(): serie.append(values[name]-values["Logística"])
    return {"metodo":"Bootstrap móvil circular de días, pareado entre modelos, intervalo percentil",
            "dias_prueba":n_dias,"longitud_bloque_dias":dias_bloque,"repeticiones":repeticiones,
            "semilla":SEED+201,"nivel":.95,
            "IC_delta_AP_vs_logistica":{k:list(map(float,np.quantile(values,[.025,.975])))
                                        for k,values in differences.items()}}


def interacciones(df):
    tr=df.loc[df.periodo.eq("train")]
    test=df.loc[df.periodo.eq("test")]
    q=tr.groupby("moneda",observed=True).monto.quantile(.9)
    rates=tr.loc[tr.categoria_mcc.ne("desconocido")].groupby("categoria_mcc",observed=True).is_fraud.agg(["mean","size"])
    high=rates.loc[rates["size"].ge(500)].sort_values("mean",ascending=False).head(2).index.tolist()
    amount=test.monto.ge(test.moneda.astype("string").map(q).astype(float)) & test.monto.notna() & test.moneda.ne("desconocido")
    night=test.timestamp.dt.hour.lt(6)
    cat=test.categoria_mcc.isin(high)
    desc={}
    for label,mask in {"Prueba completa":np.ones(len(test),dtype=bool),"Monto alto":amount,
                       "Madrugada":night,"MCC de mayor tasa":cat,"Tres condiciones":amount&night&cat,
                       "Monto alto y madrugada":amount&night}.items():
        n=int(mask.sum()); positives=int(test.loc[mask,"is_fraud"].sum())
        desc[label]={"n":n,"positivos":positives,"tasa":(positives/n if n else None)}
    complete=test.monto.notna() & test.moneda.ne("desconocido") & test.categoria_mcc.isin(high)
    cells={}
    for a in [False,True]:
        for n in [False,True]:
            mask=complete&amount.eq(a)&night.eq(n)
            cells[f"alto{int(a)}_noche{int(n)}"]={"n":int(mask.sum()),"tasa":float(test.loc[mask,"is_fraud"].mean())}
    effect=(cells["alto1_noche1"]["tasa"]-cells["alto0_noche1"]["tasa"]-
            cells["alto1_noche0"]["tasa"]+cells["alto0_noche0"]["tasa"])
    return {"categorias_mayor_tasa_train":high,"p90_monto_train_por_moneda":{str(k):float(v) for k,v in q.items()},
            "grupos_prueba":desc,"celdas_en_mcc_alto":cells,"contraste_interaccion_absoluta":effect}


def entrenar(args):
    args.salida.mkdir(exist_ok=True,parents=True)
    database=args.datos/"fintech_fraud.db"
    df,audit=leer(database,args.por_mil)
    x,y,prep=preparar(df)
    subsets={name:df.periodo.eq(name).to_numpy() for name in ["train","val","test"]}
    train,val,test=[subsets[k] for k in ["train","val","test"]]
    xt,xv=x[train],x[val]; yt,yv=y[train],y[val]
    dist={}
    for key,mask in subsets.items():
        frame=df.loc[mask]; n=int(mask.sum())
        dist[key]={"n":n,"positivos":int(y[mask].sum()),"negativos":int(n-y[mask].sum()),
                   "inicio":str(frame.timestamp.min()),"fin":str(frame.timestamp.max()),"tasa":float(y[mask].mean())}
    assert df.loc[train,"timestamp"].max()<df.loc[val,"timestamp"].min()<df.loc[test,"timestamp"].min()
    print("Partición:",dist,flush=True)
    model_lr=LogisticRegression(max_iter=250,C=1,solver="lbfgs",tol=1e-5,random_state=SEED)
    model_lr.fit(xt,yt)
    params=dict(n_estimators=850,learning_rate=.05,tree_method="hist",max_bin=128,min_child_weight=10,
                subsample=.9,colsample_bytree=1,n_jobs=1,early_stopping_rounds=40,
                eval_metric="logloss",random_state=SEED,objective="binary:logistic")
    boosted={}
    for depth in [4,1]:
        model=XGBClassifier(**params,max_depth=depth)
        model.fit(xt,yt,eval_set=[(xv,yv)],verbose=False)
        boosted[depth]=model
        print("XGB profundidad",depth,"mejor ronda",model.best_iteration+1,flush=True)
    nn,history=red(xt,yt,xv,yv,epochs=args.epocas)
    _,noise=red(xt,yt,xv,yv,ruido=True,epochs=args.epocas)
    modelos={"Logística":model_lr,"XGBoost":boosted[4],"Red neuronal":nn}
    pred_val={k:m.predict_proba(xv)[:,1] for k,m in modelos.items()}
    aps={k:float(average_precision_score(yv,p)) for k,p in pred_val.items()}
    winner=max(aps,key=aps.get)
    thresholds={k:umbrales(yv,p) for k,p in pred_val.items()}  # congelados antes de prueba
    # La prueba se abre al terminar selección y elección de cortes.
    xe,ye=x[test],y[test]
    results={}; probabilidades={}
    summaries=[]
    for name,m in modelos.items():
        p=m.predict_proba(xe)[:,1]; v=pred_val[name]; a,b=thresholds[name]
        probabilidades[name]=p
        precision,recall,_=precision_recall_curve(ye,p,drop_intermediate=False)
        results[name]={"ap_validacion":aps[name],"ap_prueba":float(average_precision_score(ye,p)),
                       "pr_auc_trapezoidal_prueba":float(auc(recall,precision)),
                       "umbrales_distintos_curva_pr":int(len(recall)-1),
                       "bce_prueba":float(log_loss(ye,p)),
                       "curva_pr":compacto_pr(precision,recall),"umbrales":{}}
        for k,u in {"operativo":a,"recall80":b,"ejemplo_065":.65,"ejemplo_082":.82}.items():
            results[name]["umbrales"][k]={"val":score(yv,v,u),"test":score(ye,p,u)}
        # Frontera completa resumida en validación, solo para presentar la zona operativa.
        f,t,_=roc_curve(yv,v)
        for lim in np.linspace(0,.05,101):
            summaries.append({"modelo":name,"limite_fpr":float(lim),"recall":float(t[f<=lim].max())})
    stump=boosted[1]
    ablation={"ap_validacion":float(average_precision_score(yv,stump.predict_proba(xv)[:,1])),
              "ap_prueba":float(average_precision_score(ye,stump.predict_proba(xe)[:,1])),
              "rondas":int(stump.best_iteration+1)}
    # Resultado agregado, sin identificadores ni ejemplos por persona.
    with database.open("rb") as fh: source_hash=hashlib.file_digest(fh,"sha256").hexdigest()
    metrics={"SHA256":source_hash,"por_mil":args.por_mil,"auditoria":audit,"preparacion":prep,
             "distribucion":dist,"semilla":SEED,"logistica_iteraciones":int(model_lr.n_iter_[0]),
             "xgb_rondas":int(boosted[4].best_iteration+1),
             "xgb_rondas_entrenadas":int(len(boosted[4].evals_result()["validation_0"]["logloss"])),
             "modelos":results,"seleccion_validacion":winner,"tocones":ablation,
             "interacciones":interacciones(df),"red":history,"red_ruido_extra":noise,
             "remuestreo_AP":remuestreo_ap(ye,probabilidades,df.loc[test,"timestamp"])}
    json_file(args.salida/"metricas.json",metrics)
    pd.DataFrame(summaries).to_csv(args.salida/"frontera_validacion.csv",index=False)
    pd.DataFrame(history["historia"]).merge(pd.DataFrame(noise["historia"]),on="epoca",
            suffixes=("_observada","_ruido_extra")).to_csv(args.salida/"epocas_red.csv",index=False)
    from informe import elaborar
    elaborar(metrics,args.salida)
    print("Resultados terminados.",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos",type=Path,default=Path.home()/"Downloads")
    ap.add_argument("--salida",type=Path,default=BASE/"resultados")
    ap.add_argument("--por-mil",type=int,default=50,help="Fracción determinista de claves, por millar")
    ap.add_argument("--epocas",type=int,default=80)
    args=ap.parse_args()
    if not (0<args.por_mil<=1000) or args.epocas<50:
        ap.error("Usar de 1 a 1000 por mil y al menos 50 épocas")
    with threadpool_limits(limits=1): entrenar(args)
