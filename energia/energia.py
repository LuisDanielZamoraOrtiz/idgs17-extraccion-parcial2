"""Pronóstico horario y diagnóstico de carga. Ejecutar desde cualquier carpeta."""
import os
for clave in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[clave]="1"
import argparse
import copy
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, log_loss,
                             mean_absolute_error, precision_recall_curve, r2_score, root_mean_squared_error)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from preparacion import SEMILLA, XCOL, PHYS, preparar

BASE=Path(__file__).resolve().parent
OUT=BASE/"resultados"
N=len(XCOL)
LAB=["Nominal","Elevada","Crítica relativa"]
AZUL="#216A81"
NARANJA="#CC7033"
VERDE="#467A5B"


def grabar(path, obj):
    def nativo(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=nativo,allow_nan=False),encoding="utf8")


def porcion(a):
    fis=a[:,N:N+4]
    m=a[:,N+4:]
    return a[:,:N],fis,{"y":m[:,0].astype(int),"kwh":m[:,1],"industrial":np.isin(m[:,2],[0,3]),
                         "zona":m[:,2].astype(int),"antigua":m[:,3]<2005,"estacion":m[:,4].astype(int),
                         "hora":m[:,5].astype(int),"temp_evento":m[:,6],"clase_previa":m[:,7].astype(int)}


def clasificacion(y, pred):
    return {"macro_f1":f1_score(y,pred,average="macro",labels=[0,1,2],zero_division=0),
            "exactitud":accuracy_score(y,pred),"confusion":confusion_matrix(y,pred,labels=[0,1,2])}


def alarma(y,pred):
    if len(y)==0:
        return {"n":0,"criticas":0,"alertas":0,"VP":0,"FP":0,"FN":0,"recall":None,"precision":None}
    real=y==2
    tp=int((real&pred).sum())
    return {"n":len(y),"criticas":int(real.sum()),"alertas":int(pred.sum()),"VP":tp,
            "FP":int((~real&pred).sum()),"FN":int((real&~pred).sum()),
            "recall":tp/int(real.sum()) if real.any() else None,
            "precision":tp/int(pred.sum()) if pred.any() else None}


def porcent(x):
    return "sin denominador" if x is None else f"{100*x:.1f} %".replace(".",",")


def num(x,d=0):
    return f"{x:,.{d}f}".replace(","," ").replace(".",",")


def regresion(real,pred):
    return {"MAE":mean_absolute_error(real,pred),"RMSE":root_mean_squared_error(real,pred),"R2":r2_score(real,pred)}


def barras(ax,grupo,series,etiquetas,colores):
    ind=np.arange(len(grupo)); w=.34
    for i,(serie,eti,col) in enumerate(zip(series,etiquetas,colores)):
        barras=ax.bar(ind+(i-.5)*w,serie,width=w,label=eti,color=col)
        ax.bar_label(barras,fmt="%.1f",fontsize=10,padding=3)
    ax.set_xticks(ind,grupo);ax.set_ylim(0, min(105,max(8,max(max(v) for v in series)*1.25+2)))
    ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.11),ncol=2)


def describir(cache,r):
    x,_,meta=porcion(np.load(cache/"train.npy",mmap_mode="r"))
    def resumen(v):
        datos=v[np.isfinite(v)]
        q1,mediana,q3=np.quantile(datos,[.25,.5,.75])
        return {"presentes":len(datos),"faltantes":len(v)-len(datos),"q1":q1,"mediana":mediana,"q3":q3}
    r["distribuciones_train"]={col:resumen(x[:,i]) for i,col in enumerate(XCOL[:7])}
    kwh=meta["kwh"]
    h,bins=np.histogram(kwh,bins=40)
    r["histograma_objetivo"]={"conteos":h,"bordes":bins,"mediana":float(np.median(kwh))}


def graficar(res):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    def fig():
        return plt.subplots(figsize=(8.3,4.3),layout="constrained")
    def guardar(g,nombre):
        g.savefig(OUT/nombre,dpi=155,facecolor="white");plt.close(g)
    if "histograma_objetivo" in res:
        g,ax=fig(); hist=res["histograma_objetivo"]
        bordes=np.asarray(hist["bordes"]); ax.bar(bordes[:-1],hist["conteos"],width=np.diff(bordes),align="edge",color=AZUL,edgecolor="white",linewidth=.6)
        ax.axvline(hist["mediana"],c=NARANJA,ls="--",lw=2,label=f"Mediana: {num(hist['mediana'],1)} kWh")
        ax.set(xlabel="Consumo medio por lectura de la hora objetivo (kWh)",ylabel="Pares de entrenamiento")
        ax.legend(frameon=False); guardar(g,"distribucion_consumo.png")
    g,ax=fig()
    curvas=res["historia_red"]
    ax.plot(curvas["train"],color=AZUL,label="Entrenamiento");ax.plot(curvas["val"],color=NARANJA,label="Validación")
    # El índice de arrays empieza en cero; la etiqueta externa cuenta épocas desde 1.
    ax.set_xticks([0,29,59,89,119,len(curvas["val"])-1],[1,30,60,90,120,len(curvas["val"])])
    ax.axvline(res["epoca_optima"]-1,color=VERDE,ls="--",label=f"Elegida: {res['epoca_optima']}")
    ax.set(xlabel="Época de la red",ylabel="Pérdida logarítmica")
    ax.legend(frameon=False,ncol=3,loc="upper center");guardar(g,"convergencia.png")
    g,ax=fig(); casos=[res["calor"][s] for s in ["argmax","operativa"]]
    items=[r["recall"] for r in casos]; prec=[r["precision"] for r in casos]
    x=np.arange(2)
    for i,(val,l,c) in enumerate([(items,"Sensibilidad",AZUL),(prec,"Precisión",NARANJA)]):
        for j,v in enumerate(val):
            pos=j+(i-.5)*.32
            if v is None:
                ax.text(pos,3,"n/d",ha="center",fontsize=10)
            else:
                ax.bar(pos,100*v,.30,color=c);ax.text(pos,100*v+1,num(v*100,1)+" %",ha="center",fontsize=10)
    ax.set_xticks(x,["Clase más probable","Umbral operativo"]);ax.set(ylim=(0,105),ylabel="Porcentaje en prueba")
    ax.legend(handles=[Patch(facecolor=AZUL,label="Sensibilidad"),Patch(facecolor=NARANJA,label="Precisión")],
              frameon=False,loc="upper left");guardar(g,"calor.png")
    g,ax=fig(); pc=res["pca"]
    cargas=pc["correlaciones_eje"];nom=["Voltaje","Corriente","Factor de potencia","Temperatura"]
    ax.barh(nom,cargas,color=[AZUL if v>=0 else NARANJA for v in cargas])
    for i,v in enumerate(cargas):
        ax.text(v+(.025 if v>=0 else -.025),i,num(v,2),va="center",ha="left" if v>=0 else "right")
    ax.axvline(0,color="#455963",linewidth=.8);ax.set(xlim=(-1.18,1.18),xlabel=f"Correlación con PC{pc['eje']+1} en entrenamiento");ax.invert_yaxis()
    guardar(g,"cargas_pca.png")
    g,ax=fig();gpos=[res["infraestructura"][s] for s in ["antes_2005","desde_2005"]]
    barras(ax,["Antes de 2005","Desde 2005"],[[100*r["error"] for r in gpos],[100*r["tasa_critica"] for r in gpos]],["Error de clasificación","Horas críticas relativas"],[AZUL,NARANJA])
    ax.set_ylabel("Porcentaje de horas evaluadas");guardar(g,"infraestructura.png")


def calcular(cache):
    OUT.mkdir(exist_ok=True)
    a=json.loads((cache/"etl.json").read_text("utf8"))
    bloques=[porcion(np.load(cache/(s+".npy"),mmap_mode="r")) for s in ["train","val","test"]]
    (xt,ft,mt),(xv,fv,mv),(xs,fs,ms)=bloques
    im=SimpleImputer(strategy="median").fit(xt)
    z=StandardScaler().fit(im.transform(xt))
    def preparar_x(x):return z.transform(im.transform(x)).astype("float32")
    xt,xv,xs=map(preparar_x,[xt,xv,xs])
    yt,yv,ys=mt["y"],mv["y"],ms["y"]
    r={"etl":a,"clases":{k:np.bincount(m["y"],minlength=3) for k,m in zip(["train","val","test"],[mt,mv,ms])}}
    def ajuste_rl(X):
        return LogisticRegression(max_iter=250,tol=1e-4,solver="lbfgs").fit(X,yt)
    lr=ajuste_rl(xt)
    xg=XGBClassifier(n_estimators=850,learning_rate=.05,max_depth=4,min_child_weight=8,
                     reg_lambda=2,objective="multi:softprob",num_class=3,eval_metric="mlogloss",device="cpu",
                     early_stopping_rounds=35,n_jobs=1,random_state=SEMILLA,tree_method="hist")
    xg.fit(xt,yt,eval_set=[(xv,yv)],verbose=False)
    print("XGBoost terminado",xg.best_iteration+1,flush=True)
    net=MLPClassifier(hidden_layer_sizes=(24,12),activation="relu",solver="adam",alpha=.002,
                     batch_size=512,learning_rate_init=.0008,random_state=np.random.RandomState(SEMILLA))
    ht,hv=[],[]; best_loss=np.inf
    for ep in range(1,131):
        net.partial_fit(xt,yt,classes=np.array([0,1,2]))
        ht.append(float(log_loss(yt,net.predict_proba(xt),labels=[0,1,2])))
        hv.append(float(log_loss(yv,net.predict_proba(xv),labels=[0,1,2])))
        if hv[-1]<best_loss:
            best_loss=hv[-1]; best=copy.deepcopy(net);best_ep=ep
        if ep % 25==0:print("Red: época",ep,flush=True)
    r.update(epoca_optima=best_ep,historia_red={"train":ht,"val":hv},iteraciones_rl=int(lr.n_iter_[0]),
             mejor_xgb=xg.best_iteration+1,historia_xgb=xg.evals_result()["validation_0"]["mlogloss"])
    r["modelos"]={}
    for nombre,mdl in [("Logística",lr),("XGBoost",xg),("Red",best)]:
        p=mdl.predict_proba(xs)
        r["modelos"][nombre]={**clasificacion(ys,p.argmax(1)), "PR_AP_critica":average_precision_score(ys==2,p[:,2])}
    r["persistencia_clase"]=clasificacion(ys,ms["clase_previa"])
    r["nominal_constante"]=clasificacion(ys,np.zeros_like(ys))
    rng=np.random.default_rng(SEMILLA); copia=xt.copy()
    # Ruido exclusivamente sobre 2 % de celdas de los siete sensores ya estandarizados.
    alteracion=rng.random((len(xt),7))<.02
    copia[:,:7]+=np.where(alteracion,rng.normal(0,.5,alteracion.shape),0).astype("float32")
    lr_ruido=ajuste_rl(copia)
    r["ruido"]={"celdas":int(alteracion.sum()),"total":int(alteracion.size),"escala_sd":.5,
                "macro_f1":f1_score(ys,lr_ruido.predict(xs),average="macro")}
    rd=Ridge(alpha=10).fit(xt,mt["kwh"])
    r["demanda"]={"ridge":regresion(ms["kwh"],rd.predict(xs)),"persistencia":regresion(ms["kwh"],bloques[2][0][:,3]),
                   "media_entrenamiento":regresion(ms["kwh"],np.full(len(ys),mt["kwh"].mean()))}
    pv=best.predict_proba(xv)[:,2]; ps=best.predict_proba(xs)
    hotval=mv["industrial"]&(mv["temp_evento"]>30)
    filtro_seleccion="industrial >30 °C"
    if np.sum(hotval&(yv==2))<20:
        hotval=np.ones_like(hotval);filtro_seleccion="toda la validación por soporte térmico insuficiente"
    pr,re,ths=precision_recall_curve(yv[hotval]==2,pv[hotval]);f1=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-12)
    # En un empate se usa el umbral más alto.
    ix=np.flatnonzero(np.isclose(f1,f1.max(),rtol=0,atol=1e-12))[-1];thr=float(ths[ix])
    hot=ms["industrial"]&(ms["temp_evento"]>30)
    caliente120=ms["industrial"]&(ms["temp_evento"]>(120-32)*5/9)
    pred=ps.argmax(1)
    serie_temp=ms["temp_evento"][ms["industrial"] & np.isfinite(ms["temp_evento"])]
    r["calor"]={"umbral":thr,"seleccion":filtro_seleccion,"n_seleccion":int(hotval.sum()),
                "validacion":alarma(yv[hotval],pv[hotval]>=thr),"max_temp_industrial_prueba":float(serie_temp.max()),
                "sin_temp_industrial_prueba":int((ms["industrial"] & ~np.isfinite(ms["temp_evento"])).sum()),
                "argmax":alarma(ys[hot],pred[hot]==2),"operativa":alarma(ys[hot],ps[hot,2]>=thr),
                "a120F":alarma(ys[caliente120],ps[caliente120,2]>=thr)}
    for k,zi in [("bajio",0),("norte",3)]:
        w=hot&(ms["zona"]==zi);r["calor"][k]=alarma(ys[w],ps[w,2]>=thr)
    # PCA descriptivo contemporáneo. Nunca se incorpora a los predictores de anticipación.
    imp=SimpleImputer(strategy="median").fit(ft)
    esc=StandardScaler().fit(imp.transform(ft))
    train_f=esc.transform(imp.transform(ft));test_f=esc.transform(imp.transform(fs))
    pca=PCA(n_components=4,svd_solver="full").fit(train_f)
    train_pc=pca.transform(train_f);test_pc=pca.transform(test_f)
    # Diferencia de centros normalizada por desviación conjunta (ponderada por tamaños).
    def efectos(coord,y):
        n0,n1=(y==0).sum(),(y==1).sum()
        return (coord[y==1].mean(0)-coord[y==0].mean(0))/np.sqrt(
            ((n0-1)*coord[y==0].var(0,ddof=1)+(n1-1)*coord[y==1].var(0,ddof=1))/(n0+n1-2))
    et=efectos(train_pc,yt);e=int(np.abs(et).argmax());signo=1 if et[e]>=0 else -1
    r["pca"]={"eje":e,"varianza":pca.explained_variance_ratio_, "efectos_train":et,
              "efecto_test":float(efectos(test_pc,ys)[e]*signo),"pesos_eje":pca.components_[e]*signo,
              "correlaciones_eje":np.corrcoef(train_f.T,(train_pc[:,e]*signo).T)[-1,:4],
              "centros_test":[float((test_pc[ys==i,e]*signo).mean()) for i in [0,1]],
              "filas_completas_train":int(np.isfinite(ft).all(1).sum()),"filas_completas_test":int(np.isfinite(fs).all(1).sum())}
    r["infraestructura"]={}
    def grupo(mask):
        ids=ms["estacion"][mask]; i,c=np.unique(ids,return_counts=True)
        cnt=np.column_stack([c,[int((mask&(ms["estacion"]==j)&(ys!=pred)).sum()) for j in i],
                              [int((mask&(ms["estacion"]==j)&(ys==2)).sum()) for j in i]])
        suma=cnt.sum(0)
        return {"n":suma[0],"errores":suma[1],"criticas":suma[2],"estaciones":len(i),
                "error":suma[1]/suma[0],"tasa_critica":suma[2]/suma[0]},cnt
    old,co=grupo(ms["antigua"]);new,cn=grupo(~ms["antigua"])
    boot=np.random.default_rng(SEMILLA)
    def tasas(c):
        o=c[boot.integers(len(c),size=(1000,len(c)))].sum(1)
        return o[:,1:]/o[:,:1]
    intervalos=np.quantile(tasas(co)-tasas(cn),[.025,.975],axis=0)
    r["infraestructura"]={"antes_2005":old,"desde_2005":new,"ic_diferencia_error":intervalos[:,0],"ic_diferencia_frecuencia":intervalos[:,1]}
    describir(cache,r)
    grabar(OUT/"metricas.json",r)
    graficar(r)
    from texto_energia import redactar
    grabar(BASE/"informe.json",redactar(r))
    print("Energía terminada",json.dumps({"calor":r["calor"],"epoca":best_ep},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datos",type=Path,default=Path.home()/"Downloads",help="Carpeta de smart_grid_energy.db")
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix="etl_energia_") as tm,threadpool_limits(limits=1):
        carpeta=Path(tm);preparar(args.datos/"smart_grid_energy.db",carpeta);calcular(carpeta)


if __name__=="__main__":
    main()
