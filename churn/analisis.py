"""Train and evaluate the three customer attrition models. Python 3.11."""
import argparse
import copy
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable]="1"

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, log_loss,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits

from etl import CHANNELS, FEATURES, prepare, sha256

HERE=Path(__file__).resolve().parent
SEED=2026
NTRAIN, NVAL, NTEST=80000,20000,120000


def write_json(path,data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def evaluate(y,p):
    pred=np.argmax(p,axis=1)
    return {"macro_f1":float(f1_score(y,pred,average="macro")), "exactitud":float(accuracy_score(y,pred)),
            "perdida":float(log_loss(y,p,labels=[0,1,2])), "confusion":confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            "clases":classification_report(y,pred,labels=[0,1,2],output_dict=True,zero_division=0)}


def apply_training_noise(x):
    """Uniformly corrupt selected observed features in the training copy, then mark them missing."""
    rng=np.random.default_rng(SEED+17)
    raw=x.copy()
    subset=[list(FEATURES).index(f) for f in ("duracion_promedio_sesion","paginas_promedio","pago_promedio","dias_envio")]
    changed={}
    for i in subset:
        candidates=np.flatnonzero(np.isfinite(raw[:,i]))
        n=int(np.floor(0.02*len(candidates)))
        idx=rng.choice(candidates, n, replace=False)
        part=n//2
        raw[idx[:part],i]=np.nan
        raw[idx[part:],i]=999999.9
        changed[list(FEATURES)[i]]={"validas_iniciales":int(len(candidates)), "nulos_inyectados":int(part), "centinelas_inyectados":int(n-part)}
        raw[raw[:,i]==np.float32(999999.9),i]=np.nan
    return raw,changed


def split_data(y):
    # Sampling is disjoint and stratified over unique users; no event from a user occurs in another set.
    universe=np.arange(len(y),dtype=np.int32)
    chosen,_=train_test_split(universe,train_size=NTRAIN+NVAL+NTEST,stratify=y,random_state=SEED)
    trva,te=train_test_split(chosen,test_size=NTEST,stratify=y[chosen],random_state=SEED+1)
    tr,va=train_test_split(trva,test_size=NVAL,stratify=y[trva],random_state=SEED+2)
    assert len(np.unique(np.concatenate([tr,va,te])))==NTRAIN+NVAL+NTEST
    return tr,va,te


def logistic(xt,y, xv,yv):
    mod=LogisticRegression(max_iter=250, solver="lbfgs",tol=1e-5,random_state=SEED)
    mod.fit(xt,y)
    return mod,{"max_iter":250,"iteraciones":int(max(mod.n_iter_))}


def boosted(xt,y,xv,yv):
    mod=XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=4, min_child_weight=8,
                      subsample=0.85,colsample_bytree=1.0,tree_method="hist",max_bin=128,n_jobs=1,
                      random_state=SEED,eval_metric="mlogloss",early_stopping_rounds=25,verbosity=0)
    mod.fit(xt,y,eval_set=[(xv,yv)],verbose=False)
    loss=mod.evals_result()["validation_0"]["mlogloss"]
    return mod,{"techo_rondas":800,"learning_rate":0.05,"paciencia":25,
                "ronda_elegida":int(mod.best_iteration+1),"rondas_ejecutadas":len(loss),
                "arboles_por_ronda":3,"historial_val_loss":loss}


def neural(xt,y,xv,yv):
    mod=MLPClassifier(hidden_layer_sizes=(24,12),activation="relu",solver="adam",batch_size=512,
                      learning_rate_init=0.001,alpha=0.001,random_state=SEED,shuffle=False)
    rng=np.random.default_rng(SEED+3)
    history=[]
    best_epoch=0
    best_loss=np.inf
    winner=None
    # Training may continue beyond 50 epochs. Restore the smallest observed validation loss.
    for epoch in range(1,121):
        idx=rng.permutation(len(y))
        mod.partial_fit(xt[idx],y[idx],classes=np.arange(3))
        train=log_loss(y, mod.predict_proba(xt),labels=[0,1,2])
        val=log_loss(yv,mod.predict_proba(xv),labels=[0,1,2])
        history.append({"epoca":epoch,"perdida_entrenamiento":float(train),"perdida_validacion":float(val)})
        if val<best_loss:
            best_loss=val
            best_epoch=epoch
            winner=copy.deepcopy(mod)
        if epoch>=50 and epoch-best_epoch>=12:
            break
    return winner,{"arquitectura":[24,12],"algoritmo":"retropropagación, Adam, ReLU", "lote":512,
                   "minimo":50,"techo":120,"paciencia":12,"epocas_ejecutadas":len(history),
                   "epoca_elegida":best_epoch,"criterio":"menor entropía cruzada observada en validación","historial":history}


def markers(x_imputed_tr, ytr, x_imputed_te, x_raw_te, yt):
    tr=ytr>0
    te=yt>0
    results=[]
    for name in ["tasa_rebote","duracion_promedio_sesion","dias_envio","tasa_devoluciones"]:
        j=list(FEATURES).index(name)
        # Direction of the marker is determined with training labels, then frozen.
        increasing=roc_auc_score(ytr[tr]==2,x_imputed_tr[tr,j])>=0.5
        score=x_imputed_te[te,j]*(1 if increasing else -1)
        values=x_raw_te[:,j]
        spec={"variable":name,"auc":float(roc_auc_score(yt[te]==2,score)),
              "direccion_alto":"mayor" if increasing else "menor",
              "n_medio":int(sum(yt==1)),"n_alto":int(sum(yt==2))}
        for cls in [1,2]:
            present=values[(yt==cls)&np.isfinite(values)]
            spec[f"n_observados_{cls}"]=len(present)
            spec[f"media_{cls}"]=float(np.mean(present,dtype=np.float64))
            spec[f"mediana_{cls}"]=float(np.median(present))
        results.append(spec)
    return sorted(results,key=lambda r:r["auc"],reverse=True)


def channel_errors(ch,yt,pred):
    result=[]
    for i,name in enumerate(CHANNELS):
        inds=ch==i
        high=inds & (yt==2)
        n=int(sum(high))
        k=int(sum(high & (pred!=2)))
        if n==0:
            result.append({"canal":name,"usuarios":int(sum(inds)),"positivos":0,"fn":0,"tasa":None,"IC95":None})
            continue
        z=1.95996398454
        freq=k/n
        denom=1+z*z/n
        center=(freq+z*z/(2*n))/denom
        margin=z*np.sqrt(freq*(1-freq)/n+z*z/(4*n*n))/denom
        result.append({"canal":name,"usuarios":int(sum(inds)),"positivos":n,"fn":k,"tasa":freq,
                       "IC95":[float(center-margin),float(center+margin)]})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos",type=Path,default=Path.home()/"Downloads")
    parser.add_argument("--cache",type=Path,help="Ruta privada local opcional .npz para acelerar recálculos. No publicar.")
    args=parser.parse_args()
    src=args.datos/"ecommerce_churn_ltv.db"
    dest=HERE/"resultados"
    dest.mkdir(exist_ok=True)
    if args.cache and args.cache.exists():
        with np.load(args.cache,allow_pickle=False) as cached:
            x,y,ch=cached["x"],cached["y"],cached["ch"]
            audit=json.loads(str(cached["audit"]))
        if audit["sha256"]!=sha256(src):
            raise ValueError("La huella de la base cambió. Retire la caché o use otra ruta.")
    else:
        x,y,ch,audit=prepare(src)
        if args.cache:
            np.savez_compressed(args.cache,x=x,y=y,ch=ch,audit=json.dumps(audit))
    print("ETL verificado; preparando particiones",flush=True)
    tr,va,te=split_data(y)
    xtr,noise=apply_training_noise(x[tr])
    imputer=SimpleImputer(strategy="median")
    imtr=imputer.fit_transform(xtr)
    imte=imputer.transform(x[te])
    scaler=StandardScaler()
    xt=scaler.fit_transform(imtr)
    xv=scaler.transform(imputer.transform(x[va]))
    xe=scaler.transform(imte)
    pca=PCA(n_components=5,svd_solver="full",random_state=SEED)
    pt=pca.fit_transform(xt)
    pv=pca.transform(xv)
    pe=pca.transform(xe)
    result={"semilla":SEED,"auditoria":audit,"ruido_entrenamiento":noise,"medianas_imputacion":dict(zip(FEATURES,imputer.statistics_.astype(float).tolist())),
            "particiones":{name:np.bincount(y[idx],minlength=3).tolist() for name,idx in [("entrenamiento",tr),("validacion",va),("prueba",te)]},
            "variables":list(FEATURES), "pca_varianza":pca.explained_variance_ratio_.astype(float).tolist(),
            "pca_cargas":pca.components_.astype(float).tolist(),"modelos":{}}
    val_scores={}
    predictions={}
    for label,fn,train,val,test in [("Logística",logistic,xt,xv,xe),("XGBoost",boosted,xt,xv,xe),("Red neuronal",neural,xt,xv,xe),
                                     ("Logística PCA",logistic,pt,pv,pe),("XGBoost PCA",boosted,pt,pv,pe)]:
        print("Entrenando", label,flush=True)
        model,stats=fn(train,y[tr],val,y[va])
        proba=model.predict_proba(test)
        valid=evaluate(y[va],model.predict_proba(val))
        result["modelos"][label]={"ajuste":stats,"validacion":valid,"prueba":evaluate(y[te],proba)}
        if label in ["Logística","XGBoost","Red neuronal"]:
            val_scores[label]=valid["macro_f1"]
            predictions[label]=np.argmax(proba,axis=1)
    winner=max(val_scores,key=val_scores.get)
    result["seleccion"]=winner
    result["b1"]=markers(imtr,y[tr],imte,x[te],y[te])
    result["b3"]=channel_errors(ch[te],y[te],predictions[winner])
    base_f1=f1_score(y[te],np.full_like(y[te],np.argmax(np.bincount(y[tr]))),average="macro")
    result["base_mayoritaria_macro_f1"]=float(base_f1)
    write_json(dest/"metricas.json",result)
    from informe import make_report
    make_report(result,HERE)
    print("Caso B listo",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
