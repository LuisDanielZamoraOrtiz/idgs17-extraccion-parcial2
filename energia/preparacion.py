"""Depuración por lotes y construcción causal de pares de horas consecutivas."""
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ZONAS = ["Bajio_Manufactura", "Centro_Urbano", "Costa_Turistica", "Norte_Industrial", "Sur_Residencial"]
SENSORES = ["voltaje_v", "corriente_a", "factor_potencia", "consumo_kwh"]
CLIMA = ["temperatura_c", "humedad_pct", "velocidad_viento"]
RANGOS = {"voltaje_v": (0, 1000), "corriente_a": (0, 10000), "factor_potencia": (0, 1),
          "consumo_kwh": (0, 100000), "temperatura_c": (-40, 60), "humedad_pct": (0, 100),
          "velocidad_viento": (0, 200)}
XCOL = SENSORES + CLIMA + ["n_lecturas", "consumo_24h", "n_horas_24h", "seno_24h", "coseno_24h",
                          "capacidad_mw", "anio_instalacion"] + ZONAS
PHYS = ["voltaje_v", "corriente_a", "factor_potencia", "temperatura_c"]
SEMILLA = 2026


def por_clave(con, tabla, clave, auditoria):
    """Una lectura por ID. La primera versión en el archivo prevalece; las diferencias se auditan."""
    max_id = con.execute(f'SELECT MAX({clave}) FROM {tabla}').fetchone()[0]
    vistos = np.zeros(max_id + 1, dtype=bool)
    firmas = np.zeros(max_id + 1, dtype=np.uint64)
    for datos in pd.read_sql_query(f'SELECT * FROM {tabla} ORDER BY rowid', con, chunksize=65000):
        auditoria[tabla + "_origen"] += len(datos)
        if datos[clave].isna().any() or (datos[clave] < 1).any():
            raise ValueError(f"Clave sin integridad en {tabla}")
        hashes = pd.util.hash_pandas_object(datos, index=False).to_numpy(dtype=np.uint64)
        ids = datos[clave].to_numpy(dtype=np.int64)
        antes = vistos[ids]
        conservar = ~antes & ~datos[clave].duplicated().to_numpy()
        auditoria[tabla + "_duplicados"] += int((~conservar).sum())
        vistos[ids] = True
        firmas[ids[conservar]] = hashes[conservar]
        auditoria[tabla + "_copias_con_diferencias"] += int((hashes!=firmas[ids]).sum())
        yield datos.loc[conservar].copy()


def limpiar_numeros(datos, columnas, auditoria):
    for c in columnas:
        originales = datos[c]
        if c == "voltaje_v":
            s = originales.astype("string").str.strip()
            auditoria["voltajes_con_unidad"] += int(s.str.endswith("V").sum())
            s = s.str.removesuffix("V").str.strip().str.replace(",", ".", regex=False)
        else:
            s = originales
        v = pd.to_numeric(s, errors="coerce").astype("float64")
        inferior, superior = RANGOS[c]
        valido = v.between(inferior, superior) & ~v.isin([-999, -50, 999999.9])
        if c == "voltaje_v":
            valido &= v > 0
        auditoria[c + "_faltantes"] += int(originales.isna().sum())
        auditoria[c + "_invalidos"] += int((~valido & originales.notna()).sum())
        datos[c] = v.where(valido)


def acumular(pares, grupos, hora, datos, columnas):
    for j, c in enumerate(columnas):
        v = datos[c].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        np.add.at(pares[..., 2*j], (grupos[ok], hora[ok]), v[ok])
        np.add.at(pares[..., 2*j+1], (grupos[ok], hora[ok]), 1)


def promedios(a):
    sumas, conteos = a[..., ::2], a[..., 1::2]
    return np.divide(sumas, conteos, out=np.full_like(sumas, np.nan), where=conteos > 0)


def diagnostico_capacidad(med, capacidades):
    maximo, excedencias, totales = 0., 0, Counter()
    for muestra, cap in zip(med,capacidades):
        conteos = muestra[:,7].astype(int)
        observada = conteos > 0
        fraccion = muestra[observada,6] / (1000*cap)
        if len(fraccion):
            maximo=max(maximo,float(fraccion.max()))
            excedencias+=int((fraccion >= 1).sum())
        totales.update(dict(zip(*[v.tolist() for v in np.unique(conteos[observada],return_counts=True)])))
    centro=(sum(totales.values())+1)/2
    mediana=next(n for n in sorted(totales) if sum(v for k,v in totales.items() if k<=n)>=centro)
    return {"fraccion_capacidad_max":maximo,"horas_al_menos_capacidad":excedencias,
            "mediana_lecturas_por_hora":mediana,"max_lecturas_por_hora":max(totales)}


def preparar(origen, temporal):
    con = sqlite3.connect(origen.resolve().as_uri() + "?mode=ro", uri=True)
    estaciones = pd.read_sql_query('SELECT * FROM subestaciones ORDER BY substation_id', con).set_index("substation_id")
    zonas = {k.lower():i for i,k in enumerate(ZONAS)}
    zonas_est = estaciones.zona.str.strip().str.lower().map(zonas)
    if estaciones.index.has_duplicates or estaciones.isna().any().any() or zonas_est.isna().any():
        raise ValueError("El catálogo de subestaciones requiere conciliación")
    if not estaciones.index.equals(pd.Index(range(1,len(estaciones)+1))):
        raise ValueError("No coincide la numeración continua de las subestaciones")
    tmin, tmax = con.execute("SELECT MIN(hora_idx), MAX(hora_idx) FROM lecturas_medidores").fetchone()
    if tmin != 0:
        raise ValueError("El índice horario no inicia en cero")
    horas = tmax + 1
    med = np.memmap(temporal/"agregados.dat", mode="w+", dtype="float64", shape=(len(estaciones),horas,8))
    clima = np.zeros((len(ZONAS), horas, 6))
    audit = Counter()
    for tabla, clave, cols in [("telemetria_clima","clima_id",CLIMA),("lecturas_medidores","lectura_id",SENSORES)]:
        for d in por_clave(con, tabla, clave, audit):
            limpiar_numeros(d, cols, audit)
            original = d.zona.astype("string")
            reg = original.str.strip().str.lower().map(zonas)
            audit[tabla+"_zona_normalizada"] += int((original.notna() & ~original.isin(ZONAS) & reg.notna()).sum())
            tiempo = d.hora_idx.between(0,tmax)
            if tabla == "lecturas_medidores":
                oficial = d.substation_id.map(zonas_est)
                audit["zona_recuperada_catalogo"] += int((reg.isna() & oficial.notna()).sum())
                audit["zona_discrepa_catalogo"] += int((reg.notna() & oficial.notna() & (reg != oficial)).sum())
                ok = tiempo & oficial.notna()
                gr = d.loc[ok,"substation_id"].to_numpy(dtype=int)-1
                cubo = med
            else:
                ok = tiempo & reg.notna()
                gr = reg[ok].to_numpy(dtype=int)
                cubo = clima
            audit[tabla+"_sin_clave_union"] += int((~ok).sum())
            acumular(cubo, gr, d.loc[ok,"hora_idx"].to_numpy(dtype=int), d.loc[ok], cols)
        print("ETL",tabla,dict((k,v) for k,v in audit.items() if k.startswith(tabla)),flush=True)
    con.close()
    med.flush()
    w = promedios(clima)
    audit["horas_zona_con_clima"] = int((clima[...,1::2].sum(axis=-1)>0).sum())
    audit["horas_zona_posibles"] = len(ZONAS)*horas
    audit["subestaciones"] = len(estaciones)
    audit["horas"] = horas
    train_end, val_end = int(horas*.70),int(horas*.85)
    audit["fin_train_excl"] = train_end
    audit["fin_val_excl"] = val_end
    rng = np.random.default_rng(SEMILLA)
    # Muestreo uniforme por prioridad aleatoria en train y validación. Test conserva todos los pares.
    topes = {"train":42000,"val":32000}
    partes = {k:[] for k in ["train","val","test"]}
    for s, (est,(zona,capacidad,anio)) in enumerate(estaciones.iterrows()):
        z = zonas[zona.lower()]
        tm = promedios(med[s])
        h = np.flatnonzero(np.isfinite(tm[1:,3]) & np.isfinite(tm[:-1,3])) + 1
        anterior=h-1
        vent = pd.Series(tm[:,3]).rolling(24,min_periods=1)
        umbrales = np.nanquantile(tm[:train_end,3],[.70,.90])
        if not np.isfinite(umbrales).all():
            raise ValueError(f"Sin historia suficiente para estación {est}")
        if s==0: cortes=[]
        cortes.append(umbrales)
        datos=np.column_stack([tm[anterior],w[z,anterior],med[s,anterior,7],vent.mean().to_numpy()[anterior],
                               vent.count().to_numpy()[anterior],np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),
                               np.full(len(h),capacidad),np.full(len(h),anio),
                               np.broadcast_to(np.eye(len(ZONAS))[z],(len(h),len(ZONAS)))])
        fisicas=np.column_stack([tm[h,:3],w[z,h,0]])
        meta=np.column_stack([np.digitize(tm[h,3],umbrales),tm[h,3],np.full(len(h),z),
                              np.full(len(h),anio),np.full(len(h),est),h,w[z,h,0],
                              np.digitize(tm[anterior,3],umbrales)])
        integral=np.column_stack([datos,fisicas,meta]).astype("float32")
        audit["estacion_horas_observadas"]+=int(np.isfinite(tm[:,3]).sum())
        audit["estacion_horas_posibles"]+=horas
        audit["pares_consecutivos"]+=len(integral)
        for k, mascara in [("train",h<train_end),("val",(h>=train_end)&(h<val_end)),("test",h>=val_end)]:
            arr=integral[mascara]
            audit["pares_"+k]+=len(arr)
            if k != "test":
                arr=np.column_stack([rng.random(len(arr)).astype("float32"),arr])
            partes[k].append(arr)
            if k!="test" and sum(len(a) for a in partes[k])>topes[k]+18000:
                pila=np.concatenate(partes[k]); sel=np.argpartition(pila[:,0],topes[k]-1)[:topes[k]]
                partes[k]=[pila[sel]]
    for k, piezas in partes.items():
        pila=np.concatenate(piezas)
        if k!="test":
            sel=np.argpartition(pila[:,0],topes[k]-1)[:topes[k]] if len(pila)>topes[k] else np.arange(len(pila))
            pila=pila[sel,1:]
        audit["usados_"+k]=len(pila)
        np.save(temporal/(k+".npy"),pila)
    audit["q70_rango"]=np.round(np.min(cortes,axis=0)[0],3),np.round(np.max(cortes,axis=0)[0],3)
    audit["q90_rango"]=np.round(np.min(cortes,axis=0)[1],3),np.round(np.max(cortes,axis=0)[1],3)
    audit.update(diagnostico_capacidad(med,estaciones.capacidad_mw))
    (temporal/"etl.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf8")
    del med
    print("Pares elegibles y de prueba",audit["pares_consecutivos"],audit["pares_test"],flush=True)
