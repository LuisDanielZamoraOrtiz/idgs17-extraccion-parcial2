"""Extracción de la muestra temporal y diseño de variables disponibles al autorizar."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalizar(serie):
    return serie.astype("string").str.strip().str.lower().replace("", pd.NA)


def maestras(db, tabla, llave, columnas):
    fields = ", ".join([llave, *columnas])
    frame = pd.read_sql_query(f"SELECT {fields} FROM {tabla}", db)
    raw = len(frame)
    for col in columnas:
        frame[col] = normalizar(frame[col])
    variantes = frame.drop_duplicates()
    conflictos = variantes[llave].duplicated(keep=False)
    unicas = variantes.loc[~conflictos].set_index(llave)
    return unicas, {"filas": raw, "claves": int(frame[llave].nunique()),
                    "claves_conflictivas_campos_usados": int(variantes.loc[conflictos, llave].nunique())}


def leer(ruta, por_mil=50):
    # La muestra depende solo de la clave. Copias de una misma operación siempre viajan juntas.
    db = sqlite3.connect(Path(ruta).resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-16384")
    try:
        auditoria = dict(zip(
            ("filas", "claves", "positivos_brutos", "sin_monto_bruto", "montos_negativos_brutos", "monto_marcador_alto_bruto"),
            db.execute("""SELECT COUNT(*),COUNT(DISTINCT tx_id),SUM(is_fraud=1),
                        SUM(monto IS NULL),SUM(monto<0),SUM(monto=999999.9) FROM transacciones""").fetchone()))
        auditoria["clientes"] = dict(zip(("filas", "claves"),
            db.execute("SELECT COUNT(*),COUNT(DISTINCT client_id) FROM clientes").fetchone()))
        merchants, auditoria["comercios"] = maestras(db, "comercios", "merchant_id", ["categoria_mcc", "pais"])
        client_keys = np.fromiter((row[0] for row in db.execute("SELECT DISTINCT client_id FROM clientes")), dtype="int64")
        query = """SELECT tx_id,client_id,merchant_id,timestamp,monto,moneda,dispositivo,intentos_pin,is_fraud
                   FROM transacciones WHERE ((tx_id*48271)%2147483647)%1000 < ?"""
        chunks = []
        for batch in pd.read_sql_query(query, db, params=(por_mil,), chunksize=30000):
            for col in ["moneda", "dispositivo"]:
                batch[col] = normalizar(batch[col]).astype("category")
            batch["timestamp"] = pd.to_datetime(batch.timestamp, format="%Y-%m-%dT%H:%M:%S", errors="coerce")
            chunks.append(batch)
    finally:
        db.close()
    frame = pd.concat(chunks, ignore_index=True)
    auditoria["muestra_filas"] = len(frame)
    auditoria["muestra_claves"] = int(frame.tx_id.nunique())
    originales = frame.drop_duplicates()
    conflicts = originales.tx_id.duplicated(keep=False)
    auditoria["copias_identicas_muestra"] = len(frame) - len(originales)
    auditoria["claves_conflictivas_muestra"] = int(originales.loc[conflicts, "tx_id"].nunique())
    df = originales.loc[~conflicts].copy()
    bad = df.timestamp.isna() | ~df.is_fraud.isin([0, 1])
    auditoria["sin_fecha_o_etiqueta_valida_muestra"] = int(bad.sum())
    df = df.loc[~bad].copy()
    auditoria["sin_cliente_maestro_muestra"] = int((~df.client_id.isin(client_keys)).sum())
    auditoria["sin_comercio_maestro_util_muestra"] = int((~df.merchant_id.isin(merchants.index)).sum())
    df = df.join(merchants, on="merchant_id", validate="many_to_one")
    # Los datos no especifican códigos de devolución. Los importes negativos y el
    # pico repetido de 999999.9 se conservan como montos desconocidos, usando indicador.
    invalido = df.monto.lt(0) | df.monto.eq(999999.9)
    auditoria["monto_invalido_muestra"] = int(invalido.sum())
    auditoria["monto_nulo_original_muestra"] = int(df.monto.isna().sum())
    df.loc[invalido, "monto"] = np.nan
    auditoria["monto_ausente_total_muestra"] = int(df.monto.isna().sum())
    for col in ["moneda", "dispositivo", "categoria_mcc", "pais"]:
        auditoria["faltantes_" + col] = int(df[col].isna().sum())
        df[col] = df[col].astype("string").fillna("desconocido").astype("category")
    auditoria["filas_analizadas"] = len(df)
    auditoria["sin_pin_valido"] = int((df.intentos_pin.isna() | df.intentos_pin.lt(0)).sum())
    df.loc[df.intentos_pin.lt(0), "intentos_pin"] = np.nan
    df["periodo"] = np.select([df.timestamp.lt("2025-09-01"), df.timestamp.lt("2025-11-01")],
                              ["train", "val"], default="test")
    return df.sort_values(["timestamp","tx_id"]).reset_index(drop=True), auditoria


def preparar(df):
    t = df.periodo.eq("train")
    categorias = ["moneda", "dispositivo", "categoria_mcc", "pais"]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    ohe.fit(df.loc[t, categorias])
    # Indicador explícito y mediana nominal calculada por moneda solo en train.
    medianas = df.loc[t].groupby("moneda",observed=True).monto.median()
    respaldo = df.loc[t,"monto"].median()
    monto = df.monto.fillna(df.moneda.astype("string").map(medianas)).fillna(respaldo).astype(float)
    pin = df.intentos_pin.fillna(df.loc[t,"intentos_pin"].median())
    hora = df.timestamp.dt.hour.to_numpy() + df.timestamp.dt.minute.to_numpy()/60
    dia = df.timestamp.dt.dayofweek.to_numpy()
    nums = np.column_stack([np.log1p(monto),df.monto.isna(),pin,df.intentos_pin.isna(),
                            np.sin(2*np.pi*hora/24),np.cos(2*np.pi*hora/24),hora<6,
                            np.sin(2*np.pi*dia/7),np.cos(2*np.pi*dia/7)]).astype("float32")
    sc = StandardScaler().fit(nums[t])
    x = np.column_stack([sc.transform(nums),ohe.transform(df[categorias])]).astype("float32")
    assert np.isfinite(x).all() and df.tx_id.is_unique
    return x,df.is_fraud.to_numpy(dtype="int8"), {
        "columnas": int(x.shape[1]),
        "nombres":["log_monto","falta_monto","intentos_pin","falta_pin","hora_seno","hora_coseno","madrugada","dia_seno","dia_coseno",*ohe.get_feature_names_out(categorias)],
        "medianas_monto_train": {str(k):float(v) for k,v in medianas.items()},
        "categorias_train":{c:list(map(str,items)) for c,items in zip(categorias,ohe.categories_)}}
