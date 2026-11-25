"""Prepare a customer-level, anonymous numerical array from a read-only database."""
from pathlib import Path
from contextlib import closing
import hashlib
import sqlite3
import tempfile

import numpy as np


FEATURES = {
    "n_sesiones": "Sesiones registradas",
    "tasa_rebote": "Tasa de rebote",
    "duracion_promedio_sesion": "Duración media de sesión",
    "paginas_promedio": "Páginas por sesión",
    "fraccion_movil": "Sesiones en móvil",
    "n_pedidos": "Pedidos registrados",
    "pago_promedio": "Pago medio por pedido",
    "dias_envio": "Días medios de envío",
    "tasa_devoluciones": "Tasa de devoluciones",
    "estrellas": "Valoración media",
}
CHANNELS = ["Email", "Organico", "Paid_Search", "Referido", "Social_Ads", "Desconocido"]
NUMERIC = {
    "usuarios": {"edad": "edad BETWEEN 0 AND 110"},
    "sesiones_web": {"duracion_seg": "duracion_seg >= 0 AND duracion_seg != 999999.9",
                     "paginas_vistas": "paginas_vistas >= 0 AND paginas_vistas != 999999.9 AND paginas_vistas = CAST(paginas_vistas AS INTEGER)"},
    "ordenes_compra": {"total_pagado": "total_pagado >= 0 AND total_pagado != 999999.9",
                       "dias_envio": "dias_envio >= 0 AND dias_envio != 999999.9",
                       "calificacion_estrellas": "calificacion_estrellas BETWEEN 1 AND 5 AND calificacion_estrellas = CAST(calificacion_estrellas AS INTEGER)"},
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(db_path):
    with tempfile.TemporaryDirectory(prefix="churn-etl-") as tmp:
        with closing(sqlite3.connect(str(Path(tmp) / "etl.sqlite"), uri=True)) as con:
            return _build(con, db_path)


def _build(con, db_path):
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("PRAGMA cache_size=-16384")
    con.execute("ATTACH DATABASE ? AS raw", (db_path.resolve().as_uri() + "?mode=ro",))
    audit = {"sha256": sha256(db_path), "tablas": {}, "numericas": {}, "categorias": {}}
    keys = {"usuarios": "user_id", "sesiones_web": "session_id", "ordenes_compra": "order_id"}
    for table, key in keys.items():
        n_raw = con.execute(f"SELECT COUNT(*) FROM raw.{table}").fetchone()[0]
        con.execute(f"CREATE TABLE {table} AS SELECT DISTINCT * FROM raw.{table}")
        # A unique index rejects conflicting versions of the same original key.
        con.execute(f"CREATE UNIQUE INDEX idx_{table} ON {table}({key})")
        n_clean = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        invalid_key = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {key} IS NULL OR {key}<=0").fetchone()[0]
        if invalid_key:
            raise ValueError(f"Claves inválidas en {table}")
        audit["tablas"][table] = {"brutas": n_raw, "unicas": n_clean, "duplicadas": n_raw - n_clean}
        conds = NUMERIC[table]
        aggregations = []
        for col, valid in conds.items():
            aggregations.extend([f"SUM({col} IS NULL)", f"SUM({col} IS NOT NULL AND NOT ({valid}))",
                                 f"MIN({col})", f"MAX({col})",
                                 f"MIN(CASE WHEN {valid} THEN {col} END)", f"MAX(CASE WHEN {valid} THEN {col} END)"])
        values = con.execute(f"SELECT {','.join(aggregations)} FROM {table}").fetchone()
        for idx, col in enumerate(conds):
            audit["numericas"][col] = dict(zip(["nulos", "invalidos", "min_antes", "max_antes", "min_despues", "max_despues"], values[idx*6:idx*6+6]))
    for table, col, values in [("usuarios","riesgo_abandono","0,1,2"),("usuarios","suscripcion_prime","0,1"),
                               ("sesiones_web","rebote","0,1"),("ordenes_compra","devuelto","0,1")]:
        invalid = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL OR {col} NOT IN ({values})").fetchone()[0]
        if invalid:
            raise ValueError(f"Dominio inválido de {col}: {invalid}")
    for table in ["sesiones_web","ordenes_compra"]:
        orphans=con.execute(f"SELECT COUNT(*) FROM {table} t LEFT JOIN usuarios u ON u.user_id=t.user_id WHERE u.user_id IS NULL").fetchone()[0]
        if orphans:
            raise ValueError(f"{orphans} referencias huérfanas en {table}")
        audit["tablas"][table]["huerfanas"]=orphans
    for table,col in [("usuarios","canal_adquisicion"),("sesiones_web","dispositivo_os")]:
        audit["categorias"][col]=[{"valor":v,"n":n} for v,n in con.execute(f"SELECT {col},COUNT(*) FROM {table} GROUP BY {col}")]
    sessions=NUMERIC["sesiones_web"]
    orders=NUMERIC["ordenes_compra"]
    con.execute(f"""CREATE TABLE s AS SELECT user_id, count(*) AS ns, avg(rebote) AS reb,
           avg(CASE WHEN {sessions['duracion_seg']} THEN duracion_seg END) AS dur,
           avg(CASE WHEN {sessions['paginas_vistas']} THEN paginas_vistas END) AS pags,
           avg(CASE WHEN lower(trim(dispositivo_os)) IN ('ios','android') THEN 1.0
                    WHEN lower(trim(dispositivo_os)) IN ('windows','macos') THEN 0.0 END) AS movil
           FROM sesiones_web GROUP BY user_id""")
    con.execute("CREATE UNIQUE INDEX idx_s ON s(user_id)")
    con.execute(f"""CREATE TABLE o AS SELECT user_id, count(*) AS no,
           avg(CASE WHEN {orders['total_pagado']} THEN total_pagado END) AS pago,
           avg(CASE WHEN {orders['dias_envio']} THEN dias_envio END) AS dias,
           avg(devuelto) AS dev,
           avg(CASE WHEN {orders['calificacion_estrellas']} THEN calificacion_estrellas END) AS stars
           FROM ordenes_compra GROUP BY user_id""")
    con.execute("CREATE UNIQUE INDEX idx_o ON o(user_id)")
    size = audit["tablas"]["usuarios"]["unicas"]
    x = np.empty((size, len(FEATURES)), dtype=np.float32)
    y = np.empty(size, dtype=np.int8)
    ch = np.empty(size, dtype=np.int8)
    chmap = {v.lower(): i for i,v in enumerate(CHANNELS)}
    query = con.execute("""SELECT riesgo_abandono, lower(trim(canal_adquisicion)),
        coalesce(ns,0),reb,dur,pags,movil,coalesce(no,0),pago,dias,dev,stars FROM usuarios u
        LEFT JOIN s ON s.user_id=u.user_id LEFT JOIN o ON o.user_id=u.user_id ORDER BY u.user_id""")
    offset=0
    while rows:=query.fetchmany(6000):
        end=offset+len(rows)
        y[offset:end]=[r[0] for r in rows]
        ch[offset:end]=[chmap.get(r[1],len(CHANNELS)-1) for r in rows]
        x[offset:end]=[r[2:] for r in rows]
        offset=end
    assert offset == size
    audit["clases"]=np.bincount(y,minlength=3).tolist()
    audit["sin_sesiones"]=int((x[:,0]==0).sum())
    audit["sin_pedidos"]=int((x[:,5]==0).sum())
    audit["faltantes_agregados"] = dict(zip(FEATURES, np.isnan(x).sum(axis=0).tolist()))
    audit["resumen_variables"]={}
    for j,feature in enumerate(FEATURES):
        low,q1,median,q3,high=np.nanpercentile(x[:,j],[0,25,50,75,100])
        audit["resumen_variables"][feature]={"min":float(low),"q1":float(q1),"mediana":float(median),"q3":float(q3),"max":float(high)}
    audit["canales"] = dict(zip(CHANNELS,np.bincount(ch,minlength=len(CHANNELS)).tolist()))
    return x,y,ch,audit
