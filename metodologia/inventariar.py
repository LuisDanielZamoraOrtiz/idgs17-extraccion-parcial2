"""Inventario técnico reproducible de las tres bases locales. Solo biblioteca estándar."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from catalogo import TABLAS


BASE = Path(__file__).resolve().parent


def huella(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identificador(nombre):
    return '"' + nombre.replace('"', '""') + '"'


def inventario(path, diccionario):
    wal=Path(str(path)+"-wal")
    con_wal=wal.is_file() and wal.stat().st_size>0
    # Una base inmutable puede abrirse sin crear archivos auxiliares. Si hay un
    # WAL con contenido, SQLite debe incorporarlo al snapshot de lectura.
    uri=path.resolve().as_uri()+("?mode=ro" if con_wal else "?mode=ro&immutable=1")
    fichas={}
    inicio=path.stat()
    with sqlite3.connect(uri,uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("BEGIN")
        tablas=dict(db.execute("""SELECT name,sql FROM sqlite_schema
                                  WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"""))
        if tablas.keys()!=diccionario.keys():
            raise ValueError(f"El catálogo de tablas difiere del esquema en {path.name}.")
        for tabla,sql in tablas.items():
            clave,semantica=diccionario[tabla]
            table,pk=identificador(tabla),identificador(clave)
            cols=list(db.execute(f"PRAGMA table_info({table})"))
            campos=[{"orden":c[0]+1,"campo":c[1],"tipo_sqlite":c[2],"not_null_declarado":bool(c[3]),
                     "valor_predeterminado_sql":c[4],"posicion_pk_declarada":c[5]} for c in cols]
            if [c["campo"] for c in campos]!=[c[0] for c in semantica]:
                raise ValueError(f"El diccionario no coincide con las columnas de {tabla}.")
            n,no_nulas,unicas=db.execute(f"SELECT COUNT(*), COUNT({pk}), COUNT(DISTINCT {pk}) FROM {table}").fetchone()
            fks=[{"tabla_destino":r[2],"columna":r[3],"columna_destino":r[4]} for r in db.execute(f"PRAGMA foreign_key_list({table})")]
            fichas[tabla]={"filas_brutas":n,"clave_logica":clave,"claves_distintas_no_nulas":unicas,
                           "claves_nulas":n-no_nulas,"repeticiones_excedentes_clave_no_nula":no_nulas-unicas,
                           "campos":campos,"foreign_keys_declaradas":fks,"ddl":sql}
        digest=huella(path)
        digest_wal=huella(wal) if con_wal else None
    fin=path.stat()
    if (inicio.st_mtime_ns,inicio.st_size)!=(fin.st_mtime_ns,fin.st_size):
        raise RuntimeError(f"La base cambió durante el inventario de {path.name}.")
    return {"archivo":path.name,"bytes":fin.st_size,"sha256":digest,"sha256_wal_si_tiene_datos":digest_wal,
            "tablas":fichas}


def leer_metricas(caso):
    target=BASE.parent/caso/"resultados/metricas.json"
    return json.loads(target.read_text(encoding="utf-8"))


def validar_con_casos(datos):
    a,b,c=[leer_metricas(caso) for caso in ["fraude","churn","energia"]]
    fa=datos["fintech_fraud.db"]; ch=datos["ecommerce_churn_ltv.db"]; en=datos["smart_grid_energy.db"]
    comprobados={}
    def comprobar(nombre, calculado, referencia):
        if calculado!=referencia:
            raise ValueError(f"Discrepancia en {nombre}: inventario={calculado}, caso={referencia}")
        comprobados[nombre]=calculado
    comprobar("Fraude SHA-256",fa["sha256"],a["SHA256"])
    for tabla,r in fa["tablas"].items():
        audit=a["auditoria"] if tabla=="transacciones" else a["auditoria"][tabla]
        comprobar("Fraude "+tabla+" filas",r["filas_brutas"],audit["filas"])
        comprobar("Fraude "+tabla+" claves",r["claves_distintas_no_nulas"],audit["claves"])
    comprobar("Abandono SHA-256",ch["sha256"],b["auditoria"]["sha256"])
    for tabla,r in ch["tablas"].items():
        audit=b["auditoria"]["tablas"][tabla]
        comprobar("Abandono "+tabla+" filas",r["filas_brutas"],audit["brutas"])
        comprobar("Abandono "+tabla+" claves",r["claves_distintas_no_nulas"],audit["unicas"])
    for tabla,r in en["tablas"].items():
        label=tabla if tabla=="subestaciones" else tabla+"_origen"
        comprobar("Energía "+tabla+" filas",r["filas_brutas"],c["etl"][label])
        if tabla!="subestaciones":
            comprobar("Energía "+tabla+" claves",r["claves_distintas_no_nulas"],
                      c["etl"][label]-c["etl"][tabla+"_duplicados"])
        else:
            comprobar("Energía subestaciones claves",r["claves_distintas_no_nulas"],c["etl"]["subestaciones"])
    return comprobados


def escribir_diccionario(datos,destino):
    lineas=["# Diccionario de campos de origen","",
            "Los tipos corresponden al esquema SQLite de las bases entregadas. Los roles describen su uso efectivo en los scripts de la práctica.",
            "Las relaciones se implementan en el ETL mediante claves lógicas. Los nueve esquemas no declaran restricciones de clave primaria o foránea.",
            "Las unidades no documentadas se conservan en su escala de origen.",""]
    for base,tablas in TABLAS.items():
        lineas.extend(["## "+base,""])
        for tabla,(_,campos) in tablas.items():
            ficha=datos[base]["tablas"][tabla]
            lineas.extend(["### "+tabla,"",f"Filas de origen {ficha['filas_brutas']:,}. Claves distintas no nulas {ficha['claves_distintas_no_nulas']:,}.".replace(","," "),
                          "","| Campo | Tipo SQLite | Unidad o escala | Significado y función en el análisis |",
                          "| --- | --- | --- | --- |"])
            for i,(campo,unidad,funcion) in enumerate(campos):
                lineas.append(f"| `{campo}` | {ficha['campos'][i]['tipo_sqlite']} | {unidad} | {funcion} |")
            lineas.append("")
    destino.write_text("\n".join(lineas),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos",type=Path,default=Path.home()/"Downloads")
    parser.add_argument("--salida",type=Path,default=BASE)
    parser.add_argument("--solo-fuentes",action="store_true",help="Permite inventariar incluso antes de entrenar los casos")
    args=parser.parse_args()
    datos={name:inventario(args.datos/name,tables) for name,tables in TABLAS.items()}
    result={"algoritmo_huella":"SHA-256 de todos los bytes del archivo, más WAL separado si entró en la lectura",
            "fuentes":datos}
    if not args.solo_fuentes:
        result["contraste_con_metricas"]=validar_con_casos(datos)
    args.salida.mkdir(parents=True,exist_ok=True)
    (args.salida/"inventario.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    escribir_diccionario(datos,args.salida/"diccionario.md")
    print(f"{len(datos)} bases, {sum(len(base['tablas']) for base in datos.values())} tablas, "
          f"{sum(len(t['campos']) for b in datos.values() for t in b['tablas'].values())} campos.")


if __name__=="__main__":
    main()
