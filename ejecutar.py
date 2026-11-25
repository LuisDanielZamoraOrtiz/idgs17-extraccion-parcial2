"""Reproduce los tres análisis y compone la memoria del Proyecto Parcial 2."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CASES = (
    ("A · Fraude financiero", "fraude/entrenar.py", "fintech_fraud.db"),
    ("B · Abandono de clientes", "churn/analisis.py", "ecommerce_churn_ltv.db"),
    ("C · Estabilidad de red", "energia/energia.py", "smart_grid_energy.db"),
)


def run_step(title, script, *arguments):
    print(f"\n{title}", flush=True)
    code = subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], cwd=ROOT).returncode
    if code:
        raise SystemExit(f"Falló «{title}» con código {code}. Revisa el mensaje de la etapa anterior.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=Path.home() / "Downloads",
                        help="Carpeta que contiene las tres bases .db (por defecto, Descargas del usuario).")
    parser.add_argument("--solo-reporte", action="store_true",
                        help="Compone solo el PDF a partir de los informes y figuras del repositorio.")
    parser.add_argument("--alumno", default="Estudiante de IDGS 17", help="Nombre para la portada local.")
    args = parser.parse_args()
    if not args.solo_reporte:
        directory = args.datos.expanduser().resolve()
        missing = [filename for _, _, filename in CASES if not (directory / filename).is_file()]
        if missing:
            parser.error("Faltan archivos en la carpeta de --datos: " + ", ".join(missing))
        for name, script, _ in CASES:
            run_step(name, script, "--datos", directory)
        run_step("Inventario y contraste de fuentes", "metodologia/inventariar.py", "--datos", directory)
    missing_reports = [case + "/informe.json" for case in ("fraude", "churn", "energia")
                       if not (ROOT / case / "informe.json").is_file()]
    if missing_reports:
        parser.error("No se pueden maquetar los resultados. Faltan " + ", ".join(missing_reports))
    run_step("Composición de la memoria", "memoria/generar.py", "--alumno", args.alumno)
    pdf = ROOT / "memoria" / "Proyecto_Parcial_2_IDGS17.pdf"
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise SystemExit(f"La composición terminó sin generar el PDF esperado en {pdf}.")
    print(f"Memoria disponible en {pdf.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
