"""Compone la memoria a partir de los informes y figuras generados por cada caso."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from matplotlib import get_data_path
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / "Proyecto_Parcial_2_IDGS17.pdf"
REPO = "https://github.com/LuisDanielZamoraOrtiz/idgs17-extraccion-parcial2"
W, H = A4
LEFT = 48
WIDTH = W - 2 * LEFT
INK, MUTED, GREEN = "#172E36", "#465E63", "#17644F"
DATE = "24 de noviembre de 2026"


def fonts():
    directory = Path(get_data_path()) / "fonts" / "ttf"
    for name, filename in [("Sans", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf"),
                           ("Serif", "DejaVuSerif.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(directory / filename)))


def para(text, size=10.35, face="Sans", color=INK):
    style = ParagraphStyle("texto", fontName=face, fontSize=size, leading=size * 1.48,
                           textColor=HexColor(color), splitLongWords=True)
    return Paragraph(escape(text), style)


def textblock(canvas, body, top, width=WIDTH, x=LEFT):
    _, height = body.wrap(width, H)
    if canvas is not None:
        body.drawOn(canvas, x, top-height)
    return top - height


@dataclass
class Sheet:
    section: str
    title: str
    blocks: list[tuple[str, str]] = field(default_factory=list)
    image: Path | None = None
    caption: str = ""
    fig: int = 0
    anchor: str = ""


def clean(text):
    return str(text).replace("lectura(s)", "lectura").replace("XGB ", "XGBoost ")


def make_pages():
    sheets = []
    references = [("Documentación y datos", "fuentes")]
    introductory = Sheet("Diseño del estudio", "Introducción y metodología", anchor="introduccion")
    introductory.blocks = [
        ("p", "¿Qué ocurre cuando una predicción se convierte en una decisión? En una compra, una alerta equivocada puede bloquear un pago legítimo. En comercio electrónico, omitir un usuario de riesgo alto cambia a quién se podría contactar. En una subestación, una alarma disparada por alta demanda puede exigir revisar miles de horas. Este proyecto mide esos errores en tres bases SQLite y explica qué decisión permite sostener cada resultado."),
        ("h", "Fase 1   ·   Extracción, transformación y carga"),
        ("p", "Se verificaron las claves antes de unir tablas. En comercio electrónico se resumieron los eventos por usuario. En energía se consolidaron los registros por subestación y hora. Cada caso conserva un recuento de duplicados y valores inválidos. El ruido adicional se aplicó a copias del entrenamiento con una semilla reproducible."),
        ("h", "Fase 2   ·   Caracterización"),
        ("p", "Se documentaron tipos, unidades, ausencias y frecuencia de las clases. La unidad de análisis cambia en cada base: una operación, un usuario o un par de horas consecutivas. Ese cambio determina cómo se separan los datos y qué significa el denominador de cada tasa."),
        ("h", "Fase 3   ·   Entrenamiento y prueba"),
        ("p", "Se compararon regresión logística, XGBoost y una red neuronal entrenada por retropropagación. La validación se utilizó para elegir ajustes, épocas o umbrales según el caso. Las métricas finales corresponden a observaciones reservadas. Los apéndices reproducibles del repositorio contienen las curvas de aprendizaje, los parámetros, los conteos y las reglas de limpieza."),
        ("p", "La logística combina las entradas de forma lineal para estimar probabilidades. XGBoost añade árboles sucesivos que corrigen parte del error anterior. La red ajusta los pesos de sus capas al propagar el error hacia atrás. Una época recorre el conjunto de entrenamiento. El límite de iteraciones de la logística y el de rondas de XGBoost se refieren a operaciones distintas."),
    ]
    sheets.append(introductory)
    names = [("fraude", "A", "Fraude financiero"), ("churn", "B", "Abandono de clientes"),
             ("energia", "C", "Demanda eléctrica")]
    toc = [("Introducción y metodología", "introduccion")]
    for folder, code, short in names:
        base = ROOT / folder
        info = json.loads((base / "informe.json").read_text(encoding="utf-8"))
        section = f"Resultados   /   Caso {code}   /   {short}"
        def pic(key):
            return base / info[key] if info.get(key) else None
        toc.append((f"{code}.  {short}", folder))
        sheets.append(Sheet(section, "Datos de origen y limpieza",
            [("lead", info["introduccion"])] + [("p", s) for s in info["etl"]],
            pic("figura_etl"), info.get("pie_etl", ""), anchor=folder))
        if info.get("figura_entrenamiento"):
            sheets.append(Sheet(section, "Caracterización de las variables",
                [("p", s) for s in info["variables"]] +
                [("h", "Alcance de la medición")] + [("p", s) for s in info["limitaciones"]]))
            sheets.append(Sheet(section, "Entrenamiento y evaluación",
                [("p", s) for s in info["entrenamiento"]], pic("figura_entrenamiento"),
                info.get("pie_entrenamiento", "")))
        else:
            sheets.append(Sheet(section, "Variables y entrenamiento",
                [("p", s) for s in info["variables"]] + [("h", "Ajuste y selección")] +
                [("p", s) for s in info["entrenamiento"]] + [("h", "Alcance de la medición")] +
                [("p", s) for s in info["limitaciones"]]))
        for j, question in enumerate(info["preguntas"], 1):
            sheets.append(Sheet(section, question["titulo"], [("p", s) for s in question["respuesta"]],
                base / question["figura"], question["pie"], anchor=f"{code}{j}"))
    toc.append(("Conclusiones", "conclusiones"))
    toc.extend(references)
    sheets.append(Sheet("Discusión", "Conclusiones", conclusions(), anchor="conclusiones"))
    sheets.append(Sheet("Reproducibilidad", "Documentación y datos", source_text(), anchor="fuentes"))
    figures = 0
    for sheet in sheets:
        if sheet.image:
            if not sheet.image.is_file():
                raise FileNotFoundError(sheet.image)
            figures += 1
            sheet.fig = figures
    return sheets, toc


def conclusions():
    def metrics(case):
        path = ROOT / case / "resultados" / "metricas.json"
        return json.loads(path.read_text(encoding="utf-8"))
    def fmt(value, digits=0):
        return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    fraud, churn, energy = (metrics(case) for case in ("fraude", "churn", "energia"))
    classifier = fraud["seleccion_validacion"]
    threshold = fraud["modelos"][classifier]["umbrales"]["operativo"]["test"]
    divergence = fraud["red"]["divergencia"]
    signal = (f"desde la época {divergence['inicio']}" if divergence
              else "sin una secuencia que cumpliera el criterio fijado")
    shipping = next(row for row in churn["b1"] if row["variable"] == "dias_envio")
    score = churn["modelos"][churn["seleccion"]]["prueba"]["macro_f1"]
    hot = energy["calor"]["operativa"]
    cohorts = energy["infraestructura"]
    gap = 100 * (cohorts["antes_2005"]["tasa_critica"] - cohorts["desde_2005"]["tasa_critica"])
    return [
        ("h", "Fraude   ·   El límite de falsas alertas determina la cobertura"),
        ("p", f"Ninguno de los tres clasificadores reunió en validación las dos metas exigidas. Con el umbral elegido bajo el límite de falsas alarmas, {classifier} detectó {fmt(threshold['TP'])} fraudes y señaló {fmt(threshold['FP'])} pagos legítimos en prueba. La comprobación de la red situó la divergencia persistente {signal}. Solo se conservó la etiqueta observada de cada operación. Atribuir individualmente esa divergencia al ruido original requeriría conocer qué etiquetas se invirtieron."),
        ("h", "Abandono   ·   El tiempo de envío separa los dos niveles de riesgo"),
        ("p", f"El envío medio fue de {fmt(shipping['media_1'],2)} días en riesgo medio y de {fmt(shipping['media_2'],2)} en alto. El modelo elegido obtuvo un macro F1 de {fmt(score,4)} en prueba. Reducir los datos a cinco componentes bajó el rendimiento de logística y XGBoost. El canal orgánico presentó la mayor tasa puntual de omisiones y Social Ads el mayor número. El solapamiento de intervalos aconseja comprobar cualquier regla de retención en una cohorte nueva y con seguimiento."),
        ("h", "Energía   ·   Una hora de anticipación tiene un coste en falsas alarmas"),
        ("p", f"La red alertó sobre {fmt(hot['VP'])} de {fmt(hot['criticas'])} horas de consumo alto en zonas industriales cálidas, con {fmt(hot['FP'])} falsas alarmas. El umbral operativo recuperó el {fmt(100*hot['recall'],1)} % de los episodios y acertó en el {fmt(100*hot['precision'],1)} % de los avisos. La carga se definió respecto del historial de cada estación. Para evaluar sobrecargas físicas se necesitan eventos operativos confirmados y cobertura de medidores. La diferencia de frecuencia entre estaciones antiguas y modernas fue de {fmt(gap,2)} puntos porcentuales y su intervalo incluyó cero."),
        ("p", "En los tres casos, la interpretación depende de la etiqueta disponible, del periodo de prueba y de quién entra en cada denominador. Esas condiciones permiten repetir el análisis y delimitar el uso que podría darse a sus resultados."),
    ]


def source_text():
    return [
        ("h", "Repositorio público del proyecto"),
        ("link", REPO),
        ("p", "El README presenta el entorno, el orden de ejecución y los comandos por caso. Cada directorio contiene el ETL, el entrenamiento, los resultados agregados y el código de sus gráficas. La memoria también puede regenerarse a partir de esos resultados."),
        ("h", "Origen y acceso a los datos"),
        ("p", "Se utilizaron las tres bases SQLite proporcionadas en la asignatura. La carpeta metodologia contiene el inventario de tablas y las huellas SHA-256 para comprobar que una reproducción utiliza las mismas entradas. Los archivos originales se suministran localmente al ejecutar el proyecto. El repositorio público contiene únicamente código, documentación, recursos institucionales y evidencia agregada."),
        ("h", "Referencias de la práctica"),
        ("p", "Proyecto Parcial 2. IDGS 17, Extracción de Conocimiento en Bases de Datos. Enunciado consultado en el documento compartido del curso, noviembre de 2026."),
        ("link", "https://scikit-learn.org/stable/modules/model_evaluation.html"),
        ("link", "https://scikit-learn.org/stable/modules/decomposition.html"),
        ("link", "https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html"),
        ("p", "Kassis, T., Agarwal, V., He, Y., Patel, D. y Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. Las guías de scikit-learn y visualización se consultaron para la separación de los datos y la revisión de figuras."),
        ("link", "https://doi.org/10.48550/arXiv.2609.00065"),
    ]


def draw_chrome(c, page, section):
    c.setFillColor(HexColor(GREEN))
    c.rect(0, H-6, W, 6, fill=1, stroke=0)
    c.setFont("Bold", 8.4)
    c.drawString(LEFT, H-37, section.upper())
    c.setFont("Sans", 8.2)
    c.setFillColor(HexColor(MUTED))
    c.drawRightString(W-LEFT, H-37, "PROYECTO PARCIAL 2")
    c.setStrokeColor(HexColor("#C7D8D1"))
    c.line(LEFT, 42, W-LEFT, 42)
    c.drawString(LEFT, 26, "Universidad Tecnológica de Querétaro   ·   IDGS 17")
    c.drawRightString(W-LEFT, 26, str(page))


def cover(c, alumno):
    c.setFillColor(HexColor("#F4F7F3"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(HexColor(GREEN))
    c.rect(0, 0, 12, H, fill=1, stroke=0)
    logo = ImageReader(str(HERE / "recursos" / "logo_uteq.png"))
    c.drawImage(logo, LEFT, H-153, 184, 97,
                preserveAspectRatio=True, mask="auto", anchor="w")
    y = H-178
    for label, size, face in [("Universidad Tecnológica de Querétaro", 18, "Bold"),
        ("Ingeniería en Desarrollo y Gestión de Software", 12.4, "Sans")]:
        y = textblock(c, para(label, size, face), y)-7
    y -= 43
    y = textblock(c, para("MEMORIA TÉCNICA   /   PROYECTO PARCIAL 2", 10, "Bold", GREEN), y)-18
    y = textblock(c, para("Extracción de conocimiento en bases de datos", 29, "Serif"), y)-14
    y = textblock(c, para("Fraude financiero, abandono de clientes y demanda eléctrica", 14.5, "Sans", MUTED), y)
    c.setStrokeColor(HexColor("#A7BCB2"))
    c.line(LEFT, y-31, W-LEFT, y-31)
    y -= 57
    for label, value in [("ALUMNO", alumno), ("DOCENTE", "M.C.C. Emmanuel Martinez Hernandez"),
                         ("GRUPO", "IDGS 17   ·   Big data, análisis de datos")]:
        y = textblock(c, para(label, 8.3, "Bold", GREEN), y)-3
        y = textblock(c, para(value, 13.4), y)-15
    c.setFont("Sans", 10.3)
    c.setFillColor(HexColor(MUTED))
    c.drawString(LEFT, 55, "Santiago de Querétaro, Querétaro   ·   " + DATE)
    c.showPage()


def index(c, sheets, toc):
    draw_chrome(c, 2, "Contenido")
    y = textblock(c, para("Índice", 27, "Serif"), H-64)-20
    positions = {s.anchor: i+3 for i,s in enumerate(sheets) if s.anchor}
    for title, key in toc:
        if key == "fraude":
            c.setFont("Bold", 9)
            c.setFillColor(HexColor(GREEN))
            c.drawString(LEFT, y-1, "RESULTADOS")
            y -= 21
        c.setFillColor(HexColor(INK))
        c.setFont("Bold", 12.5)
        c.drawString(LEFT, y, title)
        c.drawRightString(W-LEFT, y, str(positions[key]))
        c.linkRect("", key, (LEFT, y-5, W-LEFT, y+15), relative=0, thickness=0)
        y -= 24
        if key in ("fraude", "churn", "energia"):
            code = {"fraude":"A","churn":"B","energia":"C"}[key]
            topics = {"A":["Umbral operativo", "No linealidad del riesgo", "Ruido de etiqueta"],
                      "B":["Señales de riesgo medio y alto", "Reducción a cinco componentes", "Omisiones por canal"],
                      "C":["Alerta industrial con calor", "Cargas físicas del PCA", "Infraestructura anterior a 2005"]}[code]
            for j, name in enumerate(topics, 1):
                destination = code+str(j)
                c.setFont("Sans", 10.2)
                c.setFillColor(HexColor(MUTED))
                c.drawString(LEFT+15, y, destination+"   "+name)
                c.drawRightString(W-LEFT, y, str(positions[destination]))
                c.linkRect("", destination, (LEFT+12, y-4, W-LEFT, y+13), relative=0, thickness=0)
                y -= 19
            y -= 7
    y -= 13
    y = textblock(c, para("Repositorio de código y resultados", 11, "Bold", GREEN), y)-5
    link(c, REPO, y)
    c.showPage()


def link(c, url, y):
    line = para(url, 9.0, "Sans", GREEN)
    bottom = textblock(c, line, y)
    if c is not None:
        c.linkURL(url, (LEFT, bottom, W-LEFT, y), relative=0, thickness=0)
    return bottom-9


def body(c, sheet, size, figure_height):
    y = textblock(c, para(sheet.title, 22, "Serif"), H-65)-14
    for kind, text in sheet.blocks:
        if kind=="link":
            y = link(c, text, y)
        elif kind=="h":
            y = textblock(c, para(clean(text), size+.7, "Bold", GREEN), y-3)-5
        else:
            y = textblock(c, para(clean(text), size + (.45 if kind=="lead" else 0)), y)-8
    if sheet.image:
        image = ImageReader(str(sheet.image))
        iw, ih = image.getSize()
        draw_width = min(WIDTH, figure_height*iw/ih)
        draw_height = draw_width*ih/iw
        y -= 5
        if c is not None:
            c.drawImage(image, LEFT+(WIDTH-draw_width)/2, y-draw_height, draw_width, draw_height, mask="auto")
        y -= draw_height+8
        y = textblock(c, para(f"Gráfica {sheet.fig}.  "+clean(sheet.caption), 9.3, "Sans", MUTED), y)
    return y


def render(alumno):
    fonts()
    sheets, toc = make_pages()
    c = Canvas(str(OUTPUT), pagesize=A4, pageCompression=1, invariant=1)
    c.setTitle("Proyecto Parcial 2 · Extracción de conocimiento en bases de datos")
    c.setAuthor(alumno)
    cover(c, alumno)
    index(c, sheets, toc)
    for n, sheet in enumerate(sheets, 3):
        draw_chrome(c, n, sheet.section)
        if sheet.anchor:
            c.bookmarkPage(sheet.anchor)
            c.addOutlineEntry(sheet.title if sheet.anchor not in ("fraude","churn","energia") else sheet.section, sheet.anchor)
        # Shrink only when the actual paragraphs and figure need the space.
        for font_size, height in [(10.8,315),(10.5,298),(10.2,280),(10.0,252),(9.8,234),(9.5,205)]:
            if body(None, sheet, font_size, height) >= 55:
                body(c, sheet, font_size, height)
                break
        else:
            raise ValueError(f"El contenido desborda la página {n}: {sheet.title}")
        c.showPage()
    c.save()
    print(f"{OUTPUT.name}, {len(sheets)+2} páginas y {sum(bool(s.image) for s in sheets)} gráficas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alumno", default="Estudiante de IDGS 17", help="Nombre que se imprimirá en la copia local.")
    render(parser.parse_args().alumno)
