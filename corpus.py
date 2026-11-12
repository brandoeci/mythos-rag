"""descarga y pre-procesamiento del corpus.

responsable: hildebrando.

las obras se bajan de project gutenberg una sola vez y quedan en `data/`. en
corridas posteriores se leen del disco, igual que en el taller 2.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

DATA_DIR = Path("data")


@dataclass(frozen=True)
class Source:
    """una obra del corpus.

    `author` y `work` terminan como metadata de cada chunk: son los que
    permiten citar y los que permiten detectar que dos fuentes discrepan.
    """

    author: str
    work: str
    gutenberg_id: int
    filename: str
    translation: str  # cual traduccion es. dos traducciones dan textos distintos

    # si el archivo trae **varias** obras, `author` y `work` de aqui no sirven
    # para citar: hay que sacarlos del encabezado de cada seccion al chunkear.
    # mientras eso no este implementado, esta fuente NO se indexa, porque
    # produciria citas falsas.
    is_collection: bool = False

    @property
    def url(self) -> str:
        return f"https://www.gutenberg.org/cache/epub/{self.gutenberg_id}/pg{self.gutenberg_id}.txt"

    @property
    def path(self) -> Path:
        return DATA_DIR / self.filename


# ids verificados en gutenberg.org. si se cambia una edicion hay que cambiar
# tambien `translation`: es lo que nos dice que texto es el que estamos citando.
#
# apolodoro (la Biblioteca) **no esta en gutenberg**, y era una de las fuentes
# primarias que queriamos para las contradicciones. se reemplaza por dos cosas:
# la Argonautica de apolonio (fuente primaria, da su propia version de jason y
# medea) y el compendio de berens, que cuenta los mismos mitos que bulfinch y
# por eso choca con el en los detalles.
SOURCES: list[Source] = [
    # OJO: el #348 no es solo la Teogonia. el archivo trae unas treinta obras:
    # los Trabajos y Dias, la Teogonia, el Escudo de Heracles, los Catalogos de
    # Mujeres, **los Himnos Homericos** (que son de homero, no de hesiodo) y los
    # fragmentos del Ciclo Epico. citarlo entero como "hesiodo, teogonia" seria
    # una cita falsa, que es exactamente lo que el proyecto dice evitar.
    Source(
        "varios",
        "hesiodo, himnos homericos y homerica",
        348,
        "hesiodo.txt",
        "Evelyn-White, Loeb 1914",
        is_collection=True,
    ),
    Source("homero", "iliada", 2199, "iliada.txt", "Samuel Butler, prosa"),
    Source("homero", "odisea", 1727, "odisea.txt", "Samuel Butler, prosa"),
    Source("apolonio", "argonautica", 830, "argonautica.txt", "R. C. Seaton, 1912"),
    # OJO: la edicion de riley alterna el poema con el comentario del propio
    # riley (siglo XIX). la estructura es `FABLE N. [refs]` = texto de ovidio,
    # y `EXPLANATION.` = nota de riley. son 109 bloques EXPLANATION entre los
    # dos tomos, y 28 de ellos discuten versiones de otros autores. hay que
    # separarlos: un chunk de una nota citado como "ovidio" es una cita falsa.
    Source("ovidio", "metamorfosis", 21765, "ovidio_i_vii.txt", "Henry T. Riley, prosa"),
    Source("ovidio", "metamorfosis", 26073, "ovidio_viii_xv.txt", "Henry T. Riley, prosa"),
    Source("bulfinch", "the age of fable", 3327, "bulfinch.txt", "edicion de 1913"),
    Source("berens", "myths and legends", 22381, "berens.txt", "edicion de 1894"),
]


def download(source: Source) -> str:
    """baja la obra si no esta en disco y devuelve su texto crudo."""
    if source.gutenberg_id == 0:
        raise ValueError(
            f"falta el gutenberg_id de {source.work}. ver el TODO en SOURCES."
        )

    if source.path.exists():
        return source.path.read_text(encoding="utf-8")

    source.path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(source.url) as response:
        text = response.read().decode("utf-8")
    source.path.write_text(text, encoding="utf-8")
    return text


def strip_boilerplate(text: str) -> str:
    """quita las notas de gutenberg de antes y despues de los marcadores.

    si no encuentra los marcadores devuelve el texto tal cual: hay ediciones
    viejas que no los traen.
    """
    start = text.find("*** START OF")
    if start != -1:
        start = text.find("\n", start)
        text = text[start:]

    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]

    return text.strip()


def normalize(text: str) -> str:
    """normaliza espacios y saltos de linea, conservando los parrafos.

    a diferencia del taller 2 no se pasa a minusculas ni se botan los signos:
    el texto se va a **citar verbatim**, asi que se conserva como esta. la
    limpieza agresiva va despues, solo sobre la copia que indexa bm25.

    los ocho archivos vienen **a doble espacio**: hay una linea en blanco entre
    cada linea de texto, y los parrafos se separan con dos. medido en la
    iliada: 12.388 rachas de dos saltos (fin de linea) contra 1.165 de cuatro
    (fin de parrafo).

    por eso se parte por rachas de cuatro o mas y no por "\\n\\n". partir por
    "\\n\\n" daba 13.292 "parrafos" de 63 caracteres de mediana, que son lineas
    sueltas, y encima rompia las frases: buscar "fairest among the deathless
    gods" no encontraba nada porque la frase cruzaba un salto.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [" ".join(block.split()) for block in re.split(r"\n{3,}", text)]
    return "\n\n".join(block for block in paragraphs if block)


@dataclass(frozen=True)
class Section:
    """un tramo de texto con su atribucion resuelta.

    es la unidad que se puede citar. para las fuentes de una sola obra hay una
    seccion por libro o capitulo; para la antologia #348 hay una por obra.
    """

    author: str
    work: str
    locator: str
    text: str

    def __str__(self) -> str:
        return f"{self.author}, {self.work} ({self.locator})"


# --- la antologia #348 ------------------------------------------------------

# lo que NO es texto citable: la nota del preparador, el prefacio, el estudio
# introductorio de evelyn-white, la bibliografia y las notas finales.
_FRONT_MATTER = {
    "PREPARER'S NOTE",
    "PREFACE",
    "INTRODUCTION",
    "BIBLIOGRAPHY",
    "ENDNOTES",
    "HESIOD",  # es solo el rotulo del grupo, no una obra
}

# a quien se le atribuye cada obra de la antologia. el edition de evelyn-white
# agrupa cosas que NO son de hesiodo (los himnos son homericos, el ciclo epico
# es de varios autores perdidos), y de ahi venia el riesgo de la cita falsa.
_HESIOD_CORE = {"HESIOD'S WORKS AND DAYS", "THE THEOGONY", "THE SHIELD OF HERACLES"}
_EPIC_CYCLE = {
    "THE EPIC CYCLE", "THE WAR OF THE TITANS", "THE STORY OF OEDIPUS", "THE THEBAID",
    "THE EPIGONI", "THE CYPRIA", "THE AETHIOPIS", "THE LITTLE ILIAD",
    "THE SACK OF ILIUM", "THE RETURNS", "THE TELEGONY",
}


def _attribute(title: str) -> str:
    """de quien es una obra de la antologia.

    se marca "(atribuido)" donde la atribucion es tradicional y discutida: la
    propia edicion dice de La Astronomia que esta "attributed forsooth to
    Hesiod". afirmar autoria donde la fuente duda seria el mismo problema que
    las citas falsas, un paso mas arriba.
    """
    if title in _HESIOD_CORE:
        return "hesiodo"
    if title in _EPIC_CYCLE:
        return "ciclo epico (fragmentos)"
    if title.startswith("THE HOMERIC HYMNS") or title.startswith("THE EPIGRAMS OF HOMER"):
        return "homero (atribuido)"
    if re.match(r"^[IVXL]+\. TO ", title):  # los himnos numerados
        return "homero (atribuido)"
    if title == "THE CONTEST OF HOMER AND HESIOD":
        return "anonimo"
    if title in {"HOMERICA", "THE BATTLE OF FROGS AND MICE", "THE MARGITES", "THE CERCOPES"}:
        return "anonimo"
    return "hesiodo (atribuido)"


def _clean_title(title: str) -> str:
    """normaliza un encabezado para poder compararlo.

    dos cosas estorban. una, el numero de nota soldado al final: en el archivo
    aparece "THE CATALOGUES OF WOMEN AND EOIAE1701". dos, el apostrofo
    tipografico U+2019 de "HESIOD'S WORKS AND DAYS", que no es el apostrofo
    ASCII y hacia fallar la comparacion en silencio.

    solo se normaliza para comparar. el texto que se cita queda intacto.
    """
    title = title.replace("’", "'").replace("‘", "'")
    return re.sub(r"\d+$", "", title).strip()


#  un titulo de obra: corto, en mayusculas, y solo letras y puntuacion de
#  titulo. el filtro de caracteres es lo que deja fuera las lineas de la
#  bibliografia, que tambien son cortas y en mayusculas pero traen años y
#  corchetes, como "1973).-DBK.]".
_HEADING = re.compile(r"^(?:[IVXL]+\.\s+)?[A-Z][A-Z\s'’,.\-]*$")


def _is_heading(para: str) -> bool:
    if len(para) >= 60 or para != para.upper() or not re.search(r"[A-Z]{3}", para):
        return False
    # se limpia **antes** de validar: el numero de nota soldado al final es
    # legitimo ("...AND EOIAE1701") y rechazarlo hacia que la Teogonia se
    # tragara la obra siguiente.
    return bool(_HEADING.match(_clean_title(para)))


def _sections_anthology(text: str) -> list[Section]:
    """parte el #348 en una seccion por obra, con su autor resuelto.

    el archivo arranca con un indice donde los titulos aparecen en parrafos
    consecutivos, y los mismos titulos reaparecen despues como encabezados de
    verdad. el indice se descarta quedandose con la **ultima** aparicion de
    cada titulo.
    """
    paras = [" ".join(p.split()) for p in text.split("\n\n")]

    last_at: dict[str, int] = {}
    for i, para in enumerate(paras):
        if _is_heading(para):
            last_at[_clean_title(para)] = i

    # el indice del inicio repite todos los titulos en parrafos consecutivos, y
    # algunos solo aparecen ahi. se descarta todo lo anterior al final del
    # material preliminar, que cierra con la bibliografia.
    content_starts = max(
        (last_at.get(h, -1) for h in ("BIBLIOGRAPHY", "INTRODUCTION", "PREFACE")),
        default=-1,
    )

    starts = sorted(
        (i, title) for title, i in last_at.items() if i >= content_starts
    )

    sections = []
    for (start, title), nxt in zip(starts, starts[1:] + [(len(paras), "")]):
        if title in _FRONT_MATTER:
            continue
        body = "\n\n".join(paras[start + 1 : nxt[0]]).strip()
        if len(body) < 200:  # rotulos de grupo sin texto propio
            continue
        sections.append(
            Section(author=_attribute(title), work=title.lower(), locator=title.lower(), text=body)
        )
    return sections


# --- ovidio: cinco capas y solo dos son citables ----------------------------

_OV_BOOK = re.compile(r"^BOOK THE ([A-Z]+)\.")
_OV_FABLE = re.compile(r"^(FABLE [IVXL]+|THE ARGUMENT)\.\s*(\[[^\]]*\])?")

def _roman(n: int) -> str:
    """el numero de libro va en romano, como lo escribiria una cita."""
    vals = ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out or "?"


_ORDINALS = {
    "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5, "SIXTH": 6,
    "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10, "ELEVENTH": 11,
    "TWELFTH": 12, "THIRTEENTH": 13, "FOURTEENTH": 14, "FIFTEENTH": 15,
}


def _sections_ovid(text: str) -> list[Section]:
    """saca solo el texto de ovidio, dejando fuera lo de riley.

    la edicion intercala cinco cosas y solo el verso traducido es ovidio:

    1. `BOOK THE FIRST.`            el libro
    2. `FABLE N. [I.5-31]`          encabezado, con el rango de versos latinos
    3. una sinopsis de un parrafo   editorial, NO es ovidio
    4. el texto traducido           **esto si**
    5. `EXPLANATION.` y lo que sigue, mas los parrafos `[Footnote N: ...]`
       repartidos por todo el archivo: riley, siglo XIX. NO es ovidio.

    `THE ARGUMENT` si entra: es el proemio del propio ovidio (I.1-4).
    """
    paras = [p for p in text.split("\n\n") if p.strip()]

    sections: list[Section] = []
    book = 0
    fable = locator = ""
    buf: list[str] = []
    skip_synopsis = False
    in_note = False

    def flush() -> None:
        if buf and fable:
            sections.append(
                Section("ovidio", "metamorfosis", locator, "\n\n".join(buf))
            )
        buf.clear()

    for para in paras:
        if m := _OV_BOOK.match(para):
            flush()
            book = _ORDINALS.get(m.group(1), book)
            fable = locator = ""
            in_note = False
            continue

        if m := _OV_FABLE.match(para):
            flush()
            fable = m.group(1).lower().replace("fable", "fabula")
            refs = f" {m.group(2)}" if m.group(2) else ""
            locator = f"libro {_roman(book)}, {fable}{refs}"
            skip_synopsis = m.group(1).startswith("FABLE")
            in_note = False
            continue

        if para.startswith("EXPLANATION"):
            flush()
            in_note = True
            continue

        if in_note or para.lstrip().startswith("[Footnote"):
            continue

        if skip_synopsis:  # la sinopsis editorial que sigue al encabezado
            skip_synopsis = False
            continue

        buf.append(para)

    flush()
    return sections


# --- las obras de una sola pieza: un tramo por libro o capitulo -------------

_DIVISIONS = {
    "iliada.txt": (re.compile(r"^BOOK ([IVXL]+)\.\s*$"), "libro"),
    # la odisea marca los libros SIN punto, a diferencia de la iliada, aunque
    # son la misma traduccion y el mismo traductor.
    "odisea.txt": (re.compile(r"^BOOK ([IVXL]+)\s*$"), "libro"),
    "argonautica.txt": (re.compile(r"^BOOK ([IVXL]+)\s*$"), "libro"),
    "bulfinch.txt": (re.compile(r"^Chapter ([IVXL]+)\s*$"), "capitulo"),
}

# berens no se parte por sus cuatro PARTES, que darian locators inservibles de
# 26.000 palabras, sino por sus secciones con nombre: "HEPHAESTUS (VULCAN).",
# "THE SIRENS.", "MURDER OF ABSYRTUS.--".
_BERENS_HEADING = re.compile(r"^[A-Z][A-Z\s()'’,\-]*\.(?:--|—)?$")


def _sections_berens(source: Source, text: str) -> list[Section]:
    paras = [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]

    last_at: dict[str, int] = {}
    for i, para in enumerate(paras):
        if len(para) < 70 and _BERENS_HEADING.match(para):
            last_at[para] = i

    starts = sorted((i, title) for title, i in last_at.items())
    sections = []
    for (start, title), nxt in zip(starts, starts[1:] + [(len(paras), "")]):
        body = "\n\n".join(paras[start + 1 : nxt[0]]).strip()
        if len(body) < 200:  # entradas del indice y rotulos sin texto propio
            continue
        sections.append(
            Section(source.author, source.work, title.rstrip(".-—").lower(), body)
        )
    return sections


def _sections_divided(source: Source, text: str) -> list[Section]:
    """parte una obra por sus libros o capitulos.

    lo anterior a la primera division (portada, indice, prefacio) se descarta:
    no es texto de la obra y citarlo seria el mismo problema de siempre.
    """
    pattern, kind = _DIVISIONS[source.filename]
    paras = [p for p in text.split("\n\n") if p.strip()]

    sections: list[Section] = []
    current = ""
    buf: list[str] = []

    def flush() -> None:
        if buf and current:
            sections.append(
                Section(source.author, source.work, f"{kind} {current.lower()}", "\n\n".join(buf))
            )
        buf.clear()

    for para in paras:
        if m := pattern.match(para.strip()):
            flush()
            current = m.group(1)
            continue
        buf.append(para)

    flush()
    return sections


_BRACKET_NOTE = re.compile(r"\s*\[\d{1,4}\]")
_LOOSE_NOTE = re.compile(r"(?<=[a-zA-Z,;:.]) \d{3,4}\b")


def strip_markers(text: str, loose: bool = False) -> str:
    """saca los marcadores de nota soldados al texto.

    hay 1.813 en el corpus citable y saldrian dentro de las citas: el pasaje de
    pandora se lee *"he called this woman Pandora 1302, because..."*. tambien
    ensucian los tokens de bm25.

    dos formas. entre corchetes (`[12]`, 1.452 casos) **siempre** es una nota y
    se quita en todas las obras. suelta (`jar 1303`) solo es nota en la
    antologia, donde evelyn-white las numera 1301, 1302, 1303 seguidas; en las
    demas obras esos numeros son años y referencias de linea de verdad
    ("year 1820", "line 252", "in 1646"), asi que ahi no se toca.

    el limite conocido: dentro de la antologia tambien se iria un año de cuatro
    cifras si apareciera en el cuerpo de una obra. las secciones citables no
    incluyen el estudio introductorio ni la bibliografia, que es donde viven
    los años, asi que en la practica no ocurre.
    """
    text = _BRACKET_NOTE.sub("", text)
    if loose:
        text = _LOOSE_NOTE.sub("", text)
    return text


def iter_sections(source: Source, text: str) -> list[Section]:
    """los tramos citables de una fuente, con autor, obra y locator resueltos.

    es lo que `index.chunk_document` debe consumir: nunca el texto crudo, que
    trae prefacios, notas y, en el #348, obras de otros autores.
    """
    if source.is_collection:
        sections = _sections_anthology(text)
    elif source.filename.startswith("ovidio"):
        sections = _sections_ovid(text)
    elif source.filename == "berens.txt":
        sections = _sections_berens(source, text)
    elif source.filename in _DIVISIONS:
        sections = _sections_divided(source, text)
    else:
        raise ValueError(f"no se como partir {source.filename}")

    loose = source.is_collection
    return [
        Section(s.author, s.work, s.locator, strip_markers(s.text, loose=loose))
        for s in sections
    ]


def load_corpus(sources: list[Source] | None = None) -> dict[Source, str]:
    """devuelve el corpus completo, listo para indexar."""
    sources = sources if sources is not None else SOURCES
    return {source: normalize(strip_boilerplate(download(source))) for source in sources}


def summarize(corpus: dict[Source, str]) -> list[dict]:
    """la tabla de tokens y parrafos por obra que va en el notebook."""
    return [
        {
            "autor": source.author,
            "obra": source.work,
            "caracteres": len(text),
            "tokens": len(text.split()),
            "parrafos": text.count("\n\n") + 1,
        }
        for source, text in corpus.items()
    ]
