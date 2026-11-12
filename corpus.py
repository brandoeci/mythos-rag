"""descarga y pre-procesamiento del corpus.

responsable: hildebrando.

las obras se bajan de project gutenberg una sola vez y quedan en `data/`. en
corridas posteriores se leen del disco, igual que en el taller 2.
"""

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
    Source("hesiodo", "teogonia", 348, "hesiodo.txt", "Evelyn-White, Loeb 1914"),
    Source("homero", "iliada", 2199, "iliada.txt", "Samuel Butler, prosa"),
    Source("homero", "odisea", 1727, "odisea.txt", "Samuel Butler, prosa"),
    Source("apolonio", "argonautica", 830, "argonautica.txt", "R. C. Seaton, 1912"),
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

    los .txt de gutenberg vienen con finales de linea CRLF, asi que hay que
    normalizarlos **antes** de partir por parrafos: si no, "\\n\\n" no encuentra
    nada y el libro entero queda como un solo parrafo.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [" ".join(block.split()) for block in text.split("\n\n")]
    return "\n\n".join(block for block in paragraphs if block)


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
