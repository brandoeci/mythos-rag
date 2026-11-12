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

    @property
    def url(self) -> str:
        return f"https://www.gutenberg.org/cache/epub/{self.gutenberg_id}/pg{self.gutenberg_id}.txt"

    @property
    def path(self) -> Path:
        return DATA_DIR / self.filename


# TODO(hildebrando): verificar cada gutenberg_id en gutenberg.org antes de usarlo.
# los ids de abajo estan en cero a proposito: no se adivinan, se confirman uno
# por uno abriendo la ficha de la obra y copiando el numero del ebook.
# ojo al escoger la edicion: varias de estas obras tienen varias traducciones en
# gutenberg, y la traduccion cambia el texto que vamos a citar.
SOURCES: list[Source] = [
    Source("hesiodo", "teogonia", 0, "teogonia.txt"),
    Source("apolodoro", "biblioteca", 0, "biblioteca.txt"),
    Source("homero", "iliada", 0, "iliada.txt"),
    Source("homero", "odisea", 0, "odisea.txt"),
    Source("bulfinch", "the age of fable", 0, "age_of_fable.txt"),
    Source("ovidio", "metamorfosis", 0, "metamorfosis.txt"),
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
    """
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
