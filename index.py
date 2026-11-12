"""chunking, embeddings y los dos indices: chroma (denso) y bm25 (lexico).

responsable: persona a.

el indice queda persistido en `index/` para que el notebook del entregable no
tenga que reconstruirlo el dia de la revision.
"""

from dataclasses import dataclass
from pathlib import Path

INDEX_DIR = Path("index")
COLLECTION = "mitologia"

# modelo multilingue: el corpus esta en ingles y las preguntas van en español,
# asi que el embedding tiene que cruzar los dos idiomas.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

CHUNK_WORDS = 220
CHUNK_OVERLAP = 50


@dataclass(frozen=True)
class Chunk:
    """un fragmento citable del corpus.

    `locator` es lo que ve el usuario en la cita, asi que tiene que servirle
    para encontrar el pasaje en el libro: canto, libro, seccion.
    """

    id: str
    text: str
    author: str
    work: str
    locator: str
    offset: int

    @property
    def metadata(self) -> dict:
        return {
            "author": self.author,
            "work": self.work,
            "locator": self.locator,
            "offset": self.offset,
        }


def chunk_document(text: str, author: str, work: str) -> list[Chunk]:
    """parte una obra en chunks con solape.

    el solape existe para que una frase partida por la mitad siga siendo
    recuperable desde los dos lados.
    """
    raise NotImplementedError("persona a")


def build_dense_index(chunks: list[Chunk]):
    """mete los chunks en chroma, persistido en INDEX_DIR.

    los embeddings se calculan localmente con sentence-transformers: indexar
    por api quemaria la cuota que necesitamos para generar.
    """
    raise NotImplementedError("persona a")


def build_lexical_index(chunks: list[Chunk]):
    """indice bm25 sobre los mismos chunks.

    aqui si conviene limpiar duro (minusculas, sin signos): bm25 compara
    tokens, no lee. el texto verbatim para citar vive en el Chunk.
    """
    raise NotImplementedError("persona a")


class HybridRetriever:
    """denso + lexico con fusion de rankings.

    el denso entiende la pregunta parafraseada; el lexico acierta en nombres
    propios, que es justo donde el denso falla. la comparacion entre los dos
    es una fila de la tabla de ablacion, asi que ambos tienen que poder
    correrse solos.
    """

    def __init__(self, dense, lexical, alpha: float = 0.5) -> None:
        self.dense = dense
        self.lexical = lexical
        self.alpha = alpha  # 1.0 = solo denso, 0.0 = solo lexico

    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        """devuelve los k chunks mas relevantes con su score normalizado."""
        raise NotImplementedError("persona a")
