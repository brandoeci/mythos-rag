"""chunking, embeddings y los dos indices: chroma (denso) y bm25 (lexico).

responsable: hildebrando.

el indice queda persistido en `index/` para que el notebook del entregable no
tenga que reconstruirlo el dia de la revision.

el bm25 lo escribimos nosotros en vez de usar rank-bm25, igual que los n-gramas
del taller 1 y el cbow del taller 2: son treinta lineas, se entiende que hace
cada parte y nos quita una dependencia del entregable.
"""

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

INDEX_DIR = Path("index")
COLLECTION = "mitologia"

# modelo multilingue: el corpus esta en ingles y las preguntas van en español,
# asi que el embedding tiene que cruzar los dos idiomas.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

CHUNK_WORDS = 220
CHUNK_OVERLAP = 50

# constantes de bm25. son los valores estandar, no los calibramos.
K1 = 1.5
B = 0.75


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


def _slug(text: str) -> str:
    """un id corto y estable para el chunk.

    chroma exige ids unicos y los usamos tambien para volver del indice al
    Chunk, asi que no pueden cambiar entre corridas.
    """
    plain = unicodedata.normalize("NFD", text.lower())
    plain = "".join(c for c in plain if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", plain).strip("-")


def chunk_section(section) -> list[Chunk]:
    """parte una seccion en chunks con solape.

    el solape existe para que una frase partida por la mitad siga siendo
    recuperable desde los dos lados.

    el texto queda con espacios simples: dentro de la seccion los parrafos
    estan separados por "\\n\\n" y al cortar por ventana de palabras esa marca
    se pierde. no nos importa porque la cita es un pasaje, no la pagina, pero
    hay que saberlo si algun dia se quiere mostrar el parrafo completo.
    """
    words = section.text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    base = _slug(f"{section.work} {section.locator}")

    chunks = []
    for offset, start in enumerate(range(0, len(words), step)):
        window = words[start : start + CHUNK_WORDS]

        # el ultimo pedazo de una seccion puede quedar en nada mas que el
        # solape del anterior. es texto repetido, no aporta y si ensucia los
        # conteos del bm25.
        if offset > 0 and len(window) <= CHUNK_OVERLAP:
            break

        chunks.append(
            Chunk(
                id=f"{base}-{offset:03d}",
                text=" ".join(window),
                author=section.author,
                work=section.work,
                locator=section.locator,
                offset=offset,
            )
        )
    return chunks


def build_chunks(corpus: dict) -> list[Chunk]:
    """todos los chunks del corpus, en orden.

    recibe lo que devuelve `corpus.load_corpus()` y pasa por `iter_sections`,
    nunca por el texto crudo: el crudo trae prefacios, notas y, en la antologia
    #348, obras de otros autores.
    """
    from corpus import iter_sections

    chunks = []
    for source, text in corpus.items():
        for section in iter_sections(source, text):
            chunks.extend(chunk_section(section))
    return chunks


# --- indice lexico (bm25) ----------------------------------------------------

# palabras que aparecen en casi toda pregunta y no discriminan nada. las
# sacamos de los dos lados (documentos y consulta) para que no inflen el score.
STOPWORDS = {
    "a", "al", "ante", "con", "como", "cual", "cuando", "de", "del", "donde",
    "el", "en", "entre", "es", "esta", "este", "la", "las", "lo", "los", "mas",
    "o", "para", "por", "que", "quien", "se", "segun", "si", "sin", "sobre",
    "su", "sus", "un", "una", "uno", "y",
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "he",
    "her", "his", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "they", "to", "was", "were", "which", "who", "with",
}


def tokenize(text: str) -> list[str]:
    """tokens para bm25: minusculas, sin tildes y solo letras.

    bm25 compara tokens, no lee, asi que aqui si conviene limpiar duro. el
    texto verbatim para citar vive aparte, en el Chunk.

    quitamos las tildes porque el corpus esta en ingles y las preguntas en
    español: asi "Penelope" y "Penélope" caen en el mismo token, que es de los
    pocos casos en que el lexico puede cruzar los dos idiomas.
    """
    plain = unicodedata.normalize("NFD", text.lower())
    plain = "".join(c for c in plain if unicodedata.category(c) != "Mn")
    return [t for t in re.findall(r"[a-z]+", plain) if t not in STOPWORDS]


# los nombres propios en español no aparecen **ni una vez** en el corpus, que
# esta en ingles: "afrodita" da 0 y "aphrodite" da 143. sin esta tabla el bm25
# no devuelve nada para nuestras preguntas y el hibrido no aporta nada sobre el
# denso, que si cruza los dos idiomas por su cuenta.
#
# lo que no esperabamos: para cada dios el nombre **romano** es mas frecuente
# que el griego, porque butler traduce la iliada y la odisea con los romanos.
# jove 949 contra zeus 878, juno 306 contra hera 180, venus 199 contra aphrodite
# 143, vulcan 114 contra hephaestus 76. por eso cada entrada lleva las dos
# formas: preguntar por zeus y no buscar jove deja afuera la mitad del corpus.
#
# solo se expande la consulta, nunca los documentos: asi la tabla se puede
# cambiar sin reindexar, y un chunk nunca queda con tokens que no tiene.
NOMBRES = {
    "absirto": ["absyrtus", "apsyrtus"],
    "adonis": ["adonis"],
    "afrodita": ["aphrodite", "venus", "cytherea"],
    "apolo": ["apollo", "phoebus"],
    "aquiles": ["achilles"],
    "ares": ["ares", "mars"],
    "argo": ["argo", "argonauts"],
    "argonautas": ["argonauts"],
    "artemisa": ["artemis", "diana"],
    "atenea": ["athena", "athene", "minerva"],
    "cerbero": ["cerberus"],
    "citerea": ["cytherea"],
    "cronos": ["cronos", "cronus", "saturn"],
    "demeter": ["demeter", "ceres"],
    "demiurgo": ["demiurge"],
    "eros": ["eros", "cupid"],
    "escila": ["scylla"],
    "gea": ["gaea"],
    "hades": ["hades", "pluto"],
    "hefesto": ["hephaestus", "vulcan"],
    "hera": ["hera", "juno"],
    "hermes": ["hermes", "mercury"],
    "hiperborea": ["hyperborea", "hyperborean", "hyperboreans"],
    "jason": ["jason"],
    "jupiter": ["jupiter", "jove", "zeus"],
    "medea": ["medea"],
    "mitra": ["mithras", "mithra"],
    "olimpo": ["olympus"],
    "ovidio": ["ovid"],
    "pandora": ["pandora"],
    "persefone": ["persephone", "proserpine"],
    "poseidon": ["poseidon", "neptune"],
    "prometeo": ["prometheus"],
    "tetis": ["thetis"],
    "tule": ["thule"],
    "ulises": ["ulysses", "odysseus"],
    "venus": ["venus", "aphrodite", "cytherea"],
    "vulcano": ["vulcan", "hephaestus"],
    "zeus": ["zeus", "jove", "jupiter"],
    # estos tres son del grupo "fuera": se escriben casi igual en los dos
    # idiomas, asi que si el bm25 no los encuentra es porque de verdad no estan.
    "arquimedes": ["archimedes"],
    "gilgamesh": ["gilgamesh"],
    "ishtar": ["ishtar"],
}


def tokenize_query(text: str) -> list[str]:
    """tokens de la consulta, con los nombres propios traducidos al corpus.

    el token en español se deja ademas del traducido: no estorba (no esta en
    ningun documento, asi que su idf no suma) y deja ver en el debug que la
    expansion se aplico.
    """
    tokens = []
    for token in tokenize(text):
        tokens.append(token)
        tokens.extend(NOMBRES.get(token, []))
    return tokens


class Bm25Index:
    """bm25 sobre los chunks.

    score(q, d) = suma sobre los terminos t de q de
        idf(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

    el idf castiga los terminos que estan en todas partes y el denominador
    normaliza por el largo del documento, que es justo lo que le falta a un
    conteo pelado.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.docs = [tokenize(chunk.text) for chunk in chunks]
        self.lengths = np.array([len(doc) for doc in self.docs], dtype=float)
        self.avgdl = self.lengths.mean()

        # en cuantos documentos aparece cada termino
        self.df = Counter()
        for doc in self.docs:
            self.df.update(set(doc))

        # de termino a {indice del documento: frecuencia}. recorrer solo los
        # documentos que contienen el termino es lo que hace que la busqueda no
        # tenga que tocar los 4 mil chunks en cada consulta.
        self.postings: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(self.docs):
            for token, freq in Counter(doc).items():
                self.postings.setdefault(token, {})[i] = freq

        n = len(self.docs)
        self.idf = {
            token: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for token, count in self.df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """el score de bm25 de la consulta contra cada chunk."""
        scores = np.zeros(len(self.docs))
        for token in tokenize_query(query):
            if token not in self.postings:
                continue
            idf = self.idf[token]
            for i, freq in self.postings[token].items():
                norm = K1 * (1 - B + B * self.lengths[i] / self.avgdl)
                scores[i] += idf * freq * (K1 + 1) / (freq + norm)
        return scores

    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        scores = self.scores(query)
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]


# --- indice denso (chroma) ---------------------------------------------------

# e5 pide estos prefijos: fue entrenado con ellos y sin ellos la similitud baja.
# el asimetrico es a proposito, la pregunta y el pasaje no cumplen el mismo rol.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def _encoder():
    """carga el modelo de embeddings una sola vez.

    se importa aqui adentro y no arriba porque sentence-transformers se demora
    en cargar torch, y los modulos que solo usan el bm25 no tienen por que
    pagar eso.
    """
    from sentence_transformers import SentenceTransformer

    global _ENCODER
    if _ENCODER is None:
        _ENCODER = SentenceTransformer(EMBEDDING_MODEL)
    return _ENCODER


_ENCODER = None


def embed_passages(texts: list[str]) -> np.ndarray:
    return _encoder().encode(
        [PASSAGE_PREFIX + t for t in texts],
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=True,
    )


@lru_cache(maxsize=512)
def embed_query(text: str) -> np.ndarray:
    """el embedding de la pregunta.

    se cachea porque el pipeline la pide dos veces: una para el ranking y otra
    para la confianza con la que se decide si abstenerse.
    """
    return _encoder().encode(QUERY_PREFIX + text, normalize_embeddings=True)


class DenseIndex:
    """los chunks en chroma, persistido en INDEX_DIR.

    los embeddings se calculan localmente con sentence-transformers: indexar
    por api quemaria la cuota que necesitamos para generar.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        import chromadb

        self.by_id = {chunk.id: chunk for chunk in chunks}
        self.client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.collection = self.client.get_or_create_collection(
            COLLECTION,
            # coseno y no la l2 que trae por defecto. los embeddings ya salen
            # normalizados, asi que la distancia es 1 - similitud.
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, chunks: list[Chunk]) -> None:
        """calcula los embeddings y los guarda. solo se corre una vez."""
        if self.collection.count() == len(chunks):
            print(f"el indice ya tiene los {len(chunks)} chunks, no lo reconstruimos")
            return

        vectors = embed_passages([chunk.text for chunk in chunks])
        # chroma se atora si se le mandan los 4 mil de una
        for start in range(0, len(chunks), 500):
            batch = chunks[start : start + 500]
            self.collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=vectors[start : start + 500].tolist(),
            )
        print(f"indexados {self.collection.count()} chunks en {INDEX_DIR}")

    def search(self, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
        result = self.collection.query(
            query_embeddings=[embed_query(query).tolist()],
            n_results=k,
        )
        pairs = []
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
            # la pasamos a similitud para que, como en el bm25, mas sea mejor
            pairs.append((self.by_id[chunk_id], 1 - distance))
        return pairs


# --- hibrido -----------------------------------------------------------------


def _normalize(pairs: list[tuple[Chunk, float]]) -> dict[str, float]:
    """lleva los scores de un ranking a [0, 1].

    hace falta porque bm25 y coseno no viven en la misma escala: el bm25 de una
    pregunta con nombres propios pasa de 20 y el coseno nunca pasa de 1. sin
    esto, alpha no significaria nada.
    """
    if not pairs:
        return {}
    values = [score for _, score in pairs]
    low, high = min(values), max(values)
    if high == low:
        return {chunk.id: 1.0 for chunk, _ in pairs}
    return {chunk.id: (score - low) / (high - low) for chunk, score in pairs}


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
        """devuelve los k chunks mas relevantes con su score normalizado.

        se le pide a cada indice el doble de k porque la fusion solo puede
        premiar a un chunk que al menos uno de los dos haya visto, y los dos
        rankings casi no se solapan.
        """
        chunks: dict[str, Chunk] = {}
        dense_pairs = lexical_pairs = []

        if self.alpha > 0:
            dense_pairs = self.dense.search(query, k=2 * k)
        if self.alpha < 1:
            lexical_pairs = self.lexical.search(query, k=2 * k)

        for chunk, _ in list(dense_pairs) + list(lexical_pairs):
            chunks[chunk.id] = chunk

        dense_scores = _normalize(dense_pairs)
        lexical_scores = _normalize(lexical_pairs)

        fused = {
            chunk_id: self.alpha * dense_scores.get(chunk_id, 0.0)
            + (1 - self.alpha) * lexical_scores.get(chunk_id, 0.0)
            for chunk_id in chunks
        }
        top = sorted(fused, key=lambda chunk_id: -fused[chunk_id])[:k]
        return [(chunks[chunk_id], fused[chunk_id]) for chunk_id in top]


# --- construir y guardar -----------------------------------------------------

CHUNKS_PATH = INDEX_DIR / "chunks.json"


def save_chunks(chunks: list[Chunk]) -> None:
    """guarda los chunks para no volver a partir el corpus en cada corrida."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(
        json.dumps([vars(chunk) for chunk in chunks], ensure_ascii=False),
        encoding="utf-8",
    )


def load_chunks() -> list[Chunk]:
    return [Chunk(**item) for item in json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))]


def build_all() -> tuple[list[Chunk], Bm25Index, DenseIndex]:
    """parte el corpus y arma los dos indices. es lo que corre el notebook."""
    from corpus import load_corpus

    if CHUNKS_PATH.exists():
        chunks = load_chunks()
        print(f"{len(chunks)} chunks leidos de {CHUNKS_PATH}")
    else:
        chunks = build_chunks(load_corpus())
        save_chunks(chunks)
        print(f"{len(chunks)} chunks nuevos, guardados en {CHUNKS_PATH}")

    lexical = Bm25Index(chunks)
    dense = DenseIndex(chunks)
    dense.build(chunks)
    return chunks, lexical, dense


if __name__ == "__main__":
    build_all()
