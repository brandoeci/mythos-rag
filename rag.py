"""el pipeline rag: el grafo de langgraph que responde citando el corpus.

responsable: lancheros.

mientras los nodos esten en construccion, `answer()` corre en **modo stub** y
devuelve datos falsos con la forma definitiva. asi la interfaz (`app.py`) y el
arnes de evaluacion (`evaluate.py`) se construyen desde el dia 1 sin esperar a
que el pipeline funcione. poner `STUB = False` cuando los nodos esten listos.

este archivo define el contrato entre los tres modulos. si cambia `Answer`,
se avisa al grupo antes de tocarlo.
"""

from dataclasses import dataclass, field
from typing import Literal, TypedDict
from zlib import crc32

STUB = True

# umbral de confianza del retrieval por debajo del cual el sistema se abstiene.
# se calibra con el grupo 3 de las preguntas de evaluacion, no a ojo.
ABSTAIN_THRESHOLD = 0.35


@dataclass(frozen=True)
class Citation:
    """de donde salio el dato. `text` va verbatim para que se pueda verificar."""

    author: str
    work: str
    locator: str
    text: str

    def __str__(self) -> str:
        return f"{self.author}, {self.work} ({self.locator})"


@dataclass(frozen=True)
class Version:
    """una de las lecturas en conflicto sobre el mismo hecho."""

    claim: str
    citation: Citation


@dataclass
class Answer:
    """lo que devuelve el sistema. es el contrato con app.py y evaluate.py."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    conflicting_versions: list[Version] = field(default_factory=list)
    abstained: bool = False

    @property
    def has_conflict(self) -> bool:
        return len(self.conflicting_versions) > 1


class State(TypedDict, total=False):
    """el estado que viaja por el grafo."""

    question: str
    chunks: list
    scores: list[float]
    relevant: bool
    versions: list[Version]
    answer: Answer


# --- nodos del grafo ---------------------------------------------------------
# cada uno recibe el State y devuelve solo las claves que modifica.


def retrieve(state: State) -> State:
    """retrieval hibrido: denso + bm25, con fusion de los dos rankings."""
    raise NotImplementedError("lancheros")


def grade(state: State) -> State:
    """decide si los chunks recuperados realmente responden la pregunta."""
    raise NotImplementedError("lancheros")


def detect_conflict(state: State) -> State:
    """agrupa por fuente, extrae la afirmacion de cada una y las compara."""
    raise NotImplementedError("lancheros")


def generate(state: State) -> State:
    """redacta la respuesta en español con citas obligatorias."""
    raise NotImplementedError("lancheros")


def abstain(state: State) -> State:
    """responde que el corpus no tiene el dato, en vez de inventarlo."""
    return {
        "answer": Answer(
            text="eso no esta en el corpus que estoy consultando.",
            abstained=True,
        )
    }


def route_after_grade(state: State) -> Literal["detect_conflict", "abstain"]:
    """la arista condicional: responder o abstenerse."""
    return "detect_conflict" if state.get("relevant") else "abstain"


def build_graph():
    """arma el grafo de langgraph.

    retrieve -> grade -> (abstain | detect_conflict -> generate)
    """
    raise NotImplementedError("lancheros")


# --- punto de entrada --------------------------------------------------------


def answer(question: str) -> Answer:
    """responde una pregunta sobre el corpus.

    esta es la unica funcion que app.py y evaluate.py deberian llamar.
    """
    if STUB:
        return _stub_answer(question)

    graph = build_graph()
    return graph.invoke({"question": question})["answer"]


def _stub_answer(question: str) -> Answer:
    """datos falsos con la forma real, para desbloquear ui y evaluacion.

    el caso se escoge por hash de la pregunta, asi que dos corridas del arnes
    de evaluacion dan lo mismo. se usa crc32 y no hash(), porque hash() de un
    str cambia en cada proceso de python y la evaluacion dejaria de ser
    reproducible.
    """
    demo = Citation(
        author="(stub)",
        work="(stub)",
        locator="libro i",
        text="pasaje de ejemplo, aqui iria el texto citado verbatim.",
    )

    case = crc32(question.encode("utf-8")) % 3
    if case == 0:
        return Answer(text=f"respuesta de ejemplo a: {question}", citations=[demo])
    if case == 1:
        return Answer(
            text="las fuentes no coinciden en esto.",
            citations=[demo],
            conflicting_versions=[
                Version("version de la fuente a", demo),
                Version("version de la fuente b", demo),
            ],
        )
    return abstain({"question": question})["answer"]
