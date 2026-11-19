"""arnes de evaluacion: corre las configuraciones y arma la tabla de ablacion.

responsable: ximena.

la medida principal **no necesita un juez**. Las 20 preguntas contestables
traen los pasajes verbatim que las respaldan, copiados del corpus al
verificarlas a mano, asi que se puede comprobar directo si el retrieval
recupero el pasaje bueno. Eso mide lo unico que cambia entre configuraciones,
que es el retrieval, y no depende de como quedo redactada la respuesta ni de
cuota de api.

el juez con llm queda para la unica cosa que si exige leer, que es si la
respuesta redactada dice el hecho correcto. Corre solo si hay GOOGLE_API_KEY y
la columna sale vacia si no.
"""

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import rag
from rag import Answer

QUESTIONS_PATH = Path("eval/questions.json")

# las filas de la tabla de ablacion. cada una cambia una sola cosa respecto a
# la anterior, asi se ve que aporto cada pieza en vez de solo "el sistema
# funciona". `llm_solo` es el control: el modelo sin corpus.
CONFIGS = {
    "llm_solo": None,
    "rag_lexico": {"alpha": 0.0, "buscar_conflictos": False},
    "rag_denso": {"alpha": 1.0, "buscar_conflictos": False},
    "rag_hibrido": {"alpha": 0.5, "buscar_conflictos": False},
    "rag_hibrido_conflictos": {"alpha": 0.5, "buscar_conflictos": True},
}

GROUPS = {
    "directa": "respuesta directa en el corpus",
    "conflicto": "versiones contradictorias",
    "fuera": "fuera del corpus",
}

# cuantas palabras seguidas tienen que coincidir para dar un pasaje por
# recuperado. con menos de 8 empiezan a pegar frases de relleno ("and when he
# had said this") que estan en cualquier parte del corpus.
SHINGLE = 8


@dataclass
class Question:
    id: str
    group: str
    question: str
    reference: str
    expected_sources: list[str]
    # los pasajes verbatim que respaldan `reference`, copiados del corpus al
    # verificar la pregunta a mano. son el patron de oro del retrieval.
    passages: list[str] = field(default_factory=list)

    @property
    def should_abstain(self) -> bool:
        return self.group == "fuera"

    @property
    def should_report_conflict(self) -> bool:
        return self.group == "conflicto"

    @property
    def gold(self) -> list[str]:
        """los pasajes en ingles, sin la glosa en español que los introduce.

        cada entrada viene como `Hesiodo, Teogonia (ll. 116-138): "..."`, asi
        que lo citable es lo que esta entre comillas.
        """
        return [cita for p in self.passages for cita in re.findall(r'"([^"]+)"', p)]


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Question(**item) for item in raw["questions"]]


# --- el patron de oro del retrieval -----------------------------------------


def _plano(text: str) -> list[str]:
    """palabras comparables: sin tildes, sin puntuacion y en minuscula.

    hay que normalizar los dos lados porque los pasajes se copiaron a mano y
    traen comillas curvas, guiones largos y los `[...]` con que acortamos las
    citas largas.
    """
    limpio = unicodedata.normalize("NFD", text.lower())
    limpio = "".join(c for c in limpio if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z]+", limpio)


def _shingles(palabras: list[str], n: int = SHINGLE) -> set[tuple]:
    return {tuple(palabras[i : i + n]) for i in range(len(palabras) - n + 1)}


def pasaje_recuperado(question: Question, chunks: list) -> bool:
    """¿alguno de los chunks recuperados contiene un pasaje de oro?

    se compara por secuencias de 8 palabras y no por el texto completo, porque
    un pasaje de oro puede caer partido entre dos chunks y porque algunos
    llevan `[...]` en la mitad. basta con que una secuencia caiga adentro.
    """
    if not question.gold:
        return False

    recuperado = set()
    for chunk in chunks:
        recuperado |= _shingles(_plano(chunk.text))

    return any(_shingles(_plano(cita)) & recuperado for cita in question.gold)


def citations_ok(result: Answer, question: Question) -> bool:
    """¿las citas apuntan a las obras donde si esta el dato?"""
    if not result.citations:
        return False
    citadas = {citation.work for citation in result.citations}
    return bool(citadas & set(question.expected_sources))


# --- el juez, solo si hay cuota ---------------------------------------------

_JUEZ = """Compara una respuesta con la respuesta de referencia.

Pregunta: {question}
Referencia: {reference}
Respuesta: {answer}

¿La respuesta afirma el mismo hecho que la referencia? Puede estar redactada
distinta y puede ser mas corta. Contesta solo SI o NO.
"""


def hay_cuota() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY"))


def is_correct(result: Answer, question: Question) -> bool | None:
    """compara la respuesta contra la de referencia, con el llm como juez.

    el prompt del juez no ve que configuracion la produjo, para no sesgarlo.
    devuelve None si no hay cuota, y esa columna sale vacia en la tabla.
    """
    if not hay_cuota():
        return None
    if result.abstained:
        return False

    veredicto = rag._gemini(
        _JUEZ.format(
            question=question.question,
            reference=question.reference,
            answer=result.text,
        )
    )
    return veredicto.strip().upper().startswith("SI")


# --- correr --------------------------------------------------------------


def _llm_solo(question: Question) -> tuple[Answer, list]:
    """el control: el modelo contestando de memoria, sin corpus.

    es la fila que justifica todo lo demas. no tiene citas porque no tiene de
    donde sacarlas, y por eso tampoco se puede abstener bien.
    """
    if not hay_cuota():
        return Answer(text="(sin cuota de api, no se corrio)"), []

    texto = rag._gemini(
        f"Responde en maximo cuatro oraciones: {question.question}"
    )
    return Answer(text=texto), []


def run(question: Question, config: str) -> tuple[Answer, list]:
    """corre una pregunta bajo una configuracion."""
    flags = CONFIGS[config]
    if flags is None:
        return _llm_solo(question)
    return rag.run(question.question, **flags)


def score(questions: list[Question], config: str) -> dict:
    """una fila de la tabla de ablacion.

    los denominadores: 20 preguntas contestables (10 directa + 10 conflicto),
    10 de conflicto y 10 de fuera.
    """
    resultados = [(q, *run(q, config)) for q in questions]

    contestables = [(q, r, c) for q, r, c in resultados if not q.should_abstain]
    conflicto = [(q, r, c) for q, r, c in resultados if q.should_report_conflict]
    directa = [(q, r, c) for q, r, c in resultados if q.group == "directa"]
    fuera = [(q, r, c) for q, r, c in resultados if q.should_abstain]

    aciertos = [is_correct(r, q) for q, r, _ in contestables]

    return {
        "configuracion": config,
        "pasaje_de_oro_20": sum(pasaje_recuperado(q, c) for q, _, c in contestables),
        "citas_a_la_obra_20": sum(citations_ok(r, q) for q, r, _ in contestables),
        "conflictos_detectados_10": sum(r.has_conflict for _, r, _ in conflicto),
        "conflictos_falsos_10": sum(r.has_conflict for _, r, _ in directa),
        "se_abstuvo_10": sum(r.abstained for _, r, _ in fuera),
        "alucino_10": sum(not r.abstained for _, r, _ in fuera),
        "acierto_juez_20": "" if None in aciertos else sum(aciertos),
    }


def ablation_table(questions: list[Question] | None = None) -> pd.DataFrame:
    """la tabla que sostiene el Resultados y Discusion del notebook."""
    questions = questions if questions is not None else load_questions()
    return pd.DataFrame([score(questions, config) for config in CONFIGS])


# --- calibrar el umbral de abstencion ---------------------------------------


def calibracion(questions: list[Question] | None = None) -> pd.DataFrame:
    """la confianza del retrieval por grupo, para fijar ABSTAIN_THRESHOLD.

    el umbral se escoge con esta tabla y no a ojo: tiene que quedar entre la
    confianza de las preguntas que estan en el corpus y la de las que no.
    """
    from index import DenseIndex, load_chunks

    questions = questions if questions is not None else load_questions()
    dense = DenseIndex(load_chunks())

    filas = []
    for q in questions:
        mejor = dense.search(q.question, k=1)
        filas.append({
            "grupo": q.group,
            "id": q.id,
            "confianza": round(mejor[0][1], 4) if mejor else 0.0,
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    print(ablation_table().to_string(index=False))
