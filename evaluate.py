"""arnes de evaluacion: corre las 4 configuraciones y arma la tabla de ablacion.

responsable: ximena.

funciona contra el stub de rag.answer(), asi que la tabla se puede armar antes
de que el pipeline exista: al principio da numeros sin sentido, pero la
maquinaria queda probada.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rag import Answer, answer

QUESTIONS_PATH = Path("eval/questions.json")

# las 4 filas de la tabla de ablacion. cada una agrega una pieza, asi se ve que
# aporto cada una en vez de solo "el sistema funciona".
CONFIGS = ["llm_solo", "rag_denso", "rag_hibrido", "rag_hibrido_conflictos"]

GROUPS = {
    "directa": "respuesta directa en el corpus",
    "conflicto": "versiones contradictorias",
    "fuera": "fuera del corpus",
}


@dataclass
class Question:
    id: str
    group: str
    question: str
    reference: str
    expected_sources: list[str]
    # los pasajes verbatim que respaldan `reference`, copiados del corpus al
    # verificar la pregunta a mano. no los usa el arnes: son la evidencia de que
    # la pregunta se reviso de verdad y el material para la discusion.
    passages: list[str] = field(default_factory=list)

    @property
    def should_abstain(self) -> bool:
        return self.group == "fuera"

    @property
    def should_report_conflict(self) -> bool:
        return self.group == "conflicto"


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Question(**item) for item in raw["questions"]]


def run(question: Question, config: str) -> Answer:
    """corre una pregunta bajo una configuracion.

    mientras rag.STUB este en True todas las configuraciones devuelven lo
    mismo. lancheros expone los flags para diferenciarlas.
    """
    return answer(question.question)


def is_correct(result: Answer, question: Question) -> bool:
    """compara la respuesta contra la de referencia.

    se califica con el llm como juez, pasandole pregunta, referencia y
    respuesta, y pidiendole solo si coinciden en el hecho. el prompt del juez
    no ve que configuracion la produjo, para no sesgarlo.
    """
    raise NotImplementedError("ximena")


def citations_ok(result: Answer, question: Question) -> bool:
    """¿las citas apuntan a las fuentes donde si esta el dato?"""
    if not result.citations:
        return False
    cited = {citation.work for citation in result.citations}
    return bool(cited & set(question.expected_sources))


def score(questions: list[Question], config: str) -> dict:
    """una fila de la tabla de ablacion."""
    results = [(q, run(q, config)) for q in questions]

    answerable = [(q, r) for q, r in results if not q.should_abstain]
    conflicting = [(q, r) for q, r in results if q.should_report_conflict]
    unanswerable = [(q, r) for q, r in results if q.should_abstain]

    return {
        "configuracion": config,
        "aciertos": sum(is_correct(r, q) for q, r in answerable),
        "citas_correctas": (
            "n/a" if config == "llm_solo" else sum(citations_ok(r, q) for q, r in answerable)
        ),
        "conflictos_detectados": sum(r.has_conflict for _, r in conflicting),
        "abstuvo_cuando_debia": sum(r.abstained for _, r in unanswerable),
        "alucino": sum(not r.abstained for _, r in unanswerable),
    }


def ablation_table(questions: list[Question] | None = None) -> pd.DataFrame:
    """la tabla que sostiene el Resultados y Discusion del notebook."""
    questions = questions if questions is not None else load_questions()
    return pd.DataFrame([score(questions, config) for config in CONFIGS])


if __name__ == "__main__":
    print(ablation_table().to_string(index=False))
