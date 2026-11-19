"""el pipeline rag: el grafo de langgraph que responde citando el corpus.

responsable: lancheros.

el grafo es

    retrieve -> grade -> (abstain | detect_conflict -> generate)

la arista de `grade` es condicional: si lo recuperado no se parece a la
pregunta, el sistema se abstiene en vez de contestar con lo que haya.

la decision de diseño que sostiene todo: **el modelo nunca escribe la cita**.
Se le pasan los pasajes numerados y solo responde cuales uso; la cita la
armamos nosotros desde el chunk, con el autor, la obra y el locator que ya
traia. Asi una cita inventada no es improbable, es imposible.

si no hay `GOOGLE_API_KEY` el nodo `generate` responde de forma extractiva:
devuelve el pasaje recuperado tal cual. es peor de leer pero no puede alucinar,
y deja correr el sistema completo sin cuota.

este archivo define el contrato entre los tres modulos. si cambia `Answer`,
se avisa al grupo antes de tocarlo.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Literal, TypedDict

# cuantos chunks se le pasan al generador. con mas de 6 la respuesta se diluye
# y se gasta contexto; con menos se queda sin el segundo lado de la
# contradiccion, que casi siempre esta en otra obra.
TOP_K = 6

# piso de confianza del retrieval. se calibro con el grupo "fuera" y el
# resultado fue que **no sirve**: las similitudes de los tres grupos se
# superponen (fuera 0.766-0.805, directa 0.789-0.826), asi que cualquier umbral
# que atrape las preguntas sin respuesta tumba tambien las buenas. queda en
# 0.70, debajo de todo lo observado, o sea solo como piso para una consulta
# degenerada. la abstencion de verdad la decide `sin_evidencia_lexica`.
ABSTAIN_THRESHOLD = 0.70

MODEL = "gemini-2.5-flash"


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
    alpha: float  # como se mezclan los dos indices. lo fija la configuracion
    usar_llm: bool  # si el nodo generate redacta con gemini o es extractivo
    buscar_conflictos: bool  # si corre detect_conflict. es una fila de la tabla
    chunks: list
    scores: list[float]
    confidence: float
    relevant: bool
    versions: list[Version]
    answer: Answer


# --- los indices, cargados una sola vez -------------------------------------

_INDICES = None


def indices():
    """carga los dos indices. la primera llamada se demora, las demas no."""
    global _INDICES
    if _INDICES is None:
        import index

        chunks = index.load_chunks()
        _INDICES = (chunks, index.Bm25Index(chunks), index.DenseIndex(chunks))
    return _INDICES


def _citation(chunk, text: str | None = None) -> Citation:
    """la cita de un chunk. el texto sale del chunk, no del modelo."""
    return Citation(
        author=chunk.author,
        work=chunk.work,
        locator=chunk.locator,
        text=text if text is not None else chunk.text,
    )


# --- nodos del grafo ---------------------------------------------------------
# cada uno recibe el State y devuelve solo las claves que modifica.


def retrieve(state: State) -> State:
    """retrieval hibrido: denso + bm25, con fusion de los dos rankings."""
    from index import HybridRetriever

    _, lexical, dense = indices()
    alpha = state.get("alpha", 0.5)
    retriever = HybridRetriever(dense, lexical, alpha=alpha)
    pairs = retriever.search(state["question"], k=TOP_K)

    # la confianza para abstenerse NO puede salir del score fusionado: la
    # fusion normaliza cada ranking por consulta, asi que el primer resultado
    # sale siempre en 1.0 aunque sea basura. el coseno del denso si es absoluto
    # y comparable entre preguntas, y es el que se compara con el umbral.
    mejor = dense.search(state["question"], k=1)
    confidence = mejor[0][1] if mejor else 0.0

    return {
        "chunks": [chunk for chunk, _ in pairs],
        "scores": [score for _, score in pairs],
        "confidence": confidence,
    }


def entidades_de_la_pregunta(question: str) -> list[list[str]]:
    """los nombres propios de la pregunta, cada uno con sus formas en el corpus.

    dos fuentes: la tabla de nombres (afrodita -> aphrodite, venus, cytherea) y
    las palabras con mayuscula que no abren la pregunta, para los nombres que no
    estan en la tabla porque se escriben igual en los dos idiomas (Gilgamesh,
    Sigurd, Cuchulainn).
    """
    from index import NOMBRES, STOPWORDS, STOPWORDS_CONSULTA, tokenize

    entidades = []
    for token in tokenize(question):
        if token in NOMBRES:
            entidades.append([token] + NOMBRES[token])

    for palabra in re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}", question[1:]):
        token = next(iter(tokenize(palabra)), "")
        if not token or token in STOPWORDS or token in STOPWORDS_CONSULTA:
            continue
        if not any(token in formas for formas in entidades):
            entidades.append([token])

    return entidades


def sin_evidencia_lexica(question: str) -> bool:
    """¿la pregunta nombra entidades y **ninguna** esta en el corpus?

    es la señal de abstencion, y sale del indice **lexico**, no del denso. no
    era lo que esperabamos: medimos la similitud del coseno en los tres grupos
    de preguntas y las distribuciones se superponen casi del todo (fuera llega
    a 0.805 y directa arranca en 0.789), asi que **no existe un umbral que las
    separe**. el bm25 si: para "Sigurd", "Gilgamesh" o "Tule" devuelve cero
    porque esos terminos no estan, y para las que si estan devuelve de 4.84
    para arriba.

    la condicion es "nombra entidades **y** ninguna aparece", no "el bm25 dio
    cero". son distintas: "¿quien creo a la humanidad?" y "¿que habia al
    principio del mundo?" tambien dan cero, porque no tienen ningun nombre
    propio, y esas dos **si** tienen respuesta en el corpus. sin esa distincion
    el sistema se abstendria de dos preguntas buenas.
    """
    _, lexical, _ = indices()
    entidades = entidades_de_la_pregunta(question)
    if not entidades:
        return False
    return not any(
        forma in lexical.postings for formas in entidades for forma in formas
    )


def grade(state: State) -> State:
    """decide si los chunks recuperados realmente responden la pregunta.

    tres condiciones, y la que de verdad decide es la segunda:

    1. que haya chunks. con `alpha = 0.0` la confianza sale del indice denso
       pero los chunks salen del lexico, y sin este chequeo `generate` recibiria
       una lista vacia y se caeria en `chunks[0]`.
    2. que alguna entidad de la pregunta exista en el corpus. ver
       `sin_evidencia_lexica`.
    3. el umbral sobre el coseno, que quedo como piso para consultas
       degeneradas. lo medimos y **no separa nada** en este set de preguntas, y
       asi queda dicho en el notebook en vez de dejarlo puesto como si aportara.
    """
    hay_con_que = bool(state.get("chunks"))
    hay_evidencia = not sin_evidencia_lexica(state["question"])
    sobre_el_piso = state["confidence"] >= ABSTAIN_THRESHOLD
    return {"relevant": hay_con_que and hay_evidencia and sobre_el_piso}


# palabras que empiezan con mayuscula pero no son nombres propios. en ingles
# toda oracion arranca en mayuscula, asi que sin esta lista el detector de
# conflictos ve nombres donde no hay.
_NO_ES_NOMBRE = {
    "the", "and", "but", "for", "then", "when", "there", "this", "that", "these",
    "those", "he", "she", "it", "they", "his", "her", "him", "them", "their",
    "a", "an", "of", "in", "on", "at", "by", "to", "from", "with", "as", "so",
    "now", "thus", "yet", "if", "while", "after", "before", "who", "which",
    "what", "all", "one", "two", "three", "first", "last", "some", "no", "not",
    "nor", "or", "both", "such", "my", "your", "our", "i", "o", "oh", "ll",
    "book", "chapter", "part", "god", "gods", "goddess", "king", "son", "sons",
    "daughter", "father", "mother", "lord", "men", "man",
    # gentilicios y comunes que salieron en las pruebas haciendose pasar por
    # nombres: "The Venus of the Romans", "the Old man of the Sea"
    "greek", "greeks", "roman", "romans", "christian", "english", "latin",
    "old", "sea", "earth", "heaven", "muse", "muses", "sun", "moon", "day",
    "night", "war", "love", "death", "fate", "hope",
}


def _plano(text: str) -> list[str]:
    """las palabras de una oracion, en minuscula y sin puntuacion."""
    return re.findall(r"[a-z]+", text.lower())


def _oraciones(text: str) -> list[str]:
    return [o.strip() for o in re.split(r"(?<=[.;:!?])\s+", text) if o.strip()]


def _nombres_propios(oracion: str, excluir: set[str]) -> set[str]:
    """los nombres propios de una oracion.

    se salta la primera palabra porque en ingles siempre va en mayuscula, y
    descarta los nombres que ya venian en la pregunta: si las dos fuentes
    hablan de Hefesto eso no es el desacuerdo, el desacuerdo es *quien mas*
    aparece en cada una.
    """
    palabras = re.findall(r"\b[A-Z][a-z]{2,}\b", oracion[1:])
    return {
        p.lower() for p in palabras
        if p.lower() not in _NO_ES_NOMBRE and p.lower() not in excluir
    }


def _nombres_de_la_pregunta(question: str) -> set[str]:
    """los nombres de la pregunta, ya traducidos a como aparecen en el corpus."""
    from index import NOMBRES, tokenize

    nombres = set()
    for token in tokenize(question):
        if token in NOMBRES:
            nombres.add(token)
            nombres.update(NOMBRES[token])
    return nombres


def _afirmacion(question: str, chunk, relaciones: set[str] = frozenset()) -> str:
    """la oracion del chunk que responde la pregunta.

    es la que se le muestra al usuario como "esto dice esta fuente", asi que
    tiene que ser una oracion del texto y no un resumen nuestro.

    la palabra de la relacion pesa el triple que un nombre cualquiera. sin eso,
    para "quien era la madre de Afrodita" la oracion ganadora era "The Venus of
    the Romans was identified with the Aphrodite of the Greeks": tiene el nombre
    dos veces y no responde nada.
    """
    from index import tokenize, tokenize_query

    pedidos = set(tokenize_query(question))
    oraciones = _oraciones(chunk.text)
    if not oraciones:
        return chunk.text[:300]

    def puntaje(oracion: str) -> int:
        return len(pedidos & set(tokenize(oracion))) + 3 * len(
            relaciones & set(_plano(oracion))
        )

    return max(oraciones, key=puntaje)


def _nombres_cerca(
    oracion: str, relaciones: set[str], excluir: set[str], ventana: int = 6
) -> set[str]:
    """los nombres propios que estan pegados a la palabra de la relacion.

    "and Dione ... she was the mother of Aphrodite" contesta con Dione; los
    demas nombres de la oracion son decorado. mirar solo la ventana alrededor
    de la relacion es lo que separa el nombre que responde de los que solo
    estaban ahi, y es lo que hizo que el detector dejara de ver conflictos
    donde las fuentes si coinciden.
    """
    palabras = oracion.split()
    plano = [re.sub(r"[^a-z]", "", p.lower()) for p in palabras]
    posiciones = [i for i, p in enumerate(plano) if p in relaciones]

    nombres = set()
    for i in posiciones:
        for j in range(max(1, i - ventana), min(len(palabras), i + ventana + 1)):
            limpia = palabras[j].strip(".,;:!?\"'()[]")
            if not re.fullmatch(r"[A-Z][a-z]{2,}", limpia):
                continue
            if plano[j] not in _NO_ES_NOMBRE and plano[j] not in excluir:
                nombres.add(plano[j])
    return nombres


# que se esta preguntando, traducido a como lo dice el corpus. una fuente solo
# cuenta como una version del hecho si su oracion contiene alguna de estas
# palabras.
#
# sin este filtro el detector comparaba nombres de pasajes que ni siquiera
# responden la pregunta, y marcaba conflicto en 8 de las 10 preguntas de
# respuesta directa. las preguntas van en español y el corpus esta en ingles,
# asi que hay que traducir la relacion igual que se traducen los nombres.
RELACIONES = {
    "madre": ["mother", "bore", "born", "birth"],
    "padre": ["father", "begot", "sire", "parent"],
    "hijo": ["son", "offspring", "child", "begotten"],
    "hija": ["daughter"],
    "esposa": ["wife", "wedded", "bride", "spouse"],
    "esposo": ["husband", "wedded"],
    "contenia": ["contained", "within", "inside", "jar"],
    "murio": ["died", "death", "slain", "killed", "slew"],
    "arrojo": ["threw", "cast", "hurled", "flung", "dropped"],
    "recogio": ["received", "took", "caught", "nursed", "saved"],
    "creo": ["created", "made", "formed", "fashioned", "moulded"],
    "principio": ["beginning", "first", "chaos", "before"],
    "nombre": ["named", "called", "name"],
    "entregaron": ["gave", "given", "brought"],
    "construyo": ["built", "made", "builded"],
    "nacieron": ["born", "bore", "sprung"],
    "reparcio": ["divided", "shared", "apportioned"],
    "rescataron": ["received", "saved", "caught", "nursed"],
}


def _relaciones_de_la_pregunta(question: str) -> set[str]:
    from index import tokenize

    palabras: set[str] = set()
    for token in tokenize(question):
        palabras.update(RELACIONES.get(token, []))
    return palabras


def _agrupar(por_obra: dict) -> list[list[str]]:
    """agrupa las fuentes que comparten algun nombre en su afirmacion.

    si todas caen en un solo grupo, estan de acuerdo. dos grupos o mas quieren
    decir que el corpus da respuestas distintas al mismo hecho.

    se agrupa por transitividad y no de a pares: comparando pares, con seis
    pasajes casi siempre hay alguno que no se cruza con otro, y todo salia
    marcado como conflicto.
    """
    grupos: list[tuple[list[str], set[str]]] = []
    for obra, (_, _, nombres) in por_obra.items():
        juntar = [g for g in grupos if g[1] & nombres]
        if not juntar:
            grupos.append(([obra], set(nombres)))
            continue
        # cae en varios grupos: los une a todos en uno
        obras = [obra]
        todos: set[str] = set(nombres)
        for g in juntar:
            obras += g[0]
            todos |= g[1]
            grupos.remove(g)
        grupos.append((obras, todos))
    return [g[0] for g in grupos]


def detect_conflict(state: State) -> State:
    """agrupa por fuente, extrae la afirmacion de cada una y las compara.

    dos fuentes se consideran en conflicto cuando sus afirmaciones nombran
    personajes **distintos y sin cruce**. si Hesiodo responde "Urano" y el
    himno homerico responde "Dione", los conjuntos son disjuntos y hay dos
    versiones; si las dos dicen "Dione", coinciden y no hay nada que reportar.

    es una heuristica, no entiende el texto. lo que la hace util es que no
    puede inventar: las dos versiones que muestra son oraciones del corpus con
    su cita, y el que lee decide.
    """
    if not state.get("buscar_conflictos", True):
        return {"versions": []}

    de_la_pregunta = _nombres_de_la_pregunta(state["question"])
    relaciones = _relaciones_de_la_pregunta(state["question"])
    if not relaciones:
        # no sabemos que se esta preguntando, asi que no hay con que decidir si
        # dos fuentes hablan del mismo hecho. preferimos no reportar nada antes
        # que reportar un conflicto que no existe.
        return {"versions": []}

    # una sola afirmacion por obra: dos chunks de la Iliada no son dos fuentes
    por_obra: dict[str, tuple] = {}
    for chunk in state["chunks"]:
        if chunk.work in por_obra:
            continue
        afirmacion = _afirmacion(state["question"], chunk, relaciones)

        # la oracion tiene que nombrar al **sujeto de la pregunta**. sin esta
        # condicion, "¿quien arrojo a Hefesto del Olimpo?" devolvia tres
        # oraciones sobre Pandora, Cicreo y un tripode de Hesiodo, que no
        # mencionan a Hefesto por ningun lado: la palabra de la relacion
        # matcheaba y los nombres de alrededor eran cualquier cosa.
        if not (set(_plano(afirmacion)) & de_la_pregunta):
            continue

        nombres = _nombres_cerca(afirmacion, relaciones, de_la_pregunta)
        if nombres:
            por_obra[chunk.work] = (chunk, afirmacion, nombres)

    grupos = _agrupar(por_obra)
    if len(grupos) < 2:
        return {"versions": []}

    # una version por grupo: la de la obra que quedo mas arriba en el ranking
    orden = [chunk.work for chunk in state["chunks"]]
    versions: list[Version] = []
    for grupo in grupos:
        obra = min(grupo, key=orden.index)
        chunk, afirmacion, _ = por_obra[obra]
        versions.append(Version(afirmacion, _citation(chunk, afirmacion)))

    return {"versions": versions[:4]}


_PROMPT = """Eres un asistente que solo puede usar los pasajes que se le dan.

Pregunta: {question}

Pasajes:
{pasajes}

Instrucciones:
- Responde en español, en maximo cuatro oraciones.
- Usa unicamente lo que digan los pasajes. No agregues nada que no este ahi.
- Si los pasajes no responden la pregunta, responde exactamente: NO ESTA.
- Si dos pasajes se contradicen, di las dos versiones y de quien es cada una.
- Termina con una linea "PASAJES: n, m" con los numeros que usaste.
"""


def _gemini(prompt: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
    return llm.invoke(prompt).content


def generate(state: State) -> State:
    """redacta la respuesta en español con citas obligatorias."""
    chunks = state["chunks"]
    versions = state.get("versions", [])

    if not state.get("usar_llm", True) or not os.environ.get("GOOGLE_API_KEY"):
        return {"answer": _extractiva(state)}

    pasajes = "\n\n".join(
        f"[{i + 1}] {chunk.author}, {chunk.work} ({chunk.locator}):\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    salida = _gemini(_PROMPT.format(question=state["question"], pasajes=pasajes))

    if "NO ESTA" in salida.upper():
        return abstain(state)

    # los numeros que dijo que uso. la cita se arma desde el chunk, no desde su
    # texto: por eso no puede citar algo que no este en el corpus.
    usados = [
        chunks[int(n) - 1]
        for n in re.findall(r"\d+", salida.split("PASAJES:")[-1])
        if 1 <= int(n) <= len(chunks)
    ]
    texto = salida.split("PASAJES:")[0].strip()

    # si redacto pero no cito nada, no lo dejamos pasar: una respuesta sin cita
    # es exactamente lo que el proyecto dice no hacer.
    if not usados:
        return abstain(state)

    return {
        "answer": Answer(
            text=texto,
            citations=[
                _citation(chunk, _afirmacion(state["question"], chunk, relaciones))
                for chunk in usados
            ],
            conflicting_versions=versions,
        )
    }


def _extractiva(state: State) -> Answer:
    """la respuesta sin modelo: el pasaje recuperado, tal cual.

    se lee mal, pero no puede alucinar porque no redacta nada. es el piso
    contra el que se compara lo que agrega gemini.
    """
    chunks = state["chunks"]
    versions = state.get("versions", [])
    relaciones = _relaciones_de_la_pregunta(state["question"])
    afirmacion = _afirmacion(state["question"], chunks[0], relaciones)

    if len(versions) > 1:
        texto = "las fuentes no coinciden. " + " | ".join(
            f"{version.citation.author}: {version.claim}" for version in versions
        )
    else:
        texto = afirmacion

    return Answer(
        text=texto,
        citations=[_citation(chunks[0], afirmacion)],
        conflicting_versions=versions,
    )


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
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("detect_conflict", detect_conflict)
    builder.add_node("generate", generate)
    builder.add_node("abstain", abstain)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade)
    builder.add_edge("detect_conflict", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("abstain", END)
    return builder.compile()


# --- punto de entrada --------------------------------------------------------

_GRAPH = None


def run(
    question: str,
    alpha: float = 0.5,
    usar_llm: bool = True,
    buscar_conflictos: bool = True,
) -> tuple[Answer, list]:
    """corre el grafo y devuelve la respuesta **y** los chunks recuperados.

    los tres flags son las configuraciones de la tabla de ablacion: alpha
    mezcla los dos indices (1.0 solo denso, 0.0 solo bm25) y los otros dos
    prenden y apagan el generador y el detector de conflictos.

    la evaluacion necesita los chunks aparte de la respuesta, porque medir el
    retrieval y medir la redaccion son dos cosas distintas: si el pasaje bueno
    nunca se recupero, el generador no tenia con que acertar.
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()

    estado = _GRAPH.invoke({
        "question": question,
        "alpha": alpha,
        "usar_llm": usar_llm,
        "buscar_conflictos": buscar_conflictos,
    })
    return estado["answer"], estado.get("chunks", [])


def answer(question: str, **flags) -> Answer:
    """responde una pregunta sobre el corpus.

    esta es la unica funcion que necesita app.py.
    """
    return run(question, **flags)[0]


if __name__ == "__main__":
    import sys

    pregunta = " ".join(sys.argv[1:]) or "¿Quien era la madre de Afrodita?"
    resultado = answer(pregunta)
    print(resultado.text)
    print()
    for cita in resultado.citations:
        print(" -", cita)
    if resultado.has_conflict:
        print()
        print("versiones en conflicto:")
        for version in resultado.conflicting_versions:
            print(f" - {version.citation}: {version.claim[:140]}")
