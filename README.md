# mythos-rag

asistente de consulta sobre un corpus de mitologia griega de dominio publico que responde en
lenguaje natural **citando el pasaje exacto**, y que reconoce cuando el corpus no tiene la
respuesta o cuando las fuentes se contradicen.

proyecto final del curso de inteligencia artificial generativa con grandes modelos del lenguaje,
maestria en informatica, escuela colombiana de ingenieria julio garavito.

> **estado: en construccion.** este readme describe el diseño acordado. lo que ya esta hecho
> aparece marcado en [estado del proyecto](#estado-del-proyecto).

## la problematica

los llm responden con total seguridad sobre temas especificos, se equivocan, y nunca dicen de
donde sacaron el dato.

en mitologia eso es peor, porque no existe *una* respuesta. quien nace de quien, quien mato a
quien, cuantos trabajos fueron: hesiodo, apolodoro y ovidio no coinciden. si le preguntas a un
llm pelado, mezcla las versiones en una sola respuesta que suena perfecta y no es de nadie. no
tienes como verificarla.

a quien le sirve: a un estudiante de literatura o historia que necesita citar una fuente, no
"algo que dijo una ia". y a cualquiera que quiera una respuesta que se pueda ir a comprobar al
texto.

por que no basta un ctrl+f: buscar por palabra clave no responde "por que medea mato a sus
hijos", porque eso esta repartido en parrafos que no contienen la palabra "por que". se necesita
lenguaje natural sobre el corpus. y el requisito de *citar* es lo que obliga a rag y no a un llm
solo.

## objetivos

**general.** construir un asistente de consulta sobre un corpus de mitologia griega de dominio
publico que responda en lenguaje natural citando el pasaje exacto, y que reconozca cuando el
corpus no tiene la respuesta o tiene mas de una version.

**especificos.**

1. armar el corpus e indexarlo: descargar las fuentes, partir en chunks, generar embeddings y
   guardarlas en un vector store con metadata de autor y obra.
2. implementar el pipeline rag con citacion obligatoria y un guardrail para que responda "eso no
   esta en el corpus" en vez de inventar.
3. evaluar contra un set de preguntas comparando llm solo vs. rag, midiendo respuestas correctas,
   inventadas y abstenciones.

## arquitectura

```
corpus/ (gutenberg, varias obras)
   |  chunking con metadata {autor, obra, libro, offset}
chromadb (persistido en disco)  +  indice bm25
   |
+- grafo langgraph ------------------------------------+
|  retrieve (hibrido: denso + lexico)                  |
|     |                                                |
|  grade: los chunks responden la pregunta?            |
|     +-- no --> abstain  ("no esta en el corpus")     |
|     +-- si --> |                                     |
|  detect_conflict: las fuentes discrepan?             |
|     |                                                |
|  generate: respuesta + citas (+ ambas versiones)     |
+------------------------------------------------------+
   |
gradio
```

### decisiones y por que

| decision | razon |
| --- | --- |
| **langgraph** para el pipeline | el flujo tiene aristas condicionales reales (abstenerse o responder, hay conflicto o no). es para lo que sirve, no esta metido a la fuerza |
| **embeddings locales** (`sentence-transformers`) | indexar un corpus grande por api quema cuota; local es gratis e ilimitado |
| **gemini** solo para generacion | la cuota se reserva para donde si hace falta un modelo bueno |
| **modelo de embeddings multilingue** | el corpus esta en ingles, que es donde esta lo bueno en dominio publico, y las preguntas y respuestas van en español. el retrieval cruza idiomas |
| **chromadb** en vez de faiss | persiste a disco y filtra por metadata sin codigo extra. lo segundo se necesita para citar por autor |
| **codigo en modulos `.py`**, el notebook importa | tres personas editando el json de un `.ipynb` es un infierno de merges, y el codigo limpio se evalua |
| **cache en disco de toda llamada al llm** | permite re-correr sin gastar cuota, y garantiza que el entregable corra el dia de la revision aunque la api este caida |

## corpus

fuentes de dominio publico (project gutenberg). varias obras a proposito: **sin varias fuentes no
hay contradicciones**, y la contradiccion es la identidad del proyecto.

- teogonia, hesiodo
- biblioteca, apolodoro
- iliada y odisea, homero
- the age of fable, bulfinch
- metamorfosis, ovidio

el corpus se baja a `data/` en la primera corrida y despues se lee del disco. `data/` e `index/`
no estan en el repo porque se regeneran solos.

## estructura del repo

```
corpus.py          descarga y pre-procesamiento
index.py           chunking, embeddings, chroma y bm25
rag.py             el grafo de langgraph: retrieve, grade, conflict, generate
app.py             interfaz gradio
evaluate.py        arnes de evaluacion y tablas
eval/
  questions.json   las 30 preguntas con respuesta de referencia
proyecto.ipynb     el entregable, importa los modulos
```

el contrato entre modulos, acordado antes de escribir codigo:

```python
def answer(question: str) -> Answer:
    """Answer: text, citations[], conflicting_versions[], abstained: bool"""
```

con ese contrato y un stub que devuelve datos falsos, la interfaz y la evaluacion se construyen
sin esperar a que el pipeline este listo.

## evaluacion

30 preguntas en tres grupos de 10:

| grupo | que prueba | metrica |
| --- | --- | --- |
| respuesta directa en el corpus | retrieval y fidelidad | aciertos, y si la cita apunta al pasaje correcto |
| versiones contradictorias | que no escoja una en silencio | reporto las dos? cito ambas fuentes? |
| fuera del corpus | que no invente | tasa de abstencion vs. tasa de alucinacion |

las preguntas las escriben los tres, 10 cada uno, con revision cruzada. si las escribe solo quien
construyo el retrieval, salen preguntas que el sistema ya sabe responder y la evaluacion no vale
nada.

y la tabla de ablacion, que es el argumento del proyecto. cada fila agrega una pieza, asi se ve
que aporto cada una:

| configuracion | aciertos | citas correctas | conflictos detectados | abstuvo cuando debia |
| --- | --- | --- | --- | --- |
| llm solo, sin corpus | | n/a | | |
| rag denso | | | | |
| rag hibrido | | | | |
| rag hibrido + conflictos | | | | |

no se promete "eliminamos las alucinaciones". se reporta *reducimos y hacemos verificables*, con
los numeros de esta tabla.

## estado del proyecto

### nucleo

- [ ] corpus multi-fuente indexado, cada chunk con autor/obra/libro
- [ ] retrieval hibrido: embeddings + bm25
- [ ] respuesta con cita obligatoria al pasaje
- [ ] abstencion cuando la respuesta no esta en el corpus
- [ ] deteccion de versiones en conflicto
- [ ] evaluacion con ablacion de las 4 configuraciones
- [ ] interfaz gradio: respuesta + citas + panel de conflictos

### ambicioso (stretch, se corta el 23 de noviembre si no esta empezado)

- [ ] grafo de entidades: relaciones (hijo-de, mato-a, casado-con) extraidas con el llm
- [ ] aristas en conflicto marcadas en el grafo segun la fuente

## reparto

| persona | responsable de | entrega |
| --- | --- | --- |
| a | corpus e indice | `corpus.py`, `index.py`, el indice en disco, tabla de tokens y chunks por obra |
| b | pipeline rag | `rag.py` con `answer()` funcionando y el diagrama del grafo |
| c | evaluacion, interfaz y video | `eval/questions.json`, `evaluate.py`, `app.py`, las tablas y el video |

## cronograma

| fechas | fase | hito que cierra la fase |
| --- | --- | --- |
| 12-13 nov | arranque | corpus elegido, contrato de `answer()` firmado, repo con modulos vacios y el stub |
| 14-17 nov | rebanada vertical | **una** pregunta se responde de punta a punta con cita. fea, un solo libro, sin hibrido. pero funciona |
| 18-22 nov | nucleo | hibrido, abstencion, conflictos. corpus completo. las 30 preguntas escritas |
| 23-26 nov | evaluacion y ui | las 4 corridas hechas, tabla de ablacion lista, gradio funcionando. **el 23 se decide si el grafo entra o se corta** |
| 27-28 nov | escritura | **code freeze el 27.** notebook escrito con los numeros reales |
| 29 nov | video | grabado y montado |
| 30 nov | entrega | enviado en la mañana, no a las 23:59 |

dos reglas sobre el calendario:

**el hito del 17 es el que importa.** si el 17 no hay una respuesta de punta a punta, por fea que
sea, hay que recortar ahi mismo y no el 28. un pipeline que integra el dia 25 es un pipeline que
no integra.

**el code freeze del 27 es sagrado.** el video vale 30%. un video grabado a la carrera a
medianoche cuesta mas puntos que cualquier feature que alcance a entrar el 28.

## riesgos

| riesgo | como se mata |
| --- | --- |
| se acaba la cuota de la api en la semana final | embeddings locales + cache en disco desde el dia 1, no despues |
| el notebook no corre el dia de la revision | indice persistido y salidas guardadas en el `.ipynb`. cero celdas que tarden 40 minutos |
| merge hell en el notebook | codigo en `.py`, el notebook solo importa y muestra |
| scope creep con el grafo | es stretch y tiene fecha de corte: 23 de noviembre |
| el detector de conflictos no detecta nada | verificar el 18 que el corpus **si** se contradice sobre las entidades elegidas, antes de construir el detector encima |

el ultimo es el riesgo silencioso: si las fuentes elegidas no se contradicen, la pieza estrella no
tiene nada que mostrar. se chequea a mano, en media hora.

## como correrlo

todavia no hay nada que correr. cuando lo haya:

```bash
python -m venv .venv
source .venv/bin/activate        # windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # y poner la api key de gemini
python app.py
```

## licencia del corpus

todas las obras del corpus son de dominio publico via project gutenberg. el codigo es para uso
academico del curso.
