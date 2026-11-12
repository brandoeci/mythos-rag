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

ids verificados y descargados. **952.000 tokens en total**, ocho veces el corpus del taller 2:

| autor | obra | id | traduccion | tokens | parrafos |
| --- | --- | --- | --- | --- | --- |
| varios | antologia (ver abajo) | 348 | Evelyn-White, Loeb 1914 | 90.815 | 1.595 |
| homero | iliada | 2199 | Samuel Butler, prosa | 152.757 | 1.166 |
| homero | odisea | 1727 | Samuel Butler, prosa | 129.571 | 1.321 |
| apolonio | argonautica | 830 | R. C. Seaton, 1912 | 61.281 | 490 |
| ovidio | metamorfosis i-vii | 21765 | Henry T. Riley, prosa | 125.333 | 1.445 |
| ovidio | metamorfosis viii-xv | 26073 | Henry T. Riley, prosa | 139.460 | 1.404 |
| bulfinch | the age of fable | 3327 | edicion de 1913 | 145.933 | 1.624 |
| berens | myths and legends | 22381 | edicion de 1894 | 106.475 | 1.606 |

**apolodoro quedo fuera: la Biblioteca no esta en gutenberg.** era una de las fuentes primarias
que queriamos. se reemplaza por la Argonautica de apolonio (primaria, da su propia version de
jason y medea) y el compendio de berens, que cuenta los mismos mitos que bulfinch y choca con el
en los detalles.

> **corregido:** la columna de parrafos decia antes 8.225, 13.292, etc. esos eran **conteos de
> linea**: los ocho archivos vienen a doble espacio, con una linea en blanco entre cada linea y dos
> entre parrafos. medido en la iliada, 12.388 rachas de dos saltos contra 1.165 de cuatro. partir
> por `

` daba "parrafos" de 63 caracteres de mediana y **rompia las frases**: buscar "fairest
> among the deathless gods" no encontraba nada porque la frase cruzaba un salto. `normalize()` ya
> parte por rachas de tres o mas.

### atribucion: de texto crudo a secciones citables

`corpus.iter_sections()` convierte cada obra en **secciones con autor, obra y locator resueltos**.
es lo que `index.chunk_document` debe consumir; el texto crudo nunca, porque trae prefacios, notas
y, en el #348, obras de otros autores.

| obra | secciones | palabras citables | del total | ejemplo de locator |
| --- | --- | --- | --- | --- |
| #348 (antologia) | 62 | 72.756 | 80% | `the theogony`, `xix. to pan` |
| iliada | 24 | 152.639 | 100% | `libro xiii` |
| odisea | 24 | 126.689 | 98% | `libro xiii` |
| argonautica | 4 | 59.580 | 97% | `libro iii` |
| ovidio i-vii | 70 | 65.138 | 52% | `libro iii, fabula vi [III.339-401]` |
| ovidio viii-xv | 42 | 54.999 | 39% | `libro x, fabula ix [X.519-707]` |
| the age of fable | 57 | 145.799 | 100% | `capitulo vi` |
| myths and legends | 166 | 106.001 | 100% | `oceanides`, `the sirens` |
| **total** | **449** | **783.601** | **82%** | |

el 18% que se cae es justo lo que produciria citas falsas: el estudio de evelyn-white, los
prefacios, las 109 notas de riley, las sinopsis por fabula y los indices. en ovidio se cae casi la
mitad del archivo, y esta bien que se caiga.

la antologia quedo partida en 62 obras con su autor resuelto: 3 de hesiodo con atribucion firme
(Trabajos y Dias, Teogonia, Escudo de Heracles), 13 atribuidas, 32 himnos y epigramas de "homero
(atribuido)", 10 fragmentos del ciclo epico y 3 anonimos. se marca "(atribuido)" donde la propia
edicion duda: de La Astronomia dice que esta *"attributed forsooth to Hesiod"*. afirmar autoria
donde la fuente duda seria el mismo problema de las citas falsas, un paso mas arriba.

**marcadores de nota soldados al texto.** habia 1.813 en el texto citable, y salian dentro de las
citas: el pasaje de pandora se leia *"he called this woman Pandora 1302, because..."*. los 1.452
entre corchetes se quitan en todas las obras. los sueltos (`jar 1303`) **solo** en la antologia,
donde evelyn-white los numera seguidos; en las demas obras esos numeros son años y referencias de
linea de verdad ("year 1820", "line 252", "in 1646") y ahi no se tocan. quedan 0 de corchete y 54
sueltos, todos legitimos.

### tres cosas que salieron de verificar el corpus

**1. el corpus se divide en nombres griegos y latinos, y la division es casi total.**

| | heracles / hercules | odysseus / ulysses | zeus / jove |
| --- | --- | --- | --- |
| hesiodo, apolonio, berens | 263 / 4 | 136 / 2 | 698 / 0 |
| homero, ovidio, bulfinch | 6 / 279 | 7 / 995 | 35 / 563 |

preguntar por "heracles" recupera solo tres de las ocho obras, y "hercules" las otras cinco. sin
el mapa de variantes **el retrieval pierde la mitad del corpus en silencio**, y peor: las
contradicciones suelen estar entre una fuente que usa nombres griegos y otra que usa latinos, asi
que `detect_conflict` nunca veria los dos lados. esto sube la tarea 4 de hildebrando de deseable
a bloqueante.

**2. el corpus si se contradice, verificado con el texto a la vista.** el nacimiento de afrodita:

- hesiodo, teogonia: nace de la espuma del mar despues de que crono mutila a urano, sin madre.
  *"a white foam spread around them [...] and in it there grew a maiden [...] Her gods and men call
  Aphrodite, and the foam-born goddess"*
- homero, iliada: venus tiene madre. *"he commits him to the care of Apollo and goes to Olympus,
  where she is tended by her mother Dione"*

y notese que cruza justo la division del punto 1: una dice "aphrodite" y la otra "venus".

**3. quedan restos de gutenberg despues de limpiar**, en cuatro obras: ovidio i-vii (3
"transcriber", 1 "copyright"), ovidio viii-xv (1 "gutenberg", 4 "transcriber"), hesiodo y bulfinch
(1 "gutenberg" cada uno). son pocos pero hay que sacarlos: si un chunk de esos entra al indice, el
sistema puede citar una nota del transcriptor como si fuera homero.

**4. el #348 no es la Teogonia: es una antologia de unas treinta obras.** trae los Trabajos y
Dias, la Teogonia, el Escudo de Heracles, los Catalogos de Mujeres, los **Himnos Homericos** (que
son de homero, no de hesiodo) y los fragmentos del Ciclo Epico. citar el archivo entero como
"hesiodo, teogonia" produciria **citas falsas**, que es justo lo que el proyecto promete evitar.
queda marcado con `is_collection=True` y **no se indexa** hasta que `chunk_document` saque autor y
obra del encabezado de cada seccion.

y esto sube de prioridad, porque la Teogonia vive dentro de ese archivo: **sin el #348 indexado,
tres de las cinco contradicciones verificadas se quedan con un solo lado.** partir la antologia
por secciones no es limpieza, es ruta critica para la demo.

ademas el archivo trae la **introduccion de evelyn-white**, un estudio del siglo XX sobre hesiodo,
antes de los textos. citarla como "hesiodo" es el mismo problema que las notas de riley en ovidio,
y es facil caer: la introduccion resume la teogonia, asi que una busqueda de "chaos, earth and
eros" cae primero en el resumen del traductor y no en el verso.

### formato de los encabezados, obra por obra

esto define el `locator` de la tarea 3 de hildebrando. son cuatro formatos distintos, no uno:

| obra | encabezado | locator propuesto |
| --- | --- | --- |
| iliada, odisea | `BOOK I.` (24 y 24 libros) | `libro xii` |
| argonautica | `BOOK I` sin punto (4 libros) | `libro iii` |
| bulfinch | `Chapter I` | `capitulo xv` |
| berens | `PART I.--MYTHS.` + titulo de seccion | `parte i, las sirenas` |
| ovidio | `BOOK THE FIRST.` **y** `FABLE I. [I.89-112]` | `libro i, fabula iii [i.89-112]` |
| #348 (antologia) | titulo de obra en mayusculas | hay que sacar autor y obra de ahi |

**ovidio es un regalo para las citas:** los 126 encabezados de fabula traen el rango de versos
latinos entre corchetes, asi que la cita puede apuntar al verso y no solo al libro. es el locator
mas preciso del corpus y conviene aprovecharlo en la demo.

**pero el mismo archivo trae una trampa, y son cuatro capas:** de todo lo que hay en los dos tomos
de ovidio, **solo una parte es ovidio**. hay que separar al chunkear:

| capa | que es | ¿citable como ovidio? |
| --- | --- | --- |
| resumen que encabeza cada libro | sinopsis editorial en presente | **no** |
| `FABLE N. [refs]` | el encabezado, con el rango de versos | es el locator |
| la sinopsis de un parrafo justo despues del encabezado | ej. *"God reduces Chaos into order"* | **no** |
| el texto que sigue a esa sinopsis | la traduccion de riley del verso | **si** |
| `EXPLANATION.` | nota de riley, siglo XIX. 109 bloques | **no** |

caer en esto es facil: buscando un pasaje de la creacion para una pregunta, lo primero que
devuelve la busqueda es el resumen del libro y no el verso. un chunk de esas capas citado como
"ovidio, metamorfosis" es una cita falsa de diecinueve siglos.

el lado bueno: **28 de esos 109 bloques discuten explicitamente versiones en conflicto** ("some
say", "other writers", "according to some"). son un mapa de contradicciones escrito por un
erudito, y citado como "riley, notas a ovidio" es una fuente legitima. el limite es que casi
siempre citan autores que no estan en nuestro corpus (pausanias, diodoro, higino), asi que sirven
para **encontrar** candidatos, no como los dos lados.

### catalogo de contradicciones (en `eval/questions.json`)

**las 10 verificadas**, con los pasajes de cada lado en el campo `passages`. y resultaron de tres
clases distintas, lo cual es mejor que diez de la misma:

**(a) dos autores que discrepan** — el caso esperado:

| pregunta | las fuentes dicen |
| --- | --- |
| ¿que contenia la jarra de pandora? | hesiodo dice males. berens dice bendiciones. **bulfinch recoge las dos y se inclina por la segunda** |
| ¿quien era la madre de afrodita? | en hesiodo no tiene, nace de la espuma. en la iliada es dione. **bulfinch enuncia las dos** |
| ¿de quien era hijo eros? | en la teogonia no tiene padres, es de los primeros seres. en berens es hijo de afrodita |
| ¿quien era la esposa de hefesto? | bulfinch dice venus. berens anota que segun algunos relatos era charis |
| ¿como murio absirto? | en apolonio y berens es un **adulto** al mando de la flota perseguidora. en bulfinch es un **nino** al que medea manda matar |
| ¿hefesto tuvo padre? | en la teogonia hera lo pario sin union con zeus. berens y bulfinch lo hacen hijo de los dos |
| ¿quien creo a la humanidad? | en hesiodo los dioses del olimpo hicieron una raza de oro. en bulfinch fue prometeo |
| ¿que habia al principio del mundo? | en hesiodo el caos es un ser que engendra a erebo y la noche. en ovidio el caos no engendra nada, es el nombre del desorden que un dios separa |

**(b) una misma obra que se contradice a si misma** — la iliada, y es la mas vistosa:

| pregunta | que pasa |
| --- | --- |
| ¿quien arrojo a hefesto del olimpo y quien lo recogio? | en el **libro I** hefesto dice que fue zeus, que cayo en lemnos y lo atendieron los sintios. en el **libro XVIII** dice que fue su madre y que lo recogieron eurinome y tetis en una cueva del oceano |

no es que dos autores discrepen: es el mismo personaje contando su propia caida de dos maneras en
el mismo poema. eso no se resuelve escogiendo la fuente mas confiable.

**(c) dos personajes distintos con el mismo nombre** — el riesgo no es la discrepancia sino la
fusion:

| pregunta | que pasa |
| --- | --- |
| ¿quien era escila? | en la odisea es un monstruo marino de seis cabezas, hija de crateis. en ovidio es la hija del rey niso de megara, que le corta el mechon a su padre por amor a minos y acaba convertida en ave |

si el sistema las junta en una sola respuesta, produce algo coherente y completamente falso. es un
modo de fallo distinto de los otros dos y conviene tenerlo medido.

### el grupo "fuera": las 10 listas, y son de dos clases

el grupo que mas pesa en la demo esta completo y verificado. tiene **dos clases de pregunta**, y
las dos hacen falta:

**(a) entidad ausente** — el sujeto no esta en el corpus. el retrieval no trae nada relevante y el
umbral lo atrapa. es el caso facil. van 8: hiperborea, mitra, arquimedes, tule, el demiurgo,
gilgamesh, cuchulainn e ishtar.

dos de esas son tramposas a proposito: **arquimedes**, porque el corpus si menciona a pitagoras
(36 veces), platon (13) y eratostenes (11), asi que esperarlo es razonable; y **cuchulainn**,
porque bulfinch si trae mitologia nordica (odin 29, thor 77), asi que esperar material celta
tampoco es absurdo. el corpus es mucho mas ancho que "mitologia griega".

**(b) dato ausente** — la entidad **si** esta y esta descrita en detalle, pero el dato puntual no.
el retrieval trae chunks muy relevantes y el modelo queda tentado a completar. **esta es la clase
que de verdad prueba la abstencion**, y es la que no se puede improvisar. van 2:

| pregunta | por que esta verificada |
| --- | --- |
| ¿cuanto tardo hefesto en forjar el escudo de aquiles? | la iliada describe la forja en detalle, pero "vulcan" nunca aparece a menos de 400 caracteres de "days" ni de "hours" |
| ¿cuantas colas tenia cerbero? | se habla de sus **cabezas** en 6 pasajes y de colas en **cero**. un modelo sin corpus suele responder con la cola de serpiente de otras tradiciones |

### una leccion del metodo: contar palabras, no substrings

el primer barrido dio "thor" 349 veces en las ocho obras, lo cual era imposible: estaba contando
substrings y "thor" vive dentro de "author". por palabra completa son 77. igual "hera", que daba
522 por "whereas" y "hereafter", y son 157.

las parejas de la tabla de nombres de arriba aguantaron el cambio (heracles, hercules, odysseus,
ulysses, zeus y jove varian entre 0 y 8 sobre cientos, porque son palabras distintivas), asi que
esa conclusion se mantiene. pero **una pregunta del grupo "fuera" validada con substrings seria
basura**, y `search.py` ya busca por palabra completa.

### el grupo "directa": las 10 listas

diez preguntas con su pasaje y su fuente, repartidas entre las siete obras: hesiodo (3), berens
(2), bulfinch (1), iliada (1), odisea (1), argonautica (1) y ovidio (1). el nombre de los caballos
de aquiles y de quien nacieron, quien construyo el argo, por que pandora se llama asi, a quien le
dejo persefone el entendimiento en la muerte, como repartio zeus el ano de adonis.

**las 30 estan listas.** la herramienta con la que se verificaron es `search.py`:

```bash
python search.py pandora          # el contexto del termino en las ocho obras
python search.py "foam-born" 2    # 2 coincidencias por fuente
python search.py tiresias 3 600   # y 600 caracteres de contexto
```

cuando un termino **no** aparece en ninguna obra, lo dice: eso es justo lo que hay que confirmar
para las preguntas del grupo "fuera".

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

### listo

- [x] repo, readme y contrato de `answer()` definido
- [x] esqueleto de los cinco modulos, con firmas, tipos y docstrings
- [x] modo stub: `app.py` y `evaluate.py` corren sin que el pipeline exista
- [x] `proyecto.ipynb` con las secciones de la plantilla del curso
- [x] los ocho `gutenberg_id` verificados y el corpus descargado (952k tokens)
- [x] `strip_boilerplate` y `normalize` corriendo sobre las ocho obras
- [x] verificado que el corpus **si** se contradice (afrodita), que era el riesgo silencioso
- [x] `search.py` para verificar preguntas a mano, buscando por palabra completa
- [x] 5 preguntas de "conflicto" verificadas, con los pasajes de cada lado
- [x] las 10 preguntas de "fuera" verificadas, en sus dos clases
- [x] las 10 preguntas de "directa" verificadas, con el pasaje y la fuente de cada una
- [x] **las 30 preguntas listas y verificadas**, 10 por grupo

### nucleo

- [ ] corpus multi-fuente indexado, cada chunk con autor/obra/libro
- [ ] retrieval hibrido: embeddings + bm25
- [ ] respuesta con cita obligatoria al pasaje
- [ ] abstencion cuando la respuesta no esta en el corpus
- [ ] deteccion de versiones en conflicto
- [ ] evaluacion con ablacion de las 4 configuraciones
- [ ] interfaz gradio: respuesta + citas + panel de conflictos

### ambicioso (stretch, se corta el martes 17 si no esta empezado)

- [ ] grafo de entidades: relaciones (hijo-de, mato-a, casado-con) extraidas con el llm
- [ ] aristas en conflicto marcadas en el grafo segun la fuente

## reparto y orden de tareas

| persona | responsable de | archivos |
| --- | --- | --- |
| hildebrando | corpus e indice | `corpus.py`, `index.py` |
| lancheros | pipeline rag | `rag.py` |
| ximena | evaluacion, interfaz y video | `evaluate.py`, `app.py`, `eval/questions.json` |

cada bloque va **en orden**: la tarea 3 asume que la 2 esta lista. debajo de cada
tarea esta lo que hay que *asegurar* antes de darla por hecha, porque es ahi
donde esto se cae.

si alguien necesita cambiar `Answer` en `rag.py`, se avisa al grupo primero: ese
cambio rompe los otros dos modulos.

### hildebrando — corpus e indice

1. **verificar los seis `gutenberg_id`** en gutenberg.org.
   *asegurar:* dejar anotada en `SOURCES` la traduccion exacta de cada obra.
   varias tienen varias traducciones, y la traduccion cambia el texto que vamos
   a citar en el video.
2. **limpiar las seis obras** con `strip_boilerplate` y `normalize`.
   *asegurar:* buscar "gutenberg", "transcriber" y "copyright" en el texto
   limpio y que de cero. si el pie de licencia entra al corpus, el sistema va a
   citar la licencia como si fuera homero, y eso se ve en la demo.
3. **`chunk_document` con un `locator` que sirva.**
   *asegurar:* el locator tiene que dejar a un humano encontrar el pasaje en el
   libro ("canto xii", "libro iii"), no `chunk_4871`. cada obra marca sus
   divisiones distinto, asi que son seis parsers pequeños y no uno. prueba:
   tomar 10 chunks al azar y buscar el pasaje a mano con el locator. si no lo
   encuentras, el locator no sirve.
4. **mapa de variantes de nombres.** heracles/hercules, odiseo/ulises,
   zeus/jove/jupiter. el corpus mezcla nombres griegos y latinos segun el autor
   y el traductor; sin esto, preguntar por "heracles" no recupera los pasajes
   que dicen "hercules". se aplica a la consulta y a la copia que indexa bm25.
   *asegurar:* probar cada par y que devuelva los pasajes de las dos formas.
5. **deduplicar pasajes casi identicos.** bulfinch parafrasea a los otros
   autores, asi que hay chunks que dicen lo mismo. sin dedup, el retrieval
   devuelve cinco veces el mismo hecho y tapa a la fuente que lo contradice:
   eso mata la pieza de conflictos, que es la del proyecto.
   **ya esta comprobado**, y de la forma mas cruda: el relato de la creacion de
   bulfinch es ovidio casi palabra por palabra. ovidio dice *"a rude and
   undigested mass, and nothing more than an inert weight, and the discordant
   atoms of things not harmonizing"*; bulfinch, *"a rough heap of inert weight
   and discordant beginnings of things clashing together"*, y lo admite en la
   linea anterior: *"The Roman poet Ovid gives us a connected narrative of
   creation"*.
   *asegurar:* medir cuantos pares pasan de 0.95 de similitud, decidir que se
   hace con ellos y dejar la decision escrita.
6. **chroma con embeddings locales, persistido en `index/`.**
   *asegurar:* cerrar python, volver a abrirlo y cargar el indice **sin**
   recalcular embeddings. cronometrar los dos caminos y guardar los numeros, van
   al notebook.
7. **bm25 sobre la copia limpiada duro.**
   *asegurar:* el `text` del `Chunk` no se toca nunca. ese es el que se cita.
8. **`HybridRetriever.search` con fusion de los dos rankings.**
   *asegurar:* que `alpha=1.0` (denso puro) y `alpha=0.0` (lexico puro)
   funcionen solos. la tabla de ablacion necesita denso e hibrido como filas
   separadas; si solo funciona en 0.5, no hay tabla.
9. **la tabla de tokens y chunks por obra** (`summarize`), para el notebook.

### lancheros — pipeline rag

1. **la rebanada vertical, antes que nada.** retrieval denso sobre **una** obra
   y `generate` con cita. fea, sin hibrido, sin grading.
   *asegurar:* que exista el **viernes 13**. es el hito que decide si el
   proyecto va bien o si hay que recortar, y recortar el 13 se puede, el 18 no.
2. **`generate` con citacion obligatoria.** lo mas importante del proyecto.
   *asegurar:* que cada cita devuelta exista **literal** en el corpus. se
   verifica solo: buscar `citation.text` como substring del chunk de origen. si
   el modelo parafrasea la cita, la cita es falsa — y una cita falsa es peor que
   no citar, porque es exactamente el problema que decimos resolver.
3. **cache en disco de toda llamada al llm.** hoy, no cuando duela.
   *asegurar:* correr el notebook completo dos veces y que la segunda no gaste
   cuota.
4. **`grade` y abstencion, con el umbral calibrado.**
   *asegurar:* `ABSTAIN_THRESHOLD` se calibra con el grupo 3 de las preguntas,
   no a ojo. y se reportan los dos numeros juntos: cuanto abstiene de lo que
   debe y cuanto de lo que no. un umbral que abstiene de todo saca 10/10 en el
   grupo 3 y arruina el grupo 1.
5. **`detect_conflict`.**
   *asegurar:* el riesgo es el falso positivo. dos fuentes que dicen lo mismo
   con otras palabras no son una contradiccion. probar con 5 casos donde si se
   contradicen y 5 donde solo cambia la redaccion, y que distinga.
6. **el grafo de langgraph** con las aristas condicionales.
   *asegurar:* exportar el diagrama (png o mermaid) para el notebook y el video.
7. **flags de configuracion** para que `evaluate.py` pueda pedir `rag_denso`,
   `rag_hibrido` y `rag_hibrido_conflictos` sin editar codigo.
8. **reescritura de la consulta antes del retrieval.** la pregunta viene en
   español y el corpus esta en ingles; expandirla a los terminos del corpus
   deberia mejorar el retrieval.
   *asegurar:* medirlo con y sin. si no mejora, se dice en la discusion.
9. **pase de auto-verificacion.** el modelo revisa su propia respuesta contra
   los chunks y retira lo que no quede soportado.

### ximena — evaluacion, interfaz y video

1. **las 30 preguntas, 10 por grupo.**
   *asegurar (grupo 2):* verificar a mano, leyendo los dos pasajes, que las
   fuentes **si** se contradicen. una pregunta donde no se contradicen no mide
   nada. es media hora y hay que hacerla el **sabado 14**: lancheros la necesita
   el lunes para construir `detect_conflict`.
   *asegurar (grupo 3):* que suenen plausibles. el padre de un personaje que no
   aparece en el corpus, no "cual es la capital de francia": una pregunta
   absurda la rechaza cualquier modelo y no prueba nada.
   *asegurar:* cada `reference` se escribe despues de leer el pasaje, y
   `expected_sources` apunta a obras que existen en `corpus.SOURCES`.
2. **la interfaz gradio contra el stub**, desde hoy.
   *asegurar:* que las tres formas de respuesta se vean bien — normal,
   conflicto y abstencion. el stub ya produce las tres.
3. **`is_correct` con el llm como juez.**
   *asegurar:* validar al juez antes de confiarle la tabla. calificar 10
   respuestas a mano y comparar. si el juez no coincide con ustedes en al menos
   8 de 10, esta mal y los numeros no valen nada. ese dato va al notebook: es lo
   que hace creible la evaluacion.
   *asegurar:* el juez no ve que configuracion produjo la respuesta.
4. **`citations_ok` estricto.**
   *asegurar:* no basta con que la obra citada sea la correcta; el texto citado
   tiene que existir en el corpus.
5. **correr las 4 configuraciones y armar la tabla de ablacion.**
   *asegurar:* mismo set, misma semilla y **misma version del corpus** en las
   cuatro corridas. si el corpus cambia a mitad, la tabla no compara nada.
   anotar el commit con el que se corrio.
6. **tres ejemplos cualitativos lado a lado**: la misma pregunta respondida por
   el llm solo y por el rag, uno de cada grupo. es lo primero que se mira.
7. **guion del video y ensayo cronometrado.**
   *asegurar:* entre 5 y 10 minutos es requisito. ensayar con cronometro antes
   de grabar, y que hablen los tres.
8. **reportar tambien lo que fallo:** los falsos positivos de conflicto, las
   preguntas que nadie acerto y que tipo de pregunta rompe el sistema. una
   discusion que solo muestra lo que funciono se cae en preguntas.

### orden entre los tres

- **jue 12 (hoy):** los tres clonan y dejan el entorno **corriendo y probado**
  esta noche. un problema de instalacion descubierto el lunes cuesta un dia que
  no existe. se ratifica `Answer`. hildebrando verifica los `gutenberg_id`,
  ximena arranca las preguntas contra el stub, lancheros se hace un indice
  rapido de una sola obra para no esperar a nadie.
- **vie 13:** la rebanada vertical responde una pregunta de punta a punta. si no,
  se recorta ese mismo dia.
- **sab 14:** ximena verifica a mano el grupo 2, que lancheros necesita el lunes.
  hildebrando entrega chroma persistido y bm25.
- **dom 15:** hildebrando entrega el hibrido con los alphas sueltos. al cerrar
  el dia, **integracion**: el pipeline real corre contra el indice real y
  `STUB` pasa a `False`.
- **lun 16:** ximena corre las 4 configuraciones aunque esten incompletas, para
  tener el arnes probado. lancheros construye `detect_conflict`.
- **mar 17:** nucleo cerrado y corte del stretch.
- **mie 18:** code freeze y corridas finales.
- **jue 19:** escritura del notebook y ensayo del video.
- **vie 20:** entrega.

## cronograma

entrega: **viernes 20 de noviembre**. son nueve dias contando hoy, con un solo
fin de semana. el alcance es el mismo de arriba; lo que desaparecio es la
holgura, asi que los hitos se adelantan y se integra todos los dias.

| fechas | fase | hito que cierra la fase |
| --- | --- | --- |
| jue 12 (hoy) | arranque | los tres con el repo clonado y el entorno **probado**, contrato de `answer()` ratificado, los seis `gutenberg_id` verificados |
| vie 13 | rebanada vertical | **una** pregunta se responde de punta a punta con cita. una sola obra, sin hibrido. pero funciona |
| sab 14 - dom 15 | el motor del proyecto | indice real persistido, citacion verificada, abstencion, las 30 preguntas escritas y el grupo 2 verificado a mano. **integracion el domingo en la noche: `STUB = False`** |
| lun 16 - mar 17 | nucleo cerrado | hibrido, dedup, conflictos, grafo de langgraph, flags de configuracion. **el 17 se decide si el grafo de entidades entra o se corta** |
| mie 18 | congelar | **code freeze.** las 4 corridas finales y la tabla de ablacion definitiva |
| jue 19 | escritura | notebook completo con los numeros reales, guion del video y ensayo cronometrado |
| vie 20 | entrega | grabar y montar en la mañana, enviar en la tarde. no a las 23:59 |

cinco reglas sobre este calendario:

**el hito del viernes 13 es el que importa.** si el viernes no hay una respuesta de punta a punta,
por fea que sea, se recorta ese dia y no el miercoles. un pipeline que integra el 17 no integra.

**el fin de semana no es opcional.** sabado y domingo son los dos unicos dias completos que hay.
el trabajo pesado en paralelo va ahi; si se pierden, no hay de donde sacarlos.

**se integra todos los dias, no al final.** con nueve dias no hay espacio para descubrir el 17 que
los modulos no encajan. por eso la integracion esta agendada el domingo y no el miercoles.

**sync diario de 15 minutos a hora fija.** cada uno dice que cerro y que lo tiene bloqueado.
alguien atascado en silencio medio dia cuesta mas del 5% del calendario.

**el alcance queda congelado desde hoy.** no entran ideas nuevas: no hay holgura para absorberlas.
lo que sobre tiempo se gasta en el stretch que ya esta en la lista, no en algo que se ocurra el
lunes.

## riesgos

| riesgo | como se mata |
| --- | --- |
| se acaba la cuota de la api en la semana final | embeddings locales + cache en disco desde el dia 1, no despues |
| el notebook no corre el dia de la revision | indice persistido y salidas guardadas en el `.ipynb`. cero celdas que tarden 40 minutos |
| merge hell en el notebook | codigo en `.py`, el notebook solo importa y muestra |
| scope creep con el grafo | es stretch y tiene fecha de corte: martes 17 |
| el detector de conflictos no detecta nada | verificar el **sabado 14** que el corpus **si** se contradice sobre las entidades elegidas, antes de construir el detector encima |
| nueve dias sin holgura | se integra a diario y el hito del viernes 13 es el canario: si falla, se recorta ese dia |

el ultimo es el riesgo silencioso: si las fuentes elegidas no se contradicen, la pieza estrella no
tiene nada que mostrar. se chequea a mano, en media hora.

## como correrlo

```bash
python -m venv .venv
source .venv/bin/activate        # windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # y poner la api key de gemini
```

el pipeline todavia no existe, pero **el esqueleto corre en modo stub**: con
`rag.STUB = True` (el default), `rag.answer()` devuelve datos falsos con la
forma definitiva.

```bash
python app.py        # la interfaz, funcionando contra el stub
```

eso es a proposito. ximena construye y prueba la interfaz y el arnes de
evaluacion desde hoy, sin esperar a que lancheros termine el pipeline. cuando
los nodos esten listos se pone `rag.STUB = False` y nada mas cambia.

lo que falta implementar esta marcado con `raise NotImplementedError("<nombre>")`
en cada modulo, asi que `grep -rn NotImplementedError` da la lista de pendientes
con su dueño.

## licencia del corpus

todas las obras del corpus son de dominio publico via project gutenberg. el codigo es para uso
academico del curso.
