# aporte a Mnemosine-Oracle

esto **no es otro proyecto**. es lo que hay que correr sobre las 664 extracciones de
[Mnemosine-Oracle](https://github.com/Karol-Reyes/Mnemosine-Oracle) **antes de cargar Neo4j**,
porque despues costaria re-extraer, y re-extraer es cuota.

```bash
python consolidar.py ../ruta/a/Mnemosine-Oracle salida/
```

## que arregla, y por que ahora

**1. procedencia.** las 3.806 relaciones salen del extractor con `properties: {}`. esa arista dice
"Apolo HIJO_DE Zeus" pero no puede decir **de donde salio**, asi que no hay cita posible y la tesis
de las contradicciones es imposible: "Wikipedia dice X, la Iliada dice Y" exige que cada arista
sepa su fuente. peor, dos versiones distintas se fusionarian en silencio.

el nombre del archivo si lo sabe (`iliada_01_007.json` -> Iliada, canto VII), asi que se recupera
sin gastar un token.

**2. union de nodos.** cada archivo numera sus nodos desde `"0"`, asi que los mismos personajes
viven 664 veces. 6.007 instancias quedan en **1.332 nodos**.

**3. normalizacion.** alias de `equivalencias.csv`, caja y tildes, y los typos de tipo del LLM.

**4. conflictos.** una vez que cada arista sabe su fuente, las contradicciones se **calculan**:
mismo sujeto, mismo predicado, objetos distintos.

## lo que sale

| archivo | que es |
| --- | --- |
| `grafo.json` | 1.332 nodos y 2.373 aristas, cada una con sus fuentes. listo para cargar |
| `conflictos.json` | 441 candidatos, 60 de cardinalidad uno (los sospechosos de verdad) |
| `sugerencias_equivalencias.csv` | 125 pares de nodos que quizas son el mismo. **para revisar a mano**, no se unen solos |

## numeros de la corrida

| | |
| --- | --- |
| relaciones leidas | 3.806 |
| aristas unicas, todas con procedencia | 2.373 |
| nodos unicos | 1.332 |
| tipos de relacion corregidos | 10 (`VIATJA_A`, `HIHIJO_DE`, `AYUda_A`, `CREE`...) |
| grafias unidas por caja o tilde | 29 |
| `Hades` desambiguado como inframundo | 17 |
| alias aplicados | 84 |
| **aristas que violan el esquema de tipos** | **154 (6% del grafo)** |

## los dos hallazgos que importan

**el grafo es sobre todo Wikipedia, no los poemas.**

| el hecho lo atestiguan | aristas |
| --- | --- |
| solo Wikipedia | 1.475 |
| solo los poemas | 666 |
| ambas | 232 |

o sea **62% del grafo sale solo de Wikipedia**, que es una fuente terciaria y moderna que
*armoniza* los mitos. para un proyecto sobre fuentes primarias y contradicciones eso hay que
medirlo y discutirlo, no dejarlo pasar. tambien es una oportunidad: Wikipedia da la genealogia
canonica y el poema cuenta otra cosa, y ese choque es material gratis para las preguntas.

**el 6% de las aristas no respeta los tipos.** el esquema declara que `VIVE_EN` apunta a un
`Lugar` y `PERTENECE_A` a un `Grupo`. sin esa regla el grafo acepta `Apolo VIVE_EN Zeus`,
`Aquiles VIVE_EN Peleo` y `Priamo PERTENECE_A Aqueos`, que no son versiones alternativas sino
errores del extractor. la regla los separa de las contradicciones de verdad, que es lo que permite
que el informe de conflictos sea revisable.

**ese 6% es un piso, no un techo.** la regla caza errores de tipo, no errores bien tipados:
`Priamo PERTENECE_A Aqueos` es `Personaje -> Grupo`, cumple el esquema y es falso. para esos hace
falta la evaluacion con preguntas.

## lo que queda para revision humana

`sugerencias_equivalencias.csv` **no se aplica solo**, a proposito. hay dos clases mezcladas:

- las que si son el mismo: `Aquiles VIVE_EN Ptia` (como lo escribe Segala) contra `Ftia` (como lo
  escribe Wikipedia) son el mismo lugar. igual `Trinacia`/`Trinacria`, `Dionisio`/`Dioniso`,
  `Belerofonte`/`Belerofontes`.
- las que **no**: `Isandro`/`Pisandro` y `Menestio`/`Menetio` son personajes distintos de la
  Iliada. unirlos inventaria datos.

las de la primera clase van a `equivalencias.csv`, que hoy cubre dioses y epitetos pero **no
toponimos**, y los toponimos son justo donde la traduccion y Wikipedia se separan. como el 62% del
grafo es Wikipedia, cada toponimo partido deja dos subgrafos que nunca se conectan.
