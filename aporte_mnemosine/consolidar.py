"""consolida las 664 extracciones en un grafo cargable, con procedencia.

va sobre el repo Mnemosine-Oracle. hace cuatro cosas que el extractor dejo
pendientes y que hay que resolver **antes** de cargar Neo4j, porque despues
costarian re-extraer:

1. **procedencia.** las 3.857 relaciones salen con `properties: {}`, asi que
   una arista no puede decir de donde salio. el nombre del archivo si lo sabe
   (`iliada_01_001.json`), asi que se recupera sin gastar cuota. sin esto no
   hay cita posible y la tesis de las contradicciones no se puede sostener.

2. **union de nodos.** cada archivo numera sus nodos desde "0", asi que los
   mismos personajes viven 664 veces. se unen por nombre canonico.

3. **normalizacion.** nombres por `equivalencias.csv` y por caja (`aqueos` y
   `Aqueos` eran dos nodos distintos, 157 veces), y tipos de relacion con los
   typos del LLM (`VIATJA_A`, `HIHIJO_DE`, `AYUda_A`).

4. **conflictos.** una vez que cada arista sabe su fuente, las contradicciones
   se calculan: mismo sujeto y mismo predicado con objetos distintos.

    python consolidar.py <ruta-al-repo-Mnemosine-Oracle>
"""

import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher, get_close_matches
from itertools import combinations
from pathlib import Path

# los tipos que el extractor debia emitir. lo que no caiga aqui se corrige por
# distancia de edicion y, si no se parece a ninguno, se reporta sin tocar.
TIPOS = [
    "HIJO_DE", "HERMANO_DE", "ESPOSO_DE", "AMA_A", "MATA_A", "ENFRENTA_A",
    "AYUDA_A", "APOYA_A", "ORDENA_A", "VIAJA_A", "VIVE_EN", "PERTENECE_A",
    "POSEE", "CREA",
]

# que tipo de nodo puede estar a cada lado de cada relacion. sin esto el grafo
# acepta "Apolo VIVE_EN Zeus" y "Priamo PERTENECE_A Aqueos", que no son
# contradicciones entre fuentes sino errores del extractor. la regla los separa:
# lo que viola el esquema es un error; lo que lo respeta y discrepa es una
# contradiccion de verdad, y esa es la que le interesa al proyecto.
PERSONA = {"Personaje", "Deidad", "Criatura"}
ESQUEMA = {
    "HIJO_DE": (PERSONA, PERSONA),
    "HERMANO_DE": (PERSONA, PERSONA),
    "ESPOSO_DE": (PERSONA, PERSONA),
    "AMA_A": (PERSONA, PERSONA),
    "MATA_A": (PERSONA, PERSONA | {"Grupo"}),
    "ENFRENTA_A": (PERSONA, PERSONA | {"Grupo"}),
    "AYUDA_A": (PERSONA, PERSONA | {"Grupo"}),
    "APOYA_A": (PERSONA, PERSONA | {"Grupo"}),
    "ORDENA_A": (PERSONA, PERSONA | {"Grupo"}),
    "VIVE_EN": (PERSONA | {"Grupo"}, {"Lugar"}),
    "VIAJA_A": (PERSONA | {"Grupo"}, {"Lugar"}),
    "PERTENECE_A": (PERSONA, {"Grupo"}),
    "POSEE": (PERSONA, {"Objeto"}),
    "CREA": (PERSONA, {"Objeto", "Lugar", "Criatura"}),
}


def valida_esquema(tipo: str, tipo_sujeto: str, tipo_objeto: str) -> str | None:
    """devuelve el motivo del incumplimiento, o None si la arista es plausible."""
    regla = ESQUEMA.get(tipo)
    if not regla:
        return None
    permitido_sujeto, permitido_objeto = regla
    if tipo_sujeto not in permitido_sujeto:
        return f"sujeto {tipo_sujeto} no vale para {tipo}"
    if tipo_objeto not in permitido_objeto:
        return f"objeto {tipo_objeto} no vale para {tipo}"
    return None


ROMANOS = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
    9: "IX", 10: "X", 11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV",
    16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX", 21: "XXI",
    22: "XXII", 23: "XXIII", 24: "XXIV",
}


def plano(s: str) -> str:
    """clave de comparacion: sin tildes, sin caja, sin espacios de sobra.

    es lo que hace que `aqueos`, `Aqueos` y `AQUEOS` sean un solo nodo.
    """
    s = unicodedata.normalize("NFD", s.strip().lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def procedencia(nombre_archivo: str) -> dict:
    """de que fuente y que parte de ella salio una extraccion.

    `iliada_01_007.json`      -> Iliada, canto I, trozo 7
    `wiki_Helena_(mitologia)_001.json` -> Wikipedia, articulo Helena (mitologia)
    """
    tallo = nombre_archivo.removesuffix(".json")
    tallo, _, trozo = tallo.rpartition("_")

    if tallo.startswith("wiki_"):
        articulo = tallo.removeprefix("wiki_").replace("_", " ")
        return {
            "obra": "Wikipedia",
            "tipo_fuente": "terciaria",
            "locator": f"articulo {articulo}",
            "trozo": int(trozo),
        }

    poema, _, canto = tallo.rpartition("_")
    obra = {"iliada": "Iliada", "odisea": "Odisea"}.get(poema, poema)
    return {
        "obra": obra,
        "tipo_fuente": "primaria",
        "locator": f"canto {ROMANOS.get(int(canto), canto)}",
        "trozo": int(trozo),
    }


def cargar_tablas(repo: Path) -> tuple[dict, dict]:
    """entidades.csv (lista blanca con tipo) y equivalencias.csv (alias)."""
    entidades = {}
    with (repo / "data" / "entidades.csv").open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            if fila.get("nombre"):
                entidades[plano(fila["nombre"])] = (fila["nombre"], fila["tipo"])

    alias = {}
    with (repo / "data" / "equivalencias.csv").open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            destino = (fila.get("entidad") or "").strip()
            # "Atrida,Agamenon|Menelao" queda marcado ambiguo y no se resuelve
            # solo: forzarlo a uno de los dos inventaria datos.
            if destino and "|" not in destino:
                alias[plano(fila["nombre_en_texto"])] = destino
    return entidades, alias


def normalizar_tipo(tipo: str, informe: dict) -> str:
    recto = tipo.strip().upper()
    if recto in TIPOS:
        if recto != tipo:
            informe["tipos_corregidos"][f"{tipo} -> {recto}"] += 1
        return recto
    cerca = get_close_matches(recto, TIPOS, n=1, cutoff=0.75)
    if cerca:
        informe["tipos_corregidos"][f"{tipo} -> {cerca[0]}"] += 1
        return cerca[0]
    informe["tipos_desconocidos"][tipo] += 1
    return recto


def normalizar_nodo(nodo: dict, entidades: dict, alias: dict, informe: dict) -> tuple[str, str]:
    """devuelve (nombre canonico, tipo). la lista blanca manda sobre el extractor."""
    crudo = (nodo.get("properties") or {}).get("name", "").strip()
    etiqueta = nodo.get("label", "").strip() or "Desconocido"
    if etiqueta == "Group":  # typo de etiqueta, 2 casos
        etiqueta = "Grupo"

    clave = plano(crudo)
    if clave in alias:
        informe["alias_aplicados"][f"{crudo} -> {alias[clave]}"] += 1
        crudo = alias[clave]
        clave = plano(crudo)

    # Hades es el dios Y el inframundo, y la propia lista los separa. cuando el
    # extractor lo saco como Lugar, es el inframundo: se manda al nodo que ya
    # existe en vez de fusionar dos cosas distintas.
    if clave == "hades" and etiqueta == "Lugar":
        informe["hades_desambiguado"]["Hades (Lugar) -> Hades (inframundo)"] += 1
        crudo, clave = "Hades (inframundo)", plano("Hades (inframundo)")

    if clave in entidades:
        canon, tipo_lista = entidades[clave]
        if tipo_lista != etiqueta:
            informe["tipo_por_lista"][f"{canon}: extraido={etiqueta} lista={tipo_lista}"] += 1
        return canon, tipo_lista

    informe["fuera_de_lista"][crudo] += 1
    return crudo, etiqueta


def clave_nodo(nombre: str) -> str:
    """la clave con la que se unen los nodos.

    para los nombres que **no** estan en la lista blanca tambien hay que
    normalizar, o `Solimos` y `solimos` quedan como dos nodos. son 5 pares asi,
    todos entidades que el extractor descubrio y la lista no preveia.
    """
    return plano(nombre)


def consolidar(repo: Path) -> tuple[dict, dict, dict]:
    entidades, alias = cargar_tablas(repo)
    informe = defaultdict(lambda: defaultdict(int))

    nodos: dict[str, dict] = {}
    aristas: dict[tuple[str, str, str], dict] = {}
    n_rel = 0

    for ruta in sorted((repo / "data" / "extracciones").glob("*.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        proc = procedencia(ruta.name)

        locales = {}
        for nodo in datos.get("nodes", []):
            canon, tipo = normalizar_nodo(nodo, entidades, alias, informe)
            clave = clave_nodo(canon)
            locales[nodo["id"]] = clave
            registro = nodos.setdefault(
                clave, {"nombre": canon, "tipo": tipo, "fuentes": [], "grafias": defaultdict(int)}
            )
            registro["grafias"][canon] += 1
            if proc not in registro["fuentes"]:
                registro["fuentes"].append(proc)

        for rel in datos.get("relationships", []):
            sujeto = locales.get(rel.get("start_node_id"))
            objeto = locales.get(rel.get("end_node_id"))
            if not sujeto or not objeto or sujeto == objeto:
                if sujeto and sujeto == objeto:
                    informe["autoreferencias_descartadas"][f"{sujeto} {rel.get('type')}"] += 1
                continue
            n_rel += 1
            tipo = normalizar_tipo(rel.get("type", ""), informe)
            clave = (sujeto, tipo, objeto)
            arista = aristas.setdefault(
                clave, {"sujeto_clave": sujeto, "tipo": tipo, "objeto_clave": objeto, "fuentes": []}
            )
            if proc not in arista["fuentes"]:
                arista["fuentes"].append(proc)

    for registro in nodos.values():
        grafias = registro.pop("grafias")
        registro["nombre"] = max(grafias.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if len(grafias) > 1:
            informe["grafias_unidas"][" / ".join(sorted(grafias))] += 1

    for clave, arista in aristas.items():
        obras = sorted({f["obra"] for f in arista["fuentes"]})
        arista["obras"] = obras
        arista["n_fuentes"] = len(arista["fuentes"])
        arista["solo_wikipedia"] = obras == ["Wikipedia"]
        arista["solo_poemas"] = "Wikipedia" not in obras
        arista["sujeto"] = nodos[clave[0]]["nombre"]
        arista["objeto"] = nodos[clave[2]]["nombre"]
        motivo = valida_esquema(
            arista["tipo"],
            nodos[clave[0]]["tipo"],
            nodos[clave[2]]["tipo"],
        )
        arista["viola_esquema"] = motivo
        if motivo:
            informe["violaciones_de_esquema"][motivo] += 1

    informe["totales"]["relaciones_leidas"] = n_rel
    informe["totales"]["nodos_unicos"] = len(nodos)
    informe["totales"]["aristas_unicas"] = len(aristas)
    return nodos, aristas, informe


def conflictos(aristas: dict) -> list[dict]:
    """mismo sujeto y predicado con objetos distintos: candidatos a contradiccion.

    no toda divergencia es contradiccion. HIJO_DE con dos objetos suele ser
    padre y madre, y HERMANO_DE admite varios; esos predicados se marcan como
    `multiple_legitimo` para que el que revise no pierda tiempo. los de
    cardinalidad uno, como ESPOSO_DE o VIVE_EN, si son sospechosos de verdad.
    """
    multiples = {"HIJO_DE", "HERMANO_DE", "MATA_A", "AYUDA_A", "APOYA_A",
                 "ENFRENTA_A", "VIAJA_A", "POSEE", "ORDENA_A", "AMA_A", "CREA"}

    por_sujeto = defaultdict(list)
    for arista in aristas.values():
        # una arista que viola el esquema es un error de extraccion, no una
        # version alternativa. si se cuela, el informe de conflictos se
        # llena de ruido y nadie lo revisa.
        if arista["viola_esquema"]:
            continue
        por_sujeto[(arista["sujeto"], arista["tipo"])].append(arista)

    salida = []
    for (sujeto, tipo), grupo in por_sujeto.items():
        if len(grupo) < 2:
            continue
        obras = {o for a in grupo for o in a["obras"]}
        salida.append({
            "sujeto": sujeto,
            "predicado": tipo,
            "objetos": [a["objeto"] for a in grupo],
            "multiple_legitimo": tipo in multiples,
            "fuentes_discrepan": len(obras) > 1,
            "detalle": [{"objeto": a["objeto"], "obras": a["obras"]} for a in grupo],
        })

    salida.sort(key=lambda c: (c["multiple_legitimo"], not c["fuentes_discrepan"]))
    return salida


def sugerir_equivalencias(nodos: dict) -> list[dict]:
    """pares de nodos tan parecidos que probablemente sean el mismo.

    **no se unen solos.** "Trinacia/Trinacria" y "Dionisio/Dioniso" si lo son,
    pero "Isandro/Pisandro" y "Menestio/Menetio" son personajes distintos de la
    Iliada. unirlos automaticamente inventaria datos, asi que esto sale como
    lista para que alguien la revise y la pase a `equivalencias.csv`.
    """
    nombres = [n["nombre"] for n in nodos.values() if len(n["nombre"]) > 3]
    por_nombre = {n["nombre"]: n for n in nodos.values()}
    salida = []
    for a, b in combinations(sorted(nombres), 2):
        if abs(len(a) - len(b)) > 2:
            continue
        r = SequenceMatcher(None, plano(a), plano(b)).ratio()
        if r < 0.86:
            continue
        salida.append({
            "a": a, "b": b, "parecido": round(r, 3),
            "obras_a": sorted({f["obra"] for f in por_nombre[a]["fuentes"]}),
            "obras_b": sorted({f["obra"] for f in por_nombre[b]["fuentes"]}),
        })
    salida.sort(key=lambda s: -s["parecido"])
    return salida


def main(repo: Path, salida: Path) -> None:
    nodos, aristas, informe = consolidar(repo)
    conf = conflictos(aristas)

    salida.mkdir(parents=True, exist_ok=True)
    (salida / "grafo.json").write_text(
        json.dumps(
            {"nodos": list(nodos.values()), "aristas": list(aristas.values())},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    sug = sugerir_equivalencias(nodos)
    with (salida / "sugerencias_equivalencias.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nombre_a", "nombre_b", "parecido", "obras_a", "obras_b"])
        for s in sug:
            w.writerow([s["a"], s["b"], s["parecido"], "|".join(s["obras_a"]), "|".join(s["obras_b"])])

    (salida / "conflictos.json").write_text(
        json.dumps(conf, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    t = informe["totales"]
    print(f"relaciones leidas      {t['relaciones_leidas']:>6}")
    print(f"aristas unicas         {t['aristas_unicas']:>6}   (todas con procedencia)")
    print(f"nodos unicos           {t['nodos_unicos']:>6}")
    print()
    for seccion in ("tipos_corregidos", "hades_desambiguado", "tipos_desconocidos"):
        casos = informe[seccion]
        if casos:
            print(f"{seccion} ({sum(casos.values())}):")
            for k, v in sorted(casos.items(), key=lambda x: -x[1])[:10]:
                print(f"   {k:46} {v}")
            print()
    print(f"alias aplicados        {sum(informe['alias_aplicados'].values()):>6}")
    print(f"tipo tomado de la lista{sum(informe['tipo_por_lista'].values()):>6}")
    print(f"nombres fuera de lista {len(informe['fuera_de_lista']):>6} distintos")
    print()
    solo_wiki = sum(1 for a in aristas.values() if a["solo_wikipedia"])
    solo_poem = sum(1 for a in aristas.values() if a["solo_poemas"])
    print("de donde viene cada hecho:")
    print(f"   solo Wikipedia      {solo_wiki:>6}")
    print(f"   solo los poemas     {solo_poem:>6}")
    print(f"   ambas               {len(aristas) - solo_wiki - solo_poem:>6}")
    print()
    malas = [a for a in aristas.values() if a["viola_esquema"]]
    print(f"aristas que violan el esquema de tipos: {len(malas)}"
          f"  ({100*len(malas)/len(aristas):.0f}% del grafo)")
    for k, v in sorted(informe["violaciones_de_esquema"].items(), key=lambda x: -x[1])[:6]:
        print(f"   {k:46} {v}")
    print()
    print(f"grafias unidas por caja/tilde: {sum(informe['grafias_unidas'].values())}")
    print(f"pares a revisar para equivalencias.csv: {len(sug)}")
    print()
    duros = [c for c in conf if not c["multiple_legitimo"]]
    print(f"candidatos a conflicto {len(conf):>6}   de cardinalidad uno: {len(duros)}")
    print(f"\nescrito en {salida}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path("salida"))
