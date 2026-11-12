"""busca un termino en las ocho obras y muestra el contexto, fuente por fuente.

es la herramienta para **verificar preguntas a mano**: traer los dos lados de una
contradiccion, leerlos y copiar los pasajes a `eval/questions.json`. no es parte
del pipeline.

    python search.py pandora
    python search.py "foam-born" 2        # 2 coincidencias por fuente
    python search.py hope 3 600           # y 600 caracteres de contexto
"""

import sys
import textwrap

from corpus import SOURCES, load_corpus


def search(term: str, per_source: int = 1, width: int = 420) -> None:
    term = term.lower()
    corpus = load_corpus(SOURCES)

    found = False
    for source, text in corpus.items():
        low = text.lower()
        total = low.count(term)
        if not total:
            continue
        found = True
        print(f"### {source.author}/{source.work[:34]}  [{source.filename}]  ({total} veces)")
        i = 0
        for _ in range(per_source):
            i = low.find(term, i)
            if i == -1:
                break
            snippet = " ".join(text[max(0, i - 120) : i + width].split())
            print(textwrap.fill(f"    ...{snippet}...", 112, subsequent_indent="    "))
            print()
            i += len(term)

    if not found:
        print(f"'{term}' no aparece en ninguna de las {len(corpus)} obras.")
        print("util para el grupo 'fuera': confirma que algo NO esta en el corpus.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    search(
        sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else 1,
        int(sys.argv[3]) if len(sys.argv) > 3 else 420,
    )
