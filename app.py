"""interfaz de la demo.

responsable: ximena.

la demo tiene el control de alpha a la vista a proposito: mover el deslizador
de 1.0 a 0.0 cambia el sistema de "solo denso" a "solo bm25" y se ve en vivo lo
mismo que dice la tabla de ablacion.
"""

import os

import gradio as gr

from rag import Answer, answer

# una de cada grupo del set de evaluacion: una que esta, una donde las fuentes
# se contradicen, y una que no esta en el corpus y tiene que abstenerse.
EXAMPLES = [
    "¿Por que recibio Pandora ese nombre?",
    "¿Quien era la madre de Afrodita?",
    "¿Quien arrojo a Hefesto del Olimpo y quien lo recogio?",
    "¿Quien acompaño a Gilgamesh en su viaje?",
]


def format_citations(result: Answer) -> str:
    if result.abstained:
        return "_sin citas: la respuesta no esta en el corpus._"
    if not result.citations:
        return "_sin citas._"

    lines = []
    for citation in result.citations:
        lines.append(f"**{citation}**")
        lines.append(f"> {citation.text}")
        lines.append("")
    return "\n".join(lines)


def format_conflict(result: Answer) -> str:
    if not result.has_conflict:
        return "_las fuentes no se contradicen en esta respuesta._"

    lines = ["las fuentes dan versiones distintas:", ""]
    for version in result.conflicting_versions:
        lines.append(f"- {version.claim}  \n  — {version.citation}")
    return "\n".join(lines)


def ask(question: str, alpha: float) -> tuple[str, str, str]:
    question = (question or "").strip()
    if not question:
        return "escribe una pregunta.", "", ""

    result = answer(question, alpha=alpha)
    prefix = "**no esta en el corpus.** " if result.abstained else ""
    return prefix + result.text, format_citations(result), format_conflict(result)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="mythos-rag") as demo:
        gr.Markdown(
            "# mythos-rag\n"
            "consultas sobre un corpus de mitologia griega de dominio publico. "
            "toda respuesta viene con el pasaje que la respalda, y cuando las "
            "fuentes no coinciden se muestran las dos versiones."
        )

        if not os.environ.get("GOOGLE_API_KEY"):
            gr.Markdown(
                "> sin `GOOGLE_API_KEY`: la respuesta es el pasaje recuperado "
                "tal cual, sin redactar. las citas y los conflictos funcionan igual."
            )

        question = gr.Textbox(
            label="pregunta",
            placeholder="quien fue la madre de heracles?",
            lines=2,
        )
        alpha = gr.Slider(
            0.0, 1.0, value=0.5, step=0.1,
            label="alpha: 1.0 solo embeddings, 0.0 solo bm25",
        )
        submit = gr.Button("consultar", variant="primary")

        response = gr.Markdown(label="respuesta")
        with gr.Row():
            citations = gr.Markdown(label="citas")
            conflict = gr.Markdown(label="versiones en conflicto")

        gr.Examples(examples=EXAMPLES, inputs=question)

        outputs = [response, citations, conflict]
        submit.click(ask, inputs=[question, alpha], outputs=outputs)
        question.submit(ask, inputs=[question, alpha], outputs=outputs)

    return demo


if __name__ == "__main__":
    build_ui().launch()
