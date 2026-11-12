"""interfaz de la demo.

responsable: ximena.

funciona contra el stub de rag.answer(), asi que se puede construir y probar
antes de que el pipeline exista. cuando rag.STUB pase a False, esto no cambia.
"""

import gradio as gr

from rag import Answer, answer

EXAMPLES = [
    "quien fue la madre de heracles?",
    "cuantos trabajos de heracles cuentan las fuentes?",
    "que dice el corpus sobre la guerra de las galias?",
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


def ask(question: str) -> tuple[str, str, str]:
    question = (question or "").strip()
    if not question:
        return "escribe una pregunta.", "", ""

    result = answer(question)
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

        question = gr.Textbox(
            label="pregunta",
            placeholder="quien fue la madre de heracles?",
            lines=2,
        )
        submit = gr.Button("consultar", variant="primary")

        response = gr.Markdown(label="respuesta")
        with gr.Row():
            citations = gr.Markdown(label="citas")
            conflict = gr.Markdown(label="versiones en conflicto")

        gr.Examples(examples=EXAMPLES, inputs=question)

        outputs = [response, citations, conflict]
        submit.click(ask, inputs=question, outputs=outputs)
        question.submit(ask, inputs=question, outputs=outputs)

    return demo


if __name__ == "__main__":
    build_ui().launch()
