from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from persona_support_agent.classifier import PersonaClassifier  # noqa: E402
from persona_support_agent.rag import LocalRAGPipeline  # noqa: E402
from persona_support_agent.response import AdaptiveResponder  # noqa: E402


def main():
    console = Console()
    classifier = PersonaClassifier()
    rag = LocalRAGPipeline()
    responder = AdaptiveResponder()

    console.print("[bold green]Persona Adaptive Support Agent[/bold green]\n")
    console.print("Type 'exit' to quit.\n")

    history: list[dict] = []

    while True:
        user_message = console.input("[bold blue]User message:[/bold blue] ")

        if not user_message or user_message.strip().lower() in {"exit", "quit"}:
            console.print("Goodbye!\n")
            break

        history.append({"role": "user", "content": user_message})

        persona_result = classifier.classify(user_message)
        persona = persona_result.get("persona", "Technical Expert")
        confidence = persona_result.get("confidence", 0.0)
        reasoning = persona_result.get("reasoning", "")

        retrieved = rag.retrieve_context(user_message, top_k=3)
        generate_result = responder.generate(
            user_message,
            persona,
            retrieved,
            conversation_history=history,
        )

        answer = generate_result.get("answer", "")
        escalate = responder.should_escalate(user_message, persona, retrieved)

        history.append({"role": "assistant", "content": answer})

        console.print("\n[bold yellow]Detected persona:[/bold yellow] ", persona)
        console.print("[bold yellow]Persona confidence:[/bold yellow] ", f"{confidence:.2f}")
        console.print("[bold yellow]Reasoning:[/bold yellow] ", reasoning)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Source")
        table.add_column("Score", justify="right")
        table.add_column("Text", overflow="fold")
        for chunk in retrieved:
            page = chunk.get("page_number")
            section = chunk.get("section")
            provenance = chunk.get("source", "unknown")
            if page is not None:
                provenance += f" (page {page})"
            elif section:
                provenance += f" ({section})"

            table.add_row(
                provenance,
                f"{chunk['score']:.2f}",
                chunk["text"][:140].replace("\n", " "),
            )
        console.print(table)

        console.print("\n[bold cyan]Response:[/bold cyan]\n", answer)
        console.print("\n[bold red]Escalation needed:[/bold red] ", "YES" if escalate else "NO")
        if escalate:
            handoff = responder.build_handoff_summary(
                user_message,
                persona,
                retrieved,
                conversation_history=history,
            )

            console.print("\n[bold red]Handoff summary:[/bold red]\n", handoff)

        console.print("\n---\n")


if __name__ == "__main__":
    main()

