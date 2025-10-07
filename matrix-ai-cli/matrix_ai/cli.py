import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import json
import os

from .core.ai import call_ai_agent
from .core.sdv import generate_sdv_data_optimized

app = typer.Typer()
console = Console()

@app.command()
def main(
    description: str = typer.Option(..., "--description", "-d", help="A natural language description of the desired dataset."),
    records: int = typer.Option(..., "--records", "-r", help="The number of records to generate."),
    output: str = typer.Option("./output", "--output", "-o", help="The directory to save the generated data."),
):
    """Generates synthetic data based on a natural language description."""

    console.print("[bold green]Matrix AI - Synthetic Data Generator[/bold green]")

    # Step 1: Design Schema
    with console.status("[bold green]Calling AI to design schema...[/bold green]"):
        try:
            ai_output = call_ai_agent(description, records)
        except Exception as e:
            console.print(f"[bold red]Error during schema design: {e}[/bold red]")
            raise typer.Exit(1)

    console.print("[bold green]Schema design complete. Please review.[/bold green]")

    # Step 2: Review and Refine
    while True:
        metadata_preview = ai_output.get("metadata_dict", {})
        seed_data_preview = ai_output.get("seed_tables_dict", {})

        console.print("\n[bold yellow]Generated Schema:[/bold yellow]")
        console.print(json.dumps(metadata_preview, indent=2))

        console.print("\n[bold yellow]Seed Data Preview:[/bold yellow]")
        for table_name, data in seed_data_preview.items():
            table = Table(title=table_name)
            if data:
                headers = data[0].keys()
                for header in headers:
                    table.add_column(header)
                for row in data:
                    table.add_row(*[str(v) for v in row.values()])
            console.print(table)

        action = typer.prompt("\nDo you want to [A]pprove, [R]efine, or [Q]uit?", default="A").lower()

        if action == 'a':
            break
        elif action == 'r':
            refinement_prompt = typer.prompt("Describe the changes you want to make:")
            description = refinement_prompt
            with console.status("[bold green]Calling AI to refine schema...[/bold green]"):
                try:
                    ai_output = call_ai_agent(description, records, existing_metadata_json=json.dumps(metadata_preview))
                except Exception as e:
                    console.print(f"[bold red]Error during schema refinement: {e}[/bold red]")
                    continue
        elif action == 'q':
            raise typer.Exit()

    # Step 3: Synthesize Data
    console.print("\n[bold green]Schema approved. Starting data synthesis...[/bold green]")
    with Progress() as progress:
        task = progress.add_task("[cyan]Generating data...[/cyan]", total=100)

        def progress_callback(status, step, percent, **kwargs):
            progress.update(task, completed=percent, description=f"[cyan]{step}[/cyan]")

        try:
            synthetic_data = generate_sdv_data_optimized(
                num_records=records,
                metadata_dict=ai_output["metadata_dict"],
                seed_tables_dict=ai_output["seed_tables_dict"],
                data_description=description,
                progress_callback=progress_callback
            )
            progress.update(task, completed=100, description="[bold green]Synthesis complete.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error during data synthesis: {e}[/bold red]")
            raise typer.Exit(1)

    # Step 4: Save Data
    console.print(f"\n[bold green]Saving data to {output}...[/bold green]")
    try:
        os.makedirs(output, exist_ok=True)
        for table_name, df in synthetic_data.items():
            file_path = os.path.join(output, f"{table_name}.csv")
            df.to_csv(file_path, index=False)
            console.print(f"- Saved {file_path}")

        # Save metadata and reports
        with open(os.path.join(output, "metadata.json"), "w") as f:
            json.dump(ai_output["metadata_dict"], f, indent=2)
        console.print(f"- Saved {os.path.join(output, 'metadata.json')}")

    except Exception as e:
        console.print(f"[bold red]Error saving data: {e}[/bold red]")
        raise typer.Exit(1)

    console.print("\n[bold blue]Data generation complete.[/bold blue]")

if __name__ == "__main__":
    app()
