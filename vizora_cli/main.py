# Typer app for Vizora CLI

import typer
import json
import time
import pandas as pd
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.table import Table

from vizora.llm.client import get_orchestrator, Summarizer
from vizora.steps.profiling import build_dataset_profile, resolve_target_column
from vizora.steps.executor import execute_plan

app = typer.Typer(help="Vizora - AI-powered data analysis agent")
console = Console()


@app.command()
def analyze(
    dataset: str = typer.Argument(..., help="Path to CSV dataset"),
    goal: str = typer.Option(..., "--goal", "-g", help="User analysis goal"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    mode: str = typer.Option("eda", "--mode", "-m", help="eda | predictive | hybrid"),
    show_profile: bool = typer.Option(False, "--show-profile", help="Show the dataset profile sent to agent"),
):
    """
    Analyze a dataset and generate an execution plan.
    """
    # Validate mode
    if mode not in ("eda", "predictive", "hybrid"):
        console.print(f"[red]Error: Invalid mode '{mode}'. Must be eda, predictive, or hybrid[/red]")
        raise typer.Exit(1)

    # Load dataset
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading dataset:[/blue] {dataset_path.name}")
    df = pd.read_csv(dataset_path)
    console.print(f"[green]Loaded {len(df):,} rows x {len(df.columns)} columns[/green]")

    # Resolve target column if provided
    target_match = None
    if target:
        try:
            target_match = resolve_target_column(df.columns, target)
            console.print(f"[green]Target column:[/green] {target_match}")
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    elif mode in ("predictive", "hybrid"):
        console.print("[yellow]Warning: No target column specified for predictive/hybrid mode[/yellow]")

    # Build profile
    console.print("[blue]Building dataset profile...[/blue]")
    profile = build_dataset_profile(df, goal, target_match, analysis_mode=mode)

    if show_profile:
        console.print(Panel(JSON(json.dumps(profile, indent=2)), title="Dataset Profile"))

    profile_json = json.dumps(profile)
    console.print(f"[dim]Profile size: {len(profile_json):,} bytes[/dim]")

    # Get orchestrator and generate plan
    console.print(f"[blue]Generating {mode.upper()} plan...[/blue]")
    orchestrator = get_orchestrator(mode)

    start_time = time.time()
    result = orchestrator.get_plan(profile_json)
    elapsed = time.time() - start_time

    # Display results
    if result["error"]:
        console.print(f"[red]Error: {result['error']}[/red]")
        if result["raw"]:
            console.print(Panel(result["raw"], title="Raw Response"))
        raise typer.Exit(1)

    console.print(f"[green]Plan generated in {elapsed:.1f}s[/green]")

    # Show token usage
    usage = result.get("usage", {})
    if usage:
        console.print(
            f"[dim]Tokens: {usage.get('prompt_tokens', 0):,} prompt + "
            f"{usage.get('completion_tokens', 0):,} completion = "
            f"{usage.get('total_tokens', 0):,} total[/dim]"
        )

    # Display the plan
    console.print(Panel(JSON(json.dumps(result["plan"], indent=2)), title="Execution Plan"))

    return result["plan"], df, target_match


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Path to CSV dataset"),
    goal: str = typer.Option(..., "--goal", "-g", help="User analysis goal"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    mode: str = typer.Option("eda", "--mode", "-m", help="eda | predictive | hybrid"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="Directory to save figures"),
):
    """
    Generate a plan and execute it, saving all outputs.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path as P

    print(">>> RUN COMMAND STARTED <<<")
    
    # Validate mode
    if mode not in ("eda", "predictive", "hybrid"):
        console.print(f"[red]Error: Invalid mode '{mode}'. Must be eda, predictive, or hybrid[/red]")
        raise typer.Exit(1)

    # Load dataset
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading dataset:[/blue] {dataset_path.name}")
    df = pd.read_csv(dataset_path)
    console.print(f"[green]Loaded {len(df):,} rows x {len(df.columns)} columns[/green]")

    # Resolve target column if provided
    target_match = None
    if target:
        try:
            target_match = resolve_target_column(df.columns, target)
            console.print(f"[green]Target column:[/green] {target_match}")
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    # Build profile
    console.print("[blue]Building dataset profile...[/blue]")
    profile = build_dataset_profile(df, goal, target_match, analysis_mode=mode)
    profile_json = json.dumps(profile)
    console.print(f"[dim]Profile size: {len(profile_json):,} bytes[/dim]")

    # Generate plan
    console.print(f"[blue]Generating {mode.upper()} plan...[/blue]")
    orchestrator = get_orchestrator(mode)

    start_time = time.time()
    result = orchestrator.get_plan(profile_json)
    plan_time = time.time() - start_time

    if result["error"]:
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Plan generated in {plan_time:.1f}s[/green]")
    plan = result["plan"]

    # Execute plan
    console.print("\n[blue]Executing plan...[/blue]")
    start_time = time.time()
    ctx = execute_plan(df, plan, target_column=target_match, show_progress=True)
    exec_time = time.time() - start_time

    console.print(f"\n[green]Execution completed in {exec_time:.1f}s[/green]")

    # Show errors if any
    if ctx.errors:
        console.print(f"\n[yellow]Warnings/Errors ({len(ctx.errors)}):[/yellow]")
        for err in ctx.errors:
            console.print(f"  [yellow]- {err}[/yellow]")

    # Show model metrics if available
    if "model_metrics" in ctx.results:
        console.print("\n[bold]Model Metrics:[/bold]")
        metrics = ctx.results["model_metrics"]

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Model")

        # Get all metric names from first model
        first_model = list(metrics.values())[0]
        for metric_name in first_model.keys():
            table.add_column(metric_name.upper())

        for model_name, model_metrics in metrics.items():
            row = [model_name]
            for val in model_metrics.values():
                row.append(str(val) if not isinstance(val, float) else f"{val:.4f}")
            table.add_row(*row)

        console.print(table)

    # Save figures
    if ctx.figures:
        output_path = P(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[blue]Saving {len(ctx.figures)} figures to {output_dir}/[/blue]")
        for i, (fig_type, fig_name, fig) in enumerate(ctx.figures):
            filename = f"{i+1:02d}_{fig_type}_{fig_name}.png"
            filepath = output_path / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            console.print(f"  [dim]Saved: {filename}[/dim]")

        console.print(f"[green]All figures saved to {output_dir}/[/green]")

    # Save results JSON (including preprocessing artifacts for reproducibility)
    results_path = P(output_dir) / "results.json"
    # Convert any non-serializable items
    serializable_results = {}
    for k, v in ctx.results.items():
        if isinstance(v, dict):
            serializable_results[k] = v
        else:
            serializable_results[k] = str(v)

    # Add preprocessing artifacts for reproducibility
    serializable_results["preprocessing"] = ctx.preprocessing

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    console.print(f"[green]Results saved to {results_path}[/green]")

    # Generate summary
    console.print("\n[blue]Generating summary...[/blue]")
    summarizer = Summarizer()

    dataset_info = {
        "rows": len(df),
        "columns": len(df.columns),
        "target": target_match,
    }

    summary_result = summarizer.summarize(
        goal=goal,
        mode=mode,
        dataset_info=dataset_info,
        results=ctx.results,
        plan_notes=plan.get("notes", [])
    )

    if summary_result["error"]:
        console.print(f"[yellow]Warning: {summary_result['error']}[/yellow]")
    else:
        from rich.markdown import Markdown
        console.print("\n")
        console.print(Panel(Markdown(summary_result["summary"]), title="Analysis Summary", border_style="green"))

        # Save summary
        summary_path = P(output_dir) / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_result["summary"])
        console.print(f"[green]Summary saved to {summary_path}[/green]")

    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    app()