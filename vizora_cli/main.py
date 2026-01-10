# Typer app

import typer
import json
from typing import Optional
from vizora.llm.client import orchestrator_agent
from vizora.steps.profiling import build_dataset_profile, resolve_target_column, df

app = typer.Typer() # initialize Typer app

# "Perform basic EDA to show the relationship of the features with heart disease. Put features with high correlation into the spotlight, using data vizualization tools. Summarize which features showed the highest correlation and try to explain why."


@app.command()
def profile(
    goal: str = typer.Option(..., "--goal", "-g", help="User analysis goal"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    mode: str = typer.Option("eda", "--mode", "-m", help="eda | predictive | both"),
):
    # resolve target column if present
    target_match = resolve_target_column(df.columns, target) if target else None
    # build dataset profile
    profile = build_dataset_profile(df, goal, target_match)

    print(orchestrator_agent.get_response(json.dumps(profile)))

if __name__ == "__main__":
    app()