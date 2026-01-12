"""
Analysis orchestration service - wraps core Vizora functionality.
"""

import json
import pandas as pd
from typing import Optional, Callable

from vizora.llm.client import get_orchestrator, Summarizer
from vizora.steps.profiling import build_dataset_profile, resolve_target_column
from vizora.steps.executor import execute_plan

from vizora.web.models.responses import AnalysisResult, FigureData
from vizora.web.services.figure_converter import convert_figures
from vizora.web.services.inference import build_metadata, build_model_pipeline, save_artifacts


class AnalysisService:
    """
    Orchestrates the full analysis pipeline.

    This service wraps the core Vizora functions to provide
    a clean interface for the web API.
    """

    def __init__(self):
        self.summarizer = Summarizer()

    def run_analysis(
        self,
        df: pd.DataFrame,
        mode: str,
        goal: str,
        target_column: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        run_id: Optional[str] = None
    ) -> AnalysisResult:
        """
        Run the full analysis pipeline.

        Args:
            df: The dataset to analyze.
            mode: Analysis mode (eda, predictive, hybrid).
            goal: User's analysis goal.
            target_column: Target column name (optional).
            progress_callback: Optional callback for progress updates.

        Returns:
            AnalysisResult with figures, metrics, summary, and plan.
        """
        def update_progress(step: str, percentage: int):
            if progress_callback:
                progress_callback(step, percentage)

        # Step 1: Resolve target column
        update_progress("Resolving target column...", 5)
        target_match = None
        if target_column:
            try:
                target_match = resolve_target_column(df.columns, target_column)
            except ValueError:
                # If no match found, leave as None
                pass

        # Step 2: Build dataset profile
        update_progress("Building dataset profile...", 10)
        profile = build_dataset_profile(df, goal, target_match, analysis_mode=mode)
        profile_json = json.dumps(profile)

        # Step 3: Get plan from orchestrator
        update_progress("Generating analysis plan...", 25)
        orchestrator = get_orchestrator(mode)
        plan_result = orchestrator.get_plan(profile_json)

        if plan_result["error"]:
            raise ValueError(f"Failed to generate plan: {plan_result['error']}")

        plan = plan_result["plan"]

        # Step 4: Execute plan
        update_progress("Executing analysis plan...", 40)
        ctx = execute_plan(df, plan, target_column=target_match, show_progress=False)

        # Step 4b: Persist model artifacts if available
        if run_id and ctx.model is not None:
            try:
                model_pipeline = build_model_pipeline(ctx)
                if model_pipeline is not None:
                    meta = build_metadata(ctx, run_id)
                    save_artifacts(run_id, model_pipeline, meta)
            except Exception as e:
                ctx.errors.append(f"Artifact persistence failed: {e}")

        # Step 5: Convert figures to base64
        update_progress("Processing visualizations...", 75)
        figures = convert_figures(ctx.figures) if ctx.figures else []

        # Step 6: Generate summary
        update_progress("Generating AI summary...", 85)
        dataset_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "target": target_match,
        }

        summary_result = self.summarizer.summarize(
            goal=goal,
            mode=mode,
            dataset_info=dataset_info,
            results=ctx.results,
            plan_notes=plan.get("notes", [])
        )

        summary_markdown = summary_result.get("summary", "")
        if summary_result.get("error"):
            summary_markdown = f"*Summary generation failed: {summary_result['error']}*"

        # Step 7: Build result
        update_progress("Complete!", 100)

        return AnalysisResult(
            figures=figures,
            metrics=ctx.results.get("model_metrics"),
            summary_markdown=summary_markdown,
            plan=plan,
            errors=ctx.errors,
            preprocessing=ctx.preprocessing
        )


# Global instance
analysis_service = AnalysisService()
