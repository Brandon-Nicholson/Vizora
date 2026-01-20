# openai wrapper
from openai import OpenAI
from dotenv import load_dotenv
from vizora.llm.prompts import (
    EDA_PLANNER_SYSTEM_PROMPT,
    MODEL_PLANNER_SYSTEM_PROMPT,
    HYBRID_PLANNER_SYSTEM_PROMPT,
    FORECAST_PLANNER_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    FORECAST_SUMMARIZER_SYSTEM_PROMPT,
)
import os
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
orchestrator_model = os.getenv("ORCHESTRATOR_MODEL")

# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """
    OpenAI client wrapper optimized for structured JSON output.

    Key optimizations for speed:
    1. Uses response_format=json_object for guaranteed JSON output
    2. Sets reasonable max_tokens to prevent over-generation
    3. Uses temperature=0 for deterministic, faster responses
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key: str = api_key,
        max_tokens: int = 2000,  # limit tokens to 2000
        temperature: float = 0.0,  # Deterministic
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def get_plan(self, profile_json: str) -> dict:
        """
        Get a structured plan from the orchestrator.

        Args:
            profile_json: JSON string of the dataset profile

        Returns:
            dict with:
            - "plan": parsed JSON plan (dict) or None if parsing failed
            - "raw": raw response string
            - "usage": token usage dict
            - "error": error message if any
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": profile_json}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            raw_content = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}

            # Parse JSON
            try:
                plan = json.loads(raw_content)
                return {
                    "plan": plan,
                    "raw": raw_content,
                    "usage": usage,
                    "error": None,
                }
            except json.JSONDecodeError as e:
                return {
                    "plan": None,
                    "raw": raw_content,
                    "usage": usage,
                    "error": f"JSON parse error: {e}",
                }

        except Exception as e:
            return {
                "plan": None,
                "raw": "",
                "usage": {},
                "error": f"API error: {e}",
            }


# =============================================================================
# ORCHESTRATOR FACTORY
# =============================================================================

def get_orchestrator(mode: str = "eda") -> LLMClient:
    """
    Get an orchestrator agent configured for the specified mode.

    Args:
        mode: One of "eda", "predictive", "hybrid", or "forecast"

    Returns:
        Configured LLMClient instance
    """
    prompts = {
        "eda": EDA_PLANNER_SYSTEM_PROMPT,
        "predictive": MODEL_PLANNER_SYSTEM_PROMPT,
        "hybrid": HYBRID_PLANNER_SYSTEM_PROMPT,
        "forecast": FORECAST_PLANNER_SYSTEM_PROMPT,
    }

    if mode not in prompts:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {list(prompts.keys())}")

    # Forecast mode needs more tokens for time series analysis
    max_tokens = 2500 if mode in ("hybrid", "forecast") else 2000

    return LLMClient(
        model=orchestrator_model,
        system_prompt=prompts[mode],
        max_tokens=max_tokens,
    )


# Default orchestrator for backwards compatibility
orchestrator_agent = get_orchestrator("eda")


# =============================================================================
# SUMMARIZER
# =============================================================================

class Summarizer:
    """
    Generates a user-friendly summary of analysis results.
    """

    def __init__(self, model: str = None, api_key: str = api_key):
        self.model = model or orchestrator_model
        self.client = OpenAI(api_key=api_key)

    def summarize(
        self,
        goal: str,
        mode: str,
        dataset_info: dict,
        results: dict,
        plan_notes: list = None,
        forecast_config: dict = None,
    ) -> dict:
        """
        Generate a summary of the analysis results.

        Args:
            goal: User's original analysis goal
            mode: Analysis mode (eda, predictive, hybrid, forecast)
            dataset_info: Basic dataset info (rows, cols, target)
            results: Execution results from ctx.results
            plan_notes: Notes from the plan (optional)
            forecast_config: Forecast configuration (for forecast mode)

        Returns:
            dict with "summary" (markdown string) and "error" (if any)
        """
        # Choose appropriate prompt based on mode
        if mode == "forecast":
            system_prompt = FORECAST_SUMMARIZER_SYSTEM_PROMPT
            # Build forecast-specific context
            context = {
                "goal": goal,
                "mode": mode,
                "dataset_info": dataset_info,
                "time_series_info": results.get("decomposition", {}),
                "forecast_metrics": results.get("forecast_metrics", {}),
                "forecast_config": forecast_config or {},
                "notes": plan_notes or []
            }
        else:
            system_prompt = SUMMARIZER_SYSTEM_PROMPT
            # Build standard context
            context = {
                "goal": goal,
                "mode": mode,
                "dataset_info": dataset_info,
                "model_metrics": results.get("model_metrics"),
                "feature_importance": None,
                "describe_stats": results.get("describe"),
                "notes": plan_notes or []
            }

            # Extract feature importance if available, including the source model name
            for key, value in results.items():
                if key.startswith("feature_importance_"):
                    model_name = key.replace("feature_importance_", "")
                    context["feature_importance"] = {
                        "model": model_name,
                        "features": value
                    }
                    break

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, indent=2, default=str)}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=1500,
                temperature=0.3,  # Slightly creative for natural language
            )

            summary = response.choices[0].message.content or ""
            return {"summary": summary, "error": None}

        except Exception as e:
            return {"summary": None, "error": f"Summarizer error: {e}"}