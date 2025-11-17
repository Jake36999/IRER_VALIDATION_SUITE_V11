#!/usr/bin/env python
"""
ai_assistant_core.py
CLASSIFICATION: Agnostic AI Debugging Co-Pilot
GOAL: Analyze failure logs, code snippets, and transcripts to provide
      root cause analysis and actionable solutions for the ASTE project.
"""


import os
import re
import json
import argparse
from typing import Dict, List, Optional


# Conditional imports for cloud providers
try:
    # FAKE STUB for Google Vertex AI
    # import vertexai
    # from vertexai.generative_models import GenerativeModel
    pass
except ImportError:
    print("Warning: Google libraries not found. GEMINI mode will fail if invoked.")


class AgnosticAIAssistant:
    """
    Agnostic AI assistant for the ASTE project.
    Can run in BASIC (regex) or GEMINI (full AI) mode.
    """
    def __init__(self, mode: str, project_context: Optional[str] = None):
        self.mode = mode.upper()
        self.project_context = project_context or self.get_default_context()
        
        if self.mode == "GEMINI":
            print("Initializing assistant in GEMINI mode (stubbed).")
            # In a real application, the cloud client and system instruction would be set here.
            # self.client = GenerativeModel("gemini-1.5-pro")
            # self.client.system_instruction = self.project_context
        else:
            print("Initializing assistant in BASIC mode.")


    def get_default_context(self) -> str:
        """Provides the master prompt context for Gemini."""
        return """
        You are a 'Debugging Co-Pilot' for the 'ASTE' project, a complex scientific
        simulation using JAX, Python, and a Hunter-Worker architecture.
        Your task is to analyze failure logs and code to provide root cause analysis
        and actionable solutions.
        
        Our project has 6 common bug types:
        1. ENVIRONMENT_ERROR (e.g., ModuleNotFoundError)
        2. SYNTAX_ERROR (e.g., typos)
        3. JAX_COMPILATION_ERROR (e.g., ConcretizationTypeError)
        4. IMPORT_ERROR (e.g., NameError)
        5. LOGIC_ERROR (e.g., AttributeError)
        6. SCIENTIFIC_VALIDATION_ERROR (e.g., flawed physics, bad SSE scorer)
        
        Always classify the error into one of these types before explaining.
        """


    def analyze_failure(self, log_content: str, code_snippets: List[str], transcripts: List[str]) -> Dict:
        """
        Analyzes artifacts and returns a structured debug report.
        """
        if self.mode == "GEMINI":
            return self._analyze_with_gemini(log_content, code_snippets, transcripts)
        else:
            return self._analyze_with_basic(log_content)


    def _analyze_with_basic(self, log_content: str) -> Dict:
        """BASIC mode: Uses regex for simple, common errors."""
        report = {
            "classification": "UNKNOWN",
            "summary": "No root cause identified in BASIC mode.",
            "recommendation": "Re-run in GEMINI mode for deep analysis."
        }


        if re.search(r"ModuleNotFoundError", log_content, re.IGNORECASE):
            report["classification"] = "ENVIRONMENT_ERROR"
            report["summary"] = "A required Python module was not found."
            report["recommendation"] = "Identify the missing module from the log and run `pip install <module_name>`. Verify your `requirements.txt`."
            return report


        if re.search(r"SyntaxError", log_content, re.IGNORECASE):
            report["classification"] = "SYNTAX_ERROR"
            report["summary"] = "A Python syntax error was detected."
            report["recommendation"] = "Check the line number indicated in the log for typos, incorrect indentation, or missing characters."
            return report
        
        return report


    def _analyze_with_gemini(self, log_content: str, code_snippets: List[str], transcripts: List[str]) -> Dict:
        """GEMINI mode: Simulates deep semantic analysis for complex errors."""
        print("Performing deep semantic analysis (mock)...")


        if "ConcretizationTypeError" in log_content or "JAX" in log_content.upper():
            return {
                "classification": "JAX_COMPILATION_ERROR",
                "summary": "A JAX ConcretizationTypeError was detected. This typically occurs when a non-static value is used in a context requiring a compile-time constant.",
                "recommendation": "Review JIT-compiled functions. Ensure that arguments controlling Python-level control flow (e.g., if/for loops) are marked as static using `static_argnums` or `static_argnames` in `@jax.jit`. Alternatively, refactor to use `jax.lax.cond` or `jax.lax.scan`."
            }


        if "SSE" in log_content or "validation failed" in log_content.lower():
            return {
                "classification": "SCIENTIFIC_VALIDATION_ERROR",
                "summary": "The simulation completed but failed scientific validation checks. The SSE score was outside the acceptable tolerance, or a key physical metric was violated.",
                "recommendation": "Analyze the `provenance.json` artifact for the failed run. Compare the physical parameters to known-good runs. Investigate the final field state in `rho_history.h5` for signs of instability or divergence."
            }


        return {
            "classification": "GENERIC_GEMINI_ANALYSIS",
            "summary": "Gemini analysis complete. Contextual correlation was performed.",
            "recommendation": "Review the full analysis for complex discrepancies."
        }


def main():
    parser = argparse.ArgumentParser(description="ASTE Agnostic Debugging Co-Pilot")
    parser.add_argument("--log", required=True, help="Path to the failure log file.")
    parser.add_argument("--code", nargs="+", help="Paths to relevant code files.", default=[])
    parser.add_argument("--transcript", nargs="+", help="Paths to relevant project transcripts.", default=[])
    args = parser.parse_args()


    try:
        with open(args.log, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found at {args.log}", file=sys.stderr)
        exit(1)
        
    code_snippets = []
    for path in args.code:
        try:
            with open(path, 'r') as f:
                code_snippets.append(f"--- Content from {path} ---\n{f.read()}")
        except Exception as e:
            print(f"Warning: Could not read code file {path}: {e}")
            
    transcripts = []
    for path in args.transcript:
        try:
            with open(path, 'r') as f:
                transcripts.append(f"--- Transcript {path} ---\n{f.read()}")
        except Exception as e:
            print(f"Warning: Could not read transcript file {path}: {e}")


    mode = os.environ.get("AI_ASSISTANT_MODE", "BASIC")
    assistant = AgnosticAIAssistant(mode=mode)
    report = assistant.analyze_failure(log_content, code_snippets, transcripts)


    print("\n" + "="*80)
    print("--- ASTE DEBUGGING CO-PILOT REPORT ---")
    print("="*80)
    print(f"Mode:         {mode.upper()}")
    print(f"Classification: {report.get('classification', 'N/A')}")
    print("\n--- Summary ---")
    print(report.get('summary', 'N/A'))
    print("\n--- Recommendation ---")
    print(report.get('recommendation', 'N/A'))
    print("="*80)


if __name__ == "__main__":
    main()
