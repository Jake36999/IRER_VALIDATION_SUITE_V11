"""
project_api.py
CLASSIFICATION: API Gateway (ASTE V10.0)
GOAL: Exposes core system functions to external callers (e.g., a web UI).
      This is NOT a script to be run directly, but to be IMPORTED from.
      It provides a stable, high-level Python API.
"""

import os
import sys
import json
import subprocess
from typing import Dict, Any, List, Optional


try:
    import settings
except ImportError:
    print("FATAL: 'settings.py' not found. Please create it first.", file=sys.stderr)
    raise


def start_hunt_process() -> Dict[str, Any]:
    """
    Starts the main control hub server as a background process.
    """
    app_script = "app.py"
    if not os.path.exists(app_script):
        return {"status": "error", "message": f"Control Hub script '{app_script}' not found."}

    try:
        process = subprocess.Popen(
            [sys.executable, app_script],
            stdout=open("control_hub.log", "w"),
            stderr=subprocess.STDOUT
        )
        return {
            "status": "success",
            "message": "Control Hub process started in the background.",
            "pid": process.pid
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to start control hub process: {e}"}


def run_ai_analysis(log_file: str, code_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calls the ai_assistant_core.py to perform analysis on a log file.
    """
    ai_core_script = "ai_assistant_core.py"
    if not os.path.exists(ai_core_script):
        return {"status": "error", "message": f"AI Core script '{ai_core_script}' not found."}

    try:
        cmd = [sys.executable, ai_core_script, "--log", log_file]
        if code_files:
            cmd.append("--code")
            cmd.extend(code_files)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )
        
        return {
            "status": "success",
            "message": "AI Analysis Complete.",
            "report": result.stdout
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"AI Core execution failed (Exit Code: {e.returncode}).",
            "error": e.stderr,
            "output": e.stdout
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to run AI Core: {e}"}

