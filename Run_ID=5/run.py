"""
run.py
CLASSIFICATION: Command-Line Interface (ASTE V11.0)
GOAL: Provides a unified CLI for orchestrating suite tasks. The 'hunt'
      command now launches the persistent web-based Control Hub.
"""
import argparse
import subprocess
import sys
import os


def run_command(cmd: list) -> int:
    """Runs a command and returns its exit code."""
    try:
        # For the Flask app, we don't want to block, so use Popen
        if "app.py" in cmd[-1]:
            print(f"Launching Control Hub server: {' '.join(cmd)}")
            process = subprocess.Popen(cmd)
            print("Server is running. Access the UI in your browser.")
            print("Press Ctrl+C in this terminal to stop the server.")
            process.wait()
            return process.returncode
        else:
            result = subprocess.run(cmd, check=True, text=True)
            return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command '{' '.join(cmd)}' failed with exit code {e.returncode}.", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print(f"ERROR: Command not found: {cmd[0]}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nServer shutdown requested. Exiting.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="ASTE Suite Runner V11.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'hunt' command now launches the web server
    subparsers.add_parser("hunt", help="Launch the V11.0 Dynamic Control Hub (Flask server).")

    # 'validate-tda' command
    tda_parser = subparsers.add_parser("validate-tda", help="Run TDA validation on a specific hash")
    tda_parser.add_argument("hash", type=str, help="The config_hash of the run to analyze")

    args = parser.parse_args()

    cmd = []
    if args.command == "hunt":
        # Create templates directory if it doesn't exist, required by Flask
        if not os.path.exists("templates"):
            os.makedirs("templates")
        cmd = [sys.executable, "app.py"]
    elif args.command == "validate-tda":
        cmd = [sys.executable, "tda_taxonomy_validator.py", "--hash", args.hash]

    if not cmd:
        parser.print_help()
        sys.exit(1)

    print(f"--- [RUNNER] Initializing task: {args.command} ---")
    exit_code = run_command(cmd)

    if exit_code == 0:
        print(f"--- [RUNNER] Task '{args.command}' completed successfully. ---")
    else:
        print(f"--- [RUNNER] Task '{args.command}' FAILED (Exit Code: {exit_code}). ---")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
