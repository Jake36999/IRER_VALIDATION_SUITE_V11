"""
app.py
CLASSIFICATION: V11.0 Control Plane Server
GOAL: Provides a persistent, web-based meta-orchestration layer for the
      IRER suite. This is the main entrypoint for the V11.0 system.

It implements the non-blocking architecture by spawning two key threads:
1. HuntThread: Runs the core_engine.execute_hunt() function.
2. WatcherThread: Runs the ProvenanceWatcher to monitor for results.
"""

import os
import json
import logging
import threading
import time
from flask import Flask, render_template, jsonify, request, send_from_directory
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import settings
import core_engine

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ControlHub] - %(message)s')
PROVENANCE_DIR = settings.PROVENANCE_DIR
STATUS_FILE = settings.STATUS_FILE
HUNT_RUNNING_LOCK = threading.Lock()
g_hunt_in_progress = False

app = Flask(__name__, template_folder=".") # Use current dir for templates

# --- State Management ---
def update_status(new_data: dict = {}, append_file: str = None):
    """Thread-safe function to read, update, and write the central status.json file."""
    with HUNT_RUNNING_LOCK:
        status = {"hunt_status": "Idle", "last_event": "-", "last_sse": "-", "last_h_norm": "-", "final_result": {}}
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, 'r') as f:
                    status = json.load(f)
            except json.JSONDecodeError:
                pass # Overwrite corrupted file

        status.update(new_data)

        # AUDIT : This "found_files" list is an unbounded memory/IO leak.
        # It has been removed per the V11.1 remediation mandate.
        # Original flawed code:
        # if append_file and append_file not in status["found_files"]:
        #    status["found_files"].append(append_file)

        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)

# --- Watchdog Service (WatcherThread) ---
class ProvenanceWatcher(FileSystemEventHandler):
    """Monitors the provenance directory for new JSON artifacts."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            logging.info(f"Watcher: Detected new artifact: {event.src_path}")

            # This is the "Layer 2 Analysis" trigger
            # It reads the artifact and updates the central status file
            # for the UI to poll.
            try:
                with open(event.src_path, 'r') as f:
                    provenance = json.load(f)

                sse = provenance.get(settings.SSE_METRIC_KEY, -1.0)
                h_norm = provenance.get(settings.STABILITY_METRIC_KEY, -1.0)
                job_uuid = provenance.get(settings.HASH_KEY, "unknown")

                # AUDIT : These are hardcoded "magic strings"
                # that create a fragile contract with the UI.
                # A V11.1 fix is required.
                status_data = {
                    "last_event": f"Analyzed {job_uuid[:8]}...",
                    "last_sse": f"{sse:.6f}",
                    "last_h_norm": f"{h_norm:.6f}"
                }
                update_status(new_data=status_data)

            except Exception as e:
                logging.error(f"Watcher: Failed to parse {event.src_path}: {e}")

def start_watcher_service():
    """Launches the WatcherThread daemon."""
    os.makedirs(PROVENANCE_DIR, exist_ok=True)
    event_handler = ProvenanceWatcher()
    observer = Observer()
    observer.schedule(event_handler, PROVENANCE_DIR, recursive=False)
    observer.daemon = True
    observer.start()
    logging.info(f"Watcher Service: Monitoring {PROVENANCE_DIR}")

# --- Core Engine Runner (HuntThread) ---
def run_hunt_in_background(num_generations, population_size):
    """The target function for the non-blocking HuntThread."""
    global g_hunt_in_progress

    # Use lock to ensure only one hunt runs at a time
    if not HUNT_RUNNING_LOCK.acquire(blocking=False):
        logging.warning("Hunt Thread: Hunt start requested, but already running.")
        return

    g_hunt_in_progress = True
    logging.info(f"Hunt Thread: Starting hunt (Gens: {num_generations}, Pop: {population_size}).")

    try:
        # AUDIT : Hardcoded keys.
        update_status(new_data={"hunt_status": "Running", "last_event": "Initializing...", "last_sse": "-", "last_h_norm": "-", "final_result": {}})

        # Call the decoupled Data Plane engine
        final_run = core_engine.execute_hunt(num_generations, population_size)

        # AUDIT : Hardcoded keys.
        update_status(new_data={"hunt_status": "Completed", "final_result": final_run})

    except Exception as e:
        logging.error(f"Hunt Thread: CRITICAL FAILURE: {e}")
        update_status(new_data={"hunt_status": f"Error: {e}"})
    finally:
        g_hunt_in_progress = False
        HUNT_RUNNING_LOCK.release()
        logging.info("Hunt Thread: Hunt finished.")

# --- Flask API Endpoints ---
@app.route('/')
def index():
    """Serves the main Control Hub UI."""
    return render_template('index.html')

@app.route('/api/start-hunt', methods=['POST'])
def api_start_hunt():
    """
    Non-blocking endpoint to start a new hunt.
    Spawns the HuntThread and returns 202 immediately.
    """
    if g_hunt_in_progress:
        return jsonify({"status": "error", "message": "A hunt is already in progress."}), 409

    data = request.json or {}
    generations = data.get('generations', settings.NUM_GENERATIONS)
    population = data.get('population', settings.POPULATION_SIZE)

    # Launch the Data Plane in a separate, non-blocking thread
    thread = threading.Thread(target=run_hunt_in_background, args=(generations, population))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "ok", "message": "Hunt started."}), 202

@app.route('/api/get-status')
def api_get_status():
    """
    Asynchronous polling endpoint for the UI.
    Simply reads and returns the central status.json file.
    """
    if not os.path.exists(STATUS_FILE):
        # AUDIT : Hardcoded keys.
        return jsonify({"hunt_status": "Idle", "last_event": "-", "last_sse": "-", "last_h_norm": "-", "final_result": {}})

    try:
        with open(STATUS_FILE, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Ensure templates directory exists if app is run from root
    if not os.path.exists("templates"):
        os.makedirs("templates")
        # Write index.html to templates/index.html
        # (This assumes index.html is in the root)
        try:
            with open("index.html", "r") as f_in:
                content = f_in.read()
            with open("templates/index.html", "w") as f_out:
                f_out.write(content)
        except FileNotFoundError:
            print("WARNING: index.html not found. UI will be broken.")

    app.template_folder = "templates"
    update_status() # Initialize status file
    start_watcher_service() # Start the WatcherThread
    app.run(host='0.0.0.0', port=8080)
