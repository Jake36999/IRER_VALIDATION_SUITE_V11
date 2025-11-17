"""
app.py
CLASSIFICATION: Control Plane (V11.0 Control Hub)
GOAL: Provides a web-based meta-orchestration layer for the IRER suite.
"""
import os
import json
import logging
import threading
from flask import Flask, render_template, jsonify, request
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


import core_engine


# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
PROVENANCE_DIR = "provenance_reports"
STATUS_FILE = "status.json"
HUNT_RUNNING_LOCK = threading.Lock()
g_hunt_in_progress = False


app = Flask(__name__)


# --- State Management ---
def update_status(new_data: dict = {}, append_file: str = None):
    with HUNT_RUNNING_LOCK:
        status = {"hunt_status": "Idle", "found_files": [], "final_result": {}}
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, 'r') as f:
                    status = json.load(f)
            except json.JSONDecodeError:
                pass # Overwrite corrupted file
        
        status.update(new_data)
        if append_file and append_file not in status["found_files"]:
            status["found_files"].append(append_file)
        
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)


# --- Watchdog Service (WatcherThread) ---
class ProvenanceWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            logging.info(f"Watcher: Detected new provenance file: {event.src_path}")
            basename = os.path.basename(event.src_path)
            update_status(append_file=basename)


def start_watcher_service():
    if not os.path.exists(PROVENANCE_DIR):
        os.makedirs(PROVENANCE_DIR)
    
    event_handler = ProvenanceWatcher()
    observer = Observer()
    observer.schedule(event_handler, PROVENANCE_DIR, recursive=False)
    observer.daemon = True
    observer.start()
    logging.info(f"Watcher Service: Started monitoring {PROVENANCE_DIR}")


# --- Core Engine Runner (HuntThread) ---
def run_hunt_in_background(num_generations, population_size):
    global g_hunt_in_progress
    if not HUNT_RUNNING_LOCK.acquire(blocking=False):
        logging.warning("Hunt Thread: Hunt start requested, but already running.")
        return
    
    g_hunt_in_progress = True
    logging.info(f"Hunt Thread: Starting hunt (Gens: {num_generations}, Pop: {population_size}).")
    try:
        update_status(new_data={"hunt_status": "Running", "found_files": [], "final_result": {}})
        final_run = core_engine.execute_hunt(num_generations, population_size)
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
    return render_template('index.html')


@app.route('/api/start-hunt', methods=['POST'])
def api_start_hunt():
    if g_hunt_in_progress:
        return jsonify({"status": "error", "message": "A hunt is already in progress."}), 409
        
    data = request.json or {}
    generations = data.get('generations', 10)
    population = data.get('population', 10)
    
    # Clean up old artifacts before starting
    for d in [PROVENANCE_DIR, "simulation_data", "input_configs"]:
        if os.path.exists(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
    if os.path.exists("simulation_ledger.csv"):
        os.remove("simulation_ledger.csv")




    thread = threading.Thread(target=run_hunt_in_background, args=(generations, population))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "ok", "message": "Hunt started."})


@app.route('/api/get-status')
def api_get_status():
    if not os.path.exists(STATUS_FILE):
        return jsonify({"hunt_status": "Idle", "found_files": [], "final_result": {}})
    with open(STATUS_FILE, 'r') as f:
        return jsonify(json.load(f))


if __name__ == '__main__':
    update_status() # Initialize status file
    start_watcher_service()
    app.run(host='0.0.0.0', port=8080)
