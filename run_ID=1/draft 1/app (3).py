import os
import time
import json
import logging
import threading
import subprocess
from flask import Flask, render_template, jsonify, request, send_from_directory
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    import core_engine
    import settings
except ImportError:
    print("FATAL: core_engine.py or settings.py not found. Run the refactor first.")

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[
        logging.FileHandler("control_hub.log"),
        logging.StreamHandler()
    ]
)

PROVENANCE_DIR = settings.PROVENANCE_DIR
STATUS_FILE = "hub_status.json"
HUNT_LOG_FILE = "core_engine_hunt.log"

HUNT_RUNNING_LOCK = threading.Lock()
g_hunt_in_progress = False

class ProvenanceWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith(".json") and "provenance_" in os.path.basename(event.src_path):
            logging.info(f"Watcher: Detected new file: {event.src_path}")
            self.trigger_layer_2_analysis(event.src_path)

    def trigger_layer_2_analysis(self, provenance_file_path):
        logging.info(f"Watcher: Triggering Layer 2 analysis for {provenance_file_path}...")

        try:
            with open(provenance_file_path, 'r') as f:
                data = json.load(f)

            job_uuid = data.get(settings.HASH_KEY, "unknown_uuid")
            metrics = data.get("metrics", {})
            sse = metrics.get(settings.SSE_METRIC_KEY, 0)
            h_norm = metrics.get(settings.STABILITY_METRIC_KEY, 0)

            status_data = {
                "last_event": f"Analyzed {job_uuid[:8]}...",
                "last_sse": f"{sse:.6f}",
                "last_h_norm": f"{h_norm:.6f}"
            }

            self.update_status(status_data, append_file=provenance_file_path)

        except Exception as e:
            logging.error(f"Watcher: Failed to parse {provenance_file_path}: {e}")

    def update_status(self, new_data, append_file=None):
        try:
            with HUNT_RUNNING_LOCK:
                current_status = {"hunt_status": "Running", "found_files": [], "final_result": {}}
                if os.path.exists(STATUS_FILE):
                    with open(STATUS_FILE, 'r') as f:
                         current_status = json.load(f)

                current_status.update(new_data)
                if append_file and append_file not in current_status["found_files"]:
                    current_status["found_files"].append(append_file)

                with open(STATUS_FILE, 'w') as f:
                    json.dump(current_status, f, indent=2)
        except Exception as e:
            logging.error(f"Watcher: Failed to update status file: {e}")

def start_watcher_service():
    if not os.path.exists(PROVENANCE_DIR):
        os.makedirs(PROVENANCE_DIR)

    event_handler = ProvenanceWatcher()
    observer = Observer()
    observer.schedule(event_handler, PROVENANCE_DIR, recursive=False)
    observer.start()
    logging.info(f"Watcher Service: Started monitoring {PROVENANCE_DIR}")
    observer.join()

def run_hunt_in_background(num_generations, population_size):
    global g_hunt_in_progress

    if not HUNT_RUNNING_LOCK.acquire(blocking=False):
        logging.warning("Hunt Thread: Hunt start requested, but lock is held. Already running.")
        return

    g_hunt_in_progress = True
    logging.info(f"Hunt Thread: Lock acquired. Starting hunt (Gens: {num_generations}, Pop: {population_size}).")

    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump({"hunt_status": "Running", "found_files": [], "final_result": {}}, f, indent=2)

        final_run = core_engine.execute_hunt(num_generations, population_size)

        logging.info("Hunt Thread: `execute_hunt()` completed.")

        with open(STATUS_FILE, 'w') as f:
            json.dump({"hunt_status": "Completed", "found_files": [], "final_result": final_run}, f, indent=2)

    except Exception as e:
         logging.error(f"Hunt Thread: CRITICAL FAILURE: {e}")
         with open(STATUS_FILE, 'w') as f:
            json.dump({"hunt_status": f"Error: {e}", "found_files": [], "final_result": {}}, f, indent=2)
    finally:
        g_hunt_in_progress = False
        HUNT_RUNNING_LOCK.release()
        logging.info("Hunt Thread: Lock released. Hunt finished.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start-hunt', methods=['POST'])
def api_start_hunt():
    global g_hunt_in_progress
    logging.info("API: Received /api/start-hunt request.")

    if g_hunt_in_progress:
        logging.warning("API: Hunt start rejected, one is already in progress.")
        return jsonify({"message": "A hunt is already in progress."}), 409

    data = request.json or {}
    num_generations = data.get('num_generations') or settings.NUM_GENERATIONS
    population_size = data.get('population_size') or settings.POPULATION_SIZE

    hunt_thread = threading.Thread(
        target=run_hunt_in_background,
        args=(num_generations, population_size),
        daemon=True,
        name="CoreEngineThread"
    )
    hunt_thread.start()

    return jsonify({"status": "Hunt Started"}), 202

@app.route('/api/get-status')
def api_get_status():
    if not os.path.exists(STATUS_FILE):
        return jsonify({"hunt_status": "Idle", "found_files": [], "final_result": {}})

    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"hunt_status": f"Error reading status: {e}", "found_files": [], "final_result": {}}), 500

if __name__ == "__main__":
    os.makedirs(PROVENANCE_DIR, exist_ok=True)
    os.makedirs(settings.CONFIG_DIR, exist_ok=True)
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    watcher_thread = threading.Thread(target=start_watcher_service, daemon=True, name="WatcherThread")
    watcher_thread.start()

    logging.info("Control Hub: Starting Flask server on http://0.0.0.0:8089")
    app.run(host='0.0.0.0', port=8089)