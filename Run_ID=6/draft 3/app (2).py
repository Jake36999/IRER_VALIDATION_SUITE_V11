"""
app.py
V11.0: The Flask Meta-Orchestrator and Dynamic Control Hub.
This server provides API endpoints to start hunts and monitor status,
and uses a watchdog service to monitor for new artifacts.
"""
import os
import json
import threading
import time
import logging
import settings
import core_engine

from flask import Flask, jsonify, render_template, request, send_from_directory, send_file
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Centralized Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[
        logging.FileHandler("control_hub.log"),
        logging.StreamHandler()
    ]
)

# --- Configuration & Global State ---
PROVENANCE_DIR = settings.PROVENANCE_DIR
DATA_DIR = settings.DATA_DIR
STATUS_FILE = "hub_status.json"
HUNT_RUNNING_LOCK = threading.Lock()
g_hunt_in_progress = False

app = Flask(__name__)

class ProvenanceWatcher(FileSystemEventHandler):
    """Watches for new provenance.json files and updates the status."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".json") and "provenance_" in event.src_path:
            logging.info(f"Watcher: Detected new file: {event.src_path}")
            try:
                with open(event.src_path, 'r') as f:
                    data = json.load(f)

                job_uuid = data.get(settings.HASH_KEY, "unknown_uuid")
                metrics = data.get("metrics", {})
                sse = metrics.get(settings.SSE_METRIC_KEY, 0)
                h_norm = metrics.get(settings.STABILITY_METRIC_KEY, 0)

                status_data = {
                    "last_event": f"Analyzed {job_uuid[:8]}...",
                    "last_sse": f"{sse:.6f}",
                    "last_h_norm": f"{h_norm:.6f}",
                    "last_job_id": job_uuid
                }
                self.update_status(status_data)
            except Exception as e:
                logging.error(f"Watcher: Failed to process {event.src_path}: {e}")

    def update_status(self, new_data: dict):
        """Safely updates the central hub_status.json file."""
        with HUNT_RUNNING_LOCK:
            try:
                current_status = {}
                if os.path.exists(STATUS_FILE):
                    with open(STATUS_FILE, 'r') as f:
                        current_status = json.load(f)
                current_status.update(new_data)
                with open(STATUS_FILE, 'w') as f:
                    json.dump(current_status, f, indent=2)
            except Exception as e:
                logging.error(f"Watcher: Failed to update {STATUS_FILE}: {e}")

def run_hunt_in_background():
    """Target function for the background thread to run the hunt."""
    global g_hunt_in_progress
    with HUNT_RUNNING_LOCK:
        g_hunt_in_progress = True
        with open(STATUS_FILE, 'w') as f:
            json.dump({"status": "Running", "last_event": "Hunt initiated..."}, f, indent=2)

    logging.info("Hunt Thread: Started.")
    try:
        core_engine.execute_hunt()
        status_message = "Hunt completed successfully."
    except Exception as e:
        logging.error(f"Hunt Thread: CRITICAL FAILURE: {e}", exc_info=True)
        status_message = f"Hunt FAILED: {e}"

    with HUNT_RUNNING_LOCK:
        g_hunt_in_progress = False
        final_status = {}
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                final_status = json.load(f)
        final_status.update({"status": "Idle", "last_event": status_message})
        with open(STATUS_FILE, 'w') as f:
            json.dump(final_status, f, indent=2)
    logging.info(f"Hunt Thread: Finished. ({status_message})")

# --- Flask API Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start-hunt', methods=['POST'])
def start_hunt():
    global g_hunt_in_progress
    with HUNT_RUNNING_LOCK:
        if g_hunt_in_progress:
            return jsonify({"status": "error", "message": "A hunt is already in progress."}), 409

        thread = threading.Thread(target=run_hunt_in_background, name="HuntThread", daemon=True)
        thread.start()

    return jsonify({"status": "ok", "message": "Hunt started in the background."})

@app.route('/api/get-status')
def get_status():
    if not os.path.exists(STATUS_FILE):
        return jsonify({"status": "Idle", "last_event": "Server is ready."})
    try:
        with open(STATUS_FILE, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"status": "Error", "last_event": f"Could not read status file: {e}"}), 500

@app.route('/api/get-run-log/<job_id>')
def get_run_log(job_id):
    log_file_path = os.path.join(settings.DATA_DIR, f"run_log_{job_id}.txt") # Assuming worker writes a log per job
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        return jsonify({"job_id": job_id, "log": log_content})
    return jsonify({"error": "Log not found"}), 404

@app.route('/api/download-provenance/<job_id>')
def download_provenance(job_id):
    provenance_file_name = f"provenance_{job_id}.json"
    provenance_path = os.path.join(PROVENANCE_DIR, provenance_file_name)
    if os.path.exists(provenance_path):
        return send_from_directory(PROVENANCE_DIR, provenance_file_name, as_attachment=True)
    return jsonify({"error": "Provenance file not found"}), 404

@app.route('/api/get-full-log')
def get_full_log():
    log_file_path = "control_hub.log"
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        return jsonify({"log": log_content})
    return jsonify({"error": "Full log not found"}), 404

@app.route('/api/download-all-rho-data')
def download_all_rho_data():
    """Compresses and serves all HDF5 files from DATA_DIR."""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.h5'):
                file_path = os.path.join(DATA_DIR, filename)
                zf.write(file_path, arcname=filename)
    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='all_rho_data.zip')

def start_watcher_service():
    """Initializes and starts the watchdog observer in a new thread."""
    os.makedirs(PROVENANCE_DIR, exist_ok=True)
    event_handler = ProvenanceWatcher()
    observer = Observer()
    observer.schedule(event_handler, PROVENANCE_DIR, recursive=False)
    observer.daemon = True
    observer.start()
    logging.info(f"Watcher Service: Started monitoring {PROVENANCE_DIR}")

if __name__ == '__main__':
    start_watcher_service()
    # Use a port other than 5000 to avoid common conflicts
    app.run(host='0.0.0.0', port=5001, debug=False)
