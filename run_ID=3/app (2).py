"""
app.py
CLASSIFICATION: Meta-Orchestrator (IRER V11.0 Control Plane)
GOAL: Runs a persistent Flask server to act as the "Dynamic Control Hub."
      This build is based on the V11.0 "Hotfix" and "Audit-Fixed" architecture.
"""

import os
import time
import json
import logging
import threading
import subprocess # We need this for the watcher's Layer 2 calls
from flask import Flask, render_template, jsonify, request, send_from_directory
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Import the refactored Core Engine & Settings ---
try:
    import core_engine
    import settings
except ImportError:
    print("FATAL: core_engine.py or settings.py not found.")
    # This will be populated by the other files in this assembly.
    pass

# --- Global State & Configuration ---
app = Flask(__name__, template_folder="templates") # Ensure Flask looks in /templates

# --- Centralized Logging ---
# We will log to a file, as 'print' statements are lost by daemon threads.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[
        logging.FileHandler("control_hub.log"),
        logging.StreamHandler() # Also print to console
    ]
)

# --- Configuration (from settings.py) ---
PROVENANCE_DIR = settings.PROVENANCE_DIR
STATUS_FILE = "hub_status.json" # This hub's master status file

# --- Global State ---
# This simple lock prevents two hunts from being started.
HUNT_RUNNING_LOCK = threading.Lock()
# This global variable will be set to True when a hunt is active.
g_hunt_in_progress = False


# --- 1. The "Watcher" (Layer 2 Trigger) ---
class ProvenanceWatcher(FileSystemEventHandler):
    """Watches for new provenance files and triggers Layer 2 analysis."""
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        # Watch for the specific file that signals a job is done
        if event.src_path.endswith(".json") and "provenance_" in os.path.basename(event.src_path):
            logging.info(f"Watcher: Detected new file: {event.src_path}")
            self.trigger_layer_2_analysis(event.src_path)

    def trigger_layer_2_analysis(self, provenance_file_path):
        """
        Triggers all secondary analysis (TDA, BSSN-Check, etc.)
        This function runs in the Watcher's thread.
        """
        logging.info(f"Watcher: Triggering Layer 2 analysis for {provenance_file_path}...")
        
        # --- Call Layer 2 Scripts ---
        try:
            logging.info(f"Watcher: Calling run_tda_analysis.py for {provenance_file_path}")
            # Note: Using sys.executable ensures we use the same python interpreter
            subprocess.run([sys.executable, "run_tda_analysis.py", "--file", provenance_file_path], check=True, capture_output=True, text=True)
            logging.info(f"Watcher: Calling run_bssn_check.py for {provenance_file_path}")
            subprocess.run([sys.executable, "run_bssn_check.py", "--file", provenance_file_path], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Watcher: Layer 2 script failed for {provenance_file_path}: {e}. STDERR: {e.stderr}")
        except Exception as e:
            logging.error(f"Watcher: Layer 2 script failed for {provenance_file_path}: {e}")
        
        # Update the master status file with the latest metrics
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
        """Safely updates the central hub_status.json file."""
        try:
            # Use a lock to prevent race conditions on the status file
            with HUNT_RUNNING_LOCK:
                current_status = {"hunt_status": "Running", "found_files": [], "final_result": {}}
                if os.path.exists(STATUS_FILE):
                    try:
                        with open(STATUS_FILE, 'r') as f:
                             current_status = json.load(f)
                    except json.JSONDecodeError:
                        logging.warning("Watcher: hub_status.json was corrupted. Resetting.")
                
                current_status.update(new_data)
                if append_file and append_file not in current_status["found_files"]:
                    current_status["found_files"].append(append_file)
                
                with open(STATUS_FILE, 'w') as f:
                    json.dump(current_status, f, indent=2)
        except Exception as e:
            logging.error(f"Watcher: Failed to update status file: {e}")

def start_watcher_service():
    """Initializes and starts the watchdog observer in a new thread."""
    if not os.path.exists(PROVENANCE_DIR):
        os.makedirs(PROVENANCE_DIR)
        
    event_handler = ProvenanceWatcher()
    observer = Observer()
    observer.schedule(event_handler, PROVENANCE_DIR, recursive=False)
    observer.start()
    logging.info(f"Watcher Service: Started monitoring {PROVENANCE_DIR}")
    
    try:
        while True:
            time.sleep(3600) # Sleep indefinitely while the main thread runs
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# --- 2. The Core Engine Runner (Layer 1 Trigger) ---
def run_hunt_in_background(num_generations, population_size):
    """
    This function is the target for our background thread.
    It imports and runs the main hunt from the refactored core engine.
    """
    global g_hunt_in_progress
    
    # --- This is the key state-management step ---
    # Try to acquire the lock *without blocking*.
    if not HUNT_RUNNING_LOCK.acquire(blocking=False):
        logging.warning("Hunt Thread: Hunt start requested, but lock is held. Already running.")
        return # Another hunt is already in progress
    
    g_hunt_in_progress = True
    logging.info(f"Hunt Thread: Lock acquired. Starting hunt (Gens: {num_generations}, Pop: {population_size}).")
    
    try:
        # Update status to "Running"
        with open(STATUS_FILE, 'w') as f:
            json.dump({"hunt_status": "Running", "found_files": [], "final_result": {}}, f, indent=2)

        # --- This is the key call to the refactored module ---
        # We pass the parameters from the UI to the core engine
        final_run = core_engine.execute_hunt(num_generations, population_size)
        
        logging.info("Hunt Thread: `execute_hunt()` completed.")
        
        # Update status to "Completed"
        with open(STATUS_FILE, 'w') as f:
            json.dump({"hunt_status": "Completed", "found_files": [], "final_result": final_run}, f, indent=2)

    except Exception as e:
         logging.error(f"Hunt Thread: CRITICAL FAILURE: {e}")
         with open(STATUS_FILE, 'w') as f:
            json.dump({"hunt_status": f"Error: {e}", "found_files": [], "final_result": {}}, f, indent=2)
    finally:
        # --- This is the key state-management step ---
        g_hunt_in_progress = False
        HUNT_RUNNING_LOCK.release()
        logging.info("Hunt Thread: Lock released. Hunt finished.")

# --- 3. Flask API Endpoints (The Control Hub) ---
@app.route('/')
def index():
    """Serves the main interactive HTML hub."""
    return render_template('index.html')

@app.route('/api/start-hunt', methods=['POST'])
def api_start_hunt():
    """
    API endpoint to start the hunt in a non-blocking background thread.
    This is the explicit fix for the "blocking server" failure.
    """
    global g_hunt_in_progress
    logging.info("API: Received /api/start-hunt request.")
    
    if g_hunt_in_progress:
        logging.warning("API: Hunt start rejected, one is already in progress.")
        return jsonify({"message": "A hunt is already in progress."}), 409 # 409 Conflict

    # Get params from UI, with fallbacks to settings.py
    data = request.json or {}
    num_generations = data.get('num_generations') or settings.NUM_GENERATIONS
    population_size = data.get('population_size') or settings.POPULATION_SIZE
    
    # --- The non-blocking thread ---
    hunt_thread = threading.Thread(
        target=run_hunt_in_background,
        args=(num_generations, population_size),
        daemon=True,
        name="CoreEngineThread"
    )
    hunt_thread.start()
    
    return jsonify({"status": "Hunt Started"}), 202 # 202 Accepted

@app.route('/api/get-status')
def api_get_status():
    """
    API endpoint for the HTML dashboard to poll.
    It just reads the JSON file updated by the Watcher.
    """
    if not os.path.exists(STATUS_FILE):
        return jsonify({"hunt_status": "Idle", "found_files": [], "final_result": {}})
    
    try:
        # This guarantees we send the most up-to-date info
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"hunt_status": f"Error reading status: {e}", "found_files": [], "final_result": {}}), 500

# --- Main Application Runner ---
if __name__ == "__main__":
    # Create required directories on startup
    os.makedirs(PROVENANCE_DIR, exist_ok=True)
    os.makedirs(settings.CONFIG_DIR, exist_ok=True)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    
    # Start the Watcher service in its own thread
    watcher_thread = threading.Thread(target=start_watcher_service, daemon=True, name="WatcherThread")
    watcher_thread.start()
    
    # Start the Flask app
    # We use host='0.0.0.0' to make it accessible in Colab/Cloud VMs
    logging.info("Control Hub: Starting Flask server on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080)
