Phase 4 Report: Dynamic Control Hub Build PlanProject: IRER V11.0 "HPC-SDG"Status: Authorized Build PlanMandate: This report details the architecture and build plan for the "Dynamic Control Hub," the persistent meta-orchestration layer for the V11.0 suite.1. Executive Mandate & Strategic PivotThis build plan formally decommissions the "Celery/Dask" orchestration concept [cite: combined review docs] as "non-viable," high-overhead, and unnecessarily complex.The new V11.0 architecture is a "Web-Based Control Plane," which is lightweight, robust, and directly scalable from Colab to Cloud VMs. It consists of a Flask server (app.py) acting as the "Meta-Orchestrator" and a refactored core_engine.py (formerly adaptive_hunt_orchestrator.py) acting as a callable, threaded "Engine."2. System Architecture & Data FlowThe new architecture separates the "Control Plane" (the Hub) from the "Data Plane" (the JAX Core).The Hub (app.py) serves the index.html.A user clicks "Start Hunt" on the HTML.The HTML sends a fetch request to the Hub's /api/start-hunt endpoint.The Hub (in a new background thread) imports and calls core_engine.execute_hunt().The Core Engine (Layer 1) runs the JAX-HPC loop, saving provenance_<uuid>.json files.The Hub (in a separate "Watcher" thread) sees the new JSON files.The "Watcher" triggers Layer 2 scripts (TDA, BSSN-check) and updates a central status.json.The HTML (on a timer) continuously fetches /api/get-status to update the dashboard.This design is fully decoupled. The JAX Core (Layer 1) never talks to the UI, and the UI (Layer 2) never talks to the JAX Core.Data Flow Diagramgraph TD
    subgraph "User (Browser)"
        A[index.html] -- 1. Click --> B(Start Hunt Button);
        B -- 2. fetch('/api/start-hunt') --> C[app.py];
        A -- 7. fetch('/api/get-status') --> C;
        C -- 8. return status.json --> A;
    end

    subgraph "Control Plane (app.py)"
        C -- 3. (New Thread) --> D[core_engine.execute_hunt()];
        C -- 6. (Watcher Thread) --> E[provenance_reports/];
        E -- 6a. New File --> F[Run Layer 2 Scripts];
        F -- 6b. Update --> G[status.json];
    end

    subgraph "HPC Core (Layer 1)"
        D -- 4. Run JAX Loop --> H(worker_sncgl_sdg.py);
        H -- 5. Save --> E;
    end
3. Component Build Plan (Option 1)As specified, this plan provides simple stubs for simple functions and explicit, complete code for complex, failure-prone components.app.py (The Meta-Orchestrator)Implementation: Explicit & Complete. This is the most complex new component.Notes: This code provides the Flask server, the non-blocking /api/start-hunt endpoint, the "Watcher" thread, and the /api/get-status endpoint.%%writefile app.py
"""
app.py
CLASSIFICATION: Meta-Orchestrator (IRER V11.0 Control Plane)
GOAL: Runs a persistent Flask server to act as the "Dynamic Control Hub."
"""

import os
import time
import json
import logging
import threading
from flask import Flask, render_template, jsonify, request
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Import the refactored Core Engine ---
# This assumes adaptive_hunt_orchestrator.py has been renamed to core_engine.py
try:
    import core_engine
except ImportError:
    print("FATAL: core_engine.py not found. Run the refactor first.")
    # In a real app, you might want to handle this more gracefully
    # For Colab, we can assume the %%writefile will run.
    pass 

# --- Global State & Configuration ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Define watched directories from settings (or hardcode for simplicity)
PROVENANCE_DIR = "provenance_reports"
STATUS_FILE = "status.json"
HUNT_LOG_FILE = "aste_hunt.log" # Assumed log file for the engine

# --- 1. The "Watcher" (Layer 2 Trigger) ---
# This is a complex, critical component.
class ProvenanceWatcher(FileSystemEventHandler):
    """Watches for new provenance files and triggers Layer 2 analysis."""
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith(".json") and "provenance_" in os.path.basename(event.src_path):
            logging.info(f"Watcher: Detected new file: {event.src_path}")
            self.trigger_layer_2_analysis(event.src_path)

    def trigger_layer_2_analysis(self, provenance_file_path):
        """
        Stub for triggering all secondary analysis.
        In a real system, this would call subprocesses for:
        1. TDA / Quantule Classification
        2. BSSN-Checker (Legacy)
        3. Plotting
        """
        logging.info(f"Watcher: Triggering Layer 2 analysis for {provenance_file_path}...")
        
        # --- STUB FOR LAYER 2 SCRIPTS ---
        # e.g., subprocess.run(["python", "run_tda.py", "--file", provenance_file_path])
        # e.g., subprocess.run(["python", "run_bssn_check.py", "--file", provenance_file_path])
        
        # For this demo, we just update the status file
        try:
            with open(provenance_file_path, 'r') as f:
                data = json.load(f)
            
            job_uuid = data.get("job_uuid", "unknown")
            metrics = data.get("metrics", {})
            sse = metrics.get("log_prime_sse", 0)
            h_norm = metrics.get("sdg_h_norm_l2", 0)

            status_data = {
                "last_event": f"Processed {job_uuid[:8]}...",
                "last_sse": f"{sse:.6f}",
                "last_h_norm": f"{h_norm:.6f}"
            }
            
            # --- This is the key state-management step ---
            # It reads the old status, updates it, and writes back.
            self.update_status(status_data)

        except Exception as e:
            logging.error(f"Watcher: Failed to process {provenance_file_path}: {e}")

    def update_status(self, new_data):
        """Safely updates the central status.json file."""
        try:
            current_status = {}
            if os.path.exists(STATUS_FILE):
                with open(STATUS_FILE, 'r') as f:
                    current_status = json.load(f)
            
            current_status.update(new_data)
            
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
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# --- 2. The Core Engine Runner (Layer 1 Trigger) ---
# This is the second complex, critical component.
def run_hunt_in_background():
    """
    This function is the target for our background thread.
    It imports and runs the main hunt from the refactored core engine.
    """
    logging.info("Hunt Thread: Started.")
    try:
        # --- This is the key call to the refactored module ---
        core_engine.execute_hunt()
        logging.info("Hunt Thread: `execute_hunt()` completed.")
    except Exception as e:
        logging.error(f"Hunt Thread: CRITICAL FAILURE: {e}")

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
    logging.info("API: Received /api/start-hunt request.")
    
    # Check if a hunt is already running (simple state check)
    # A more robust system would use a global boolean or class state
    
    # --- The non-blocking thread ---
    # We launch the `run_hunt_in_background` function as a daemon thread.
    # This means the API request returns *immediately* (in 1ms),
    # while the hunt runs in the background for hours.
    hunt_thread = threading.Thread(target=run_hunt_in_background, daemon=True)
    hunt_thread.start()
    
    return jsonify({"status": "Hunt Started"}), 202

@app.route('/api/get-status')
def api_get_status():
    """
    API endpoint for the HTML dashboard to poll.
    It just reads the JSON file updated by the Watcher.
    """
    if not os.path.exists(STATUS_FILE):
        return jsonify({"status": "idle", "last_event": "No hunts running."})
    
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"status": "error", "last_event": str(e)}), 500

# --- Main Application Runner ---
if __name__ == "__main__":
    # Start the Watcher service in its own thread
    watcher_thread = threading.Thread(target=start_watcher_service, daemon=True)
    watcher_thread.start()
    
    # Start the Flask app
    # We use host='0.0.0.0' to make it accessible in Colab/Cloud VMs
    logging.info("Control Hub: Starting Flask server...")
    app.run(host='0.0.0.0', port=8080)

core_engine.py (The Refactored Orchestrator)Implementation: Simple Stub.Notes: This is a simple refactor. The task is to take the existing, working adaptive_hunt_orchestrator.py from the V11.0 build, rename it, and wrap its main() logic in a callable execute_hunt() function.%%writefile core_engine.py
"""
core_engine.py
CLASSIFICATION: Core Engine (IRER V11.0)
GOAL: Refactored orchestrator, now a callable module.
"""

import os
import json
import subprocess
import sys
import uuid
import time
import logging
import settings
import aste_hunter

# --- All the functions from the V11.0 orchestrator go here ---
# (setup_directories, run_simulation_job)

def setup_directories():
    """Ensures all required I/O directories exist."""
    logging.info("[CoreEngine] Ensuring I/O directories exist...")
    os.makedirs(settings.CONFIG_DIR, exist_ok=True)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.PROVENANCE_DIR, exist_ok=True)

def run_simulation_job(job_uuid: str, params_filepath: str) -> bool:
    """
    This is the *exact* same function from adaptive_hunt_orchestrator.py.
    It's the "Layer 1" JAX/HPC loop.
    """
    logging.info(f"--- [CoreEngine] STARTING JOB {job_uuid[:10]}... ---")
    
    worker_cmd = [
        sys.executable, settings.WORKER_SCRIPT,
        "--params", params_filepath,
        "--job_uuid", job_uuid
    ]
    try:
        subprocess.run(worker_cmd, capture_output=True, text=True, check=True, timeout=600)
    except Exception as e:
        logging.error(f"[CoreEngine] WORKER FAILED: {job_uuid[:10]}")
        return False
    
    validator_cmd = [
        sys.executable, settings.VALIDATOR_SCRIPT,
        "--job_uuid", job_uuid,
    ]
    try:
        subprocess.run(validator_cmd, capture_output=True, text=True, check=True, timeout=300)
    except Exception as e:
        logging.error(f"[CoreEngine] VALIDATOR FAILED: {job_uuid[:10]}")
        return False
        
    logging.info(f"--- [CoreEngine] JOB SUCCEEDED {job_uuid[:10]} ---")
    return True


# --- THIS IS THE KEY REFACTOR ---
# The old `main()` function is renamed `execute_hunt()`
def execute_hunt():
    """
    This is the refactored main() function.
    It's now called by app.py in a background thread.
    """
    
    # Setup logging to go to a file instead of stdout
    # This is critical so it doesn't spam the Flask server logs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(HUNT_LOG_FILE),
            logging.StreamHandler(sys.stdout) # Keep printing to console too
        ]
    )

    logging.info("--- [CoreEngine] V11.0 HUNT EXECUTION STARTED ---")
    
    setup_directories()
    hunter = aste_hunter.Hunter(ledger_file=settings.LEDGER_FILE)

    start_gen = hunter.get_current_generation()
    end_gen = start_gen + settings.NUM_GENERATIONS
    logging.info(f"[CoreEngine] Starting Hunt: {settings.NUM_GENERATIONS} generations...")

    for gen in range(start_gen, end_gen):
        logging.info(f"--- [CoreEngine] STARTING GENERATION {gen} ---")
        
        parameter_batch = hunter.get_next_generation(settings.POPULATION_SIZE)
        
        jobs_to_run = []
        jobs_to_register = []

        for phys_params in parameter_batch:
            job_uuid = str(uuid.uuid4())
            # ... (Full parameter setup logic from V11.0 orchestrator) ...
            # ... (This logic is identical to the V11.0 file) ...
            full_params = {
                settings.HASH_KEY: job_uuid,
                "global_seed": random.randint(0, 2**32 - 1),
                "simulation": {"N_grid": 32, "T_steps": 200},
                "sncgl_params": phys_params
            }
            params_filepath = os.path.join(settings.CONFIG_DIR, f"config_{job_uuid}.json")
            with open(params_filepath, 'w') as f:
                json.dump(full_params, f, indent=2)

            jobs_to_run.append({"job_uuid": job_uuid, "params_filepath": params_filepath})
            jobs_to_register.append({settings.HASH_KEY: job_uuid, "generation": gen, **phys_params})

        hunter.register_new_jobs(jobs_to_register)

        job_hashes_completed = []
        for job in jobs_to_run:
            if run_simulation_job(job["job_uuid"], job["params_filepath"]):
                job_hashes_completed.append(job["job_uuid"])

        logging.info(f"[CoreEngine] GENERATION {gen} COMPLETE. Processing results...")
        hunter.process_generation_results(settings.PROVENANCE_DIR, job_hashes_completed)
        
        best_run = hunter.get_best_run()
        if best_run:
            logging.info(f"[CoreEngine] Best Run So Far: {best_run[settings.HASH_KEY][:8]}...")

    logging.info("--- [CoreEngine] ALL GENERATIONS COMPLETE ---")

# The old `if __name__ == "__main__":` block is removed.
# This file is now a module, not a script.
templates/index.html (The Interactive Hub)Implementation: Simple Stub.Notes: This is a minimal HTML file to demonstrate the "golden run" UI. It shows the two key fetch calls.%%writefile templates/index.html
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IRER V11.0 | Dynamic Control Hub</title>
    <script src="[https://cdn.tailwindcss.com](https://cdn.tailwindcss.com)"></script>
    <script>
        tailwind.config = {
            darkMode: 'class'
        }
    </script>
</head>
<body class="bg-gray-900 text-gray-200 font-sans p-8">
    <div class="max-w-4xl mx-auto">
        <h1 class="text-3xl font-bold text-cyan-400">IRER V11.0 Control Hub</h1>
        <p class="text-gray-400 mb-6">"HPC-SDG" Core | Dynamic Analysis Layer</p>

        <!-- Layer 1 Control -->
        <div class="bg-gray-800 p-6 rounded-lg shadow-lg mb-6">
            <h2 class="text-xl font-semibold mb-4">Layer 1: HPC Core Control</h2>
            <button id="start-hunt-btn" class="bg-cyan-600 hover:bg-cyan-500 text-white font-bold py-2 px-4 rounded-lg transition-colors">
                Start New Hunt
            </button>
            <p id="hunt-status" class="text-sm text-gray-400 mt-2">Status: Idle</p>
        </div>

        <!-- Layer 2 Visualization -->
        <div class="bg-gray-800 p-6 rounded-lg shadow-lg">
            <h2 class="text-xl font-semibold mb-4">Layer 2: Live Analysis Dashboard</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-gray-700 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-gray-400">LAST EVENT</h3>
                    <p id="status-event" class="text-2xl font-bold text-white">-</p>
                </div>
                <div class="bg-gray-700 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-gray-400">LAST SSE</h3>
                    <p id="status-sse" class="text-2xl font-bold text-emerald-400">-</p>
                </div>
                <div class="bg-gray-700 p-4 rounded-lg">
                    <h3 class="text-sm font-medium text-gray-400">LAST H-NORM (SDG)</h3>
                    <p id="status-h-norm" class="text-2xl font-bold text-amber-400">-</p>
                </div>
            </div>
        </div>

    </div>

    <script>
        const startBtn = document.getElementById('start-hunt-btn');
        const huntStatus = document.getElementById('hunt-status');
        
        const statusEvent = document.getElementById('status-event');
        const statusSse = document.getElementById('status-sse');
        const statusHNorm = document.getElementById('status-h-norm');

        // --- Layer 1 Control Logic ---
        startBtn.addEventListener('click', async () => {
            huntStatus.textContent = 'Sending signal to start hunt...';
            startBtn.disabled = true;
            startBtn.textContent = 'Hunt Running...';

            try {
                const response = await fetch('/api/start-hunt', { method: 'POST' });
                if (response.status === 202) {
                    huntStatus.textContent = 'Hunt started successfully.';
                } else {
                    huntStatus.textContent = 'Error starting hunt.';
                    startBtn.disabled = false;
                    startBtn.textContent = 'Start New Hunt';
                }
            } catch (error) {
                huntStatus.textContent = 'Error: Could not connect to server.';
                startBtn.disabled = false;
                startBtn.textContent = 'Start New Hunt';
            }
        });

        // --- Layer 2 Visualization Logic ---
        async function updateStatus() {
            try {
                const response = await fetch('/api/get-status');
                const data = await response.json();
                
                statusEvent.textContent = data.last_event || '-';
                statusSse.textContent = data.last_sse || '-';
                statusHNorm.textContent = data.last_h_norm || '-';

            } catch (error) {
                statusEvent.textContent = 'Offline';
            }
        }

        // Poll the status every 3 seconds
        setInterval(updateStatus, 3000);
        // Initial call
        updateStatus();
    </script>
</body>
</html>
requirements.txt (Dependencies)Implementation: Simple Stub.%%writefile requirements.txt
# Core server
flask
# File system watcher
watchdog
# JAX (must be installed separately based on CPU/GPU/TPU)
# jax
# jaxlib
# HDF5 support
h5py
# (numpy, etc. are dependencies of jax/h5py)
4. Scalability Path (Colab to Cloud VM)This architecture is now perfectly staged for scaling:Colab (This Build): Create a templates directory (os.makedirs('templates')), write these files, install the requirements.txt, and run !python app.py. Everything (Flask, Watcher, and Core Engine) will run in one Colab instance.Cloud VM (The Scale-Up):Control VM: A cheap e2-micro VM runs app.py.HPC VMs: Your powerful JAX VMs are set up.The Change: You modify one function: core_engine.py's run_simulation_job(). Instead of subprocess.run([sys.executable...]), it now makes an ssh call: subprocess.run(["ssh", "user@hpc-vm-1", "python /path/to/worker_sncgl_sdg.py ..."]).This plan is complete, robust, and directly implements the architecture you designed.