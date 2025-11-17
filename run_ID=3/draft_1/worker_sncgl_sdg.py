import argparse
import time
import os
import json
import logging
import random
import sys
import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description="S-NCGL+SDG Worker Stub")
    parser.add_argument("--params", required=True, help="Path to the config_{job_uuid}.json file")
    parser.add_argument("--job_uuid", required=True, help="The unified job_uuid")
    args = parser.parse_args()

    log.info(f"[WorkerStub {args.job_uuid[:8]}] Starting...")

    try:
        with open(args.params, 'r') as f:
            params = json.load(f)
        log.info(f"[WorkerStub {args.job_uuid[:8]}] Loaded params (Seed: {params.get('global_seed')})")
    except Exception as e:
        log.error(f"[WorkerStub {args.job_uuid[:8]}] Failed to load params file: {e}")
        sys.exit(1)

    sleep_time = random.uniform(1, 3)
    time.sleep(sleep_time)

    log.info(f"[WorkerStub {args.job_uuid[:8]}] Work complete in {sleep_time:.2f}s.")

if __name__ == "__main__":
    main()
