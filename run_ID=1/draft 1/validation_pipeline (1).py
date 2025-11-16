import argparse
import time
import os
import json
import random
import logging
import settings
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description="Validator Stub")
    parser.add_argument("--job_uuid", required=True, help="The unified job_uuid")
    args = parser.parse_args()

    log.info(f"[ValidatorStub {args.job_uuid[:8]}] Starting...")

    time.sleep(random.uniform(0.5, 1))

    fake_sse = random.uniform(0.001, 0.5)
    fake_h_norm = random.uniform(0.001, 0.1)

    metrics = {
        settings.SSE_METRIC_KEY: fake_sse,
        settings.STABILITY_METRIC_KEY: fake_h_norm,
        "other_metric": random.random()
    }

    payload = {
        settings.HASH_KEY: args.job_uuid,
        "metrics": metrics,
        "timestamp": time.time()
    }

    output_filename = f"provenance_{args.job_uuid}.json"
    output_path = os.path.join(settings.PROVENANCE_DIR, output_filename)

    try:
        os.makedirs(settings.PROVENANCE_DIR, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2)
        log.info(f"[ValidatorStub {args.job_uuid[:8]}] Provenance file saved: {output_path}")
    except Exception as e:
        log.error(f"[ValidatorStub {args.job_uuid[:8]}] FAILED to write provenance: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
