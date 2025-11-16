import os
import csv
import json
import random
import logging
import settings

class Hunter:
    def __init__(self, ledger_file: str):
        self.ledger_file = ledger_file
        self.fieldnames = [
            settings.HASH_KEY,
            "generation",
            "fitness",
            settings.SSE_METRIC_KEY,
            settings.STABILITY_METRIC_KEY,
            "param_D",
            "param_eta"
        ]
        self.population = self._load_ledger()
        logging.info(f"[Hunter] Initialized. Loaded {len(self.population)} runs from {self.ledger_file}")

    def _load_ledger(self) -> list:
        if not os.path.exists(self.ledger_file):
            os.makedirs(os.path.dirname(self.ledger_file), exist_ok=True)
            self._save_ledger([])
            return []

        try:
            with open(self.ledger_file, 'r') as f:
                reader = csv.DictReader(f)
                pop = []
                for row in reader:
                    for key in [settings.SSE_METRIC_KEY, settings.STABILITY_METRIC_KEY, "fitness", "param_D", "param_eta"]:
                        if key in row and row[key]:
                            row[key] = float(row[key])
                    if 'generation' in row and row['generation']:
                        row['generation'] = int(row['generation'])
                    pop.append(row)
                return pop
        except Exception as e:
            logging.error(f"[Hunter Error] Failed to load ledger: {e}")
            return []

    def _save_ledger(self, rows: list = None):
        try:
            with open(self.ledger_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows if rows is not None else self.population)
        except Exception as e:
            logging.error(f"[Hunter Error] Failed to save ledger: {e}")

    def get_current_generation(self) -> int:
        if not self.population:
            return 0
        return max(int(run.get('generation', 0)) for run in self.population) + 1

    def get_next_generation(self, population_size: int) -> list:
        logging.info(f"[Hunter] Breeding Generation {self.get_current_generation()}...")
        new_generation_params = []
        for _ in range(population_size):
            params = {
                "param_D": random.uniform(0.1, 1.0),
                "param_eta": random.uniform(0.01, 0.5)
            }
            new_generation_params.append(params)
        return new_generation_params

    def register_new_jobs(self, job_list: list):
        self.population.extend(job_list)
        logging.info(f"[Hunter] Registered {len(job_list)} new jobs in ledger.")
        self._save_ledger()

    def process_generation_results(self, provenance_dir: str, job_hashes: list):
        logging.info(f"[Hunter] Processing {len(job_hashes)} new results from {provenance_dir}...")
        processed_count = 0
        for job_hash in job_hashes:
            report_path = os.path.join(provenance_dir, f"provenance_{job_hash}.json")

            try:
                with open(report_path, 'r') as f:
                    data = json.load(f)

                metrics = data.get("metrics", {})
                sse = metrics.get(settings.SSE_METRIC_KEY, 999.0)
                h_norm = metrics.get(settings.STABILITY_METRIC_KEY, 999.0)

                fitness = 1.0 / (sse + 1e-9)

                found = False
                for run in self.population:
                    if run[settings.HASH_KEY] == job_hash:
                        run[settings.SSE_METRIC_KEY] = sse
                        run[settings.STABILITY_METRIC_KEY] = h_norm
                        run["fitness"] = fitness
                        found = True
                        processed_count += 1
                        break
                if not found:
                    logging.warning(f"[Hunter] Hash {job_hash} found in JSON but not in population ledger.")

            except FileNotFoundError:
                logging.warning(f"[Hunter] Provenance file not found: {report_path}")
            except Exception as e:
                logging.error(f"[Hunter] Failed to parse {report_path}: {e}")

        logging.info(f"[Hunter] Successfully processed and updated {processed_count} runs.")
        self._save_ledger()

    def get_best_run(self) -> dict:
        if not self.population:
            return {}
        valid_runs = [r for r in self.population if r.get("fitness") is not None]
        if not valid_runs:
            return {}
        return max(valid_runs, key=lambda x: x["fitness"])
