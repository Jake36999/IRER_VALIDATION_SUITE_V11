"""
aste_hunter.py
CLASSIFICATION: Adaptive Learning Engine (ASTE V1.0)
GOAL: Acts as the "Brain" of the ASTE.
      Manages a population of parameters and "breeds"
      new generations.
"""
import os
import csv
import json
import random
import logging
import settings

# Define parameter bounds
PARAM_D_MIN, PARAM_D_MAX = 0.1, 1.0
PARAM_ETA_MIN, PARAM_ETA_MAX = 0.01, 0.5

class Hunter:
    """
    Implements the core evolutionary "hunt" logic.
    Manages a population of parameters stored in a ledger.
    """

    def __init__(self, ledger_file: str):
        self.ledger_file = ledger_file
        self.fieldnames = [
            settings.HASH_KEY,
            "generation",
            "fitness",
            settings.SSE_METRIC_KEY,
            settings.STABILITY_METRIC_KEY,
            "param_D", # Example physical parameter
            "param_eta"  # Example physical parameter
        ]
        self.population = self._load_ledger()
        logging.info(f"[Hunter] Initialized. Loaded {len(self.population)} runs from {self.ledger_file}")

    def _load_ledger(self) -> list:
        """Loads the historical population from the CSV ledger."""
        if not os.path.exists(self.ledger_file):
            os.makedirs(os.path.dirname(self.ledger_file), exist_ok=True)
            self._save_ledger([]) # Create header
            return []

        try:
            with open(self.ledger_file, 'r') as f:
                reader = csv.DictReader(f)
                pop = []
                for row in reader:
                    # Convert numeric strings back to numbers
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
        """Saves the entire population back to the CSV ledger."""
        try:
            with open(self.ledger_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows if rows is not None else self.population)
        except Exception as e:
            logging.error(f"[Hunter Error] Failed to save ledger: {e}")

    def get_current_generation(self) -> int:
        """Determines the next generation number to breed."""
        if not self.population:
            return 0
        return max(int(run.get('generation', 0)) for run in self.population) + 1

    def _select_parents(self, num_parents: int) -> list:
        """Selects parent individuals based on fitness using tournament selection."""
        # Filter for runs that have fitness calculated
        eligible_population = [run for run in self.population if run.get('fitness') is not None]

        if len(eligible_population) < 2: # Need at least two for crossover
            logging.warning("[Hunter] Not enough eligible population for selection. Generating random parents.")
            # Fallback to random if not enough fit individuals
            return [self._create_random_params() for _ in range(num_parents)]

        # Sort by fitness (descending)
        eligible_population.sort(key=lambda x: x.get('fitness', 0), reverse=True)

        parents = []
        for _ in range(num_parents):
            # Tournament selection: pick a few random candidates and select the best one
            tournament_size = min(3, len(eligible_population))
            competitors = random.sample(eligible_population, tournament_size)
            winner = max(competitors, key=lambda x: x.get('fitness', 0))
            parents.append(winner)
        return parents

    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        """Performs simple arithmetic crossover for parameters."""
        child_params = {}
        # Simple average for crossover
        child_params["param_D"] = (parent1.get("param_D", PARAM_D_MIN) + parent2.get("param_D", PARAM_D_MIN)) / 2
        child_params["param_eta"] = (parent1.get("param_eta", PARAM_ETA_MIN) + parent2.get("param_eta", PARAM_ETA_MIN)) / 2
        return child_params

    def _mutate(self, params: dict, mutation_rate: float = 0.1, mutation_strength: float = 0.1) -> dict:
        """Applies mutation to parameters within their bounds."""
        mutated_params = params.copy()

        if random.random() < mutation_rate:
            # Mutate param_D
            perturbation = random.uniform(-mutation_strength, mutation_strength)
            mutated_params["param_D"] = max(PARAM_D_MIN, min(PARAM_D_MAX, params.get("param_D", PARAM_D_MIN) + perturbation))

        if random.random() < mutation_rate:
            # Mutate param_eta
            perturbation = random.uniform(-mutation_strength, mutation_strength)
            mutated_params["param_eta"] = max(PARAM_ETA_MIN, min(PARAM_ETA_MAX, params.get("param_eta", PARAM_ETA_MIN) + perturbation))

        return mutated_params

    def _create_random_params(self) -> dict:
        """Generates a set of random parameters within defined bounds."""
        return {
            "param_D": random.uniform(PARAM_D_MIN, PARAM_D_MAX),
            "param_eta": random.uniform(PARAM_ETA_MIN, PARAM_ETA_MAX)
        }

    def get_next_generation(self, population_size: int) -> list:
        """
        Breeds a new generation of parameters using selection, crossover, and mutation.
        """
        logging.info(f"[Hunter] Breeding Generation {self.get_current_generation()}...")
        new_generation_params = []

        # If population is too small or no fitness data, generate randomly
        eligible_for_breeding = [run for run in self.population if run.get('fitness') is not None]
        if len(eligible_for_breeding) < 2: # Need at least two for meaningful breeding
            logging.warning("[Hunter] Insufficient population with fitness data for breeding. Generating random population.")
            for _ in range(population_size):
                new_generation_params.append(self._create_random_params())
            return new_generation_params

        # Elitism: Carry over the very best individual directly
        best_run = self.get_best_run()
        if best_run and population_size > 0: # Ensure best_run is not empty and population_size is positive
            new_generation_params.append({"param_D": best_run.get("param_D"), "param_eta": best_run.get("param_eta")})

        # Fill the rest of the population
        while len(new_generation_params) < population_size:
            # Select two parents from the eligible population
            parents = random.sample(eligible_for_breeding, 2)
            
            # Crossover
            child = self._crossover(parents[0], parents[1])

            # Mutation
            mutated_child = self._mutate(child)

            new_generation_params.append(mutated_child)

        # Ensure correct population size if elitism caused an extra individual
        return new_generation_params[:population_size]

    def register_new_jobs(self, job_list: list):
        """
        Called by the Orchestrator *after* it has generated
        canonical hashes for the new jobs.
        """
        self.population.extend(job_list)
        logging.info(f"[Hunter] Registered {len(job_list)} new jobs in ledger.")
        self._save_ledger()

    def process_generation_results(self, provenance_dir: str, job_hashes: list):
        """
        Reads new provenance.json files, calculates fitness,
        and updates the internal ledger.
        """
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

                # Simple fitness = 1.0 / (sse + 1e-9) (avoid division by zero)
                fitness = 1.0 / (sse + 1e-9)

                # Find the run in our population and update it
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
        """
        Utility to get the best-performing run from the ledger.
        """
        if not self.population:
            return {}
        valid_runs = [r for r in self.population if r.get("fitness") is not None]
        if not valid_runs:
            return {}
        return max(valid_runs, key=lambda x: x["fitness"])
