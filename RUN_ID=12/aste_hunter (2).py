"""
aste_hunter.py
CLASSIFICATION: Adaptive Learning Engine (ASTE V11.0)
GOAL: Acts as the "Brain" of the simulation suite. It manages a population
      of parameters, evaluates their performance based on validation reports,
      and "breeds" new generations to steer the search toward scientifically
      valid and numerically stable regimes.

This V11.0 version implements the "Paradox Fix" by evolving the
fitness function. It directly addresses the V10.0 "Geometric Crisis,"
where high physical order (scientific fidelity) paradoxically correlated
with high geometric instability. The new composite fitness function
simultaneously rewards high fidelity while penalizing instability, guiding
the search towards robust, physically meaningful solutions.
"""

import os
import json
import random
import math
from typing import List, Dict, Any, Optional

import settings

# Defines the parameter search space for the S-NCGL physics model.
# The Hunter will explore combinations of these parameters.
PARAM_SPACE = {
    "param_alpha": {"min": 0.05, "max": 0.5},
    "param_kappa": {"min": 0.5, "max": 2.0},
    "param_sigma_k": {"min": 0.1, "max": 1.5},
}
PARAM_KEYS = list(PARAM_SPACE.keys())

class Hunter:
    """Manages the evolutionary search for optimal parameters."""
    def __init__(self, population_size: int = 10):
        self.population_size = population_size
        self.population: List[Dict[str, Any]] = [] # Stores dicts with 'params', 'fitness', etc.

    def _breed(self, parent1: Dict, parent2: Dict) -> Dict:
        """Performs crossover between two parent parameter sets."""
        child = {}
        for key in PARAM_KEYS:
            # Simple average crossover
            child[key] = (parent1.get(key, 0.0) + parent2.get(key, 0.0)) / 2.0
        return child

    def _mutate(self, params: Dict) -> Dict:
        """Applies random mutations to a parameter set."""
        mutated = params.copy()
        for key in PARAM_KEYS:
            if random.random() < 0.2: # 20% chance of mutation per gene
                space = PARAM_SPACE[key]
                mutation_strength = (space['max'] - space['min']) * 0.1
                change = random.gauss(0, mutation_strength)
                mutated[key] = max(space['min'], min(space['max'], mutated[key] + change))
        return mutated

    def get_next_generation_parameters(self) -> List[Dict]:
        """
        Generates a new population of parameters using elitism, breeding,
        and mutation, reflecting its role as the system's "Brain".
        """
        new_params = []
        if not self.population:
            # Bootstrap generation
            for _ in range(self.population_size):
                params = {key: random.uniform(val['min'], val['max']) for key, val in PARAM_SPACE.items()}
                new_params.append(params)
            return new_params

        # Sort by fitness for elitism and parent selection
        self.population.sort(key=lambda r: r['fitness'], reverse=True)

        # 1. Elitism: Carry over the top 20%
        num_elites = int(self.population_size * 0.2)
        elites = self.population[:num_elites]
        new_params.extend([run['params'] for run in elites])

        # 2. Breeding & Mutation: Fill the rest of the population
        while len(new_params) < self.population_size:
            # Select parents from the top 50% of the population
            parent1 = random.choice(self.population[:len(self.population)//2])
            parent2 = random.choice(self.population[:len(self.population)//2])
            child = self._breed(parent1['params'], parent2['params'])
            mutated_child = self._mutate(child)
            new_params.append(mutated_child)
        return new_params

    def process_generation_results(self, job_uuid: str, params: Dict) -> Dict[str, Any]:
        """
        Processes a completed run's provenance report to calculate fitness
        and update the population ledger.
        Adheres to the V11.0 data contract.
        """
        provenance_file = os.path.join(settings.PROVENANCE_DIR, f"provenance_{job_uuid}.json")
        run_results = {
            "job_uuid": job_uuid,
            "params": params,
            "fitness": 0.0,
            "sse": 1e9,
            "h_norm": 1e9
        }

        if not os.path.exists(provenance_file):
            print(f"[Hunter] WARNING: Provenance not found for {job_uuid[:8]}. Assigning zero fitness.")
            self.population.append(run_results)
            return run_results

        try:
            with open(provenance_file, 'r') as f:
                provenance = json.load(f)

            # Reliably extract metrics using the data contract from settings.py
            sse = float(provenance.get(settings.SSE_METRIC_KEY, 1e9))
            h_norm = float(provenance.get(settings.STABILITY_METRIC_KEY, 1e9))

            run_results["sse"] = sse
            run_results["h_norm"] = h_norm

            # --- V11.0 "Paradox Fix" Composite Fitness Function ---
            # Solves the "Stability-Fidelity Paradox" by rewarding high fidelity (low SSE)
            # while simultaneously penalizing geometric instability (high h_norm).
            # A small epsilon prevents division by zero for a perfect SSE.
            if math.isfinite(sse) and math.isfinite(h_norm) and h_norm < 1.0:
                # The (1 + h_norm) term ensures the divisor is always >= 1 and that
                # fitness trends to zero as instability (h_norm) grows large.
                fitness = (1.0 / (sse + 1e-12)) / (1.0 + h_norm)
                run_results["fitness"] = fitness

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[Hunter] ERROR: Failed to parse provenance for {job_uuid[:8]}: {e}")
            # Fitness remains 0.0

        self.population.append(run_results)
        return run_results

    def get_best_run(self) -> Optional[Dict[str, Any]]:
        """Returns the best-performing run from the current population."""
        if not self.population:
            return None
        return max(self.population, key=lambda r: r['fitness'])
