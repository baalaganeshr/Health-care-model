"""Ant Colony + Genetic Algorithm (ACGA) for Cluster Head Selection.

Hybrid optimization algorithm combining ACO and GA for optimal
cluster head selection in wireless sensor networks.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from .manhattan import assign_to_ch, calculate_intra_cluster_distance
from .energy_model import RadioEnergyModel


@dataclass
class ACGAConfig:
    """Configuration for ACGA algorithm."""
    aco_ants: int = 20
    aco_alpha: float = 1.0  # Pheromone importance
    aco_beta: float = 2.0   # Heuristic importance
    aco_rho_init: float = 0.5  # Initial evaporation rate
    aco_max_iter: int = 30
    aco_adaptive_evap: bool = True
    
    ga_pop_size: int = 30
    ga_generations: int = 30
    ga_tournament_k: int = 3
    ga_crossover_p: float = 0.9
    ga_mutation_p: float = 0.02
    
    # Multi-objective weights
    weight_energy: float = 0.5
    weight_distance: float = 0.3
    weight_residual: float = 0.2


class ACGAOptimizer:
    """Hybrid ACO+GA optimizer for cluster head selection."""
    
    def __init__(self, config: ACGAConfig, radio_model: RadioEnergyModel):
        self.config = config
        self.radio = radio_model
        self.pheromones = None
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def _initialize_pheromones(self, n_nodes: int) -> np.ndarray:
        """Initialize pheromone matrix."""
        return np.ones((n_nodes,)) * 0.1
    
    def _calculate_heuristic(self, positions: np.ndarray, 
                           residual_energy: np.ndarray) -> np.ndarray:
        """Calculate heuristic information combining distance and energy."""
        n_nodes = len(positions)
        heuristic = np.zeros(n_nodes)
        
        # Center of the field for distance calculation
        center = np.mean(positions, axis=0)
        
        for i in range(n_nodes):
            # Inverse distance to center (closer nodes preferred)
            dist_to_center = np.linalg.norm(positions[i] - center)
            inv_distance = 1.0 / (1.0 + dist_to_center)
            
            # Normalized residual energy (higher energy preferred)
            norm_energy = residual_energy[i] / np.max(residual_energy)
            
            # Combine factors
            heuristic[i] = inv_distance * norm_energy
        
        return heuristic
    
    def _aco_construct_solution(self, positions: np.ndarray, 
                              residual_energy: np.ndarray, 
                              k_ch: int) -> List[int]:
        """Construct solution using ACO probabilistic selection."""
        n_nodes = len(positions)
        heuristic = self._calculate_heuristic(positions, residual_energy)
        
        # Calculate selection probabilities
        probabilities = np.zeros(n_nodes)
        for i in range(n_nodes):
            probabilities[i] = (self.pheromones[i] ** self.config.aco_alpha) * \
                             (heuristic[i] ** self.config.aco_beta)
        
        # Avoid division by zero
        if np.sum(probabilities) == 0:
            probabilities = np.ones(n_nodes)
        
        probabilities /= np.sum(probabilities)
        
        # Select cluster heads probabilistically
        selected = np.random.choice(n_nodes, size=k_ch, replace=False, p=probabilities)
        return sorted(selected.tolist())
    
    def _evaluate_fitness(self, ch_indices: List[int], positions: np.ndarray,
                         residual_energy: np.ndarray, bs_position: Tuple[float, float]) -> float:
        """Evaluate multi-objective fitness of cluster head selection."""
        if not ch_indices:
            return float('inf')
        
        try:
            # Assign nodes to cluster heads
            assignments = assign_to_ch(positions, ch_indices)
            
            # Calculate total transmission energy
            total_energy = 0.0
            for i, ch_idx in enumerate(assignments):
                if i != ch_idx:  # Member to CH
                    dist = np.linalg.norm(positions[i] - positions[ch_idx])
                    total_energy += self.radio.tx_energy(dist)
            
            # CH to BS energy
            for ch_idx in ch_indices:
                dist_to_bs = np.linalg.norm(positions[ch_idx] - np.array(bs_position))
                total_energy += self.radio.tx_energy(dist_to_bs)
            
            # Calculate intra-cluster distance
            intra_distance = calculate_intra_cluster_distance(positions, assignments)
            
            # Calculate average residual energy of CHs
            avg_residual = np.mean([residual_energy[i] for i in ch_indices])
            
            # Multi-objective fitness (minimize energy and distance, maximize residual)
            fitness = (self.config.weight_energy * total_energy + 
                      self.config.weight_distance * intra_distance - 
                      self.config.weight_residual * avg_residual)
            
            return fitness
            
        except Exception:
            return float('inf')
    
    def _aco_phase(self, positions: np.ndarray, residual_energy: np.ndarray,
                  k_ch: int, bs_position: Tuple[float, float]) -> List[int]:
        """ACO phase for cluster head selection."""
        n_nodes = len(positions)
        self.pheromones = self._initialize_pheromones(n_nodes)
        
        best_solution = None
        best_fitness = float('inf')
        
        for iteration in range(self.config.aco_max_iter):
            # Generate solutions
            solutions = []
            fitnesses = []
            
            for _ in range(self.config.aco_ants):
                solution = self._aco_construct_solution(positions, residual_energy, k_ch)
                fitness = self._evaluate_fitness(solution, positions, residual_energy, bs_position)
                solutions.append(solution)
                fitnesses.append(fitness)
            
            # Update best solution
            min_idx = np.argmin(fitnesses)
            if fitnesses[min_idx] < best_fitness:
                best_fitness = fitnesses[min_idx]
                best_solution = solutions[min_idx].copy()
            
            # Update pheromones
            self.pheromones *= (1 - self.config.aco_rho_init)  # Evaporation
            
            # Pheromone deposition (only best solution)
            if best_solution:
                deposit = 1.0 / (1.0 + best_fitness)
                for node_idx in best_solution:
                    self.pheromones[node_idx] += deposit
        
        return best_solution if best_solution else list(range(k_ch))
    
    def _create_individual(self, n_nodes: int, k_ch: int) -> List[int]:
        """Create random individual for GA."""
        return sorted(random.sample(range(n_nodes), k_ch))
    
    def _tournament_selection(self, population: List[List[int]], 
                            fitnesses: List[float]) -> List[int]:
        """Tournament selection for GA."""
        tournament_size = min(self.config.ga_tournament_k, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: List[int], parent2: List[int], 
                  n_nodes: int, k_ch: int) -> Tuple[List[int], List[int]]:
        """Two-point crossover with repair."""
        if random.random() > self.config.ga_crossover_p:
            return parent1.copy(), parent2.copy()
        
        # Simple approach: randomly mix genes from both parents
        combined = list(set(parent1 + parent2))
        if len(combined) < k_ch:
            # Add random nodes if not enough
            remaining = [i for i in range(n_nodes) if i not in combined]
            combined.extend(random.sample(remaining, k_ch - len(combined)))
        
        # Create two offspring
        random.shuffle(combined)
        child1 = sorted(combined[:k_ch])
        child2 = sorted(combined[:k_ch])  # Same for simplicity
        
        return child1, child2
    
    def _mutate(self, individual: List[int], n_nodes: int) -> List[int]:
        """Bit-flip mutation."""
        if random.random() > self.config.ga_mutation_p:
            return individual
        
        # Replace one randomly selected CH with a random non-CH node
        available_nodes = [i for i in range(n_nodes) if i not in individual]
        if available_nodes:
            replace_idx = random.randint(0, len(individual) - 1)
            new_node = random.choice(available_nodes)
            individual[replace_idx] = new_node
            individual.sort()
        
        return individual
    
    def _ga_phase(self, initial_solution: List[int], positions: np.ndarray,
                 residual_energy: np.ndarray, k_ch: int, 
                 bs_position: Tuple[float, float]) -> List[int]:
        """GA phase starting from ACO solution."""
        n_nodes = len(positions)
        
        # Initialize population with ACO solution and random individuals
        population = [initial_solution.copy()]
        for _ in range(self.config.ga_pop_size - 1):
            individual = self._create_individual(n_nodes, k_ch)
            population.append(individual)
        
        best_solution = initial_solution.copy()
        best_fitness = self._evaluate_fitness(initial_solution, positions, 
                                            residual_energy, bs_position)
        
        for generation in range(self.config.ga_generations):
            # Evaluate population
            fitnesses = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, positions, 
                                               residual_energy, bs_position)
                fitnesses.append(fitness)
            
            # Update best solution
            min_idx = np.argmin(fitnesses)
            if fitnesses[min_idx] < best_fitness:
                best_fitness = fitnesses[min_idx]
                best_solution = population[min_idx].copy()
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(best_solution.copy())
            
            # Generate rest of population
            while len(new_population) < self.config.ga_pop_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                
                child1, child2 = self._crossover(parent1, parent2, n_nodes, k_ch)
                
                child1 = self._mutate(child1, n_nodes)
                child2 = self._mutate(child2, n_nodes)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.config.ga_pop_size]
        
        return best_solution
    
    def select_cluster_heads(self, positions: np.ndarray, 
                           residual_energy: np.ndarray, k_ch: int,
                           bs_position: Tuple[float, float] = (250, 250)) -> List[int]:
        """Select optimal cluster heads using hybrid ACGA.
        
        Args:
            positions: Node positions as (N, 2) array
            residual_energy: Residual energy for each node
            k_ch: Number of cluster heads to select
            bs_position: Base station position
            
        Returns:
            List of selected cluster head indices
        """
        if k_ch <= 0 or k_ch > len(positions):
            k_ch = max(1, min(len(positions) // 10, len(positions)))
        
        # For small networks, use simple energy-based selection for speed
        if len(positions) <= 20 or self.config.aco_max_iter <= 5:
            # Simple heuristic: select nodes with highest energy and good positions
            scores = residual_energy.copy()
            center = np.mean(positions, axis=0)
            for i in range(len(positions)):
                dist_to_center = np.linalg.norm(positions[i] - center)
                scores[i] += 1.0 / (1.0 + dist_to_center)  # Closer to center is better
            
            selected = np.argsort(scores)[-k_ch:].tolist()
            return sorted(selected)
        
        # Phase 1: ACO (simplified)
        aco_solution = self._aco_phase(positions, residual_energy, k_ch, bs_position)
        
        # Phase 2: GA refinement (simplified)
        final_solution = self._ga_phase(aco_solution, positions, residual_energy, 
                                      k_ch, bs_position)
        
        return final_solution


def select_cluster_heads(positions: np.ndarray, residual_energy: np.ndarray,
                        k_ch: int, cfg: Dict[str, Any]) -> List[int]:
    """Convenience function for cluster head selection.
    
    Args:
        positions: Node positions as (N, 2) array  
        residual_energy: Residual energy for each node
        k_ch: Number of cluster heads to select
        cfg: Configuration dictionary
        
    Returns:
        List of selected cluster head indices
    """
    # Create ACGA config from dictionary
    acga_config = ACGAConfig(
        aco_ants=cfg.get('aco', {}).get('ants', 20),
        aco_alpha=cfg.get('aco', {}).get('alpha', 1.0),
        aco_beta=cfg.get('aco', {}).get('beta', 2.0),
        aco_rho_init=cfg.get('aco', {}).get('rho_init', 0.5),
        aco_max_iter=cfg.get('aco', {}).get('max_iter', 30),
        aco_adaptive_evap=cfg.get('aco', {}).get('adaptive_evap', True),
        ga_pop_size=cfg.get('ga', {}).get('pop_size', 30),
        ga_generations=cfg.get('ga', {}).get('generations', 30),
        ga_tournament_k=cfg.get('ga', {}).get('tournament_k', 3),
        ga_crossover_p=cfg.get('ga', {}).get('crossover_p', 0.9),
        ga_mutation_p=cfg.get('ga', {}).get('mutation_p', 0.02),
        weight_energy=cfg.get('pareto', {}).get('weights', {}).get('energy', 0.5),
        weight_distance=cfg.get('pareto', {}).get('weights', {}).get('distance', 0.3),
        weight_residual=cfg.get('pareto', {}).get('weights', {}).get('residual', 0.2)
    )
    
    # Create radio model
    radio_cfg = cfg.get('radio', {})
    radio = RadioEnergyModel(
        e_elec=radio_cfg.get('e_elec', 5.0e-8),
        eps_amp=radio_cfg.get('eps_amp', 1.0e-10),
        n=radio_cfg.get('n', 2),
        packet_bits=cfg.get('packet_bits', 512)
    )
    
    # Run ACGA optimization
    optimizer = ACGAOptimizer(acga_config, radio)
    bs_position = tuple(cfg.get('base_station_xy', [250, 250]))
    
    return optimizer.select_cluster_heads(positions, residual_energy, k_ch, bs_position)
