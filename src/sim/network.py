"""Wireless Sensor Network Simulation.

Provides network topology generation and communication simulation
for wireless sensor networks with cluster-based routing.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any
from .energy_model import RadioEnergyModel
from .manhattan import assign_to_ch
from .acga import select_cluster_heads


def synth_network(num_nodes: int, field: Tuple[int, int] = (500, 500),
                 seed: int = 42, init_energy_J: float = 0.5,
                 bs: Tuple[int, int] = (250, 250)) -> Dict[str, Any]:
    """Generate synthetic wireless sensor network.
    
    Args:
        num_nodes: Number of sensor nodes
        field: Field dimensions (width, height) in meters
        seed: Random seed for reproducibility
        init_energy_J: Initial energy per node in Joules
        bs: Base station position (x, y)
        
    Returns:
        Dictionary containing:
            - positions: Node positions as (N, 2) array
            - residual_J: Residual energy array (N,)
            - base_station: Base station position tuple
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate random node positions within field
    positions = np.random.uniform(0, field[0], size=(num_nodes, 2))
    positions[:, 1] = np.random.uniform(0, field[1], size=num_nodes)
    
    # Initialize residual energy
    residual_J = np.full(num_nodes, init_energy_J, dtype=float)
    
    return {
        'positions': positions,
        'residual_J': residual_J,
        'base_station': bs
    }


def simulate_round(positions: np.ndarray, residual_J: np.ndarray,
                  ch_indices: List[int], radio: RadioEnergyModel,
                  bs_xy: Tuple[float, float]) -> Tuple[np.ndarray, int, int, List[float]]:
    """Simulate one communication round.
    
    Args:
        positions: Node positions as (N, 2) array
        residual_J: Current residual energy per node
        ch_indices: List of cluster head indices
        radio: Radio energy model
        bs_xy: Base station position
        
    Returns:
        Tuple containing:
            - updated_residual_J: Updated energy levels
            - delivered: Number of packets delivered to BS
            - attempted: Number of packets attempted
            - per_link_energy_J: Energy consumed per link
    """
    n_nodes = len(positions)
    updated_residual = residual_J.copy()
    per_link_energy = []
    attempted = 0
    delivered = 0
    
    if not ch_indices:
        return updated_residual, delivered, attempted, per_link_energy
    
    try:
        # Assign nodes to cluster heads
        assignments = assign_to_ch(positions, ch_indices)
        
        # Phase 1: Member nodes send to cluster heads
        for i in range(n_nodes):
            if i not in ch_indices and updated_residual[i] > 0:
                ch_idx = assignments[i]
                distance = np.linalg.norm(positions[i] - positions[ch_idx])
                
                # Energy for transmission
                tx_energy = radio.tx_energy(distance)
                rx_energy = radio.rx_energy()
                
                attempted += 1
                
                # Check if sender has enough energy
                if updated_residual[i] >= tx_energy:
                    updated_residual[i] -= tx_energy
                    per_link_energy.append(tx_energy)
                    
                    # Check if receiver has enough energy
                    if updated_residual[ch_idx] >= rx_energy:
                        updated_residual[ch_idx] -= rx_energy
                        # Packet successfully received by CH
                    else:
                        # CH out of energy, packet lost
                        updated_residual[ch_idx] = 0
                else:
                    # Sender out of energy
                    updated_residual[i] = 0
        
        # Phase 2: Cluster heads send aggregated data to base station
        bs_position = np.array(bs_xy)
        for ch_idx in ch_indices:
            if updated_residual[ch_idx] > 0:
                distance_to_bs = np.linalg.norm(positions[ch_idx] - bs_position)
                tx_energy_bs = radio.tx_energy(distance_to_bs)
                
                attempted += 1
                
                if updated_residual[ch_idx] >= tx_energy_bs:
                    updated_residual[ch_idx] -= tx_energy_bs
                    per_link_energy.append(tx_energy_bs)
                    delivered += 1  # Successfully delivered to BS
                else:
                    # CH out of energy
                    updated_residual[ch_idx] = 0
    
    except Exception as e:
        # Return original state if simulation fails
        print(f"Simulation round failed: {e}")
        return residual_J.copy(), 0, 0, []
    
    return updated_residual, delivered, attempted, per_link_energy


def run_scenario(num_nodes: int, rounds: int, seed: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run complete network scenario simulation.
    
    Args:
        num_nodes: Number of nodes in network
        rounds: Number of communication rounds
        seed: Random seed
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with simulation results:
            - avg_energy_mJ: Average energy consumed in millijoules
            - throughput_pct: Throughput percentage
            - pdr_pct: Packet delivery ratio percentage  
            - final_residuals: Final energy levels
            - logs: Detailed round-by-round logs
    """
    # Extract simulation parameters
    field_size = tuple(cfg.get('field_size_m', [500, 500]))
    init_energy = cfg.get('init_energy_J', 0.5)
    bs_position = tuple(cfg.get('base_station_xy', [250, 250]))
    k_ch_ratio = cfg.get('k_ch_ratio', 0.1)
    
    # Create radio energy model
    radio_cfg = cfg.get('radio', {})
    radio = RadioEnergyModel(
        e_elec=radio_cfg.get('e_elec', 5.0e-8),
        eps_amp=radio_cfg.get('eps_amp', 1.0e-10),
        n=radio_cfg.get('n', 2),
        packet_bits=cfg.get('packet_bits', 512)
    )
    
    # Generate network
    network = synth_network(num_nodes, field_size, seed, init_energy, bs_position)
    positions = network['positions']
    residual_J = network['residual_J']
    
    # Calculate number of cluster heads
    k_ch = max(1, int(num_nodes * k_ch_ratio))
    
    # Simulation tracking
    total_delivered = 0
    total_attempted = 0
    total_energy_consumed = 0.0
    round_logs = []
    
    for round_num in range(rounds):
        # Select cluster heads using ACGA
        try:
            ch_indices = select_cluster_heads(positions, residual_J, k_ch, cfg)
        except Exception as e:
            print(f"Cluster head selection failed in round {round_num}: {e}")
            # Fallback to random selection
            alive_nodes = np.where(residual_J > 0)[0]
            if len(alive_nodes) >= k_ch:
                ch_indices = np.random.choice(alive_nodes, k_ch, replace=False).tolist()
            else:
                ch_indices = alive_nodes.tolist()
        
        # Simulate communication round
        initial_energy = np.sum(residual_J)
        residual_J, delivered, attempted, link_energies = simulate_round(
            positions, residual_J, ch_indices, radio, bs_position
        )
        final_energy = np.sum(residual_J)
        
        # Track metrics
        energy_consumed = initial_energy - final_energy
        total_energy_consumed += energy_consumed
        total_delivered += delivered
        total_attempted += attempted
        
        # Log round details
        round_log = {
            'round': round_num + 1,
            'cluster_heads': ch_indices,
            'delivered': delivered,
            'attempted': attempted,
            'energy_consumed_J': energy_consumed,
            'alive_nodes': np.sum(residual_J > 0),
            'avg_residual_J': np.mean(residual_J)
        }
        round_logs.append(round_log)
        
        # Early termination if no nodes alive
        if np.sum(residual_J > 0) == 0:
            print(f"All nodes died after round {round_num + 1}")
            break
    
    # Calculate final metrics
    avg_energy_mJ = (total_energy_consumed / rounds) * 1000  # Convert to mJ
    throughput_pct = (total_delivered / max(1, total_attempted)) * 100
    pdr_pct = throughput_pct  # Same calculation for this model
    
    results = {
        'avg_energy_mJ': avg_energy_mJ,
        'throughput_pct': throughput_pct,
        'pdr_pct': pdr_pct,
        'final_residuals': residual_J,
        'logs': round_logs,
        'total_delivered': total_delivered,
        'total_attempted': total_attempted,
        'total_energy_consumed_J': total_energy_consumed
    }
    
    return results
