#!/usr/bin/env python3
"""Simple simulation test"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sim.network import synth_network, run_scenario

def test_basic():
    print("🧪 Testing basic network generation...")
    net = synth_network(10, seed=42)
    print(f"✅ Network created with {len(net['positions'])} nodes")
    
    print("🧪 Testing simple simulation...")
    # Load minimal config
    config = {
        'packet_bits': 512,
        'field_size_m': [500, 500],
        'init_energy_J': 0.5,
        'base_station_xy': [250, 250],
        'rounds': 2,
        'radio': {
            'e_elec': 5.0e-8,
            'eps_amp': 1.0e-10,
            'n': 2
        },
        'k_ch_ratio': 0.1,
        'aco': {
            'ants': 5,
            'alpha': 1.0,
            'beta': 2.0,
            'rho_init': 0.5,
            'max_iter': 5,
            'adaptive_evap': True
        },
        'ga': {
            'pop_size': 5,
            'generations': 5,
            'tournament_k': 2,
            'crossover_p': 0.9,
            'mutation_p': 0.02
        },
        'pareto': {
            'weights': {'energy': 0.5, 'distance': 0.3, 'residual': 0.2}
        }
    }
    
    result = run_scenario(10, 2, 42, config)
    print(f"✅ Simulation completed! Energy: {result['avg_energy_mJ']:.4f} mJ")
    return result

if __name__ == "__main__":
    test_basic()