#!/usr/bin/env python3
"""Fast simulation test"""

import time
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulation import run

def fast_test():
    print("⏱️ Running fast simulation test...")
    start = time.time()
    
    # Create minimal config
    config = {
        'seed': 42,
        'sim': {
            'packet_bits': 512,
            'field_size_m': [500, 500],
            'init_energy_J': 0.5,
            'base_station_xy': [250, 250],
            'nodes_list': [50],  # Just one network size
            'rounds': 2,  # Just 2 rounds
            'radio': {
                'e_elec': 5.0e-8,
                'eps_amp': 1.0e-10,
                'n': 2
            },
            'k_ch_ratio': 0.1,
            'aco': {
                'ants': 3,
                'alpha': 1.0,
                'beta': 2.0,
                'rho_init': 0.5,
                'max_iter': 3,
                'adaptive_evap': True
            },
            'ga': {
                'pop_size': 3,
                'generations': 3,
                'tournament_k': 2,
                'crossover_p': 0.9,
                'mutation_p': 0.02
            },
            'pareto': {
                'weights': {'energy': 0.5, 'distance': 0.3, 'residual': 0.2}
            }
        }
    }
    
    # Save test config
    with open('configs/test_fast.yaml', 'w') as f:
        yaml.dump(config, f)
    
    result = run(Path('configs/test_fast.yaml'), seed=42)
    elapsed = time.time() - start
    
    print(f"✅ Fast test completed in {elapsed:.1f} seconds")
    print(f"📊 Tested networks: {result.get('nodes_tested', [])}")
    
    return result

if __name__ == "__main__":
    fast_test()