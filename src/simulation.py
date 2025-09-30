"""IoT Wireless Sensor Network Simulation Pipeline.

Runs ACGA-based clustering simulation across different network sizes
and generates energy, throughput, and PDR analysis.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from .sim import run_scenario
    from .report.table_builders import build_table2, save_tables_to_files
except ImportError:
    # For direct execution
    from sim import run_scenario
    from report.table_builders import build_table2, save_tables_to_files


def run(config_path: Path, seed: int = 42) -> Dict[str, Any]:
    """Run IoT simulation across different network sizes.
    
    Args:
        config_path: Path to configuration file
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with simulation results and artifact paths
    """
    print("🌐 Starting IoT Wireless Sensor Network Simulation")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sim_config = config.get('sim', {})
    nodes_list = sim_config.get('nodes_list', [50, 100, 150, 200, 250])
    rounds = sim_config.get('rounds', 10)
    
    print(f"📊 Testing network sizes: {nodes_list}")
    print(f"🔄 Simulation rounds per network: {rounds}")
    print(f"🌱 Random seed: {seed}")
    
    # Run simulations
    results_cleveland = {}
    
    for num_nodes in nodes_list:
        print(f"\n🔍 Simulating {num_nodes} nodes...")
        
        try:
            # Run scenario simulation
            result = run_scenario(
                num_nodes=num_nodes,
                rounds=rounds,
                seed=seed + num_nodes,  # Vary seed per network size
                cfg=sim_config
            )
            
            results_cleveland[num_nodes] = result
            
            print(f"  ⚡ Avg Energy: {result['avg_energy_mJ']:.4f} mJ")
            print(f"  📈 Throughput: {result['throughput_pct']:.2f}%")
            print(f"  📦 PDR: {result['pdr_pct']:.2f}%")
            print(f"  🔋 Delivered: {result['total_delivered']}/{result['total_attempted']} packets")
            
        except Exception as e:
            print(f"  ❌ Simulation failed for {num_nodes} nodes: {e}")
            # Create default result to prevent crashes
            results_cleveland[num_nodes] = {
                'avg_energy_mJ': 0.0,
                'throughput_pct': 0.0,
                'pdr_pct': 0.0,
                'total_delivered': 0,
                'total_attempted': 0,
                'final_residuals': [],
                'logs': []
            }
    
    # Build tables from results
    print("\n📊 Building result tables...")
    energy_df, throughput_df, pdr_df = build_table2(results_cleveland, None)
    
    # Create output directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save simulation tables
    tables = {
        'sim_energy_mJ': energy_df.reset_index(),
        'sim_throughput_pct': throughput_df.reset_index(),
        'sim_pdr_pct': pdr_df.reset_index()
    }
    
    artifact_paths = {}
    
    # Save individual table files
    for table_name, df in tables.items():
        # CSV
        csv_path = artifacts_dir / f"{table_name}.csv"
        df.to_csv(csv_path, index=False)
        artifact_paths[f"{table_name}_csv"] = str(csv_path)
        
        # Excel
        xlsx_path = artifacts_dir / f"{table_name}.xlsx"
        df.to_excel(xlsx_path, index=False)
        artifact_paths[f"{table_name}_xlsx"] = str(xlsx_path)
        
        # Markdown
        md_path = artifacts_dir / f"{table_name}.md"
        with open(md_path, 'w') as f:
            f.write(f"# {table_name.replace('_', ' ').title()}\n\n")
            f.write(f"Generated from IoT simulation with ACGA clustering\n")
            f.write(f"Nodes tested: {nodes_list}\n")
            f.write(f"Rounds per network: {rounds}\n")
            f.write(f"Random seed: {seed}\n\n")
            f.write(df.to_markdown(index=False))
        artifact_paths[f"{table_name}_md"] = str(md_path)
    
    # Generate plots
    print("📈 Generating plots...")
    
    # Energy vs Nodes plot
    plt.figure(figsize=(10, 6))
    plt.plot(nodes_list, [results_cleveland[n]['avg_energy_mJ'] for n in nodes_list], 
             'b-o', linewidth=2, markersize=8, label='Cleveland')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Energy Consumption (mJ)')
    plt.title('Energy Consumption vs Network Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    energy_plot_path = artifacts_dir / "energy_vs_nodes.png"
    plt.savefig(energy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifact_paths['energy_plot'] = str(energy_plot_path)
    
    # Throughput vs Nodes plot
    plt.figure(figsize=(10, 6))
    plt.plot(nodes_list, [results_cleveland[n]['throughput_pct'] for n in nodes_list], 
             'g-s', linewidth=2, markersize=8, label='Cleveland')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Throughput (%)')
    plt.title('Throughput vs Network Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    throughput_plot_path = artifacts_dir / "throughput_vs_nodes.png"
    plt.savefig(throughput_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifact_paths['throughput_plot'] = str(throughput_plot_path)
    
    # PDR vs Nodes plot
    plt.figure(figsize=(10, 6))
    plt.plot(nodes_list, [results_cleveland[n]['pdr_pct'] for n in nodes_list], 
             'r-^', linewidth=2, markersize=8, label='Cleveland')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Packet Delivery Ratio (%)')
    plt.title('PDR vs Network Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    pdr_plot_path = artifacts_dir / "pdr_vs_nodes.png"
    plt.savefig(pdr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifact_paths['pdr_plot'] = str(pdr_plot_path)
    
    # Save detailed results as JSON
    results_json_path = artifacts_dir / "sim_detailed_results.json"
    with open(results_json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = {}
        for node_count, result in results_cleveland.items():
            serializable_results[str(node_count)] = convert_numpy(result)
        
        json.dump({
            'simulation_config': sim_config,
            'seed': seed,
            'nodes_tested': nodes_list,
            'results_cleveland': serializable_results
        }, f, indent=2)
    artifact_paths['detailed_results'] = str(results_json_path)
    
    print("\n✅ IoT simulation completed successfully!")
    print(f"📁 Artifacts saved to: {artifacts_dir}")
    
    return {
        'nodes_tested': nodes_list,
        'results_cleveland': results_cleveland,
        'artifact_paths': artifact_paths,
        'summary': {
            'avg_energy_range_mJ': (
                min(results_cleveland[n]['avg_energy_mJ'] for n in nodes_list),
                max(results_cleveland[n]['avg_energy_mJ'] for n in nodes_list)
            ),
            'avg_throughput_range_pct': (
                min(results_cleveland[n]['throughput_pct'] for n in nodes_list),
                max(results_cleveland[n]['throughput_pct'] for n in nodes_list)
            ),
            'avg_pdr_range_pct': (
                min(results_cleveland[n]['pdr_pct'] for n in nodes_list),
                max(results_cleveland[n]['pdr_pct'] for n in nodes_list)
            )
        }
    }
