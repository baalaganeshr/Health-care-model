#!/usr/bin/env python3
"""
Generate reporting tables from actual run artifacts.
NEVER fabricates results - only uses real artifacts or paper-reported static values when explicitly requested.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_file_checksum(filepath: Path) -> Optional[str]:
    """Calculate SHA256 checksum of a file."""
    if not filepath.exists():
        return None
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_file_mtime(filepath: Path) -> Optional[str]:
    """Get file modification time in ISO format."""
    if not filepath.exists():
        return None
    
    mtime = filepath.stat().st_mtime
    return datetime.fromtimestamp(mtime).isoformat()


def load_real_metrics(project_root: Path) -> Dict[str, Any]:
    """Load real metrics from artifacts, preferring eval over train results."""
    eval_metrics_path = project_root / "artifacts" / "metrics_eval.json"
    train_metrics_path = project_root / "artifacts" / "metrics.json"
    
    # Try eval metrics first, fallback to train metrics
    metrics_path = None
    if eval_metrics_path.exists():
        metrics_path = eval_metrics_path
    elif train_metrics_path.exists():
        metrics_path = train_metrics_path
    else:
        print("❌ ERROR: No metrics files found!")
        print(f"   Expected: {eval_metrics_path}")
        print(f"   Or: {train_metrics_path}")
        sys.exit(1)
    
    print(f"📊 Metrics source: {metrics_path}")
    print(f"📅 File modified: {get_file_mtime(metrics_path)}")
    
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        return data, metrics_path
    except Exception as e:
        print(f"❌ ERROR: Failed to load {metrics_path}: {e}")
        sys.exit(1)


def create_real_tables(data: Dict[str, Any], project_root: Path) -> None:
    """Create tables from real run artifacts."""
    print("\n🔍 Creating REAL metrics tables from actual run artifacts...")
    
    # Extract model metrics
    if "models" in data:
        # Train metrics format
        models_data = data["models"]
    elif isinstance(data, dict) and "model" in data:
        # Eval metrics format - single best model
        model_name = data["model"]
        metrics = data["metrics"]
        models_data = {model_name: metrics}
    else:
        # Try to extract metrics directly
        models_data = data
    
    # Build dataframe
    rows = []
    for model_name, metrics in models_data.items():
        if isinstance(metrics, dict):
            row = {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", "N/A"),
                "Precision": metrics.get("precision", "N/A"),
                "Recall": metrics.get("recall", "N/A"),
                "F1-Score": metrics.get("f1", "N/A"),
                "ROC-AUC": metrics.get("roc_auc", "N/A")
            }
            rows.append(row)
    
    if not rows:
        print("❌ ERROR: No valid model metrics found in artifacts!")
        sys.exit(1)
    
    df = pd.DataFrame(rows)
    
    # Create output files
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save as different formats
    md_path = artifacts_dir / "tables_real.md"
    csv_path = artifacts_dir / "tables_real.csv"
    xlsx_path = artifacts_dir / "tables_real.xlsx"
    
    # Markdown
    with open(md_path, 'w') as f:
        f.write(f"# Real Metrics Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Model Performance (From Actual Run Artifacts)\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
    
    # CSV
    df.to_csv(csv_path, index=False, float_format="%.4f")
    
    # Excel
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Metrics_Eval', index=False, float_format="%.4f")
    
    print(f"✅ Created: {md_path.relative_to(project_root)}")
    print(f"✅ Created: {csv_path.relative_to(project_root)}")
    print(f"✅ Created: {xlsx_path.relative_to(project_root)}")
    
    return df


def create_paper_tables(project_root: Path) -> None:
    """Create tables with paper-reported static values."""
    print("\n📄 Creating PAPER tables with static reported values...")
    
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Paper data as specified
    paper_data = {
        "Energy_mJ": pd.DataFrame({
            "nodes": [50, 100, 150, 200, 250],
            "cleveland": [0.034, 0.0785, 0.122, 0.23, 0.28],
            "kaggle": [0.0255, 0.07525, 0.0915, 0.2245, 0.2745]
        }),
        "Throughput_pct": pd.DataFrame({
            "nodes": [50, 100, 150, 200, 250],
            "cleveland": [0.983, 0.9715, 0.9585, 0.9415, 0.9295],
            "kaggle": [0.987, 0.9803, 0.9713, 0.9557, 0.9478]
        }),
        "PDR_pct": pd.DataFrame({
            "nodes": [50, 100, 150, 200, 250],
            "cleveland": [98.93, 98.68, 98.82, 98.09, 98.26],
            "kaggle": [99.02, 98.71, 99.41, 98.21, 98.57]
        }),
        "Precision_pct": pd.DataFrame({
            "instances": [10, 20, 30, 40, 50],
            "cleveland": [96.02, 96.55, 97.02, 97.42, 97.95],
            "kaggle": [96.82, 97.35, 97.82, 98.22, 98.75]
        }),
        "Recall_pct": pd.DataFrame({
            "instances": [10, 20, 30, 40, 50],
            "cleveland": [96.22, 97.00, 97.38, 97.99, 98.17],
            "kaggle": [97.02, 97.80, 98.18, 98.79, 98.97]
        }),
        "F1_pct": pd.DataFrame({
            "instances": [10, 20, 30, 40, 50],
            "cleveland": [96.12, 96.76, 97.21, 97.78, 98.97],
            "kaggle": [96.92, 97.56, 98.01, 98.58, 99.52]
        }),
        "Accuracy_pct": pd.DataFrame({
            "instances": [10, 20, 30, 40, 50],
            "cleveland": [97.48, 98.12, 97.80, 98.53, 99.21],
            "kaggle": [97.99, 98.53, 99.05, 98.79, 99.36]
        })
    }
    
    # Create Excel file with multiple sheets
    xlsx_path = artifacts_dir / "tables_paper.xlsx"
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sheet_name, df in paper_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Create combined CSV
    csv_path = artifacts_dir / "tables_paper.csv"
    combined_df = pd.DataFrame()
    for name, df in paper_data.items():
        df_copy = df.copy()
        df_copy['Metric_Type'] = name
        combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
    combined_df.to_csv(csv_path, index=False)
    
    # Create Markdown
    md_path = artifacts_dir / "tables_paper.md"
    with open(md_path, 'w') as f:
        f.write("# Paper-Reported Tables (Static, Not Computed)\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("**Note: These are static values from research papers, not computed from current run.**\n\n")
        
        for name, df in paper_data.items():
            f.write(f"## {name.replace('_', ' ').title()}\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
    
    print(f"✅ Created: {xlsx_path.relative_to(project_root)}")
    print(f"✅ Created: {csv_path.relative_to(project_root)}")
    print(f"✅ Created: {md_path.relative_to(project_root)}")


def print_summary(data: Dict[str, Any], project_root: Path) -> None:
    """Print summary information about the run artifacts."""
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    # Find best model
    best_model = None
    if "best_model" in data:
        best_model = data["best_model"]
    elif "model" in data:
        best_model = data["model"]
    
    if best_model:
        print(f"🏆 Best model: {best_model}")
    
    # Check model files and their checksums
    models_dir = project_root / "models"
    for model_file in ["best_model.pkl", "best_model.pt"]:
        model_path = models_dir / model_file
        if model_path.exists():
            checksum = get_file_checksum(model_path)
            print(f"🔐 SHA256 ({model_file}): {checksum}")
    
    # File information
    for metrics_file in ["metrics_eval.json", "metrics.json"]:
        metrics_path = project_root / "artifacts" / metrics_file
        if metrics_path.exists():
            mtime = get_file_mtime(metrics_path)
            print(f"📅 {metrics_file} mtime: {mtime}")


def main(paper_tables: bool = False) -> None:
    """Main entry point for report generation."""
    project_root = get_project_root()
    
    print("🔍 Healthcare ML Project Report Generator")
    print("="*50)
    
    # Load real metrics
    data, metrics_path = load_real_metrics(project_root)
    
    # Create real tables
    df_real = create_real_tables(data, project_root)
    
    # Create paper tables if requested
    if paper_tables:
        create_paper_tables(project_root)
    else:
        print("\n📝 Paper tables skipped (use --paper_tables true to include)")
    
    # Print summary
    print_summary(data, project_root)
    
    print("\n✅ Report generation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate healthcare ML project reports")
    parser.add_argument("--paper_tables", type=str, choices=["true", "false"], 
                       default="false", help="Include paper-reported static tables")
    
    args = parser.parse_args()
    paper_tables = args.paper_tables.lower() == "true" or os.getenv("PAPER_TABLES") == "1"
    
    main(paper_tables=paper_tables)