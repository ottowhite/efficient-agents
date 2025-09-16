#!/usr/bin/env python3
"""
Simple NSys GPU Metrics Analysis Script
"""

import sqlite3
import sys
import os
from pathlib import Path


def get_metric_data(db_path: str, metric_id: int):
    """Extract metric data with timeout and progress."""
    print(f"  Analyzing metric {metric_id} in {Path(db_path).name}...", end=" ", flush=True)
    
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL") 
        cursor = conn.cursor()
        
        # Simple query with limit to test
        query = "SELECT COUNT(*) FROM GPU_METRICS WHERE metricId = ?"
        cursor.execute(query, (metric_id,))
        count = cursor.fetchone()[0]
        print(f"({count} records)", end=" ", flush=True)
        
        if count > 0:
            query = """
            SELECT 
                SUM(value) as total_value,
                COUNT(*) as total_ticks,
                CAST(SUM(value) AS FLOAT) / COUNT(*) as avg_value
            FROM GPU_METRICS 
            WHERE metricId = ?
            """
            cursor.execute(query, (metric_id,))
            result = cursor.fetchone()
            print("✓")
            conn.close()
            return result
        else:
            print("- no data")
            conn.close()
            return (0, 0, 0.0)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return (0, 0, 0.0)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_nsys_metrics_simple.py <sqlite_file1> <sqlite_file2> ...")
        sys.exit(1)
    
    db_paths = sys.argv[1:]
    
    print("NSys GPU Metrics Analysis (Simple)")
    print("=" * 40)
    
    results = {}
    
    for db_path in db_paths:
        if not os.path.exists(db_path):
            print(f"File not found: {db_path}")
            continue
            
        exp_name = Path(db_path).stem.replace('-', ' ').title()
        print(f"\nAnalyzing {exp_name}:")
        
        # Get SMs Active (metric 10)
        sms_data = get_metric_data(db_path, 10)
        
        # Get DRAM Read Bandwidth (metric 25)  
        dram_data = get_metric_data(db_path, 25)
        
        results[exp_name] = {
            'sms': sms_data,
            'dram': dram_data
        }
    
    # Print results table
    print(f"\n{'='*50}")
    print("RESULTS:")
    print(f"{'='*50}")
    
    experiments = list(results.keys())
    
    # Header
    header = f"| {'Metric':<20} |"
    for exp in experiments:
        header += f" {exp:<15} |"
    print(header)
    
    # Separator
    sep = f"|{'-'*22}|"
    for _ in experiments:
        sep += f"{'-'*17}|"
    print(sep)
    
    # SMs Active
    row = f"| {'SMs Active [%]':<20} |"
    for exp in experiments:
        avg = results[exp]['sms'][2] if results[exp]['sms'][1] > 0 else 0
        row += f" {avg:>13.2f}% |"
    print(row)
    
    # DRAM Read BW
    row = f"| {'DRAM Read BW [%]':<20} |"
    for exp in experiments:
        avg = results[exp]['dram'][2] if results[exp]['dram'][1] > 0 else 0
        row += f" {avg:>13.2f}% |"
    print(row)


if __name__ == "__main__":
    main()
