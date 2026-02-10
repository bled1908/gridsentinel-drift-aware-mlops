import pandas as pd
from pathlib import Path
from src.evaluation import (
    plot_mape_over_time, plot_drift_signals, plot_retrain_count_comparison,
    plot_tradeoff_scatter, plot_pareto_frontier, create_performance_table,
    create_statistical_test_table
)

def main():
    print("Starting Role 5 Evaluation...")
    figures_dir = Path("results/figures"); figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path("results/tables"); tables_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df = pd.read_csv("results/experiments/summary.csv")
    
    # 1. Tables
    create_performance_table(summary_df, output_path=tables_dir / "performance_table.tex")
    create_statistical_test_table("results/experiments", output_path=tables_dir / "statistical_tests.tex")
    
    # 2. General Plots
    plot_retrain_count_comparison(summary_df, output_path=figures_dir / "retrain_counts.png")
    plot_tradeoff_scatter(summary_df, output_path=figures_dir / "tradeoff_scatter.png")
    
    # 3. Scenario Plots
    for scenario in summary_df['scenario'].unique():
        print(f"Processing {scenario}...")
        
        # New Pareto Plot
        plot_pareto_frontier(summary_df, scenario, output_path=figures_dir / f"pareto_{scenario}.png")
        
        # Time Series
        for policy in ['policy0', 'policy1', 'policy2', 'policy3']:
            metrics_path = Path(f"results/experiments/{policy}/{scenario}/{policy}_metrics.csv")
            events_path = Path(f"results/experiments/{policy}/{scenario}/{policy}_events.csv")
            
            if metrics_path.exists():
                m_df = pd.read_csv(metrics_path, parse_dates=['timestamp'])
                e_df = pd.read_csv(events_path, parse_dates=['timestamp']) if events_path.exists() else pd.DataFrame()
                
                plot_mape_over_time(m_df, e_df, f"{policy} - {scenario}", figures_dir / f"{policy}_{scenario}_mape.png")
                
                if policy == 'policy3':
                    plot_drift_signals(m_df, f"Drift Signals {scenario}", figures_dir / f"{policy}_{scenario}_drift.png")

    print("\nAll Role 5 deliverables generated successfully.")

if __name__ == "__main__":
    main()