import pandas as pd
from pathlib import Path

def aggregate_experiments(results_dir='results/experiments'):
    print("Aggregating results...")
    results_path = Path(results_dir)
    summary_rows = []
    
    # Walk through the directory structure
    for policy_dir in results_path.iterdir():
        if not policy_dir.is_dir(): continue
        
        for scenario_dir in policy_dir.iterdir():
            if not scenario_dir.is_dir(): continue
            
            policy = policy_dir.name
            scenario = scenario_dir.name
            
            metrics_file = list(scenario_dir.glob("*_metrics.csv"))
            events_file = list(scenario_dir.glob("*_events.csv"))
            
            if not metrics_file:
                continue
                
            metrics_df = pd.read_csv(metrics_file[0])
            
            # Load events and calculate retrain stats
            num_retrains = 0
            total_retrain_time = 0.0
            
            if events_file:
                events_df = pd.read_csv(events_file[0])
                num_retrains = len(events_df)
                if 'retrain_time_seconds' in events_df.columns:
                    total_retrain_time = events_df['retrain_time_seconds'].sum()
            
            # Calculate aggregate stats
            summary = {
                'policy': policy,
                'scenario': scenario,
                'mean_mape': metrics_df['mape'].mean(),
                'median_mape': metrics_df['mape'].median(),
                'std_mape': metrics_df['mape'].std(),
                'num_retrains': num_retrains,
                'total_retrain_time_seconds': total_retrain_time, # <--- Added this field
                'final_mape': metrics_df['mape'].iloc[-1] if not metrics_df.empty else 0
            }
            summary_rows.append(summary)
    
    if not summary_rows:
        print("No results found.")
        return

    summary_df = pd.DataFrame(summary_rows)
    # Sort for cleaner viewing
    summary_df = summary_df.sort_values(['scenario', 'mean_mape'])
    
    output_file = results_path / 'summary.csv'
    summary_df.to_csv(output_file, index=False)
    
    print(f"\nAggregation Complete. Saved to {output_file}")
    # Print preview including the new column
    print(summary_df[['policy', 'scenario', 'mean_mape', 'total_retrain_time_seconds']].head().to_string(index=False))

if __name__ == "__main__":
    aggregate_experiments()