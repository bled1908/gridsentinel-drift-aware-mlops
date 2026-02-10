import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette('colorblind')
plt.rcParams['figure.dpi'] = 300

# --- 1. METRIC COMPUTATION ---
def compute_policy_metrics(metrics_df: pd.DataFrame, events_df: pd.DataFrame) -> dict:
    """Compute comprehensive metrics for a single policy run."""
    return {
        'mean_mape': metrics_df['mape'].mean(),
        'median_mape': metrics_df['mape'].median(),
        'std_mape': metrics_df['mape'].std(),
        'max_mape': metrics_df['mape'].max(),
        'min_mape': metrics_df['mape'].min(),
        'num_retrains': len(events_df),
        'total_retrain_time': events_df['retrain_time_seconds'].sum() if len(events_df) > 0 and 'retrain_time_seconds' in events_df else 0,
        'mean_retrain_time': events_df['retrain_time_seconds'].mean() if len(events_df) > 0 and 'retrain_time_seconds' in events_df else 0
    }

def compute_mape_degradation_rate(metrics_df: pd.DataFrame, window_size: int = 168) -> float:
    """Compute rate of MAPE increase over time (slope)."""
    mape_smooth = metrics_df['mape'].rolling(window=window_size, min_periods=1).mean()
    timestamps_numeric = (metrics_df['timestamp'] - metrics_df['timestamp'].iloc[0]).dt.total_seconds() / 86400
    slope, _, _, _, _ = stats.linregress(timestamps_numeric, mape_smooth)
    return slope

# --- 2. TIME-SERIES VISUALIZATION ---
def plot_mape_over_time(metrics_df: pd.DataFrame, events_df: pd.DataFrame, title: str, output_path: str = None):
    """Plot MAPE over time with retrain events marked."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(metrics_df['timestamp'], metrics_df['mape'], label='MAPE', color='steelblue', alpha=0.8)
    
    if len(events_df) > 0:
        retrain_times = pd.to_datetime(events_df['timestamp'])
        mape_at_retrains = []
        for t in retrain_times:
            # Find closest timestamp in metrics to plot the marker accurately
            nearest_idx = metrics_df['timestamp'].searchsorted(t)
            if nearest_idx < len(metrics_df):
                mape_at_retrains.append(metrics_df['mape'].iloc[nearest_idx])
            else:
                mape_at_retrains.append(metrics_df['mape'].iloc[-1])
        ax.scatter(retrain_times, mape_at_retrains, color='red', marker='v', s=100, label='Retrain', zorder=5)
                   
    ax.set_title(title); ax.set_ylabel('MAPE (%)'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path: plt.savefig(output_path, bbox_inches='tight'); plt.close()

def plot_drift_signals(metrics_df: pd.DataFrame, title: str, output_path: str = None):
    """Plot drift signals (PSI, KS, MAPE)."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # PSI
    axes[0].plot(metrics_df['timestamp'], metrics_df['psi'], color='orange')
    axes[0].axhline(0.25, color='red', linestyle='--', label='Drift Threshold')
    axes[0].set_ylabel('PSI'); axes[0].legend()
    
    # KS Test - FIX: Handling the column name mismatch here
    ks_col = 'ks_drifted' if 'ks_drifted' in metrics_df.columns else 'ks_num_drifted'
    
    if ks_col in metrics_df.columns:
        axes[1].plot(metrics_df['timestamp'], metrics_df[ks_col], color='purple')
        axes[1].axhline(3, color='red', linestyle='--', label='Drift Threshold')
        axes[1].set_ylabel('# Drifted Features')
    else:
        axes[1].text(0.5, 0.5, 'KS Data Not Found', ha='center', va='center')
    
    # MAPE
    axes[2].plot(metrics_df['timestamp'], metrics_df['mape'], color='green')
    axes[2].set_ylabel('MAPE (%)'); axes[2].set_xlabel('Time')
    
    plt.suptitle(title); plt.tight_layout()
    if output_path: plt.savefig(output_path, bbox_inches='tight'); plt.close()

# --- 3. COMPARISON VISUALIZATION ---
def plot_mape_comparison_boxplot(summary_df: pd.DataFrame, scenario: str, output_path: str = None):
    """Boxplot comparing MAPE distributions across policies for a scenario."""
    results_dir = Path('results/experiments')
    policies = ['policy0', 'policy1', 'policy2', 'policy3']
    all_mapes = []
    policy_labels = []
    
    for policy in policies:
        metrics_path = results_dir / policy / scenario / f"{policy}_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            all_mapes.extend(metrics_df['mape'].values)
            policy_labels.extend([policy] * len(metrics_df))
            
    if not all_mapes:
        return

    plot_df = pd.DataFrame({'Policy': policy_labels, 'MAPE (%)': all_mapes})
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=plot_df, x='Policy', y='MAPE (%)', ax=ax)
    ax.set_title(f'MAPE Distribution by Policy ({scenario})')
    ax.set_ylabel('MAPE (%)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_retrain_count_comparison(summary_df: pd.DataFrame, output_path: str = None):
    """Bar chart of retrain counts."""
    fig, ax = plt.subplots(figsize=(10, 5))
    summary_df.pivot(index='scenario', columns='policy', values='num_retrains').plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Retrains'); ax.set_title('Retraining Frequency'); plt.tight_layout()
    if output_path: plt.savefig(output_path, bbox_inches='tight'); plt.close()

def plot_tradeoff_scatter(summary_df: pd.DataFrame, output_path: str = None):
    """Scatter plot: Cost vs Accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=summary_df, x='num_retrains', y='mean_mape', hue='scenario', s=100, ax=ax)
    for _, row in summary_df.iterrows():
        ax.text(row['num_retrains'], row['mean_mape'], row['policy'], fontsize=8)
    ax.set_title('Efficiency vs Accuracy Trade-off'); plt.tight_layout()
    if output_path: plt.savefig(output_path, bbox_inches='tight'); plt.close()

def plot_pareto_frontier(summary_df: pd.DataFrame, scenario: str, output_path: str = None):
    """Plot Pareto frontier."""
    data = summary_df[summary_df['scenario'] == scenario].copy()
    data = data.sort_values('num_retrains')
    
    pareto_points = []
    current_best_mape = float('inf')
    for _, row in data.iterrows():
        if row['mean_mape'] < current_best_mape:
            pareto_points.append(row['policy'])
            current_best_mape = row['mean_mape']
            
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['num_retrains'], data['mean_mape'], s=100, alpha=0.5, label='All Policies')
    
    pareto_data = data[data['policy'].isin(pareto_points)]
    ax.plot(pareto_data['num_retrains'], pareto_data['mean_mape'], 'r--', alpha=0.5)
    ax.scatter(pareto_data['num_retrains'], pareto_data['mean_mape'], color='red', s=120, label='Pareto Frontier')
    
    for _, row in data.iterrows():
        ax.annotate(row['policy'], (row['num_retrains'], row['mean_mape']), xytext=(5,5), textcoords='offset points')
        
    ax.set_title(f'Pareto Frontier: {scenario}')
    ax.set_xlabel('Cost (Num Retrains)'); ax.set_ylabel('Error (MAPE)')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    if output_path: plt.savefig(output_path, bbox_inches='tight'); plt.close()

# --- 4. STATISTICAL VALIDATION ---
def bootstrap_confidence_interval(data: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95):
    """Compute bootstrap CI."""
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return np.mean(data), lower, upper

def compare_policies_statistical(policy1_metrics: pd.DataFrame, policy2_metrics: pd.DataFrame) -> dict:
    """Full statistical comparison."""
    mape1, mape2 = policy1_metrics['mape'].values, policy2_metrics['mape'].values
    min_len = min(len(mape1), len(mape2))
    mape1 = mape1[:min_len]
    mape2 = mape2[:min_len]
    
    # Wilcoxon
    try: 
        stat, p_val = stats.wilcoxon(mape1, mape2)
    except: 
        p_val = 1.0
        stat = 0.0
        
    # Cohen's d
    diff_mean = np.mean(mape1) - np.mean(mape2)
    pooled_std = np.sqrt((np.std(mape1)**2 + np.std(mape2)**2) / 2)
    cohens_d = diff_mean / pooled_std if pooled_std > 0 else 0
    
    # Bootstrap CIs
    mean1, low1, high1 = bootstrap_confidence_interval(mape1)
    mean2, low2, high2 = bootstrap_confidence_interval(mape2)
    
    return {
        'policy1': {'mean': mean1, 'ci_lower': low1, 'ci_upper': high1},
        'policy2': {'mean': mean2, 'ci_lower': low2, 'ci_upper': high2},
        'wilcoxon': {'statistic': stat, 'p_value': p_val, 'significant': p_val < 0.05},
        'cohens_d': cohens_d
    }

# --- 5. TABLES ---
def create_performance_table(summary_df: pd.DataFrame, output_path: str = None):
    table = summary_df[['policy', 'scenario', 'mean_mape', 'std_mape', 'num_retrains', 'total_retrain_time_seconds']].copy()
    table.columns = ['Policy', 'Scenario', 'Mean MAPE', 'Std Dev', 'Retrains', 'Time (s)']
    if output_path:
        with open(output_path, 'w') as f: f.write(table.to_latex(index=False, float_format="%.2f"))
    return table

def create_statistical_test_table(results_dir, output_path=None):
    """Generate the statistical comparison table."""
    results_path = Path(results_dir)
    scenarios = ['baseline', 'seasonal_drift_winter', 'seasonal_drift_summer', 'holiday_drift', 'long_term_drift']
    rows = []
    
    for sc in scenarios:
        p3_path = results_path / 'policy3' / sc / 'policy3_metrics.csv'
        if not p3_path.exists(): continue
        p3_df = pd.read_csv(p3_path)
        
        for p_other in ['policy0', 'policy1']:
             other_path = results_path / p_other / sc / f'{p_other}_metrics.csv'
             if other_path.exists():
                 res = compare_policies_statistical(p3_df, pd.read_csv(other_path))
                 rows.append({
                     'Scenario': sc, 
                     'Comparison': f'P3 vs {p_other}',
                     'P-Value': res['wilcoxon']['p_value'], 
                     "Cohen's d": res['cohens_d'], 
                     'Sig': res['wilcoxon']['significant']
                 })
    
    df = pd.DataFrame(rows)
    if output_path and not df.empty:
        with open(output_path, 'w') as f: f.write(df.to_latex(index=False, float_format="%.4f"))
    return df