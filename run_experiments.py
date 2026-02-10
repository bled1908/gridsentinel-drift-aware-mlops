import subprocess
import os
import sys

# Configuration
POLICIES = ["policy0", "policy1", "policy2", "policy3"]
SCENARIOS = ["baseline", "seasonal_drift_winter", "seasonal_drift_summer", "holiday_drift", "long_term_drift"]
PYTHON_EXEC = sys.executable # Uses your current venv python

def run_experiment(policy, scenario):
    print(f"\nItems remaining... Running {policy} on {scenario}...")
    
    config_file = f"configs/{policy}_config.yaml"
    output_dir = f"results/experiments/{policy}/{scenario}"
    
    # Construct command
    cmd = [
        PYTHON_EXEC, "src/main_pipeline.py",
        "--config", config_file,
        "--scenario", scenario,
        "--output", output_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Success: {policy} | {scenario}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {policy} | {scenario}")

def main():
    print(f"Starting Experiment Suite: {len(POLICIES)} Policies x {len(SCENARIOS)} Scenarios")
    
    for policy in POLICIES:
        for scenario in SCENARIOS:
            # Check if config exists before running
            if not os.path.exists(f"configs/{policy}_config.yaml"):
                print(f"⚠️ Skipping {policy}: Config not found.")
                continue
                
            run_experiment(policy, scenario)
            
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()