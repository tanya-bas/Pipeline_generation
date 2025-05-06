import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from agent import process_complete_pipeline, eval
import pandas as pd
from tqdm import tqdm
import argparse

def run_experiment(start_n=1, end_n=10, iterations=5, model="openai/gpt-4o"):
    """
    Run the pipeline experiment with varying number of stages.
    
    Args:
        start_n: Starting number of stages
        end_n: Ending number of stages
        iterations: Number of iterations per n value
        model: Model to use for evaluation
    
    Returns:
        DataFrame with experiment results
    """
    # Load pipeline specification
    pipeline_spec = json.load(open("./data/pipeline_spec_10.json"))
    
    # Prepare results storage
    results = []
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # For each n value
    for n in range(start_n, end_n + 1):
        print(f"\n===== Running experiments for n={n} =====")
        
        # For each iteration
        for i in tqdm(range(iterations), desc=f"Iterations for n={n}"):
            try:
                # Run the evaluation
                eval_result = eval(
                    process_complete_pipeline(pipeline_spec, n),
                    model=model
                )
                
                # Extract accuracy from the result 
                accuracy = eval_result[0].samples[0].scores['validation_pipeline_scorer'].value
               
                # Store result
                results.append({
                    "n": n,
                    "iteration": i + 1,
                    "accuracy": accuracy
                })
                
                # Save intermediate results to CSV
                pd.DataFrame(results).to_csv(f"results/experiment_results.csv", index=False)
                
                # Add a delay to avoid rate limiting
                time.sleep(5)
                
            except Exception as e:
                print(f"Error in n={n}, iteration={i+1}: {str(e)}")
                # Still record the failure
                results.append({
                    "n": n,
                    "iteration": i + 1,
                    "accuracy": None,
                    "error": str(e)
                })
                # Save intermediate results to CSV
                pd.DataFrame(results).to_csv(f"results/experiment_results.csv", index=False)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save final results
    results_df.to_csv(f"results/experiment_results_final.csv", index=False)
    
    return results_df

def plot_results(results_df, output_path="results/experiment_plot.png"):
    """
    Plot the experiment results with confidence intervals.
    
    Args:
        results_df: DataFrame with experiment results
        output_path: Path to save the plot
    """
    # Group by n and calculate statistics
    stats = results_df.groupby('n')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
    
    # Calculate 95% confidence intervals
    z = 1.96  # 95% confidence
    stats['ci_lower'] = stats['mean'] - z * (stats['std'] / np.sqrt(stats['count']))
    stats['ci_upper'] = stats['mean'] + z * (stats['std'] / np.sqrt(stats['count']))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot mean accuracy line
    plt.plot(stats['n'], stats['mean'], marker='o', linestyle='-', color='blue', label='Mean Accuracy')
    
    # Plot confidence interval area
    plt.fill_between(stats['n'], stats['ci_lower'], stats['ci_upper'], color='blue', alpha=0.2, label='95% Confidence Interval')
    
    # Labels and title
    plt.xlabel('Number of Pipeline Stages (n)')
    plt.ylabel('Accuracy')
    plt.title('Pipeline Accuracy vs. Number of Stages')
    plt.xticks(range(min(stats['n']), max(stats['n'])+1))
    plt.ylim(0, 1.05)  # Set y-axis from 0 to 1.05 for clearer visualization
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plot saved to {output_path}")
    
    # Also generate and save a table of statistics
    stats[['n', 'mean', 'std', 'ci_lower', 'ci_upper']].to_csv(
        output_path.replace('.png', '_stats.csv'), 
        index=False
    )
    
    return stats

def main():
    """Main function to run the experiment."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run pipeline experiment')
    parser.add_argument('--start_n', type=int, default=1, help='Starting number of stages')
    parser.add_argument('--end_n', type=int, default=10, help='Ending number of stages')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations per n value')
    parser.add_argument('--model', type=str, default="openai/gpt-4o", help='Model to use')
    parser.add_argument('--plot_only', action='store_true', help='Only plot existing results')
    
    args = parser.parse_args()
    
    # Check if we should just plot existing results
    if args.plot_only:
        if os.path.exists("results/experiment_results_final.csv"):
            results_df = pd.read_csv("results/experiment_results_final.csv")
            plot_results(results_df)
        else:
            print("No existing results found. Please run the experiment first.")
        return
    
    # Run the experiment
    results_df = run_experiment(
        start_n=args.start_n,
        end_n=args.end_n,
        iterations=args.iterations,
        model=args.model
    )
    
    # Plot the results
    plot_results(results_df)
    
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main() 