# Pipeline Generation Experiment

This repository contains code for running and evaluating LLM-based data pipeline generation.

## Overview

The system uses a large language model to generate data processing pipelines based on specifications. Each pipeline consists of a number of stages that transform data in specific ways.

## Scripts

- `src/agent.py`: Main implementation of the agent and validation logic
- `src/validator.py`: Contains validation logic for individual pipeline stages
- `src/experiment.py`: Script for running multiple experiments and plotting results
- `src/run_single.py`: Script for running a single experiment (useful for testing)

## Running Experiments

### Single Experiment

To run a single experiment with a specific number of stages:

```bash
cd Pipeline_generation
python src/run_single.py --n 3 --model openai/gpt-4o
```

Options:
- `--n`: Number of pipeline stages to include (default: 1)
- `--model`: Model to use for evaluation (default: openai/gpt-4o)

### Full Experiment Suite

To run a full experiment suite with n from 1 to 10, with 5 iterations each:

```bash
cd Pipeline_generation
python src/experiment.py --start_n 1 --end_n 10 --iterations 5 --model openai/gpt-4o
```

Options:
- `--start_n`: Starting number of stages (default: 1)
- `--end_n`: Ending number of stages (default: 10)
- `--iterations`: Number of iterations per n value (default: 5)
- `--model`: Model to use for evaluation (default: openai/gpt-4o)
- `--plot_only`: Only plot existing results, don't run new experiments

### Plot Only

If you've already run experiments and just want to regenerate the plots:

```bash
cd Pipeline_generation
python src/experiment.py --plot_only
```

## Results

Results are saved in the `results/` directory:
- `experiment_results.csv`: Intermediate results (updated after each iteration)
- `experiment_results_final.csv`: Complete results after all experiments
- `experiment_plot.png`: Plot showing accuracy vs. number of stages with confidence intervals
- `experiment_plot_stats.csv`: Table with statistical data (mean, std, confidence intervals)

## Interpreting Results

The experiment measures how model accuracy changes with increasing pipeline complexity (more stages). The plot shows:

- **Mean accuracy** for each value of n (blue line)
- **95% confidence intervals** (blue shaded area)

A wider confidence interval indicates more variability in performance for that n value.

## Implementation Details

The system:
1. Loads a pipeline specification with n stages
2. Prompts the model to implement the pipeline
3. Validates the output by applying the same transformations to the input data
4. Calculates accuracy by comparing expected and actual outputs

The accuracy metric represents the fraction of stages that were implemented correctly.