# Pipeline Generation Experiment

This repository contains code for running and evaluating LLM-based data pipeline generation.

## Overview

The system uses a large language model to generate data processing pipelines based on specifications. Each pipeline consists of a number of stages that transform data in specific ways. The system validates the model's output by applying the same transformations to the input data and comparing with the expected results.

## Setup Instructions

### Environment Setup

You can set up the environment using either Poetry or pip:

#### Using Poetry (Recommended)

1. Ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed
2. Clone this repository and navigate to the root directory
3. Install dependencies:
   ```bash
   cd Pipeline_generation
   poetry install
   ```
4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

#### Using pip (requirements file is used by Docker mainly but can also work for env setup)

1. Clone this repository and navigate to the root directory
2. Create a virtual environment (optional but recommended):
   ```bash
   cd Pipeline_generation
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### API Key Setup

Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Directory Structure

```
Pipeline_generation/
├── data/                       # Data files and generation scripts
│   ├── data_generation.py      # Script to generate sample data
│   ├── input_data.csv          # Input data for the pipeline
│   ├── pipeline_generation.py  # Generates pipeline specifications
│   ├── pipeline_spec_10.json   # Full pipeline with 10 stages
│   ├── stages.py               # Definitions of pipeline operations
├── logs/                       # Logs directory
├── results/                    # Results and visualization outputs
├── src/                        # Source code
│   ├── agent.py                # Main agent implementation and validation
│   ├── experiment.py           # Script for running experiments
│   ├── validator.py            # Pipeline validation logic
├── Dockerfile                  # Docker configuration
├── pyproject.toml              # Poetry dependencies
├── requirements.txt            # pip dependencies
└── README.md                   # This file
```

## File Descriptions

### Data Files

- **data_generation.py**: Generates a synthetic dataset with 10,000 rows, multiple numeric columns, categorical columns, and a date column. It introduces missing values and outliers to test various transformations.
- **stages.py**: Defines all available pipeline operations, their descriptions, parameters, and default values.
- **pipeline_generation.py**: Generates pipeline specifications with a configurable number of preprocessing steps.
- **pipeline_spec_10.json**: A full pipeline specification with 10 transformation stages.

### Source Files

- **agent.py**: Implements the agent-based approach to pipeline generation and validation. Contains the `process_complete_pipeline` task and the validation scoring logic.
- **validator.py**: Contains the `PipelineValidator` class which implements and validates all transformation operations.
- **experiment.py**: Script for running experiments with different numbers of pipeline stages and visualizing results.

## Running the Code

### 1. Generate Sample Data (optional)

If you want to generate fresh sample data:

```bash
cd Pipeline_generation
python data/data_generation.py
```

This will create a new `input_data.csv` file in the data directory.

### 2. Generate Pipeline Specifications (optional)

If you want to create new pipeline specifications:

```bash
cd Pipeline_generation
python data/pipeline_generation.py
```

This will create a pipeline specification with default operations.

### 3. Running a Single Experiment

To run a single experiment with a specific number of stages:

```bash
cd Pipeline_generation
python src/experiment.py --start_n 3 --end_n 3 --iterations 1 --model openai/gpt-4o
```

Options:
- `--start_n`: Starting number of stages (default: 1)
- `--end_n`: Ending number of stages (default: 10)
- `--iterations`: Number of iterations per n value (default: 5)
- `--model`: Model to use for evaluation (default: openai/gpt-4o)

### 4. Running a Full Experiment Suite

To run a full experiment suite with n from 1 to 10, with 5 iterations each:

```bash
cd Pipeline_generation
python src/experiment.py --start_n 1 --end_n 10 --iterations 5 --model openai/gpt-4o
```

### 5. Plot Only Mode

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

### Pipeline Validation Process

1. The system loads the pipeline specification with n stages
2. It prompts the model to implement all stages in a single Python program
3. The validation system:
   - Loads the input data
   - Dynamically applies each transformation stage to the data
   - Compares the model's output with the expected output for each stage
   - Calculates accuracy as the fraction of stages implemented correctly

### Transformation Types

The system supports the following transformations:
- **fill_na**: Fill missing values with a specified value
- **outlier_removal**: Remove outliers based on z-score threshold
- **normalize**: Scale values to 0-1 range
- **log_transform**: Apply logarithmic transformation
- **clip_values**: Clip values to specified min/max
- **multiply_by**: Multiply column values by a constant
- **round_values**: Round numeric values to integers
- **upper_case**: Convert string columns to uppercase
- **time_features**: Extract date/time components
- **one_hot_encode**: One-hot encode categorical columns

## Troubleshooting

- **Missing API Key**: Ensure your OpenAI API key is set in the `.env` file
- **Import Errors**: Check that all dependencies are installed correctly
- **File Not Found Errors**: Ensure you're running the commands from the root directory
- **Validation Errors**: Check the validator log file for detailed error information

## Docker Support

The repository includes a Dockerfile to run the experiments in a containerized environment:

```bash
docker build -t pipeline-generation .
docker run -it --env-file .env pipeline-generation
```