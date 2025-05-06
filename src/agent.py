import dotenv
import os
import json
from inspect_ai import Task, task, eval
from inspect_ai.solver import basic_agent, Solver
from inspect_ai.dataset import Sample
from inspect_ai.tool import bash, python
from inspect_ai.scorer import Score, accuracy, stderr, scorer
from inspect_ai.util import sandbox
import logging
import sys
import pandas as pd
import numpy as np
import io
from pathlib import Path
import time
import traceback
# Import the validator
from validator import PipelineValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pipeline")

# Create a direct file logging method that doesn't rely on Python's logging
VALIDATOR_LOG_FILE = os.path.abspath("validator_trace.log")

def direct_log(message):
    """Write directly to a file without using logging module"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(VALIDATOR_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()  # Force write to disk
    
    # Also print to stdout as a backup
    print(f"[VALIDATOR] {message}")
    sys.stdout.flush()  # Force flush to make sure it appears in logs

# Clear the log file at the start
with open(VALIDATOR_LOG_FILE, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] === VALIDATOR LOG STARTED ===\n")
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Log file path: {VALIDATOR_LOG_FILE}\n")

print(f"Validator log file: {VALIDATOR_LOG_FILE}")

python_tool = python()
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def default_solver() -> Solver:
    # the below is the solver we will use if no task-specific solver is provided
    return basic_agent(
        tools=[bash(timeout=120), python(timeout=120)],
        # provide a more descriptive submit_description to guide the agent
        submit_description="When you have completed implementing all pipeline stages and saved the output file, call submit() with the message 'Finished'. Include this exact word so the system knows you've completed the task."
    )

# Validation-based scorer using direct column comparison
@scorer(metrics=[accuracy(), stderr()])
def validation_pipeline_scorer(pipeline_spec, input_path, ground_truth_path, n_stages):
    """
    A scorer that evaluates the accuracy of data transformations by directly comparing 
    affected columns between agent output and ground truth data.
    """
    async def score(state, target):
        answer = state.output.completion
        
        # Direct logging
        direct_log(f"===== VALIDATION STARTED =====")
        direct_log(f"Input path: {input_path}")
        direct_log(f"Ground truth path: {ground_truth_path}")
        direct_log(f"Number of stages: {n_stages}")
        
        # Default score values
        score_value = 0.0
        details = []
        successful_stages = 0
        total_stages = n_stages
        
        # Check if the output indicates the agent completed the task
        direct_log(f"Checking if agent completed task...")
        if "Finished" not in answer:
            direct_log("FAILED: Agent did not complete execution - 'Finished' not found in output")
            details.append("Agent did not complete execution")
            return Score(
                value=score_value,
                explanation="Agent did not complete the task",
                answer=answer[:100] + "..." if len(answer) > 100 else answer,
                metadata={"details": details, "successful_stages": 0, "total_stages": total_stages, "validator_log_file": VALIDATOR_LOG_FILE}
            )
        
        # Get the output file path from the pipeline spec
        output_filepath = pipeline_spec['stages'][-1]['parameters']['filepath']
        direct_log(f"Output file from spec: {output_filepath}")
        
        # Current directory info for debugging
        cwd = os.getcwd()
        direct_log(f"Current working directory: {cwd}")
        
        # List sandbox directory contents using sandbox().exec
        direct_log("Listing sandbox directory contents...")
        sandbox_ls_result = await sandbox().exec(cmd=["ls", "-la"])
        direct_log(f"Sandbox directory contents: {sandbox_ls_result.stdout}")
        
        # Try to extract the dataframe from CSV markers in agent answer
        agent_df = None
        direct_log("Checking for CSV markers in agent's answer...")
        if "===CSV_START===" in answer and "===CSV_END===" in answer:
            try:
                csv_content = answer.split("===CSV_START===")[1].split("===CSV_END===")[0].strip()
                direct_log(f"Found CSV content, length: {len(csv_content)}")
                direct_log(f"CSV content preview: {csv_content[:200]}...")
                
                # Parse the CSV content into a dataframe
                agent_df = pd.read_csv(io.StringIO(csv_content))
                direct_log(f"Successfully created dataframe from CSV markers: {agent_df.shape}")
                direct_log(f"Columns: {list(agent_df.columns)}")
            except Exception as e:
                direct_log(f"Error parsing CSV content: {str(e)}")
                direct_log(traceback.format_exc())
        else:
            direct_log("CSV markers not found in agent's answer")
        
        # If we couldn't extract a dataframe from CSV markers, try to get it from the sandbox
        if agent_df is None:
            direct_log("Could not extract dataframe from answer, trying to get file from sandbox...")
            
            # Potential paths to try within the sandbox
            potential_paths = [
                output_filepath,                 # As specified
                f"./{output_filepath}",          # With explicit relative path
                f"/tmp/{output_filepath}",       # In tmp
                f"data/{output_filepath}",       # In data subfolder
                os.path.basename(output_filepath) # Just the filename
            ]
            
            # Try each potential path in the sandbox
            output_found = False
            for path in potential_paths:
                direct_log(f"Trying to access {path} in sandbox...")
                cat_result = await sandbox().exec(cmd=["cat", path])
                
                if cat_result.returncode == 0:
                    direct_log(f"FOUND OUTPUT FILE in sandbox at: {path}")
                    
                    try:
                        # Create dataframe from the file content
                        agent_df = pd.read_csv(io.StringIO(cat_result.stdout))
                        direct_log(f"Successfully loaded dataframe from sandbox file: {agent_df.shape}")
                        direct_log(f"Columns: {list(agent_df.columns)}")
                        output_found = True
                        break
                    except Exception as e:
                        direct_log(f"Error parsing file content from sandbox: {str(e)}")
                        direct_log(traceback.format_exc())
                else:
                    direct_log(f"File not found at {path} in sandbox")
            
            if not output_found:
                direct_log("OUTPUT FILE NOT FOUND in sandbox in any of the tried locations")
                
                # Try to extract a dataframe from the printed output as last resort
                try:
                    if "df.head()" in answer:
                        direct_log("Attempting to extract dataframe from df.head() output...")
                        # Try to extract the actual dataframe content after df.head()
                        df_section = answer.split("df.head()")[1].strip()
                        # Find where the dataframe output ends (usually before "Finished")
                        if "Finished" in df_section:
                            df_section = df_section.split("Finished")[0].strip()
                        
                        direct_log(f"Found dataframe section: {df_section[:200]}...")
                        
                        # Try to parse this as a dataframe
                        import re
                        
                        # Extract column names
                        column_line = df_section.split('\n')[0] if '\n' in df_section else ""
                        column_names = re.findall(r'\s+(\w+)', column_line)
                        direct_log(f"Extracted column names: {column_names}")
                        
                        # Extract data rows
                        rows = []
                        for line in df_section.split('\n')[1:]:
                            if line.strip() and re.match(r'^\d+', line.strip()):
                                # Extract values using regex
                                values = re.findall(r'\s+([^\s]+)', line)
                                if values:
                                    rows.append(values)
                        
                        if column_names and rows:
                            # Create a small dataframe from the extracted data
                            agent_df = pd.DataFrame(rows, columns=column_names)
                            direct_log(f"Created extracted dataframe with shape: {agent_df.shape}")
                        else:
                            direct_log("Couldn't extract valid dataframe from printed output")
                except Exception as e:
                    direct_log(f"Error extracting dataframe from answer: {str(e)}")
                    direct_log(traceback.format_exc())
                
                if agent_df is None:
                    details.append(f"Output file not found in sandbox: {output_filepath}")
                    return Score(
                        value=score_value,
                        explanation="Output file not found in sandbox and couldn't extract dataframe from output",
                        answer=answer[:100] + "..." if len(answer) > 100 else answer,
                        metadata={"details": details, "successful_stages": 0, "total_stages": total_stages, "validator_log_file": VALIDATOR_LOG_FILE}
                    )
        
        try:
            # Load ground truth dataframe (from host file system, not sandbox)
            direct_log(f"Loading ground truth dataframe from: {ground_truth_path}")
            ground_truth_df = pd.read_csv(ground_truth_path)
            direct_log(f"Ground truth dataframe loaded: {ground_truth_df.shape}, first few columns: {list(ground_truth_df.columns)[:3]}")
            
            # Check if dataframes were loaded successfully
            if agent_df is None:
                direct_log("ERROR: Agent dataframe is None")
                return Score(
                    value=0.0,
                    explanation="Failed to load agent dataframe",
                    answer=answer[:100] + "..." if len(answer) > 100 else answer,
                    metadata={"details": details, "successful_stages": 0, "total_stages": total_stages, "validator_log_file": VALIDATOR_LOG_FILE}
                )
            
            # Get the processing stages (skip data loading stage)
            processing_stages = pipeline_spec['stages'][1:n_stages+1]
            direct_log(f"Processing {len(processing_stages)} stages: {[s['name'] for s in processing_stages]}")
            
            # For each stage, check if the affected columns match between agent output and ground truth
            stage_results = {}
            
            for i, stage in enumerate(processing_stages):
                stage_id = stage['id']
                stage_name = stage['name']
                stage_params = stage['parameters']
                
                direct_log(f"------- Stage {stage_id}: {stage_name} -------")
                direct_log(f"Parameters: {json.dumps(stage_params)}")
                
                # Get columns affected by this stage
                affected_columns = []
                
                # Handle different stage types
                if stage_name == 'fill_na':
                    columns = stage_params.get('columns', [])
                    if isinstance(columns, str):
                        affected_columns = [columns]
                    else:
                        affected_columns = columns
                
                elif stage_name == 'one_hot_encode':
                    # One-hot encoding creates new columns
                    col = stage_params.get('columns', '')
                    if isinstance(col, list):
                        col = col[0]  # Usually one column is one-hot encoded
                    
                    # Find columns with the prefix in both dataframes
                    agent_one_hot_cols = [c for c in agent_df.columns if c.startswith(f'{col}_')]
                    ground_truth_one_hot_cols = [c for c in ground_truth_df.columns if c.startswith(f'{col}_')]
                    
                    direct_log(f"One-hot encode columns in agent: {agent_one_hot_cols}")
                    direct_log(f"One-hot encode columns in ground truth: {ground_truth_one_hot_cols}")
                    
                    # Only compare columns that exist in both
                    affected_columns = [c for c in agent_one_hot_cols if c in ground_truth_one_hot_cols]
                
                elif stage_name == 'time_features':
                    column = stage_params.get('column', '')
                    features = stage_params.get('features', ['year', 'month', 'day'])
                    affected_columns = [f'{column}_{feature}' for feature in features]
                
                else:
                    # For most other operations, the 'columns' parameter indicates affected columns
                    columns = stage_params.get('columns', [])
                    if isinstance(columns, str):
                        affected_columns = [columns]
                    else:
                        affected_columns = columns
                
                direct_log(f"Affected columns: {affected_columns}")
                
                # Check if columns exist in both dataframes
                missing_cols = [col for col in affected_columns if col not in agent_df.columns or col not in ground_truth_df.columns]
                if missing_cols:
                    direct_log(f"ERROR: Missing columns: {missing_cols}")
                    direct_log(f"Agent columns: {list(agent_df.columns)}")
                    direct_log(f"Ground truth columns: {list(ground_truth_df.columns)}")
                    
                    stage_results[stage_id] = {
                        'name': stage_name,
                        'success': False,
                        'details': {'message': f'Missing columns: {missing_cols}'}
                    }
                    details.append(f"Stage {stage_id} ({stage_name}): ✗ Missing columns: {missing_cols}")
                    continue
                
                # Compare each affected column
                columns_match = True
                column_differences = {}
                
                for col in affected_columns:
                    # Skip non-existent columns
                    if col not in agent_df.columns or col not in ground_truth_df.columns:
                        continue
                    
                    direct_log(f"Comparing column: {col}")
                    
                    # For numeric columns, use approximate comparison
                    if pd.api.types.is_numeric_dtype(agent_df[col]) and pd.api.types.is_numeric_dtype(ground_truth_df[col]):
                        try:
                            # Log samples for comparison
                            agent_sample = agent_df[col].head(3).tolist()
                            gt_sample = ground_truth_df[col].head(3).tolist()
                            direct_log(f"Sample values - Agent: {agent_sample}, Ground Truth: {gt_sample}")
                            
                            # Use numpy's allclose for numeric comparison with tolerance
                            comparison_result = np.allclose(
                                agent_df[col].fillna(0).values, 
                                ground_truth_df[col].fillna(0).values, 
                                rtol=1e-5, atol=1e-8, equal_nan=True
                            )
                            
                            if not comparison_result:
                                columns_match = False
                                column_differences[col] = {'message': 'Values do not match within tolerance'}
                                direct_log(f"FAIL: Column {col} numeric values don't match within tolerance")
                                
                                # Log statistics to help identify the issue
                                agent_stats = {
                                    'mean': float(agent_df[col].mean()),
                                    'min': float(agent_df[col].min()),
                                    'max': float(agent_df[col].max()),
                                    'null_count': int(agent_df[col].isna().sum())
                                }
                                gt_stats = {
                                    'mean': float(ground_truth_df[col].mean()),
                                    'min': float(ground_truth_df[col].min()),
                                    'max': float(ground_truth_df[col].max()),
                                    'null_count': int(ground_truth_df[col].isna().sum())
                                }
                                direct_log(f"Column {col} stats - Agent: {agent_stats}, Ground Truth: {gt_stats}")
                            else:
                                direct_log(f"PASS: Column {col} values match within tolerance")
                        except Exception as e:
                            # Fallback to equals if allclose fails
                            direct_log(f"ERROR in numeric comparison for {col}: {str(e)}")
                            if not agent_df[col].equals(ground_truth_df[col]):
                                columns_match = False
                                column_differences[col] = {'message': f'Values do not match: {str(e)}'}
                                direct_log(f"FAIL: Column {col} equals check failed: {str(e)}")
                    
                    # For non-numeric columns, use exact comparison
                    else:
                        equal_result = agent_df[col].equals(ground_truth_df[col])
                        if not equal_result:
                            columns_match = False
                            column_differences[col] = {'message': 'Values do not match'}
                            direct_log(f"FAIL: Column {col} (non-numeric) values don't match")
                            
                            # Find the first few differences
                            try:
                                diff_mask = agent_df[col] != ground_truth_df[col]
                                diff_count = diff_mask.sum()
                                direct_log(f"Number of different values: {diff_count} out of {len(agent_df)}")
                                
                                diff_indices = diff_mask[diff_mask].index.tolist()[:3]
                                if diff_indices:
                                    for idx in diff_indices:
                                        agent_val = str(agent_df.loc[idx, col])
                                        gt_val = str(ground_truth_df.loc[idx, col])
                                        direct_log(f"Row {idx}: Agent='{agent_val}', GT='{gt_val}'")
                            except Exception as e:
                                direct_log(f"Error finding differences: {str(e)}")
                        else:
                            direct_log(f"PASS: Column {col} values match exactly")
                
                if columns_match:
                    successful_stages += 1
                    stage_results[stage_id] = {
                        'name': stage_name,
                        'success': True,
                        'details': {}
                    }
                    direct_log(f"SUCCESS: Stage {stage_id} ({stage_name}) - All columns match")
                    details.append(f"Stage {stage_id} ({stage_name}): ✓ Columns match")
                else:
                    stage_results[stage_id] = {
                        'name': stage_name,
                        'success': False,
                        'details': {'differences': column_differences}
                    }
                    direct_log(f"FAILURE: Stage {stage_id} ({stage_name}) - Columns with differences: {list(column_differences.keys())}")
                    details.append(f"Stage {stage_id} ({stage_name}): ✗ Column values don't match")
            
            # Calculate final score
            score_value = successful_stages / total_stages if total_stages > 0 else 0.0
            success_percentage = score_value * 100
            
            explanation = f"Pipeline validation: {successful_stages}/{total_stages} stages correct ({success_percentage:.1f}%)"
            direct_log(f"===== VALIDATION COMPLETE =====")
            direct_log(explanation)
            
            return Score(
                value=score_value,
                explanation=explanation,
                answer=answer[:100] + "..." if len(answer) > 100 else answer,
                metadata={
                    "details": details,
                    "stage_results": stage_results,
                    "successful_stages": successful_stages,
                    "total_stages": total_stages,
                    "success_percentage": success_percentage,
                    "validator_log_file": VALIDATOR_LOG_FILE
                }
            )
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            direct_log(f"CRITICAL ERROR: {str(e)}")
            direct_log(f"Traceback: {error_traceback}")
            
            details.append(f"Validation error: {str(e)}")
            return Score(
                value=0.0,
                explanation=f"Error during validation: {str(e)}",
                answer=answer[:100] + "..." if len(answer) > 100 else answer,
                metadata={"details": details, "error": str(e), "traceback": error_traceback, "validator_log_file": VALIDATOR_LOG_FILE}
            )

    return score

@task
def process_complete_pipeline(pipeline_spec, n):
    """Process a pipeline with n stages using Inspect AI.
    
    Args:
        pipeline_spec: The complete pipeline specification
        n: Number of processing stages to include (not counting load/save)
    """
    # Log current working directory and available files
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    
    # Always include load and save operations
    input_filepath = pipeline_spec['data_path']
    output_filepath = pipeline_spec['stages'][-1]['parameters']['filepath']
    
    # Get the first n processing stages (skip the first stage which is loading)
    stages_to_process = pipeline_spec['stages'][1:n+1]
    
    # Build the operation steps for the prompt
    operations_text = ""
    for i, stage in enumerate(stages_to_process):
        # Format each operation step
        op_text = f"   {i+1}. Perform {stage['name']}: {stage['description']} "
        
        # Add parameters info
        params = []
        for param_name, param_value in stage['parameters'].items():
            params.append(f"{param_name}={param_value}")
        
        if params:
            op_text += f"with parameters: {', '.join(params)}"
        
        operations_text += op_text + "\n"
    
    prompt = f"""
    You are a Python data engineer. You need to implement and execute a data pipeline with {n} processing operations.

    IMPORTANT: Execute ALL of the following steps in a SINGLE Python code block:
    1. Import all necessary libraries
    2. Load the data from: {input_filepath}
{operations_text}
    {n+2}. Save the processed data to: {output_filepath}
    
    Do NOT split your code into multiple separate executions. Write ONE complete script that performs all steps.
    It is VERY important that you save the result file in a single execution, otherwise you get an automatic score of zero!
    
    After saving the file successfully:
    Output the string "Finished".
    """

    logger.info(f"Input file: {input_filepath}")
    logger.info(f"Output file: {output_filepath}")
    logger.info(f"Processing {n} stages")
    # logger.info(f"Prompt: {prompt}")

    # Create a basic agent with the available tools
    sample = Sample(input=prompt)
    
    # Create a basic agent with the available tools
    tools = [bash(), python_tool]
    
    agent = basic_agent(
        tools=tools,
        instructions="""
        You are a data engineer agent that implements data pipeline stages.
        Use the available tools to write and execute code to implement all pipeline stages in sequence.
        First load the data from the specified filepath, then transform it according to the stage requirements, then save it to the specified output filepath.
        """
    )
    
    # Get file paths
    input_path = pipeline_spec['data_path']
    ground_truth_path = "/Users/tetianabas/Documents/Pipeline_generation/data/ground_truth_data.csv"
    pipeline_spec_path = "./data/pipeline_spec_10.json"
    
    return Task(
        dataset=[sample], 
        solver=agent, 
        sandbox="docker", 
        scorer=validation_pipeline_scorer(pipeline_spec, input_path, ground_truth_path, n)
    )

if __name__ == "__main__":
    pipeline_spec = json.load(open("./data/pipeline_spec_10.json"))
    input_path = "/Users/tetianabas/Documents/Pipeline_generation/data/input_data.csv"
    ground_truth_path = "/Users/tetianabas/Documents/Pipeline_generation/data/ground_truth_data.csv"

    n_stages = 1
    print(f"Processing pipeline with {n_stages} operations")
    
    # Run the task with our scorer
    print("Running pipeline evaluation...")
    results = eval(
        process_complete_pipeline(pipeline_spec, n_stages), 
        model="openai/gpt-4o"
    )
