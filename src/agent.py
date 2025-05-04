import dotenv
import os
import json
from inspect_ai import Task, task, eval
from inspect_ai.solver import basic_agent, Solver
from inspect_ai.dataset import Sample
from inspect_ai.tool import bash, python
from inspect_ai.scorer import Score, accuracy, stderr, scorer
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
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

python_tool = python()
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def default_solver() -> Solver:
    # the below is the solver we will use if no task-specific solver is provided
    return basic_agent(
        tools=[bash(timeout=120), python(timeout=120)],
        # the submit_description field can be used to provide a hint to the solver about the expected output format
        submit_description="Finished",
    )

# Debug function to log what checks are passing/failing
def debug_print(message):
    print(f"DEBUG: {message}")
    logger.info(f"DEBUG: {message}")

# Validation-based scorer using direct column comparison
@scorer(metrics=[accuracy(), stderr()])
def validation_pipeline_scorer(pipeline_spec, input_path, ground_truth_path, n_stages):
    """
    A scorer that evaluates the accuracy of data transformations by directly comparing 
    affected columns between agent output and ground truth data.
    """
    async def score(state, target):
        answer = state.output.completion
        
        # Default score values
        score_value = 0.0
        details = []
        successful_stages = 0
        total_stages = n_stages
        
        # Check if the output indicates the agent completed the task
        if "Finished" not in answer:
            details.append("Agent did not complete execution")
            return Score(
                value=score_value,
                explanation="Agent did not complete the task",
                answer=answer[:100] + "..." if len(answer) > 100 else answer,
                metadata={"details": details, "successful_stages": 0, "total_stages": total_stages}
            )
        
        # Get the output file path from the pipeline spec
        output_filepath = pipeline_spec['stages'][-1]['parameters']['filepath']
        
        # Check if output file exists
        if not os.path.exists(output_filepath):
            details.append(f"Output file not found: {output_filepath}")
            return Score(
                value=score_value,
                explanation="Output file not found",
                answer=answer[:100] + "..." if len(answer) > 100 else answer,
                metadata={"details": details, "successful_stages": 0, "total_stages": total_stages}
            )
        
        try:
            # Load dataframes
            agent_df = pd.read_csv(output_filepath)
            ground_truth_df = pd.read_csv(ground_truth_path)
            
            # Get the processing stages (skip data loading stage)
            processing_stages = pipeline_spec['stages'][1:n_stages+1]
            
            # For each stage, check if the affected columns match between agent output and ground truth
            stage_results = {}
            
            for i, stage in enumerate(processing_stages):
                stage_id = stage['id']
                stage_name = stage['name']
                stage_params = stage['parameters']
                
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
                
                # Check if columns exist in both dataframes
                missing_cols = [col for col in affected_columns if col not in agent_df.columns or col not in ground_truth_df.columns]
                if missing_cols:
                    stage_results[stage_id] = {
                        'name': stage_name,
                        'success': False,
                        'details': {'message': f'Missing columns: {missing_cols}'}
                    }
                    logger.info(f"Stage {stage_id} ({stage_name}): ✗ Missing columns: {missing_cols}")
                    continue
                
                # Compare each affected column
                columns_match = True
                column_differences = {}
                
                for col in affected_columns:
                    # Skip non-existent columns
                    if col not in agent_df.columns or col not in ground_truth_df.columns:
                        continue
                        
                    # For numeric columns, use approximate comparison
                    if pd.api.types.is_numeric_dtype(agent_df[col]) and pd.api.types.is_numeric_dtype(ground_truth_df[col]):
                        try:
                            # Use numpy's allclose for numeric comparison with tolerance
                            if not np.allclose(agent_df[col].fillna(0).values, 
                                              ground_truth_df[col].fillna(0).values, 
                                              rtol=1e-5, atol=1e-8, equal_nan=True):
                                columns_match = False
                                column_differences[col] = {'message': 'Values do not match within tolerance'}
                                debug_print(f"Column {col} values don't match")
                        except Exception as e:
                            # Fallback to equals if allclose fails
                            if not agent_df[col].equals(ground_truth_df[col]):
                                columns_match = False
                                column_differences[col] = {'message': f'Values do not match: {str(e)}'}
                                debug_print(f"Column {col} equals check failed: {str(e)}")
                    
                    # For non-numeric columns, use exact comparison
                    else:
                        if not agent_df[col].equals(ground_truth_df[col]):
                            columns_match = False
                            column_differences[col] = {'message': 'Values do not match'}
                            debug_print(f"Column {col} (non-numeric) values don't match")
                
                if columns_match:
                    successful_stages += 1
                    stage_results[stage_id] = {
                        'name': stage_name,
                        'success': True,
                        'details': {}
                    }
                    logger.info(f"Stage {stage_id} ({stage_name}): ✓ Columns match")
                else:
                    stage_results[stage_id] = {
                        'name': stage_name,
                        'success': False,
                        'details': {'differences': column_differences}
                    }
                    logger.info(f"Stage {stage_id} ({stage_name}): ✗ Column values don't match")
            
            # Calculate final score
            score_value = successful_stages / total_stages if total_stages > 0 else 0.0
            success_percentage = score_value * 100
            
            explanation = f"Pipeline validation: {successful_stages}/{total_stages} stages correct ({success_percentage:.1f}%)"
            debug_print(explanation)
            
            return Score(
                value=score_value,
                explanation=explanation,
                answer=answer[:100] + "..." if len(answer) > 100 else answer,
                metadata={
                    "details": details,
                    "stage_results": stage_results,
                    "successful_stages": successful_stages,
                    "total_stages": total_stages,
                    "success_percentage": success_percentage
                }
            )
            
        except Exception as e:
            logger.info(f"Validation error: {str(e)}")
            debug_print(f"Error during validation: {str(e)}")
            
            return Score(
                value=0.0,
                explanation=f"Error during validation: {str(e)}",
                answer=answer[:100] + "..." if len(answer) > 100 else answer,
                metadata={"details": details, "error": str(e)}
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
    
    After saving the file successfully, output the string "Finished".
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
