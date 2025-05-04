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

# Simple custom scorer with required metrics parameter
@scorer(metrics=[accuracy(), stderr()])
def simple_pipeline_scorer():
    """
    A simple scorer that checks for basic elements in the agent's code.
    """
    async def score(state, target):
        answer = state.output.completion
        score_value = 0.0
        details = []
        
        # Check for dataframe usage (check for common variable names: df, data, dataframe)
        dataframe_vars = ["df", "data", "dataframe"]
        has_dataframe = any(var in answer for var in dataframe_vars)
        if has_dataframe:
            score_value += 0.2
        
        # Check for pandas import
        pandas_imports = ["import pandas", "from pandas import"]
        has_pandas = any(imp in answer for imp in pandas_imports)
        if has_pandas:
            score_value += 0.2
        
        # Check for data operation keywords
        operation_keywords = ["fillna", "fill_na", "fill", "replace", "na", "missing", "ffill", "bfill", "interpolate"]
        found_operations = [kw for kw in operation_keywords if kw.lower() in answer.lower()]
        if found_operations:
            score_value += 0.2
        
        # Check for CSV save operation
        csv_save_patterns = [".to_csv", "save_csv", "write_csv", "save", "to_csv"]
        has_csv_save = any(pattern in answer for pattern in csv_save_patterns)
        if has_csv_save:
            score_value += 0.2
        
        # Check for task completion
        if "Finished" in answer:
            score_value += 0.2
        
        explanation = "Simple pipeline validation: " + " | ".join(details)
        
        # Add detailed metadata for debugging
        metadata = {
            "details": details,
            "code_snippet": answer[:500] + "..." if len(answer) > 500 else answer,
            "checks": {
                "dataframe_usage": has_dataframe,
                "pandas_import": has_pandas,
                "operation_usage": len(found_operations) > 0,
                "csv_save": has_csv_save,
                "task_completed": "Finished" in answer
            }
        }
        
        debug_print(f"Final score: {score_value}")
        debug_print(f"Check results: {metadata['checks']}")
        
        return Score(
            value=score_value,
            explanation=explanation,
            answer=answer[:100] + "..." if len(answer) > 100 else answer,
            metadata=metadata
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
    logger.info(f"Prompt: {prompt}")

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
    
    return Task(
        dataset=[sample], 
        solver=agent, 
        sandbox="docker", 
        scorer=simple_pipeline_scorer()
    )

if __name__ == "__main__":
    pipeline_spec = json.load(open("./data/pipeline_spec_10.json"))
    n_stages = 5  
    
    print(f"Processing pipeline with {n_stages} operations")
    
    # Run the task with our scorer
    print("Running pipeline evaluation...")
    results = eval(
        process_complete_pipeline(pipeline_spec, n_stages), 
        model="openai/gpt-4o"
    )
