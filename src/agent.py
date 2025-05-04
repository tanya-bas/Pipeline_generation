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
            details.append(f"✓ Uses dataframe variable")
            debug_print(f"Dataframe check passed. Found one of: {dataframe_vars}")
        else:
            details.append("✗ No dataframe variable usage found")
            debug_print("Dataframe check failed")
        
        # Check for pandas import
        pandas_imports = ["import pandas", "from pandas import"]
        has_pandas = any(imp in answer for imp in pandas_imports)
        if has_pandas:
            score_value += 0.2
            details.append("✓ Imports pandas")
            debug_print("Pandas import check passed")
        else:
            details.append("✗ No pandas import found")
            debug_print("Pandas import check failed")
        
        # Check for data operation keywords
        operation_keywords = ["fillna", "fill_na", "fill", "replace", "na", "missing", "ffill", "bfill", "interpolate"]
        found_operations = [kw for kw in operation_keywords if kw.lower() in answer.lower()]
        if found_operations:
            score_value += 0.2
            details.append(f"✓ Uses data operations: {', '.join(found_operations)}")
            debug_print(f"Operations check passed. Found: {found_operations}")
        else:
            details.append("✗ No data operation functions found")
            debug_print("Operations check failed")
        
        # Check for CSV save operation
        csv_save_patterns = [".to_csv", "save_csv", "write_csv", "save", "to_csv"]
        has_csv_save = any(pattern in answer for pattern in csv_save_patterns)
        if has_csv_save:
            score_value += 0.2
            details.append("✓ Saves results to CSV")
            debug_print("CSV save check passed")
        else:
            details.append("✗ No CSV save operation found")
            debug_print("CSV save check failed")
        
        # Check for task completion
        if "Finished" in answer:
            score_value += 0.2
            details.append("✓ Task completed")
            debug_print("Task completion check passed")
        else:
            details.append("✗ Task not marked as completed")
            debug_print("Task completion check failed")
        
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
def process_complete_pipeline(stage_op):
    """Process the complete 3-stage pipeline using Inspect AI."""
    # Log current working directory and available files
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    
    prompt = f"""
    You are a Python data engineer. You need to implement and execute a data pipeline.

    IMPORTANT: Execute ALL of the following steps in a SINGLE Python code block:
    1. Import all necessary libraries
    2. Load the data from: {pipeline_spec['stages'][0]['operation']['parameters']['input_filepath']}
    3. Perfrom the following operation(s): {stage_op['name']}, which requires you to {stage_op['description']} on the following column: {stage_op['parameters']['columns']}
    4. Save the processed data to: {pipeline_spec['stages'][0]['operation']['parameters']['output_filepath']}
    
    Do NOT split your code into multiple separate executions. Write ONE complete script that performs all steps.
    It is VERY important that you save the result file in a single execution, otherwise you get an automatic score of zero!
    
    After saving the file successfully, output the string "Finished".
    """
    
    print("PROMPT:")
    print(prompt)
    print("=" * 50)

    logger.info(f"Input folder: {pipeline_spec['stages'][0]['operation']['parameters']['input_filepath']}")
    logger.info(f"Output folder: {pipeline_spec['stages'][0]['operation']['parameters']['output_filepath']}")

    # Create a basic agent with the available tools
    sample = Sample(input=prompt)
    
    # Create a basic agent with the available tools
    tools = [bash(), python_tool]
    
    agent = basic_agent(
        tools=tools,
        instructions="""
        You are a data engineer agent that implements data pipeline stages.
        Use the available tools to write and execute code to implement all pipeline stages in sequence.
        First load the data from the filepath specified in the operation parameters, then transform it according to the stage requirements, then save it to a filepath specified in the operation parameters.
        """
    )
    
    return Task(
        dataset=[sample], 
        solver=agent, 
        sandbox="docker", 
        scorer=simple_pipeline_scorer()
    )

def display_detailed_results(results):
    """Display detailed results from the evaluation"""
    print("\n" + "="*50)
    print("DETAILED PIPELINE EVALUATION RESULTS")
    print("="*50)
    
    # Check if results has the expected structure
    if not hasattr(results, 'samples') or not results.samples:
        print("No sample results available.")
        return
    
    # Display metrics
    if hasattr(results, 'metrics') and results.metrics:
        print("\nMETRICS:")
        for metric_name, metric_value in results.metrics.items():
            print(f"  {metric_name}: {metric_value}")
    
    # Display sample details
    print("\nSAMPLE DETAILS:")
    for i, sample in enumerate(results.samples):
        print(f"\nSample {i+1}:")
        
        # Display sample score
        if hasattr(sample, 'score'):
            print(f"  Score: {sample.score.value}")
            print(f"  Explanation: {sample.score.explanation}")
            
            # Display metadata if available
            if hasattr(sample.score, 'metadata') and sample.score.metadata:
                print("\n  Detailed checks:")
                
                # Display check results
                if 'checks' in sample.score.metadata:
                    for check_name, check_result in sample.score.metadata['checks'].items():
                        status = "✓" if check_result else "✗"
                        print(f"    {status} {check_name}")
        
        # Display sample output
        if hasattr(sample, 'output') and hasattr(sample.output, 'completion'):
            print(f"\n  Code snippet: {sample.output.completion[:300]}...")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    pipeline_spec = json.load(open("./data/pipeline_spec1.json"))
    stage_op = pipeline_spec['stages'][0]['operation']

    dataframe_path = "./data/input_data.csv"
    
    # Run the task with our scorer
    print("Running pipeline evaluation...")
    results = eval(
        process_complete_pipeline(stage_op), 
        model="openai/gpt-4o"
    )
    
    # Save detailed results to file for inspection
    try:
        with open('pipeline_results.json', 'w') as f:
            # Convert results to dict for JSON serialization
            results_dict = {
                'metrics': results.metrics if hasattr(results, 'metrics') else {},
                'samples': []
            }
            
            # Extract sample data
            if hasattr(results, 'samples'):
                for sample in results.samples:
                    sample_dict = {}
                    
                    if hasattr(sample, 'score'):
                        sample_dict['score'] = {
                            'value': sample.score.value,
                            'explanation': sample.score.explanation
                        }
                        
                        if hasattr(sample.score, 'metadata'):
                            sample_dict['score']['metadata'] = sample.score.metadata
                    
                    if hasattr(sample, 'output') and hasattr(sample.output, 'completion'):
                        sample_dict['code'] = sample.output.completion
                    
                    results_dict['samples'].append(sample_dict)
            
            json.dump(results_dict, f, indent=2)
        print("Detailed results saved to pipeline_results.json")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Display detailed results
    display_detailed_results(results)