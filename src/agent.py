import dotenv
import os
import json
from inspect_ai import Task, task, eval
from inspect_ai.solver import basic_agent, Solver
from inspect_ai.dataset import Sample
from inspect_ai.tool import bash, python
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
        tools=[bash(timeout=60), python(timeout=60)],
        # the submit_description field can be used to provide a hint to the solver about the expected output format
        submit_description="Finished",
    )

@task
def process_complete_pipeline(stage_op):
    """Process the complete 3-stage pipeline using Inspect AI."""
    # Log current working directory and available files
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    
    prompt = f"""
    You are a Python data engineer. You need to implement and execute a data pipeline.
    Here is the name of the operation: {stage_op['name']}, here is the description: {stage_op['description']}
    load the data from the follwoing location {pipeline_spec['stages'][0]['operation']['parameters']['input_filepath']}
    Here is the name of the coolumn that opeation will be applied to: {stage_op['parameters']['columns']}
    save the data to the follwoing location {pipeline_spec['stages'][0]['operation']['parameters']['output_filepath']}
    Once you finished the operation, submit the string "Finished".  
    """

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
    
    return Task(dataset=[sample], solver=agent, sandbox="docker")

if __name__ == "__main__":
    pipeline_spec = json.load(open("./data/pipeline_spec1.json"))
    stage_op = pipeline_spec['stages'][0]['operation']

    dataframe_path = "./data/input_data.csv"
    
    # Run the evaluation with the agent
    results = eval(process_complete_pipeline(stage_op), model="openai/gpt-4o")
    
    # The agent will have executed the code and shown the results in its response
    print("Complete pipeline processing complete.")