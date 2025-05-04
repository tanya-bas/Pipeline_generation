import os
import json
import random
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from openai import OpenAI
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
import time
import json


from stages import OPERATIONS, DATA_LOADERS, SAVING_OPERATIONS


class PipelineGenerator:
    def __init__(self, n_preprocessing_steps: int, seed=None):
        """
        Initialize the pipeline generator with exact number of preprocessing steps.
        
        Args:
            n_preprocessing_steps: Exact number of preprocessing steps to include
            seed: Random seed for reproducibility
        """
        if seed:
            random.seed(seed)
        self.n_steps = n_preprocessing_steps
        self.operations = OPERATIONS
        self.pipeline_spec = {}
        
        # Define column mappings for default parameters
        self.column_mappings = {
            "column_1": "column_1",
            "column_2": "column_2",
            "column_3": "column_3",
            "column_4": "column_4",
            "column_5": "column_5",
            "column_6": "column_6",
            "column_7": "column_7",
            "column_8": "column_8",
            "fruits": "fruits",
            "cities": "cities",
            "date": "date"
        }
        
    def _fix_column_references(self, params: Dict) -> Dict:
        """Convert column references to actual column names"""
        fixed_params = params.copy()
        
        # Fix column parameter
        if "columns" in fixed_params and not isinstance(fixed_params["columns"], list):
            col_ref = str(fixed_params["columns"])
            if col_ref in self.column_mappings:
                fixed_params["columns"] = [self.column_mappings[col_ref]]
        
        # Fix single column parameter
        if "column" in fixed_params and not isinstance(fixed_params["column"], str):
            col_ref = str(fixed_params["column"])
            if col_ref in self.column_mappings:
                fixed_params["column"] = self.column_mappings[col_ref]
                
        return fixed_params
        
    def generate_pipeline(self, data_path: str) -> Dict:
        """Generate a deterministic data processing pipeline using the first n operations"""
        # Generate pipeline stages
        pipeline = {
            "name": f"Generated Pipeline ({self.n_steps} steps)",
            "description": "Data pipeline with fixed number of preprocessing steps",
            "data_path": data_path,
            "stages": []
        }

        # 1. Always start with the first DATA_LOADER
        loader_name = list(DATA_LOADERS.keys())[0]
        loader_config = DATA_LOADERS[loader_name]
        pipeline["stages"].append({
            "id": "stage_0",
            "name": loader_name,
            "description": loader_config["description"],
            "parameters": {"filepath": data_path}
        })

        # 2. Add operations in order from OPERATIONS
        op_names = list(self.operations.keys())
        for i in range(min(self.n_steps, len(op_names))):
            op_name = op_names[i]
            op_config = self.operations[op_name]
            
            # Get default parameters and fix column references
            params = op_config.get("default_params", {}).copy()
            params = self._fix_column_references(params)
            
            pipeline["stages"].append({
                "id": f"stage_{i+1}",
                "name": op_name,
                "description": op_config.get("description", ""),
                "parameters": params
            })

        # 3. Always end with save
        saver_name = list(SAVING_OPERATIONS.keys())[0]
        saver_config = SAVING_OPERATIONS[saver_name]
        
        pipeline["stages"].append({
            "id": f"stage_{len(pipeline['stages'])}",
            "name": saver_name,
            "description": saver_config["description"],
            "parameters": {"filepath": f"output_{self.n_steps}_steps.csv"}
        })

        self.pipeline_spec = pipeline
        return pipeline

    def get_pipeline_spec(self) -> Dict:
        """Return the generated pipeline specification"""
        return self.pipeline_spec
    
    def save_pipeline_spec(self, output_path: str) -> None:
        """
        Save the pipeline specification to a JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(self.pipeline_spec, f, indent=2)


if __name__ == "__main__":
    # Generate pipeline
    pipeline_generator = PipelineGenerator(n_preprocessing_steps=10)
    pipeline = pipeline_generator.generate_pipeline("./data/input_data.csv")
    pipeline_generator.save_pipeline_spec("./data/pipeline_spec_10.json")
    
    # Print generated pipeline for verification
    print("Pipeline generated and saved to pipeline_spec.json")
    print("\nPipeline specification:")
    print(json.dumps(pipeline, indent=2))