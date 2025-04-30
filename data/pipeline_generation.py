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
        self.used_operations = set()
        self.used_columns = set()
        
        # Define available columns based on data generation
        self.numeric_columns = [f"column_{i+1}" for i in range(8)]
        self.categorical_columns = ["fruits", "cities"]
        self.date_columns = ["date"]
        self.all_columns = self.numeric_columns + self.categorical_columns + self.date_columns

    def _get_available_columns(self, operation_name: str) -> List[str]:
        """Get available columns suitable for the operation type"""
        # Special case for specific operations
        if operation_name == "one_hot_encode":
            return ["fruits"] if "fruits" not in self.used_columns else []
        elif operation_name == "upper_case":
            return ["cities"] if "cities" not in self.used_columns else []
        elif operation_name in ["time_features"]:
            return [col for col in self.date_columns if col not in self.used_columns]
        else:  # Operations that work on numeric columns
            return [col for col in self.numeric_columns if col not in self.used_columns]

    def generate_pipeline(self, data_path: str) -> Dict:
        """Generate a random data processing pipeline based on difficulty with no repeating operations"""
        self.used_operations = set()
        self.used_columns = set()

        total_operations = min(self.n_steps, len(self.operations))

        # Generate pipeline stages
        pipeline = {
            "name": f"Generated Pipeline ({self.n_steps} steps)",
            "description": "Data pipeline with fixed number of preprocessing steps",
            "data_path": data_path,
            "stages": []
        }

        # 1. Always start with one from DATA_LOADERS
        loader_name = random.choice(list(DATA_LOADERS.keys()))
        loader = self._create_operation(loader_name, DATA_LOADERS[loader_name])
        loader["parameters"]["filepath"] = data_path
        pipeline["stages"].append({
            "id": "stage_0",
            "name": "Data Loading",
            "operation": loader
        })

        # 2. Always make fill_na the first operation after loading
        fill_na_config = self.operations["fill_na"]
        fill_na_operation = self._create_operation("fill_na", fill_na_config)
        pipeline["stages"].append({
            "id": "stage_1",
            "name": "Fill Missing Values",
            "operation": fill_na_operation
        })
        self.used_operations.add("fill_na")

        # 3. Add remaining operations
        available_ops = [op for op in list(self.operations.keys()) if op != "fill_na"]
        random.shuffle(available_ops)
        
        remaining_ops = min(total_operations - 1, len(available_ops))  # -1 for fill_na
        
        for i in range(remaining_ops):
            op_name = available_ops[i]
            op_config = self.operations[op_name]
            operation = self._create_operation(op_name, op_config)

            pipeline["stages"].append({
                "id": f"stage_{i+2}",  
                "name": f"Process Step {i+2}",
                "operation": operation
            })

            self.used_operations.add(op_name)

        # 3. Always end with save
        
        saver_name = random.choice(list(SAVING_OPERATIONS.keys()))
        saver = self._create_operation(saver_name, SAVING_OPERATIONS[saver_name])
        saver["parameters"]["filepath"] = f"output_{self.n_steps}_steps.csv"
        pipeline["stages"].append({
            "id": f"stage_{self.n_steps+1}",
            "name": "Save Data",
            "operation": saver
        })

        self.pipeline_spec = pipeline
        return pipeline

    def _create_operation(self, op_name: str, op_config: Dict) -> Dict:
        """Create an operation with appropriate parameters"""
        params = op_config.get("default_params", {}).copy()
        
        # Handle column selection based on operation type
        if "columns" in op_config.get("parameters", []):
            # Special cases for specific operations
            if op_name == "one_hot_encode":
                params["columns"] = ["fruits"]
            elif op_name == "upper_case":
                params["columns"] = ["cities"]
            elif op_name == "time_features":
                params["columns"] = ["date"]
            else:
                # For other operations, find an unused numeric column
                available_cols = [col for col in self.numeric_columns 
                                 if col not in self.used_columns]
                
                if available_cols:
                    selected_column = random.choice(available_cols)
                    params["columns"] = [selected_column]
                    self.used_columns.add(selected_column)
                else:
                    # If all columns used, just pick a random numeric column
                    params["columns"] = [random.choice(self.numeric_columns)]
        
        # Operation-specific parameter logic
        if op_name == "fill_na":
            params["value"] = random.choice([0, -1, "mean", "median", "mode"])
            params["method"] = random.choice(["constant", "ffill", "bfill"]) if params["value"] in ["mean", "median", "mode"] else "constant"
        elif op_name == "normalize":
            params["method"] = random.choice(["minmax", "zscore", "robust"])
        elif op_name == "one_hot_encode":
            params["drop_first"] = random.choice([True, False])
        elif op_name == "correlation":
            params["method"] = random.choice(["pearson", "spearman"])

        return {
            "name": op_name,
            "description": op_config.get("description", ""),
            "parameters": params
        }

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
    pipeline_generator = PipelineGenerator(n_preprocessing_steps=5)
    pipeline = pipeline_generator.generate_pipeline("./data/input_data.csv")
    pipeline_generator.save_pipeline_spec("./data/pipeline_spec.json")
    
    # Print generated pipeline for verification
    print("Pipeline generated and saved to pipeline_spec.json")
    print("\nPipeline specification:")
    print(json.dumps(pipeline, indent=2))