import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional


class PipelineValidator:
    """
    A simplified validator that scores pipeline stages by comparing original and modified CSV files.
    """
    def __init__(self, pipeline_spec_path: str):
        """
        Initialize the validator with the pipeline specification.
        
        Args:
            pipeline_spec_path: Path to the pipeline specification JSON file
        """
        # Load pipeline specification
        with open(pipeline_spec_path, 'r') as f:
            self.pipeline_spec = json.load(f)
        
        self.stages = self.pipeline_spec['stages']
        self.results = {
            'score': 0,
            'total_stages': len(self.stages) - 2,  # Exclude data loading and saving stages
            'stage_results': {}
        }
    
    def validate(self, original_csv_path: str, modified_csv_path: str) -> Dict:
        """
        Validate the pipeline by applying each stage to the original CSV and comparing with modified CSV.
        
        Args:
            original_csv_path: Path to the original CSV file
            modified_csv_path: Path to the modified CSV file (after pipeline execution)
            
        Returns:
            Dict containing validation results and score
        """
        # Load dataframes
        try:
            original_df = pd.read_csv(original_csv_path)
            modified_df = pd.read_csv(modified_csv_path)
        except Exception as e:
            return {
                'score': 0,
                'total_stages': self.results['total_stages'],
                'error': f'Failed to load CSV files: {str(e)}',
                'stage_results': {}
            }
        
        # Make a copy of the original dataframe to apply transformations
        current_df = original_df.copy()
        
        # Skip first stage (data loading) and last stage (saving)
        processing_stages = self.stages[1:-1]
        
        for i, stage in enumerate(processing_stages):
            stage_id = stage['id']
            stage_name = stage['name']
            stage_params = stage['parameters']
            
            try:
                # Apply the stage transformation
                current_df = self._apply_transformation(current_df, stage_name, stage_params)
                
                # Check if the transformation is correctly reflected in the modified CSV
                validation_result = self._validate_transformation(current_df, modified_df, stage_name, stage_params)
                
                if validation_result['success']:
                    self.results['score'] += 1
                
                self.results['stage_results'][stage_id] = {
                    'name': stage_name,
                    'success': validation_result['success'],
                    'details': validation_result.get('details', {})
                }
                
            except Exception as e:
                self.results['stage_results'][stage_id] = {
                    'name': stage_name,
                    'success': False,
                    'error': str(e)
                }
        
        # Add success percentage
        self.results['success_percentage'] = (self.results['score'] / self.results['total_stages']) * 100 if self.results['total_stages'] > 0 else 0
        
        return self.results
    
    def _apply_transformation(self, df: pd.DataFrame, operation: str, params: Dict) -> pd.DataFrame:
        """
        Apply a transformation operation to the dataframe.
        
        Args:
            df: Input dataframe
            operation: Operation name
            params: Operation parameters
            
        Returns:
            Transformed dataframe
        """
        result_df = df.copy()
        
        # Extract columns parameter (could be a single string or a list)
        columns = params.get('columns', [])
        if isinstance(columns, str):
            columns = [columns]
        
        # Apply the appropriate transformation based on operation name
        if operation == 'fill_na':
            value = params.get('value', 0)
            method = params.get('method', 'constant')
            
            if method == 'constant':
                result_df[columns] = result_df[columns].fillna(value)
            elif method in ['ffill', 'bfill']:
                result_df[columns] = result_df[columns].fillna(method=method)
            elif value in ['mean', 'median', 'mode']:
                for col in columns:
                    if value == 'mean':
                        fill_value = result_df[col].mean()
                    elif value == 'median':
                        fill_value = result_df[col].median()
                    elif value == 'mode':
                        fill_value = result_df[col].mode()[0]
                    result_df[col] = result_df[col].fillna(fill_value)
        
        elif operation == 'outlier_removal':
            threshold = params.get('threshold', 3.0)
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    result_df = result_df[(result_df[col] <= mean + threshold * std) & 
                                         (result_df[col] >= mean - threshold * std)]
        
        elif operation == 'normalize':
            method = params.get('method', 'minmax')
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    if method == 'minmax':
                        min_val = result_df[col].min()
                        max_val = result_df[col].max()
                        if max_val > min_val:
                            result_df[col] = (result_df[col] - min_val) / (max_val - min_val)
                    elif method == 'zscore':
                        mean = result_df[col].mean()
                        std = result_df[col].std()
                        if std > 0:
                            result_df[col] = (result_df[col] - mean) / std
        
        elif operation == 'log_transform':
            base = params.get('base', 10)
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    # Handle zero and negative values
                    min_val = result_df[col].min()
                    if min_val <= 0:
                        offset = abs(min_val) + 1
                        result_df[col] = np.log(result_df[col] + offset) / np.log(base)
                    else:
                        result_df[col] = np.log(result_df[col]) / np.log(base)
        
        elif operation == 'clip_values':
            min_value = params.get('min_value', 0)
            max_value = params.get('max_value', 100)
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    result_df[col] = result_df[col].clip(min_value, max_value)
        
        elif operation == 'one_hot_encode':
            for col in columns:
                if col in result_df.columns:
                    dummies = pd.get_dummies(result_df[col], prefix=col)
                    result_df = pd.concat([result_df, dummies], axis=1)
                    if params.get('drop_first', False):
                        result_df = result_df.drop(dummies.columns[0], axis=1)
        
        elif operation == 'multiply_by':
            value = params.get('value', 1)
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    result_df[col] = result_df[col] * value
        
        elif operation == 'round_values':
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    result_df[col] = result_df[col].round()
        
        elif operation == 'upper_case':
            for col in columns:
                if result_df[col].dtype == 'object':
                    result_df[col] = result_df[col].str.upper()
        
        elif operation == 'time_features':
            column = params.get('column', '')
            features = params.get('features', ['year', 'month', 'day'])
            
            if column in result_df.columns:
                try:
                    # Convert to datetime
                    result_df[column] = pd.to_datetime(result_df[column])
                    
                    # Extract requested features
                    for feature in features:
                        if feature == 'year':
                            result_df[f'{column}_year'] = result_df[column].dt.year
                        elif feature == 'month':
                            result_df[f'{column}_month'] = result_df[column].dt.month
                        elif feature == 'day':
                            result_df[f'{column}_day'] = result_df[column].dt.day
                        elif feature == 'weekday':
                            result_df[f'{column}_weekday'] = result_df[column].dt.weekday
                        elif feature == 'hour':
                            result_df[f'{column}_hour'] = result_df[column].dt.hour
                except Exception:
                    pass  # Skip if datetime conversion fails
        
        return result_df
    
    def _validate_transformation(self, transformed_df: pd.DataFrame, modified_df: pd.DataFrame, 
                                operation: str, params: Dict) -> Dict:
        """
        Validate if a specific transformation is correctly reflected in the modified dataframe.
        
        Args:
            transformed_df: Dataframe after applying transformations up to current stage
            modified_df: The final modified dataframe
            operation: Operation name
            params: Operation parameters
            
        Returns:
            Dict with validation results
        """
        # Extract columns parameter (could be a single string or a list)
        columns = params.get('columns', [])
        if isinstance(columns, str):
            columns = [columns]
            
        # For time_features operation, we use the column parameter instead
        if operation == 'time_features':
            column = params.get('column', '')
            features = params.get('features', ['year', 'month', 'day'])
            
            # Check if the derived columns exist in both dataframes
            derived_columns = [f'{column}_{feature}' for feature in features]
            columns_exist = all(col in transformed_df.columns and col in modified_df.columns 
                               for col in derived_columns)
            
            if not columns_exist:
                return {
                    'success': False,
                    'details': {
                        'message': 'Missing derived time feature columns',
                        'expected_columns': derived_columns
                    }
                }
            
            # Check values in derived columns
            columns_match = True
            differences = {}
            
            for col in derived_columns:
                if not pd.api.types.is_numeric_dtype(transformed_df[col]) or not pd.api.types.is_numeric_dtype(modified_df[col]):
                    if not transformed_df[col].equals(modified_df[col]):
                        columns_match = False
                        differences[col] = {
                            'message': 'Values do not match'
                        }
                else:
                    # For numeric columns, use approximate comparison
                    if not np.allclose(transformed_df[col].fillna(0).values, 
                                     modified_df[col].fillna(0).values, 
                                     rtol=1e-5, atol=1e-8, equal_nan=True):
                        columns_match = False
                        differences[col] = {
                            'message': 'Values do not match within tolerance'
                        }
            
            return {
                'success': columns_match,
                'details': {
                    'differences': differences if not columns_match else {}
                }
            }
        
        # Handle one_hot_encode separately as it creates new columns
        elif operation == 'one_hot_encode':
            if not columns:
                return {'success': False, 'details': {'message': 'No columns specified'}}
                
            col = columns[0]  # Usually one-hot encoding is applied to a single column
            
            # Check if one-hot encoded columns exist in both dataframes
            # Get unique values in the original column to determine expected one-hot columns
            unique_values = transformed_df[col].unique()
            expected_columns = [f'{col}_{val}' for val in unique_values]
            
            # Check if the derived columns exist
            encoded_cols_exist = all(any(col_name.startswith(f'{col}_') for col_name in modified_df.columns)
                                   for val in unique_values)
            
            if not encoded_cols_exist:
                return {
                    'success': False, 
                    'details': {
                        'message': 'Missing one-hot encoded columns',
                        'expected_pattern': f'{col}_*'
                    }
                }
            
            # Simple check - just verify that some one-hot encoded columns exist
            one_hot_cols = [c for c in modified_df.columns if c.startswith(f'{col}_')]
            return {
                'success': len(one_hot_cols) > 0,
                'details': {
                    'one_hot_columns_found': one_hot_cols
                }
            }
        
        # For other operations, validate the affected columns
        else:
            # Check if all specified columns exist in both dataframes
            columns_exist = all(col in transformed_df.columns and col in modified_df.columns for col in columns)
            
            if not columns_exist:
                return {'success': False, 'details': {'message': 'Missing columns'}}
            
            # For numeric operations, check if values are close enough
            columns_match = True
            differences = {}
            
            for col in columns:
                # Skip non-numeric columns for numeric operations
                if operation in ['normalize', 'log_transform', 'clip_values', 'multiply_by', 'round_values', 'outlier_removal']:
                    if not pd.api.types.is_numeric_dtype(transformed_df[col]) or not pd.api.types.is_numeric_dtype(modified_df[col]):
                        continue
                
                # For string operations, check exact equality
                if operation == 'upper_case':
                    if transformed_df[col].dtype == 'object' and modified_df[col].dtype == 'object':
                        # Check if strings are uppercase in both dataframes
                        if not all(str(val).upper() == str(val) for val in modified_df[col] if pd.notna(val)):
                            columns_match = False
                            differences[col] = {
                                'message': 'Not all strings are uppercase'
                            }
                    continue
                
                # For fill_na, check if NaN values are handled
                if operation == 'fill_na':
                    # Check if there are no NaN values in the modified dataframe
                    if modified_df[col].isna().sum() > 0:
                        columns_match = False
                        differences[col] = {
                            'message': 'NaN values still present',
                            'nan_count': int(modified_df[col].isna().sum())
                        }
                    continue
                
                # For numeric columns, use approximate comparison
                if pd.api.types.is_numeric_dtype(transformed_df[col]) and pd.api.types.is_numeric_dtype(modified_df[col]):
                    # Handle special case for round_values - check if values are integers
                    if operation == 'round_values':
                        if not all(float(val).is_integer() for val in modified_df[col].dropna()):
                            columns_match = False
                            differences[col] = {
                                'message': 'Not all values are integers'
                            }
                        continue
                    
                    # For other numeric operations, compare with tolerance
                    if not np.allclose(transformed_df[col].fillna(0).values, 
                                     modified_df[col].fillna(0).values, 
                                     rtol=1e-5, atol=1e-8, equal_nan=True):
                        columns_match = False
                        differences[col] = {
                            'message': 'Values do not match within tolerance'
                        }
            
            return {
                'success': columns_match,
                'details': {
                    'differences': differences if not columns_match else {}
                }
            }
    
    def save_results(self, output_path: str) -> None:
        """
        Save validation results to a JSON file.
        
        Args:
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)


if __name__ == "__main__":
    # Example usage
    validator = SimplePipelineValidator("./data/pipeline_spec_10.json")
    results = validator.validate("./data/input_data.csv", "./data/output_10_steps.csv")
    validator.save_results("./data/validation_results.json")
    
    print(f"Validation Score: {results['score']}/{results['total_stages']} ({results['success_percentage']:.1f}%)")
    print("Detailed results saved to validation_results.json")