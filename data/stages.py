DATA_LOADERS = {
    "csv_loader": {
        "description": "Load data from a CSV file",
        "parameters": ["filepath"],
        "default_params": {}
    },
}

OPERATIONS = {

    "fill_na": {
        "description": "Fill missing values with a specified value",
         "parameters": ["columns"],
         "default_params": {}
    },
    
    # "remove_duplicates": {
    #     "description": "Remove duplicate rows",
    #     "parameters": ["columns"],
    #     "default_params": {}
    # },

    "outlier_removal": {
        "description": "Remove outliers based on z-score threshold",
        "parameters": ["columns", "threshold"],
        "default_params": {"threshold": 3.0},
    },

    "normalize": {
        "description": "Scale values to 0-1 range",
        "parameters": ["columns", "method"],
        "default_params": {"method": "minmax"},
    },

    "one_hot_encode": {
        "description": "One-hot encode categorical columns",
        "parameters": ["columns"],
        "default_params": {}
    },

    # "create_ratio": {
    #     "description": "Create a new column as ratio of two existing columns",
    #     "parameters": ["numerator", "denominator", "new_column_name"],
    #     "default_params": {}
    # },


    "time_features": {
        "description": "Extract date/time components from timestamp column",
        "parameters": ["column", "features"],
        "default_params": {"features": ["year", "month", "day"]}
    },


    # "correlation": {
    #     "description": "Calculate correlation between columns",
    #     "parameters": ["columns"],
    #     "default_params": {}
    # },


    "multiply_by": {
        "description": "Multiply column values by a constant",
        "parameters": ["columns"],
        "default_params": {"value": 100}  # e.g., for percentage conversion
    },

    "round_values": {
        "description": "Round numeric values to integers",
        "parameters": ["columns"],
        "default_params": {}
    },

    "upper_case": {
        "description": "Convert string columns to uppercase",
        "parameters": ["columns"],
        "default_params": {}
    },
    

}

SAVING_OPERATIONS = {
    "save_csv": {
        "description": "Save data to a CSV file",
        "parameters": ["filepath"],
        "default_params": {}
    },
}