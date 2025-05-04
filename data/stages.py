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
         "default_params": {"columns" : "column_1"}
    },

    "outlier_removal": {
        "description": "Remove outliers based on z-score threshold",
        "parameters": ["columns", "threshold"],
        "default_params": {"threshold": 3.0, "columns" : "column_2"},
    },

    "normalize": {
        "description": "Scale values to 0-1 range",
        "parameters": ["columns", "method"],
        "default_params": {"method": "minmax", "columns" : "column_3"},
    },

    "log_transform": {
        "description": "Apply logarithmic transformation to numeric columns",
        "parameters": ["columns", "base"],
        "default_params": {"base": 10, "columns" : "column_4"}
    },

    "clip_values": {
        "description": "Clip values to specified minimum and maximum",
        "parameters": ["columns", "min_value", "max_value"],
        "default_params": {"min_value": 0, "max_value": 100, "columns" : "column_5"}
    },

    "multiply_by": {
        "description": "Multiply column values by a constant",
        "parameters": ["columns"],
        "default_params": {"value": 100, "columns" : "column_6"}  # e.g., for percentage conversion
    },

    "round_values": {
        "description": "Round numeric values to integers",
        "parameters": ["columns"],
        "default_params": {"columns" : "column_7"}
    },

    "one_hot_encode": {
        "description": "One-hot encode categorical columns",
        "parameters": ["columns"],
        "default_params": {"columns" : "fruits"}
    },

    "upper_case": {
        "description": "Convert string columns to uppercase",
        "parameters": ["columns"],
        "default_params": {"columns" : "cities"}
    },

    "time_features": {
        "description": "Extract date/time components from timestamp column",
        "parameters": ["column", "features"],
        "default_params": {"features": ["year", "month", "day"], "column" : "date"}
    },
    
}

SAVING_OPERATIONS = {
    "save_csv": {
        "description": "Save data to a CSV file",
        "parameters": ["filepath"],
        "default_params": {"filepath" : "output_10_steps.csv"}
    },
}