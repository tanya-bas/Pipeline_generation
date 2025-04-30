import pandas as pd
import numpy as np
import random


def generate_sample_data(filepath: str) -> None:
    """Generate sample data for pipeline testing"""
    rows = 10_000
    columns = 10

    df = pd.DataFrame()

    for i in range(columns - 2):
        if i % 3 == 0:  # Some columns with missing values
            values = np.random.normal(50, 15, rows)
            mask = np.random.random(rows) < 0.1  # 10% missing values
            values[mask] = np.nan
            df[f"column_{i+1}"] = values
        else:
            df[f"column_{i+1}"] = np.random.normal(50, 15, rows)

    # Generate categorical column
    fruits = ["Apples", "Bananas", "Cherries", "Dates", "Elderberries"]
    df["fruits"] = np.random.choice(fruits, rows)

    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Miami"]
    df["cities"] = np.random.choice(cities, rows)


    # Generate date column
    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2023-12-31")
    date_range = (end_date - start_date).days
    dates = [start_date + pd.Timedelta(days=random.randint(0, date_range)) for _ in range(rows)]
    df["date"] = dates

    # Add outliers
    outlier_indices = np.random.choice(rows, size=int(rows * 0.05), replace=False)
    for idx in outlier_indices:
        col = f"column_{random.randint(1, columns-2)}"
        df.loc[idx, col] = df[col].mean() + random.randint(5, 10) * df[col].std()

    # Save to file
    df.to_csv(filepath, index=False)
    print(f"Sample data created at {filepath}")
    return df


if __name__ == "__main__":
    df = generate_sample_data("./data/input_data.csv")