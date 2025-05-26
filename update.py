import itertools
import json
import re
from collections import Counter
from io import StringIO

import numpy as np
import pandas as pd
import requests

# Define the URLs for each vendor's Wikipedia page
data = {
    "NVIDIA": {
        "url": "https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units"
    },
    "AMD": {
        "url": "https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units",
    },
    "Intel": {
        "url": "https://en.wikipedia.org/wiki/Intel_Xe",
    },
}

# Regular expression to remove references at the end of strings
REFERENCES_AT_END = r"(?:\s*\[\d+\])+(?:\d+,)?(?:\d+)?$"

def clean_html(html: str) -> str:
    """Clean the HTML content by removing unnecessary tags and special characters"""
    html = re.sub(r'<span [^>]*style="display:none[^>]*>([^<]+)</span>', "", html)
    html = re.sub(r"<span[^>]*>([^<]+)</span>", r"\1", html)
    html = re.sub(r"(\d) (\d)", r"\1\2", html)
    html = re.sub(r" ", "", html)
    html = re.sub("\xa0", " ", html)
    html = re.sub(r" ", " ", html)
    html = re.sub(r"<br />", " ", html)
    html = re.sub("\u2012", "-", html)
    html = re.sub("\u2013", "-", html)
    html = re.sub("\u2014", "", html)
    html = re.sub(r"mm<sup>2</sup>", "mm2", html)
    html = re.sub("\u00d710<sup>6</sup>", "\u00d7106", html)
    html = re.sub("\u00d710<sup>9</sup>", "\u00d7109", html)
    html = re.sub(r"<sup>\d+</sup>", "", html)
    return html

def process_dataframe(df: pd.DataFrame, vendor: str) -> pd.DataFrame:
    """Process the DataFrame by cleaning column names and standardizing data"""
    # Clean column names by removing duplicates and unnecessary parts
    df.columns = [
        " ".join(
            a for a, b in itertools.zip_longest(col, col[1:])
            if (a != b and not a.startswith("Unnamed: "))
        ).strip()
        for col in df.columns.values
    ]
    # Use raw strings for regular expressions
    df.columns = [re.sub(r" Arc [\w]*$", "", col) for col in df.columns.values]
    df.columns = [
        " ".join([re.sub(r"[\d,]+$", "", word) for word in col.split()])
        for col in df.columns.values
    ]
    df.columns = [
        " ".join([re.sub(r"(?:\[[a-zA-Z0-9]+\])+$", "", word) for word in col.split()])
        for col in df.columns.values
    ]
    df.columns = [col.replace("- ", "") for col in df.columns.values]
    df.columns = [col.replace("/ ", "/") for col in df.columns.values]
    df.columns = df.columns.str.strip()

    # Add the vendor column
    df["Vendor"] = vendor

    # Handle the 'Launch' column
    if ("Launch" not in df.columns) and ("Release Date & Price" in df.columns):
        df["Launch"] = df["Release Date & Price"].str.extract(r"^(.*\d\d\d\d)", expand=False)
        df["Release Price (USD)"] = df["Release Date & Price"].str.extract(r"(\$[\d,]+)", expand=False)

    if "Launch" not in df.columns:
        print(f"Warning: 'Launch' column not found in {vendor}'s DataFrame")
        df["Launch"] = pd.NaT  # Set to NaT if 'Launch' column is missing

    # Clean and convert 'Launch' column to datetime format
    df["Launch"] = df["Launch"].astype(str)
    df["Launch"] = df["Launch"].str.replace(REFERENCES_AT_END, "", regex=True)
    df["Launch"] = df["Launch"].str.extract(r"^(.*?[\d]{4})", expand=False)
    df["Launch"] = pd.to_datetime(df["Launch"], errors="coerce")

    # Remove duplicate columns
    if [c for c in Counter(df.columns).items() if c[1] > 1]:
        df = df.loc[:, ~df.columns.duplicated()]

    return df

def merge_columns(df, dst, src, replace_no_with_nan=False, delete=True):
    """Merge two columns by filling NaN values in dst with values from src"""
    if src not in df.columns:
        return df
    df[src] = df[src].replace("\u2014", np.nan)  # Replace em-dash with NaN
    if replace_no_with_nan:
        df[src] = df[src].replace("No", np.nan)
    df[dst] = df[dst].fillna(df[src])
    if delete:
        df.drop(src, axis=1, inplace=True)
    return df

def remove_bracketed_references(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Remove bracketed references like [274] from specified columns"""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\[\d+\]", "", regex=True).str.strip()
    return df

def main():
    all_dfs = []
    for vendor in ["NVIDIA", "AMD", "Intel"]:
        # Fetch HTML and handle potential network errors
        try:
            response = requests.get(data[vendor]["url"])
            response.raise_for_status()  # Raise an exception for HTTP errors
            html = response.text
        except requests.RequestException as e:
            print(f"Error fetching data for {vendor}: {e}")
            continue

        cleaned_html = clean_html(html)
        # Parse HTML tables
        try:
            dfs = pd.read_html(
                StringIO(cleaned_html),
                match=re.compile("Launch|Release Date & Price"),
                parse_dates=True,
            )
        except ValueError as e:
            print(f"No tables found for {vendor}: {e}")
            continue

        processed_dfs = [process_dataframe(df, vendor) for df in dfs]
        all_dfs.extend(processed_dfs)

    # If no DataFrames are collected, exit
    if not all_dfs:
        print("No DataFrames to concatenate. Exiting.")
        return

    # Concatenate all DataFrames
    df = pd.concat(all_dfs, sort=False, ignore_index=True)

    # Clean specific columns from bracketed references
    df = remove_bracketed_references(df, ["Model", "Die size (mm2)"])

    # Convert to dictionary and output as JSON
    result_dict = {}
    records = df.to_dict(orient="records")
    for row in records:
        temp_dict = {}
        for k, v in row.items():
            if not pd.isna(v):  # Filter out NaN values
                temp_dict[k] = v
        code_name = row.get("Code name", "Unknown")
        # Use vendor and code name combination to ensure unique keys
        key = f"{row['Vendor']}_{code_name}"
        result_dict[key] = temp_dict

    with open("gpu.json", "w", encoding="utf-8") as outfile:
        json.dump(result_dict, outfile, indent=4, ensure_ascii=False, default=str)

if __name__ == "__main__":
    main()