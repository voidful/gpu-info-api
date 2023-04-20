import itertools
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import requests

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

referencesAtEnd = r"(?:\s*\[\d+\])+(?:\d+,)?(?:\d+)?$"


def clean_html(html: str) -> str:
    html = re.sub(r'<span [^>]*style="display:none[^>]*>([^<]+)</span>', "", html)
    html = re.sub(r"<span[^>]*>([^<]+)</span>", r"\1", html)
    html = re.sub(r"(\d)&#160;(\d)", r"\1\2", html)
    html = re.sub(r"&thinsp;", "", html)
    html = re.sub(r"&#8201;", "", html)
    html = re.sub("\xa0", " ", html)
    html = re.sub(r"&#160;", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
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
    df.columns = [
        " ".join(
            a
            for a, b in itertools.zip_longest(col, col[1:])
            if (a != b and not a.startswith("Unnamed: "))
        ).strip()
        for col in df.columns.values
    ]

    df.columns = [re.sub(" Arc [\w]*$", "", col) for col in df.columns.values]

    df.columns = [
        " ".join([re.sub("[\d,]+$", "", word) for word in col.split()])
        for col in df.columns.values
    ]

    df.columns = [
        " ".join(
            [re.sub(r"(?:\[[a-zA-Z0-9]+\])+$", "", word) for word in col.split()]
        )
        for col in df.columns.values
    ]

    df.columns = [col.replace("- ", "") for col in df.columns.values]
    df.columns = [col.replace("/ ", "/") for col in df.columns.values]
    df.columns = df.columns.str.strip()

    df["Vendor"] = vendor

    if ("Launch" not in df.columns.values) and (
            "Release Date & Price" in df.columns.values
    ):
        df["Launch"] = df["Release Date & Price"].str.extract(
            r"^(.*\d\d\d\d)", expand=False
        )
        df["Release Price (USD)"] = df["Release Date & Price"].str.extract(
            r"(\$[\d,]+)", expand=False
        )

    if "Launch" not in df.columns.values:
        print("Launch not in following df:\n", df)
    df["Launch"] = df["Launch"].apply(lambda x: str(x))
    df["Launch"] = df["Launch"].str.replace(referencesAtEnd, "", regex=True)
    df["Launch"] = df["Launch"].str.extract("^(.*?[\d]{4})", expand=False)
    df["Launch"] = df["Launch"].apply(
        lambda x: pd.to_datetime(x, infer_datetime_format=True, errors="coerce")
    )

    if [c for c in Counter(df.columns).items() if c[1] > 1]:
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def merge_columns(df, dst, src, replaceNoWithNaN=False, delete=True):
    """Merge two columns, replacing NaNs in dst with values from src"""
    if src not in df.columns.values:
        return df
    df[src] = df[src].replace("\u2014", np.nan)  # em-dash
    if replaceNoWithNaN:
        df[src] = df[src].replace("No", np.nan)
    df[dst] = df[dst].fillna(df[src])
    if delete:
        df.drop(src, axis=1, inplace=True)
    return df


def filter_rows_by_pattern(df, col, pattern):
    return df[~df[col].str.contains(pattern, re.UNICODE, na=False)]


def extract_shader_count(df):
    df["Pixel/unified shader count"] = (
        df["Core config"]
            .str.split(":")
            .str[0]
            .str.split("(")
            .str[0]
    )
    df["Pixel/unified shader count"] = pd.to_numeric(
        df["Pixel/unified shader count"], downcast="integer", errors="coerce"
    )
    df = merge_columns(df, "Pixel/unified shader count", "Stream processors")
    df = merge_columns(df, "Pixel/unified shader count", "Shaders Cuda cores (total)")
    df = merge_columns(df, "Pixel/unified shader count", "Shading units")  # Intel
    return df


def extract_sm_count(df):
    df["SM count (extracted)"] = df["Core config"].str.extract(
        r"\((\d+ SM[MX])\)", expand=False
    )
    df = merge_columns(df, "SM count", "SM count (extracted)")
    df["SM count (extracted)"] = df["Core config"].str.extract(
        r"(\d+) CU", expand=False
    )
    df = merge_columns(df, "SM count", "SM count (extracted)")
    df["SM count (extracted)"] = df["Core config"].str.extract(
        r"\((\d+)\)", expand=False
    )
    df = merge_columns(df, "SM count", "SM count (extracted)")
    df = merge_columns(df, "SM count", "SMX count")
    df = merge_columns(df, "SM count", "Execution units")  # Intel
    return df


def extract_fab_nm(df):
    df["Architecture (Fab) (extracted)"] = df["Architecture (Fab)"].str.extract(
        r"\((\d+) nm\)", expand=False
    )
    df = merge_columns(df, "Fab (nm)", "Architecture (Fab) (extracted)")
    df["Architecture (Fab) (extracted)"] = df["Architecture & Fab"].str.extract(
        r"(\d+) nm", expand=False
    )
    df = merge_columns(df, "Fab (nm)", "Architecture (Fab) (extracted)")

    for fab in [
        "TSMC",
        "GloFo",
        "Samsung/GloFo",
        "Samsung",
        "SGS",
        "SGS/TSMC",
        "IBM",
        "UMC",
        "TSMC/UMC",
    ]:
        df["Architecture (Fab) (extracted)"] = df["Architecture & Fab"].str.extract(
            r"%s (\d+)" % fab, expand=False
        )
        df = merge_columns(df, "Fab (nm)", "Architecture (Fab) (extracted)")
        df["Fab (nm)"] = df["Fab (nm)"].str.replace(fab, "").str.replace("nm$", "", regex=True)

    df["Architecture (Fab) (extracted)"] = df["Process"].str.extract(
        r"(\d+)", expand=False
    )
    df = merge_columns(df, "Fab (nm)", "Architecture (Fab) (extracted)")
    return df


def process_release_price(df):
    df["Release Price (USD)"] = (
        df["Release Price (USD)"]
            .str.replace(r"[,\$]", "", regex=True)
            .str.split(" ")
            .str[0]
    )
    for col in [
        "Memory Bandwidth (GB/s)",
        "TDP (Watts)",
        "Fab (nm)",
        "Release Price (USD)",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    for vendor in ["NVIDIA", "AMD", "Intel"]:
        html = requests.get(data[vendor]["url"]).text
        cleaned_html = clean_html(html)
        dfs = pd.read_html(
            cleaned_html,
            match=re.compile("Launch|Release Date & Price"),
            parse_dates=True,
        )
        processed_dfs = [process_dataframe(df, vendor) for df in dfs]
        data[vendor]["dfs"] = processed_dfs

    df = pd.concat(
        data["NVIDIA"]["dfs"] + data["AMD"]["dfs"] + data["Intel"]["dfs"],
        sort=False,
        ignore_index=True,
    )

    # Merge related columns using the merge_columns function
    merge_columns_list = [
        ("Model", "Model Units"),
        ("Processing power (TFLOPS) Single precision", "Processing power (TFLOPS)"),
        ("Processing power (TFLOPS) Single precision", "Processing power (TFLOPS) Single precision (MAD+MUL)", True),
        ("Processing power (TFLOPS) Single precision", "Processing power (TFLOPS) Single precision (MAD or FMA)", True),
        ("Processing power (TFLOPS) Double precision", "Processing power (TFLOPS) Double precision (FMA)", True),
        ("Memory Bandwidth (GB/s)", "Memory configuration Bandwidth (GB/s)"),
        ("TDP (Watts)", "TDP (Watts) Max."),
        ("TDP (Watts)", "TDP (Watts) Max"),
        ("TDP (Watts)", "TBP (W)"),
        ("TDP (Watts)", "TDP (W)"),
        ("TDP (Watts)", "Combined TDP Max. (W)"),
        ("TDP (Watts)", "TDP /idle (Watts)"),
        ("Model", "Model (Codename)"),
        ("Model", "Model (Code name)"),
        ("Model", "Model (codename)"),
        ("Model", "Code name (console model)"),
        ("Core clock (MHz)", "Shaders Base clock (MHz) MHz"),
        ("Core clock (MHz)", "Shader clock (MHz)"),
        ("Core clock (MHz)", "Clock rate Base (MHz)"),
        ("Core clock (MHz)", "Clock rate (MHz)"),
        ("Core clock (MHz)", "Clock speeds Base core clock (MHz)"),
        ("Core clock (MHz)", "Core Clock (MHz)"),
        ("Core clock (MHz)", "Clock rate Core (MHz)"),
        ("Core clock (MHz)", "Clock speed Core (MHz)"),
        ("Core clock (MHz)", "Clock speed Average (MHz)"),
        ("Core clock (MHz)", "Core Clock rate (MHz)"),
        ("Core clock (MHz)", "Clock rate (MHz) Core (MHz)"),
        ("Core clock (MHz)", "Clock speed Shader (MHz)"),
        ("Core clock (MHz)", "Clock speeds  Base core (MHz)"),
        ("Core clock (MHz)", "Core Clock (MHz) Base"),
        ("Core config", "Core Config"),
        ("Transistors Die Size", "Transistors & Die Size"),
        ("Transistors Die Size", "Transistors & die size"),
        ("Memory Bus type", "Memory RAM type"),
        ("Memory Bus type", "Memory Type"),
        ("Memory Bus type", "Memory configuration DRAM type"),
        ("Memory Bus width (bit)", "Memory configuration Bus width (bit)"),
        ("Release Price (USD)", "Release price (USD)"),
        ("Release Price (USD)", "Release price (USD) MSRP"),
        ("NVLink Support", "NVLink support"),
    ]

    for dst, src, *args in merge_columns_list:
        df = merge_columns(df, dst, src, *args)

    # Clean 'Release Price (USD)' column
    df["Release Price (USD)"] = df["Release Price (USD)"].str.extract(
        r"\$?([\d,]+)", expand=False
    )

    # Filter rows containing specific patterns
    filter_rows_list = [
        ("Chips", r"^[2-9]\u00d7"),
        ("Code name", r"^[2-9]\u00d7"),
        ("Core config", r"^[2-9]\u00d7"),
        ("Model", "[xX]2$"),
        ("Transistors (million)", r"\u00d7[2-9]$"),
        ("Die size (mm2)", r"\u00d7[2-9]$"),
    ]

    for col, pattern in filter_rows_list:
        df = filter_rows_by_pattern(df, col, pattern)

    # Refactor Processing power (GFLOPS) columns
    for prec in ["Single", "Double", "Half"]:
        for col in [
            f"Processing power (GFLOPS) {prec} precision",
            f"Processing power (GFLOPS) {prec} precision (base)",
            f"Processing power (GFLOPS) {prec} precision (boost)",
        ]:
            if col in df.columns.values:
                destcol = f"Processing power (TFLOPS) {prec} precision"
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace(",", "")  # get rid of commas
                df[col] = df[col].str.extract(r"^([\d\.]+)", expand=False)
                df[col] = pd.to_numeric(df[col]) / 1000.0  # change to TFLOPS
                df = merge_columns(df, destcol, col)

    # Merge TFLOPS columns with "Boost" column headers and rename
    for prec in ["Single", "Double", "Half"]:
        col = f"Processing power (TFLOPS) {prec} precision"
        spcol = "Single-precision TFLOPS"

        merge_columns_list = [
            # ("Processing power (TFLOPS) %s precision Base Core (Base Boost) (Max Boost 3.0)", ),
            # ("Processing power (TFLOPS) %s precision R/F.E Base Core Reference (Base Boost) F.E. (Base Boost) R/F.E. (Max Boost 4.0)", ),
            ("Processing power (TFLOPS) %s",),
        ]

        for src_col in merge_columns_list:
            df = merge_columns(df, col, src_col[0] % prec)

        if prec != "Single":
            df.loc[df[col] == "1/16 SP", col] = pd.to_numeric(df[spcol]) / 16
            df.loc[df[col] == "2x SP", col] = pd.to_numeric(df[spcol]) * 2

        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(",", "")
        df[col] = df[col].str.extract(r"^([\d\.]+)", expand=False)
        df = df.rename(columns={col: f"{prec}-precision TFLOPS"})

    # Split out 'transistors die size'
    for exponent in ["\u00d7106", "\u00d7109", "B"]:
        dftds = df["Transistors Die Size"].str.extract(
            f"^([\d\.]+){exponent} (\d+) mm2", expand=True
        )
        if exponent == "\u00d7106":
            df["Transistors (million)"] = df["Transistors (million)"].fillna(
                pd.to_numeric(dftds[0], errors="coerce")
            )
        if exponent == "\u00d7109" or exponent == "B":
            df["Transistors (billion)"] = df["Transistors (billion)"].fillna(
                pd.to_numeric(dftds[0], errors="coerce")
            )
        df["Die size (mm2)"] = df["Die size (mm2)"].fillna(
            pd.to_numeric(dftds[1], errors="coerce")
        )

    df["Core clock (MHz)"] = df["Core clock (MHz)"].astype(str).str.split(" ").str[0]
    df["Memory Bus width (bit)"] = df["Memory Bus width (bit)"].astype(str).str.split(" ").str[0]
    df["Memory Bus width (bit)"] = df["Memory Bus width (bit)"].astype(str).str.split("/").str[0]
    df["Memory Bus width (bit)"] = df["Memory Bus width (bit)"].astype(str).str.split(",").str[0]

    # Strip out bit width from combined column
    df = merge_columns(df, "Memory Bus type & width (bit)", "Memory Bus type & width")
    df["bus"] = df["Memory Bus type & width (bit)"].str.extract("(\d+)-bit", expand=False)
    df["bus"] = df["bus"].fillna(pd.to_numeric(df["bus"], errors="coerce"))
    df = merge_columns(df, "Memory Bus width (bit)", "bus", delete=False)

    # Collate memory bus type and take first word only, removing chud as appropriate
    df = merge_columns(df, "Memory Bus type", "Memory Bus type & width (bit)", delete=False)
    df["Memory Bus type"] = df["Memory Bus type"].str.split(" ").str[0]
    df["Memory Bus type"] = df["Memory Bus type"].str.split(",").str[0]
    df["Memory Bus type"] = df["Memory Bus type"].str.split("/").str[0]
    df["Memory Bus type"] = df["Memory Bus type"].str.split("[").str[0]
    df.loc[df["Memory Bus type"] == "EDO", "Memory Bus type"] = "EDO VRAM"

    # Merge transistor counts
    df["Transistors (billion)"] = df["Transistors (billion)"].fillna(
        pd.to_numeric(df["Transistors (million)"], errors="coerce") / 1000.0
    )
    df = extract_shader_count(df)
    df = extract_sm_count(df)
    df = extract_fab_nm(df)
    df = process_release_price(df)

    # remove references from end of model/transistor names
    for col in ["Model", "Transistors (million)"]:
        df[col] = df[col].str.replace(referencesAtEnd, "", regex=True)
        # then take 'em out of the middle too
        df[col] = df[col].str.replace(r"\[\d+\]", "", regex=True)

    # mark mobile processors
    df["GPU Type"] = np.where(
        df["Model"].str.contains(r" [\d]+M[X]?|\(Notebook\)"), "Mobile", "Desktop"
    )
    rdict = {}
    for i in df.to_dict(orient="records"):
        dummy_dict = {}
        for k, v in i.items():
            if not v != v:
                dummy_dict[k] = v
        rdict[i["Code name"]] = dummy_dict
    with open("gpu.json", 'w', encoding='utf-8') as outfile:
        json.dump(rdict, outfile, indent=4, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
