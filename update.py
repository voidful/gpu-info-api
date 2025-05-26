import itertools
import json
import re
from collections import Counter
from io import StringIO

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

REFERENCES_AT_END = r"(?:\s*\[\d+\])+(?:\d+,)?(?:\d+)?$"

def clean_html(html: str) -> str:
    html = re.sub(r'(colspan|rowspan)="(\d+)[^"]*"', r'\1="\2"', html)
    html = re.sub(r'<sup[^>]*>.*?</sup>', '', html, flags=re.DOTALL)
    html = re.sub(r'<br\s*/?>', ' ', html)
    html = re.sub(r'<th([^>]*)>', lambda m: '<th'+m.group(1)+'>', html)
    html = re.sub(r'<span [^>]*style="display:none[^>]*>([^<]+)</span>', "", html)
    html = re.sub(r"<span[^>]*>([^<]+)</span>", r"\1", html)
    html = re.sub(r"(\d)&#160;(\d)", r"\1\2", html)
    html = re.sub(r"&thinsp;", "", html)
    html = re.sub(r"&#8201;", "", html)
    html = re.sub("\xa0", " ", html)
    html = re.sub(r"&#160;", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"<small>.*?</small>", "", html, flags=re.DOTALL)
    html = re.sub(r"\u2012", "-", html)
    html = re.sub("\u2013", "-", html)
    html = re.sub("\u2014", "", html)
    html = re.sub(r"mm<sup>2</sup>", "mm2", html)
    html = re.sub("\u00d710<sup>6</sup>", "\u00d7106", html)
    html = re.sub("\u00d710<sup>9</sup>", "\u00d7109", html)
    html = re.sub(r"<sup>\d+</sup>", "", html)
    return html

def normalize_columns(cols):
    # 支援多層欄位，合併成 "主欄位 副欄位"
    if isinstance(cols, pd.MultiIndex):
        new_cols = []
        for col in cols.values:
            names = [str(x).strip() for x in col if str(x) != "nan" and not str(x).startswith("Unnamed")]
            name = " ".join(names)
            new_cols.append(name)
        return new_cols
    else:
        return [str(col).strip() for col in cols]

def fetch_gpu_table(url):
    print(f"Fetching from: {url}")
    resp = requests.get(url)
    html = resp.text
    html = clean_html(html)
    all_gpus = []
    # 先嘗試雙層header，失敗 fallback 單層
    try:
        dfs = pd.read_html(StringIO(html), header=[0,1], flavor="bs4")
    except ValueError:
        dfs = pd.read_html(StringIO(html), header=0, flavor="bs4")
    for df in dfs:
        # 欄位名稱整理
        df.columns = normalize_columns(df.columns)
        if df.shape[1] < 2 or df.shape[0] < 1:
            continue
        all_gpus.append(df)
    return all_gpus

def process_dataframe(df: pd.DataFrame, vendor: str) -> pd.DataFrame:
    # 保證所有欄位都是 str，避免 re.sub 因 bytes 出錯
    df.columns = [str(col) for col in df.columns]
    # 標題清理，將 Arc、重複、雜訊移除
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
    df.columns = [col.strip() for col in df.columns]
    df["Vendor"] = vendor

    # 嘗試找出 launch/release date 欄位
    possible_launch_cols = [c for c in df.columns if "Launch" in c or "Release Date" in c]
    launch_col = None
    if possible_launch_cols:
        launch_col = possible_launch_cols[0]
    if not launch_col:
        print("Launch not in following df:\n", df)
        df["Launch"] = pd.NaT
    else:
        df[launch_col] = df[launch_col].astype(str)
        df[launch_col] = df[launch_col].str.replace(REFERENCES_AT_END, "", regex=True)
        df[launch_col] = df[launch_col].str.extract(r"([A-Za-z]+\s*\d{1,2},?\s*\d{4}|\d{4})", expand=False)
        df["Launch"] = pd.to_datetime(df[launch_col], errors="coerce")
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df

def remove_bracketed_references(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\[\d+\]", "", regex=True).str.strip()
    return df

def main():
    all_dfs = []
    for vendor in data.keys():
        gpu_tables = fetch_gpu_table(data[vendor]["url"])
        for df in gpu_tables:
            if df.shape[0] < 2 or df.shape[1] < 3:
                continue
            processed_df = process_dataframe(df, vendor)
            all_dfs.append(processed_df)
    if not all_dfs:
        print("No tables found!")
        return
    df = pd.concat(all_dfs, sort=False, ignore_index=True)
    df = remove_bracketed_references(df, ["Model", "Model name", "Model (Codename)", "Model (Code name)", "Die size (mm2)", "Die size", "Code name", "Code name(s)", "Code names"])

    def get_model_name(row):
        for key in ["Model", "Model name", "Model (Codename)", "Model (Code name)", "Model Model", "Model Name"]:
            v = row.get(key)
            if v and str(v).strip().lower() not in ["nan", ""]:
                return str(v).strip()
        return str(list(row.values())[0]).strip() if len(row) else "Unknown"

    result_dict = {}
    records = df.to_dict(orient="records")
    for row in records:
        temp_dict = {}
        for k, v in row.items():
            if not pd.isna(v):
                temp_dict[k] = v
        code_name = get_model_name(row)
        if code_name in result_dict:
            i = 2
            while f"{code_name}_{i}" in result_dict:
                i += 1
            code_name = f"{code_name}_{i}"
        result_dict[code_name] = temp_dict

    with open("gpu.json", "w", encoding="utf-8") as outfile:
        json.dump(result_dict, outfile, indent=2, ensure_ascii=False, default=str)

if __name__ == "__main__":
    main()
