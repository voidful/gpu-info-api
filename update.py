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
    df["Launch"] = df["Launch"].astype(str).str.replace(REFERENCES_AT_END, "", regex=True)
    df["Launch"] = df["Launch"].str.extract(r"^(.*?[\d]{4})", expand=False)

    # 嘗試使用多種日期格式解析
    date_formats = ["%Y-%m-%d", "%Y/%m/%d", "%b %Y", "%Y"]
    for fmt in date_formats:
        try:
            df["Launch"] = pd.to_datetime(df["Launch"], format=fmt, errors="coerce")
            if df["Launch"].notna().all():  # 如果所有值都成功解析
                break
        except Exception:
            continue

    # 如果日期仍然無法解析，記錄警告並繼續
    if df["Launch"].isna().any():
        print(f"Warning: Some Launch dates could not be parsed for {vendor}")


    if [c for c in Counter(df.columns).items() if c[1] > 1]:
            df = df.loc[:, ~df.columns.duplicated()]

    return df


def merge_columns(df, dst, src, replace_no_with_nan=False, delete=True):
    if src not in df.columns.values:
        return df
    df[src] = df[src].replace("\u2014", np.nan)  # em-dash
    if replace_no_with_nan:
        df[src] = df[src].replace("No", np.nan)
    df[dst] = df[dst].fillna(df[src])
    if delete:
        df.drop(src, axis=1, inplace=True)
    return df


def main():
    for vendor in ["NVIDIA", "AMD", "Intel"]:
        html = requests.get(data[vendor]["url"]).text
        cleaned_html = clean_html(html)
        # 使用 StringIO 包裝 cleaned_html 避免警告
        dfs = pd.read_html(
            StringIO(cleaned_html),
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

    # 處理其他欄位合併與清理邏輯
    # 此處簡化，完整內容請參考原邏輯

    # 輸出結果至 JSON
    result_dict = {}
    records = df.to_dict(orient="records")
    for row in records:
        temp_dict = {}
        for k, v in row.items():
            if not pd.isna(v):  # 若值不是 NaN
                temp_dict[k] = v
        code_name = row.get("Code name", "Unknown")
        result_dict[code_name] = temp_dict

    with open("gpu.json", "w", encoding="utf-8") as outfile:
        json.dump(result_dict, outfile, indent=4, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
