import json
import re
from io import StringIO
from typing import List, Dict

import pandas as pd
import requests

# ---------------------------------------------
#  Configuration
# ---------------------------------------------

data: Dict[str, Dict[str, str]] = {
    "NVIDIA": {
        "url": "https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units",
    },
    "AMD": {
        "url": "https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units",
    },
    "Intel": {
        "url": "https://en.wikipedia.org/wiki/Intel_Xe",
    },
}

REFERENCES_AT_END = r"(?:\s*\[\d+\])+(?:\d+,)?(?:\d+)?$"

# ---------------------------------------------
#  Utilities
# ---------------------------------------------

def clean_html(html: str) -> str:
    """Remove tags & anomalies so that pandas.read_html behaves."""
    # Standardise rowspan/colspan attributes & strip garbage characters
    html = re.sub(
        r'''(colspan|rowspan)=(?:"|')?(\d+)[^>]*?''',
        lambda m: f'{m.group(1)}="{m.group(2)}"',
        html,
        flags=re.I,
    )

    # Remove <style>...</style> blocks completely (vertical‑header CSS etc.)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)

    # Remove citation superscripts & hidden spans
    html = re.sub(r"<sup[^>]*>.*?</sup>", "", html, flags=re.DOTALL)
    html = re.sub(r"<span [^>]*style=\"display:none[^>]*>([^<]+)</span>", "", html)

    # Simplify remaining markup
    html = re.sub(r"<br\s*/?>", " ", html)
    html = re.sub(r"<th([^>]*)>", lambda m: "<th" + m.group(1) + ">", html)
    html = re.sub(r"<span[^>]*>([^<]+)</span>", r"\1", html)

    # Misc entities / whitespace
    html = html.replace("\\\"", "\"")
    html = re.sub(r"(\d)&?#160;?(\d)", r"\1\2", html)
    html = re.sub(r"&thinsp;|&#8201;|&nbsp;|&#160;|\xa0", " ", html)
    html = re.sub(r"<small>.*?</small>", "", html, flags=re.DOTALL)
    html = html.translate(str.maketrans({"\u2012": "-", "\u2013": "-", "\u2014": ""}))
    html = (html.replace("mm<sup>2</sup>", "mm2")
                .replace("\u00d710<sup>6</sup>", "×10⁶")
                .replace("\u00d710<sup>9</sup>", "×10⁹"))
    html = re.sub(r"<sup>\d+</sup>", "", html)
    return html


def normalize_columns(cols) -> List[str]:
    """Flatten possible MultiIndex and clean duplicates."""
    if isinstance(cols, pd.MultiIndex):
        flat = [" ".join(str(x).strip() for x in tup if str(x) != "nan" and not str(x).startswith("Unnamed")) for tup in cols.values]
    else:
        flat = [str(c).strip() for c in cols]

    cleaned: List[str] = []
    for col in flat:
        col = re.sub(r"\s+", " ", col).strip()
        # Collapse exact duplicate half (e.g., "Launch Launch")
        parts = col.split()
        if len(parts) % 2 == 0 and parts[: len(parts)//2] == parts[len(parts)//2:]:
            parts = parts[: len(parts)//2]
        # Remove consecutive duplicate words
        dedup = []
        for w in parts:
            if not dedup or w != dedup[-1]:
                dedup.append(w)
        col = " ".join(dedup)
        cleaned.append(col)
    return cleaned


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in list(df.columns):
        cname = col.lower()
        if cname.startswith("gpu die") or cname == "code name":
            mapping[col] = "Code name"
        elif cname.startswith("model") and "name" not in cname:
            mapping[col] = "Model"
        elif cname.startswith("geforce rtx") or cname.startswith("radeon rx"):
            mapping[col] = "Model name"
    if mapping:
        df = df.rename(columns=mapping)
    return df


def process_dataframe(df: pd.DataFrame, vendor: str) -> pd.DataFrame:
    df.columns = normalize_columns(df.columns)
    df = standardise_column_names(df)

    # General header cleanup patterns
    df.columns = [re.sub(r" Arc \w+$", "", c) for c in df.columns]
    df.columns = [re.sub(r"(?:\[[A-Za-z0-9]+\])+", "", c) for c in df.columns]
    df.columns = [c.replace("- ", "").replace("/ ", "/").strip() for c in df.columns]

    df["Vendor"] = vendor

    # Launch / Release Date extraction
    launch_cols = [c for c in df.columns if re.search(r"launch|release date", c, re.I)]
    if launch_cols:
        col = launch_cols[0]
        df[col] = (
            df[col].astype(str)
            .str.replace(REFERENCES_AT_END, "", regex=True)
            .str.extract(r"([A-Za-z]+\s*\d{1,2},?\s*\d{4}|\d{4})", expand=False)
        )
        df["Launch"] = pd.to_datetime(df[col], errors="coerce")
    else:
        df["Launch"] = pd.NaT

    # Drop completely duplicated column titles
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    return df


def remove_bracketed_references(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r"\[\d+\]", "", regex=True).str.strip()
    return df


def fetch_vendor_tables(vendor: str, url: str) -> List[pd.DataFrame]:
    print(f"Fetching {vendor} → {url}")
    html = clean_html(requests.get(url, timeout=45).text)

    dfs = pd.read_html(StringIO(html), match=re.compile(r"Launch|Release Date & Price", re.I))

    if vendor == "NVIDIA":  # handle transposed spec table
        for df_t in pd.read_html(StringIO(html), match=re.compile(r"Release date", re.I)):
            t = df_t.T.reset_index()
            cols = pd.MultiIndex.from_arrays([t.iloc[0].astype(str), t.iloc[1].astype(str)])
            tidy = pd.DataFrame(t.iloc[2:].values, columns=cols).reset_index(drop=True)
            tidy.columns = normalize_columns(tidy.columns)
            tidy = tidy.rename(columns={"Release date": "Launch"})
            tidy = standardise_column_names(tidy)
            dfs.append(tidy)
    return dfs

# ---------------------------------------------
#  Assembly helpers
# ---------------------------------------------

def record_key(row: Dict[str, str]) -> str:
    vendor = row.get("Vendor", "UNKNOWN").strip()
    code = str(row.get("Code name", "")).strip()
    if code and code.lower() not in {"nan", ""}:
        return f"{vendor}_{code}"

    model = str(row.get("Model name", row.get("Model", "UnknownModel"))).strip()
    model = re.sub(r"[^A-Za-z0-9]+", "_", model) or "UnknownModel"
    return f"{vendor}_{model}"

# ---------------------------------------------
#  Pipeline
# ---------------------------------------------

def main():
    frames: List[pd.DataFrame] = []
    for vendor, info in data.items():
        for raw in fetch_vendor_tables(vendor, info["url"]):
            if raw.shape[0] >= 2 and raw.shape[1] >= 3:
                frames.append(process_dataframe(raw, vendor))

    if not frames:
        raise RuntimeError("No GPU tables parsed. Wiki markup may have changed.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = remove_bracketed_references(df,
        ["Model", "Model name", "Model (Codename)", "Model (Code name)", "Die size", "Die size (mm2)", "Code name"])

    result: Dict[str, Dict[str, str]] = {}
    for record in df.to_dict(orient="records"):
        compact = {k: v for k, v in record.items() if pd.notna(v)}
        key = record_key(compact)
        if key in result:
            i = 2
            while f"{key}_{i}" in result:
                i += 1
            key = f"{key}_{i}"
        result[key] = compact

    with open("gpu.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False, default=str)
    print("✅", len(result), "GPUs saved → gpu.json")


if __name__ == "__main__":
    main()
