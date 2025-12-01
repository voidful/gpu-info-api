"""
GPU Info API - Data Fetcher and Processor

Fetches GPU information from Wikipedia sources for NVIDIA, AMD, and Intel GPUs,
processes the data, and outputs to JSON format.

This script is designed to be run periodically (e.g., weekly via GitHub Actions)
to keep the GPU database up to date.
"""
import argparse
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import (
    VENDOR_CONFIGS,
    REFERENCES_AT_END,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_MIN_WAIT,
    RETRY_MAX_WAIT,
    DEFAULT_OUTPUT_FILE,
    JSON_INDENT,
    CREATE_BACKUP,
    MIN_TABLE_ROWS,
    MIN_TABLE_COLS,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_FILE,
)
from validators import validate_dataframe, validate_output, ValidationError

# ---------------------------------------------
#  Logging Setup
# ---------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = LOG_LEVEL, log_to_file: bool = True) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to also log to file
    """
    handlers = [logging.StreamHandler()]
    
    if log_to_file:
        handlers.append(logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )
    
    # Reduce verbosity of third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


# ---------------------------------------------
#  HTML Processing Utilities
# ---------------------------------------------

def clean_html(html: str) -> str:
    """
    Remove tags & anomalies so that pandas.read_html behaves correctly.
    
    This function standardizes Wikipedia HTML markup by:
    - Normalizing rowspan/colspan attributes
    - Removing style blocks, citations, and hidden spans
    - Simplifying markup and entities
    - Handling special characters and formatting
    
    Args:
        html: Raw HTML string from Wikipedia
        
    Returns:
        Cleaned HTML string suitable for pandas parsing
    """
    try:
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
        html = html.translate(str.maketrans({"–": "-", "—": "-", "‒": "-"}))
        html = (html.replace("mm<sup>2</sup>", "mm2")
                    .replace("×10<sup>6</sup>", "×10⁶")
                    .replace("×10<sup>9</sup>", "×10⁹"))
        html = re.sub(r"<sup>\d+</sup>", "", html)
        
        logger.debug(f"HTML cleaned successfully ({len(html)} bytes)")
        return html
        
    except Exception as e:
        logger.error(f"Error cleaning HTML: {e}")
        raise


# ---------------------------------------------
#  DataFrame Processing Utilities
# ---------------------------------------------

def normalize_columns(cols) -> List[str]:
    """
    Flatten possible MultiIndex and clean duplicates.
    
    Args:
        cols: DataFrame columns (can be MultiIndex or regular Index)
        
    Returns:
        List of cleaned column names
    """
    if isinstance(cols, pd.MultiIndex):
        flat = [
            " ".join(str(x).strip() for x in tup if str(x) != "nan" and not str(x).startswith("Unnamed"))
            for tup in cols.values
        ]
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
    """
    Standardize column names across different vendor tables.
    
    Args:
        df: DataFrame with potentially varying column names
        
    Returns:
        DataFrame with standardized column names
    """
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
        logger.debug(f"Standardizing columns: {mapping}")
        df = df.rename(columns=mapping)
    
    return df


def process_dataframe(df: pd.DataFrame, vendor: str) -> pd.DataFrame:
    """
    Process a raw DataFrame from Wikipedia tables.
    
    Performs column normalization, standardization, vendor tagging,
    date parsing, and duplicate removal.
    
    Args:
        df: Raw DataFrame from Wikipedia
        vendor: Vendor name (NVIDIA, AMD, Intel)
        
    Returns:
        Processed DataFrame ready for export
    """
    logger.debug(f"{vendor}: Processing DataFrame with shape {df.shape}")
    
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
        logger.debug(f"{vendor}: Extracted {df['Launch'].notna().sum()} launch dates")
    else:
        df["Launch"] = pd.NaT
        logger.debug(f"{vendor}: No launch date column found")

    # Drop completely duplicated column titles
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
    
    logger.debug(f"{vendor}: Processed to shape {df.shape}")
    return df


def remove_bracketed_references(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Remove citation brackets [1], [2], etc. from specified columns.
    
    Args:
        df: DataFrame to clean
        cols: List of column names to clean
        
    Returns:
        DataFrame with cleaned values
    """
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r"\[\d+\]", "", regex=True).str.strip()
    
    return df


# ---------------------------------------------
#  Network & Fetching
# ---------------------------------------------

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def fetch_url_with_retry(url: str) -> str:
    """
    Fetch URL with automatic retry on failure.
    
    Uses exponential backoff retry strategy to handle transient network issues.
    
    Args:
        url: URL to fetch
        
    Returns:
        HTML content as string
        
    Raises:
        requests.RequestException: If all retry attempts fail
    """
    logger.debug(f"Fetching URL: {url}")
    
    # Add User-Agent header to avoid Wikipedia blocking
    headers = {
        'User-Agent': 'GPU-Info-API/1.0 (https://github.com/voidful/gpu-info-api; Educational/Research)'
    }
    
    response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
    response.raise_for_status()
    logger.debug(f"Successfully fetched {len(response.text)} bytes")
    return response.text


def fetch_vendor_tables(vendor: str, url: str) -> List[pd.DataFrame]:
    """
    Fetch and parse GPU tables from vendor Wikipedia page.
    
    Args:
        vendor: Vendor name (NVIDIA, AMD, Intel)
        url: Wikipedia URL to fetch
        
    Returns:
        List of DataFrames extracted from tables
        
    Raises:
        Exception: If fetching or parsing fails
    """
    logger.info(f"Fetching {vendor} data from Wikipedia...")
    
    try:
        html = fetch_url_with_retry(url)
        html = clean_html(html)
        
        # Parse tables matching launch/release date patterns
        dfs = pd.read_html(
            StringIO(html),
            match=re.compile(r"Launch|Release Date & Price", re.I)
        )
        
        logger.info(f"{vendor}: Found {len(dfs)} tables")
        
        # NVIDIA-specific: handle transposed spec tables
        if vendor == "NVIDIA":
            try:
                for df_t in pd.read_html(StringIO(html), match=re.compile(r"Release date", re.I)):
                    t = df_t.T.reset_index()
                    cols = pd.MultiIndex.from_arrays([t.iloc[0].astype(str), t.iloc[1].astype(str)])
                    tidy = pd.DataFrame(t.iloc[2:].values, columns=cols).reset_index(drop=True)
                    tidy.columns = normalize_columns(tidy.columns)
                    tidy = tidy.rename(columns={"Release date": "Launch"})
                    tidy = standardise_column_names(tidy)
                    dfs.append(tidy)
                    logger.debug(f"{vendor}: Added transposed table")
            except Exception as e:
                logger.warning(f"{vendor}: Error processing transposed tables: {e}")
        
        return dfs
        
    except requests.RequestException as e:
        logger.error(f"{vendor}: Network error fetching data: {e}")
        raise
    except Exception as e:
        logger.error(f"{vendor}: Error parsing tables: {e}")
        raise


# ---------------------------------------------
#  Assembly & Export
# ---------------------------------------------

def record_key(row: Dict[str, str]) -> str:
    """
    Generate a unique key for a GPU record.
    
    Prefers Code name, falls back to Model name or Model.
    
    Args:
        row: GPU record dictionary
        
    Returns:
        Unique key string
    """
    vendor = row.get("Vendor", "UNKNOWN").strip()
    code = str(row.get("Code name", "")).strip()
    
    if code and code.lower() not in {"nan", ""}:
        return f"{vendor}_{code}"

    model = str(row.get("Model name", row.get("Model", "UnknownModel"))).strip()
    model = re.sub(r"[^A-Za-z0-9]+", "_", model) or "UnknownModel"
    return f"{vendor}_{model}"


def create_backup(output_path: Path) -> None:
    """
    Create a backup of existing output file.
    
    Args:
        output_path: Path to file to backup
    """
    if output_path.exists():
        backup_path = output_path.with_suffix('.json.backup')
        shutil.copy2(output_path, backup_path)
        logger.info(f"Created backup: {backup_path}")


# ---------------------------------------------
#  Pipeline
# ---------------------------------------------

def main(output_file: str = DEFAULT_OUTPUT_FILE, dry_run: bool = False) -> Dict[str, Dict]:
    """
    Main pipeline: fetch, process, validate, and export GPU data.
    
    Args:
        output_file: Path to output JSON file
        dry_run: If True, don't write output file (validation only)
        
    Returns:
        Dictionary of GPU data
        
    Raises:
        ValidationError: If data validation fails
        Exception: If processing fails
    """
    logger.info("=" * 60)
    logger.info("GPU Info API - Starting data fetch and processing")
    logger.info("=" * 60)
    
    output_path = Path(output_file)
    frames: List[pd.DataFrame] = []
    failed_vendors: List[str] = []
    
    # Fetch data from all vendors
    for vendor, info in VENDOR_CONFIGS.items():
        try:
            vendor_dfs = fetch_vendor_tables(vendor, info["url"])
            
            for idx, raw_df in enumerate(vendor_dfs):
                # Validate table dimensions
                if raw_df.shape[0] < MIN_TABLE_ROWS or raw_df.shape[1] < MIN_TABLE_COLS:
                    logger.warning(
                        f"{vendor} table {idx}: Too small ({raw_df.shape}), skipping"
                    )
                    continue
                
                # Process and validate DataFrame
                processed_df = process_dataframe(raw_df, vendor)
                is_valid, warnings = validate_dataframe(processed_df, vendor)
                
                if is_valid:
                    frames.append(processed_df)
                else:
                    logger.warning(f"{vendor} table {idx}: Validation failed, skipping")
                
        except Exception as e:
            logger.error(f"{vendor}: Failed to fetch/process data: {e}", exc_info=True)
            failed_vendors.append(vendor)
            continue
    
    # Check if we got any data
    if not frames:
        raise RuntimeError(
            "No GPU tables parsed successfully. "
            "All vendors failed or wiki markup may have changed."
        )
    
    if failed_vendors:
        logger.warning(f"Failed to process vendors: {', '.join(failed_vendors)}")
    
    # Combine all DataFrames
    logger.info(f"Combining {len(frames)} tables...")
    df = pd.concat(frames, ignore_index=True, sort=False)
    
    # Clean reference brackets from key columns
    df = remove_bracketed_references(
        df,
        ["Model", "Model name", "Model (Codename)", "Model (Code name)", 
         "Die size", "Die size (mm2)", "Code name"]
    )
    
    # Convert to dictionary format
    logger.info("Converting to JSON format...")
    result: Dict[str, Dict[str, str]] = {}
    
    for record in df.to_dict(orient="records"):
        # Remove NaN values and create compact record
        compact = {k: v for k, v in record.items() if pd.notna(v)}
        
        # Generate unique key
        key = record_key(compact)
        
        # Handle duplicate keys
        if key in result:
            i = 2
            while f"{key}_{i}" in result:
                i += 1
            original_key = key
            key = f"{key}_{i}"
            logger.debug(f"Duplicate key '{original_key}' renamed to '{key}'")
        
        result[key] = compact
    
    # Validate output
    logger.info("Validating output...")
    try:
        validate_output(result, output_path)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise
    
    # Write output
    if not dry_run:
        # Create backup if requested
        if CREATE_BACKUP:
            create_backup(output_path)
        
        logger.info(f"Writing output to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=JSON_INDENT, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Successfully saved {len(result)} GPUs to {output_path}")
    else:
        logger.info(f"✅ Dry run: {len(result)} GPUs validated (not written to file)")
    
    # Summary
    vendor_counts = {}
    for record in result.values():
        vendor = record.get("Vendor", "Unknown")
        vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
    
    logger.info("Summary by vendor:")
    for vendor, count in sorted(vendor_counts.items()):
        logger.info(f"  {vendor}: {count} GPUs")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and process GPU information from Wikipedia"
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without writing output file"
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging level (default: {LOG_LEVEL})"
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable logging to file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_to_file=not args.no_log_file)
    
    # Run main pipeline
    try:
        main(output_file=args.output, dry_run=args.dry_run)
        sys.exit(0)
    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
