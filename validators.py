"""
Validation module for GPU Info API.

Contains functions to validate data quality and output format.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd

from config import REQUIRED_FIELDS, MIN_EXPECTED_GPUS

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_dataframe(df: pd.DataFrame, vendor: str) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame for completeness and quality.
    
    Args:
        df: DataFrame to validate
        vendor: Vendor name for logging context
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    if df.empty:
        warnings.append(f"{vendor}: DataFrame is empty")
        return False, warnings
    
    # Check for reasonable number of rows
    if len(df) < 5:
        warnings.append(f"{vendor}: Only {len(df)} rows found (expected more)")
    
    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        warnings.append(f"{vendor}: Duplicate columns found: {duplicate_cols}")
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        logger.debug(f"{vendor}: Empty columns will be dropped: {empty_cols}")
    
    return True, warnings


def validate_gpu_record(record: Dict[str, Any], key: str) -> Tuple[bool, List[str]]:
    """
    Validate a single GPU record.
    
    Args:
        record: GPU record dictionary
        key: Record key for logging context
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check for required fields
    for field in REQUIRED_FIELDS:
        if field not in record or pd.isna(record.get(field)):
            warnings.append(f"{key}: Missing required field '{field}'")
    
    # Validate vendor value
    vendor = record.get("Vendor", "")
    if vendor and vendor not in ["NVIDIA", "AMD", "Intel"]:
        warnings.append(f"{key}: Unknown vendor '{vendor}'")
    
    # Check if record has any meaningful data beyond vendor
    non_vendor_fields = [k for k in record.keys() if k != "Vendor"]
    if len(non_vendor_fields) < 3:
        warnings.append(f"{key}: Very sparse record (only {len(non_vendor_fields)} fields)")
    
    return len(warnings) == 0, warnings


def validate_output(data: Dict[str, Dict[str, Any]], output_path: Path) -> bool:
    """
    Validate the final output data.
    
    Args:
        data: Complete GPU data dictionary
        output_path: Path where output will be written
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    logger.info(f"Validating output data ({len(data)} records)...")
    
    # Check minimum record count
    if len(data) < MIN_EXPECTED_GPUS:
        raise ValidationError(
            f"Output has {len(data)} GPUs but expected at least {MIN_EXPECTED_GPUS}. "
            "Wiki structure may have changed."
        )
    
    # Validate individual records
    total_warnings = 0
    critical_errors = 0
    
    for key, record in data.items():
        is_valid, warnings = validate_gpu_record(record, key)
        if warnings:
            total_warnings += len(warnings)
            for warning in warnings:
                if "Missing required field" in warning:
                    logger.error(warning)
                    critical_errors += 1
                else:
                    logger.warning(warning)
    
    # Report validation summary
    if total_warnings > 0:
        logger.warning(f"Validation found {total_warnings} warnings across {len(data)} records")
    
    if critical_errors > 0:
        raise ValidationError(
            f"Validation failed with {critical_errors} critical errors. "
            "Check logs for details."
        )
    
    # Validate JSON serializability
    try:
        json.dumps(data, default=str)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Output data is not JSON serializable: {e}")
    
    logger.info("✅ Validation passed")
    return True


def validate_json_file(json_path: Path) -> bool:
    """
    Validate a JSON file for correct structure.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not json_path.exists():
        raise ValidationError(f"File not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {e}")
    
    if not isinstance(data, dict):
        raise ValidationError("JSON root must be an object/dictionary")
    
    logger.info(f"✅ JSON file is valid ({len(data)} records)")
    return True


if __name__ == "__main__":
    """Allow running validators as standalone script."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validators.py <path_to_gpu.json>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        validate_json_file(Path(sys.argv[1]))
        print("✅ Validation successful")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
