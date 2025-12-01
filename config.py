"""
Configuration module for GPU Info API.

Contains all configuration settings, vendor URLs, and constants.
"""
import os
from typing import Dict

# ---------------------------------------------
#  Vendor Configuration
# ---------------------------------------------

VENDOR_CONFIGS: Dict[str, Dict[str, str]] = {
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

# ---------------------------------------------
#  Regex Patterns
# ---------------------------------------------

# Pattern to match reference citations at the end of strings
REFERENCES_AT_END = r"(?:\s*\[\d+\])+(?:\d+,)?(?:\d+)?$"

# ---------------------------------------------
#  Request Configuration
# ---------------------------------------------

# Timeout for HTTP requests in seconds
REQUEST_TIMEOUT = 45

# Maximum number of retry attempts for failed requests
MAX_RETRIES = 3

# Exponential backoff multiplier for retries
RETRY_MULTIPLIER = 2

# Minimum wait time between retries (seconds)
RETRY_MIN_WAIT = 2

# Maximum wait time between retries (seconds)
RETRY_MAX_WAIT = 10

# ---------------------------------------------
#  Output Configuration
# ---------------------------------------------

# Default output file path
DEFAULT_OUTPUT_FILE = "gpu.json"

# Output JSON indentation (2 for pretty print, None for compact)
JSON_INDENT = int(os.getenv("JSON_INDENT", "2"))

# Create backup before overwriting
CREATE_BACKUP = os.getenv("CREATE_BACKUP", "true").lower() == "true"

# ---------------------------------------------
#  Validation Configuration
# ---------------------------------------------

# Minimum expected GPUs in output (sanity check)
MIN_EXPECTED_GPUS = 100

# Required fields for each GPU record
REQUIRED_FIELDS = ["Vendor"]

# Minimum table dimensions to be considered valid
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 3

# ---------------------------------------------
#  Logging Configuration
# ---------------------------------------------

# Default log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Log file path
LOG_FILE = "gpu_info_api.log"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
