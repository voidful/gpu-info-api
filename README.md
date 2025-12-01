# GPU Info API

This repository provides an API for extracting GPU information from Nvidia, AMD, and Intel GPUs in JSON format. The data
is collected from wiki sources and updated weekly using GitHub Actions. The API contains detailed information on various
GPU models, including their specifications, performance metrics, and other relevant details.

## API Path

You can access the API at the following path:
https://raw.githubusercontent.com/voidful/gpu-info-api/gpu-data/gpu.json

## API Example

```json
{
  "AD104-250": {
    "Model": "GeForce RTX 4070",
    "Launch": "2023-04-13 00:00:00",
    "Code name": "AD104-250",
    "Fab (nm)": 4.0,
    "Die size (mm2)": 294.5,
    "Bus interface": "PCIe 4.0 x16",
    "Core clock (MHz)": "1920",
    "Core config": "5888:184:64:184:46 (46)(4)",
    "Memory Bandwidth (GB/s)": 504.0,
    "Memory Bus type": "GDDR6X",
    "Memory Bus width (bit)": "192",
    "Vendor": "NVIDIA",
    "Fillrate Pixel (GP/s)": 158.4,
    "Fillrate Texture (GT/s)": 455.4,
    "TDP (Watts)": 200.0,
    "Release Price (USD)": 599.0,
    "SM count": "46",
    "Process": "TSMC N4",
    "Transistors (billion)": 35.8,
    "L Cache (MB)": 36,
    "Memory Size (GB)": 12,
    "Clock speeds Memory (MT/s)": 21000,
    "Release price (USD) Founders Edition": "$599",
    "Clock speeds Boost core clock (MHz)": 2475,
    "Single-precision TFLOPS": "22.6",
    "Double-precision TFLOPS": "0.353",
    "Half-precision TFLOPS": "22.6",
    "Pixel/unified shader count": 5888.0,
    "GPU Type": "Desktop"
  },
  "AD104-400": {
    "Model": "GeForce RTX 4080",
    "Code name": "AD104-400",
    "Fab (nm)": 4.0,
    "Die size (mm2)": 294.5,
    "Bus interface": "PCIe 4.0 x16",
    "Core clock (MHz)": "2310",
    "Core config": "7680:240:80:240:60 (60)(5)",
    "Memory Bandwidth (GB/s)": 504.0,
    "Memory Bus type": "GDDR6X",
    "Memory Bus width (bit)": "192",
    "Vendor": "NVIDIA",
    "Fillrate Pixel (GP/s)": 208.8,
    "Fillrate Texture (GT/s)": 626.4,
    "TDP (Watts)": 285.0,
    "Release Price (USD)": 899.0,
    "SM count": "60",
    "Process": "TSMC N4",
    "Transistors (billion)": 35.8,
    "L Cache (MB)": 48,
    "Memory Size (GB)": 12,
    "Clock speeds Memory (MT/s)": 21000,
    "Clock speeds Boost core clock (MHz)": 2610,
    "Single-precision TFLOPS": "35.5",
    "Double-precision TFLOPS": "0.554",
    "Half-precision TFLOPS": "35.5",
    "Processing power (TFLOPS) Tensor compute (FP16) (2: sparse)": "142 (284) 160 (321)",
    "Ray-tracing Performance (TFLOPS)": 92.7,
    "Pixel/unified shader count": 7680.0,
    "GPU Type": "Desktop"
  }
}
```

## Inspiration and Credits

This repository is inspired by and borrows from the following project: https://github.com/owensgroup/gpustats

## API Usage

To use the API, you can simply make an HTTP request to the API path mentioned above. The API will return the GPU
information in JSON format, which you can then parse and use in your application.

Here's an example of the JSON data returned by the API for two GPU models:

```json
{
  "AD104-250": {
    "Model": "GeForce RTX 4070",
    "Launch": "2023-04-13 00:00:00",
    ...
    "GPU Type": "Desktop"
  },
  "AD104-400": {
    "Model": "GeForce RTX 4080",
    ...
    "GPU Type": "Desktop"
  }
}
```

You can then extract specific information about a GPU model using its key, such as "AD104-250" or "AD104-400".

## Development

### Running the Script

The `update.py` script fetches GPU data from Wikipedia and generates the `gpu.json` file.

#### Installation

Install dependencies using:
```bash
pip install -r requirements.txt
```

#### Basic Usage

```bash
# Standard run - fetches data and creates gpu.json
python update.py

# Dry run - validates data without writing file
python update.py --dry-run

# Custom output path
python update.py -o custom_output.json

# Debug mode with detailed logging
python update.py --log-level DEBUG
```

### Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. -v
```

### Features

The refactored codebase includes:

- ✅ **Comprehensive error handling** with retry logic for network failures
- ✅ **Professional logging** to both console and file (`gpu_info_api.log`)
- ✅ **Data validation** at multiple stages to ensure quality
- ✅ **Automated testing** with 19 unit tests
- ✅ **Type hints** throughout for better code safety
- ✅ **CLI arguments** for flexible usage
- ✅ **Graceful degradation** - continues if one vendor fails

### Validation

Validate the generated JSON file:

```bash
python validators.py gpu.json
```


## Contributing

If you'd like to contribute to this project or have any suggestions, feel free to open an issue or submit a pull
request. We appreciate any feedback and assistance in improving the quality and accuracy of the GPU information provided
by this API.