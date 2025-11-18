# Test Data Setup

## Overview

This directory contains tests for the paleo-emu project. To ensure tests can run reliably in CI environments, test data files are generated programmatically when needed.

## Test Data Generation

The `setup_test_data.py` script generates minimal NetCDF test files that are required by the test suite. This script is automatically run during CI before executing tests.

### What it does:

1. **Checks existing files**: The script only generates files if they're missing or appear to be corrupted (< 1KB).

2. **Training data files**: Creates minimal training data files with proper NetCDF4 structure:
   - Dimensions: id (120), lat (73), lon (96)
   - Variable: 'var' with mean ~5.28
   - Format: NETCDF4

3. **Output directories**: Ensures prediction output directories exist so tests can write their results.

### Running manually:

```bash
python -m tests.setup_test_data
```

### Why this approach?

Large binary NetCDF files in git repositories can occasionally become corrupted or cause issues in CI environments. By generating minimal test fixtures programmatically, we ensure:

- Tests always have valid input data
- Files are small and deterministic
- No dependency on Git LFS or binary file tracking
- CI remains fast and reliable

## Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -p 'test*.py'

# Run with verbose output
python -m unittest discover -s tests -p 'test*.py' -v
```

## CI Integration

The CI workflow (`.github/workflows/ci.yml`) automatically runs `setup_test_data.py` before executing tests to ensure all required test data files are present.
