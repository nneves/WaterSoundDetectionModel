# UV Setup for Modelo v3.2

This project now includes a `pyproject.toml` file for use with UV, a fast Python package manager.

## Requirements

- **Python 3.10 or higher** (required by some dependencies like contourpy, tensorflow, etc.)

## Installation and Setup

### 1. Install UV
```bash
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and Activate Virtual Environment

#### Option A: Using pyproject.toml (if working)
```bash
# Navigate to the modelo v3.2 directory
cd "ModeloAI/modelo v3.2"

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

#### Option B: Using UV with requirements file (recommended)
```bash
# Navigate to the modelo v3.2 directory
cd "ModeloAI/modelo v3.2"

# Create virtual environment
uv venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate

# Install dependencies using UV
uv pip install -r uv-requirements.txt
```

#### Option C: Traditional approach
```bash
# Install dependencies directly using original requirements.txt
uv pip install -r requirements.txt
```

## Project Structure

- `pyproject.toml` - Modern Python project configuration with UV support
- `requirements.txt` - Traditional pip requirements (legacy)
- `uv.lock` - Will be generated automatically when you run `uv sync`

## Usage Commands

```bash
# Install all dependencies
uv sync

# Install only production dependencies
uv sync --no-dev

# Add a new dependency
uv add package-name

# Remove a dependency
uv remove package-name

# Update all dependencies
uv sync --upgrade

# Run Python with the virtual environment
uv run python your_script.py

# Install development dependencies
uv sync --group dev
```

## Benefits of UV

- **Fast**: 10-100x faster than pip
- **Reliable**: Consistent dependency resolution
- **Modern**: Uses pyproject.toml standard
- **Lock file**: Ensures reproducible builds
- **Cross-platform**: Works on Windows, macOS, and Linux

## Migration Notes

This setup is completely independent from other Python environments and the original requirements.txt. The pyproject.toml contains all the same dependencies with the exact same versions for compatibility.
