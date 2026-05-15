#!/bin/bash
# Install system dependencies via Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install it first: https://brew.sh/"
    exit 1
fi

echo "Installing system dependencies..."
brew install python-tk p7zip

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Installation complete."
echo "To run MUXP, use the following commands:"
echo "source .venv/bin/activate"
echo "python3 muxp.py"
