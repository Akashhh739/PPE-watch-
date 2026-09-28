#!/bin/bash

echo "Starting PPE-watch App..."
echo "========================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "1. Checking/Installing Dependencies..."
pip3 install -r requirements.txt

echo -e "\n2. Launching Web App..."
echo "(Press Ctrl+C to stop the server at any time)"
python3 app.py
