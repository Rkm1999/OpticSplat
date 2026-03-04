#!/usr/bin/env bash

echo "Starting SHARP Camera Simulator..."

echo "Installing server dependencies..."
# Using python3 -m pip ensures we use the pip associated with the active python3 environment
python3 -m pip install -r requirements-server.txt

echo "Starting backend server..."
python3 server.py

# Keep the terminal window open to see output/errors, similar to 'pause' in Windows
read -p "Press Enter to exit..."