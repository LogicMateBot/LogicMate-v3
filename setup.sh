#!/bin/bash

echo "Setting up the environment..."
python3.10 --version || { echo "Python 3.10 is not installed"; exit 1; }
echo "Python 3.10 is installed."

echo "Creating a virtual environment..."
python3.10 -m venv .venv
if [ $? -ne 0 ]; then
    echo "Failed to create a virtual environment"
    exit 1
fi
echo "Virtual environment created."

echo "Activating the virtual environment..."
source .venv/bin/activate
echo "Virtual environment created and activated."

echo "Upgrading pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to upgrade pip"
    exit 1
fi
echo "Pip upgraded."

echo "Installing Poetry..."
pip install poetry
if [ $? -ne 0 ]; then
    echo "Failed to install Poetry"
    exit 1
fi
echo "Poetry installed."

