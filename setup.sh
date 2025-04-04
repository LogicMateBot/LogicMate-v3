#!/bin/bash

echo "Setting up the environment..."

python3.10 --version >/dev/null 2>&1 || { echo "Python 3.10 is not installed"; exit 1; }
echo "Python 3.10 is installed."

# Verify if virtual environment already exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Do you want to remove it? (y/n)"
    read -r answer
    if [ "$answer" == "y" ]; then
        echo "Removing existing virtual environment..."
        rm -rf .venv
        if [ $? -ne 0 ]; then
            echo "Failed to remove the existing virtual environment"
            exit 1
        fi
        echo "Existing virtual environment removed."
    else
        echo "Keeping the existing virtual environment."
    fi
fi

# Create a new virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating a virtual environment..."
    python3.10 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create a virtual environment"
        exit 1
    fi
    echo "Virtual environment created."
fi

# Activating the virtual environment
echo "Activating the virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment"
    exit 1
fi
echo "Virtual environment activated."

# Updating pip
echo "Upgrading pip..."
pip install --upgrade pip >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Failed to upgrade pip"
    exit 1
fi
echo "Pip upgraded."

# Check if Poetry is already installed
echo "Checking if Poetry is installed..."
if command -v poetry >/dev/null 2>&1; then
    echo "Poetry is already installed."
else
    echo "Poetry is not installed. Installing Poetry..."
    pip install poetry >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Failed to install Poetry"
        exit 1
    fi
    echo "Poetry installed."
fi

# Check if Poetry is in the PATH
poetry --version

# Install dependencies
echo "Installing dependencies..."
poetry install
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies"
    exit 1
fi
echo "Dependencies installed."

# Download model Weights
if [ ! -d "weights-v1" ]; then
    echo "Model Weights V1 not found. Cloning the repository..."
    echo "Cloning Models V1 Weights repository..."
    if [ ! -d "weights" ]; then
        mkdir weights
    fi
    cd weights
    echo "Cloning model Weights V1 repository..."
    git clone https://github.com/LogicMateBot/Models-v1.git .
    if [ $? -ne 0 ]; then
        echo "Failed to clone model Weights V1 repository"
        exit 1
    fi
    cd .. && cd ..
    echo "model Weights V1 cloned successfully."
else
    echo "model Weights V1 already exists."
fi

# Download model Weights
if [ ! -d "weights-v2" ]; then
    echo "Model Weights V2 not found. Cloning the repository..."
    echo "Cloning Models Weights V2 repository..."
    if [ ! -d "weights" ]; then
        mkdir weights
    fi
    cd weights
    echo "Cloning model Weights V2 repository..."
    git clone https://github.com/LogicMateBot/Models-v2.git .
    if [ $? -ne 0 ]; then
        echo "Failed to clone model Weights V2 repository"
        exit 1
    fi
    cd .. && cd ..
    echo "model Weights V2 cloned successfully."
else
    echo "model Weights V2 already exists."
fi

# Creating media directory
if [ ! -d "media" ]; then
    echo "Creating media directory..."
    mkdir media
    if [ $? -ne 0 ]; then
        echo "Failed to create media directory"
        exit 1
    fi
    echo "Media directory created."
else
    echo "Media directory already exists."
fi

# Creating videos directory inside media
if [ ! -d "media/videos" ]; then
    echo "Creating videos directory inside media..."
    mkdir media/videos
    if [ $? -ne 0 ]; then
        echo "Failed to create videos directory"
        exit 1
    fi
    echo "Videos directory created."
else
    echo "Videos directory already exists."
fi

# Creating images directory inside media
if [ ! -d "media/images" ]; then
    echo "Creating images directory inside media..."
    mkdir media/images
    if [ $? -ne 0 ]; then
        echo "Failed to create images directory"
        exit 1
    fi
    echo "Images directory created."
else
    echo "Images directory already exists."
fi

# Installing roff
pip install ruff
if [ $? -ne 0 ]; then
    echo "Failed to install ruff"
    exit 1
fi
echo "Ruff installed."

echo "Environment setup complete."