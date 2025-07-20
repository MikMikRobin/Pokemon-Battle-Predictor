"""
Wrapper script for training that sets the necessary environment variables.
"""
import os
import sys
import subprocess

# Set the environment variable
os.environ["SCIPY_ARRAY_API"] = "1"

print("Set SCIPY_ARRAY_API=1 environment variable")

# Get command line arguments
args = sys.argv[1:]

# Default to running the training script with both formats and random forest
if not args:
    args = ["--model-type", "random_forest", "--format", "both"]

# Construct the command
cmd = ["python", "train.py"] + args

print(f"Running command: {' '.join(cmd)}")

# Run the training script
try:
    result = subprocess.run(cmd, check=True)
    sys.exit(result.returncode)
except subprocess.CalledProcessError as e:
    print(f"Error running training script: {e}")
    sys.exit(e.returncode)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1) 