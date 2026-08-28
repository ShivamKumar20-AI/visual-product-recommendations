"""
Run the full visual product recommendation pipeline.

This script runs the workflow from data cleaning through to evaluation,
then launches the Streamlit app locally.

Usage:
    python run_pipeline.py

Assumes:
- Raw images are in: Data/raw/images/
- Output directories exist or will be created by the scripts
- Requirements are installed: pip install -r requirements.txt
"""

import os
import sys
import subprocess

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_cleaning import clean_images
from create_catalogue_embeddings import create_catalogue_embeddings
from evaluate import create_recommendation_figure


def main():
    print("=== Visual Product Recommendation Pipeline ===\n")

    # Step 1: Data cleaning
    print("Step 1: Cleaning data...")
    clean_images()
    print("Data cleaning complete.\n")

    # Step 2: Create catalogue embeddings
    # (feature_extraction helpers are used inside this step)
    print("Step 2: Creating catalogue embeddings...")
    create_catalogue_embeddings()
    print("Catalogue embeddings created.\n")

    # Step 3: Evaluate model
    print("Step 3: Evaluating model...")
    create_recommendation_figure()
    print("Evaluation complete.\n")

    print("=== Pipeline finished successfully ===")
    print("Saved artefacts are in the 'artifacts' directory.")
    print("Launching Streamlit app locally...\n")

    # Step 4: Launch Streamlit app
    app_path = os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")
    subprocess.run(["streamlit", "run", app_path])


if __name__ == "__main__":
    main()