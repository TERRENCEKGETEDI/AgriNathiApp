#!/usr/bin/env python3
"""
Setup script for Kaggle API credentials
"""

import os
from pathlib import Path

def setup_kaggle_credentials():
    """Guide user through setting up Kaggle API credentials"""

    print("Setting up Kaggle API credentials for AgriNathi")
    print("=" * 50)

    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'

    if kaggle_json.exists():
        print("[OK] Kaggle credentials already exist!")
        print(f"Location: {kaggle_json}")
        return True

    print("\nTo download agricultural datasets from Kaggle, you need API credentials:")
    print("\nStep 1: Create a Kaggle account (if you don't have one)")
    print("   Go to: https://www.kaggle.com/account/login")

    print("\nStep 2: Generate API token")
    print("   1. Go to: https://www.kaggle.com/account")
    print("   2. Scroll down to 'API' section")
    print("   3. Click 'Create New API Token'")
    print("   4. Download the kaggle.json file")

    print(f"\nStep 3: Place the file in the correct location")
    print(f"   Move kaggle.json to: {kaggle_dir}")
    print(f"   Create the directory if it doesn't exist: {kaggle_dir}")

    # Create the directory
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[OK] Created directory: {kaggle_dir}")

    # Wait for user to place the file
    input("\nPress Enter after you've placed kaggle.json in the directory above...")

    if kaggle_json.exists():
        # Set proper permissions (600 on Unix-like systems)
        try:
            os.chmod(kaggle_json, 0o600)
            print("[OK] Set proper permissions on kaggle.json")
        except:
            print("[INFO] Could not set permissions (this is normal on Windows)")

        print("\n[OK] Kaggle API credentials are ready!")
        print("You can now run: python scripts/download_kaggle_datasets.py")
        return True
    else:
        print("\n✗ kaggle.json not found. Please try again.")
        return False

def test_kaggle_connection():
    """Test Kaggle API connection"""
    try:
        import kaggle
        print("[OK] Kaggle API library imported successfully")

        # Try to list competitions (this tests the API)
        competitions = kaggle.api.competitions_list()
        print("[OK] Kaggle API connection successful!")
        return True

    except Exception as e:
        print(f"[ERROR] Kaggle API test failed: {str(e)}")
        print("Please check your kaggle.json file and internet connection.")
        return False

if __name__ == '__main__':
    if setup_kaggle_credentials():
        print("\nTesting Kaggle connection...")
        test_kaggle_connection()