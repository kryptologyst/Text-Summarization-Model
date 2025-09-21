#!/usr/bin/env python3
"""
Run script for Text Summarization Model
======================================

Simple script to run the text summarization application.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_web_app():
    """Run the Streamlit web application."""
    print("🚀 Starting Text Summarization Web App...")
    print("   The app will open in your browser at http://localhost:8501")
    print("   Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running web app: {e}")
        print("   Make sure Streamlit is installed: pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")

def run_cli():
    """Run the command line interface."""
    print("🚀 Starting Text Summarization CLI...")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "0104.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running CLI: {e}")

def run_tests():
    """Run the test suite."""
    print("🧪 Running Test Suite...")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "test.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running tests: {e}")

def main():
    """Main function with menu."""
    print("📝 Text Summarization Model - Run Script")
    print("=" * 50)
    
    while True:
        print("\nChoose an option:")
        print("1. Run Web Application (Streamlit)")
        print("2. Run Command Line Interface")
        print("3. Run Tests")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_web_app()
        elif choice == "2":
            run_cli()
        elif choice == "3":
            run_tests()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
