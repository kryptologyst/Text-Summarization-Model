#!/usr/bin/env python3
"""
Setup script for Text Summarization Model
=========================================

This script helps set up the text summarization project environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please use Python 3.8 or higher")
        return False

def create_virtual_environment():
    """Create a virtual environment."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("ℹ️  Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def install_dependencies():
    """Install required dependencies."""
    # Determine the correct pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    return run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")

def create_sample_data():
    """Create sample data by running the main script."""
    print("📚 Creating sample data...")
    try:
        # Run the main script to create sample data
        if os.name == 'nt':  # Windows
            python_cmd = "venv\\Scripts\\python"
        else:  # Unix/Linux/macOS
            python_cmd = "venv/bin/python"
        
        result = subprocess.run(
            f"{python_cmd} -c \"from 0104 import MockDatabase; db = MockDatabase(); print('Sample data created')\"",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ Sample data created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def run_tests():
    """Run the test suite."""
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    return run_command(f"{python_cmd} test.py", "Running tests")

def main():
    """Main setup function."""
    print("🚀 Text Summarization Model - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Setup failed at virtual environment creation")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Create sample data
    create_sample_data()
    
    # Run tests
    print("\n🧪 Running tests...")
    run_tests()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("\n2. Run the web interface:")
    print("   streamlit run app.py")
    
    print("\n3. Or run the command line version:")
    print("   python 0104.py")
    
    print("\n4. For more information, see README.md")

if __name__ == "__main__":
    main()
