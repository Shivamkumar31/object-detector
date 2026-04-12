#!/usr/bin/env python3
"""
Test Script for Object Detection Setup
Run this to verify everything is installed correctly
"""

import sys
import subprocess
import os

def print_header(text):
    """Print colored header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")

def check_python():
    """Check Python version"""
    print_header("1. Checking Python")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_pip():
    """Check pip"""
    print_header("2. Checking pip")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )
        print_success(result.stdout.strip())
        return True
    except Exception as e:
        print_error(f"pip not found: {e}")
        return False

def check_package(package_name):
    """Check if package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_dependencies():
    """Check required packages"""
    print_header("3. Checking Python Packages")
    
    packages = {
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "torch": "PyTorch",
        "fastapi": "FastAPI",
        "streamlit": "Streamlit",
        "ultralytics": "Ultralytics (YOLOv8)"
    }
    
    all_ok = True
    for module, name in packages.items():
        if check_package(module):
            print_success(f"{name}")
        else:
            print_error(f"{name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_docker():
    """Check if Docker is installed"""
    print_header("4. Checking Docker")
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_success(result.stdout.strip())
            return True
        else:
            print_error("Docker installed but not working")
            return False
    except FileNotFoundError:
        print_warning("Docker not found")
        print_info("Download from: https://www.docker.com/products/docker-desktop")
        return False

def check_files():
    """Check if required files exist"""
    print_header("5. Checking Project Files")
    
    required_files = [
        "app.py",
        "api.py",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "README.md"
    ]
    
    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print_success(f"{file} ({size} bytes)")
        else:
            print_error(f"{file} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_yolo_model():
    """Check if YOLOv8 model works"""
    print_header("6. Testing YOLOv8 Model")
    
    try:
        print_info("Attempting to import and load YOLOv8...")
        from ultralytics import YOLO
        
        # This will download model if not present
        print_info("Loading yolov8n (nano model)...")
        model = YOLO("yolov8n.pt")
        print_success("YOLOv8 model loaded successfully")
        return True
    except Exception as e:
        print_error(f"Failed to load YOLOv8: {e}")
        print_info("This may be OK - will download on first run")
        return False

def check_opencv():
    """Check OpenCV GPU support"""
    print_header("7. Checking OpenCV")
    
    try:
        import cv2
        print_success(f"OpenCV {cv2.__version__}")
        
        # Check GPU support (CUDA)
        if hasattr(cv2, 'cuda'):
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print_info("GPU support: ENABLED")
            else:
                print_info("GPU support: Not available (CPU only)")
        return True
    except Exception as e:
        print_error(f"OpenCV error: {e}")
        return False

def print_next_steps(all_checks_passed):
    """Print next steps"""
    print_header("Next Steps")
    
    if all_checks_passed:
        print_success("All checks passed! Ready to run the application.")
        print_info("\n  To start the application:")
        print("    docker-compose build")
        print("    docker-compose up\n")
        print_info("  Then open: http://localhost:8501")
    else:
        print_warning("Some checks failed. Install missing dependencies:")
        print("    pip install -r requirements.txt\n")
        print_info("  Then run this test again to verify installation.")

def main():
    """Main test function"""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  Object Detection Setup Verification".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    results = {
        "Python Version": check_python(),
        "pip": check_pip(),
        "Project Files": check_files(),
        "Python Packages": check_dependencies(),
        "Docker": check_docker(),
        "OpenCV": check_opencv(),
    }
    
    # Optional YOLOv8 test
    try:
        results["YOLOv8 Model"] = check_yolo_model()
    except:
        results["YOLOv8 Model"] = False
    
    # Print summary
    print_header("Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check}: {status}")
    
    print(f"\nPassed: {passed}/{total}")
    
    all_ok = passed == total
    print_next_steps(all_ok)
    
    print("\n" + "█"*60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
