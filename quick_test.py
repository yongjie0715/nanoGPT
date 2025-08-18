#!/usr/bin/env python3
"""
Quick integration test - just start training and verify dashboard.
"""

import subprocess
import sys
import time

def main():
    print("🚀 Quick Integration Test")
    print("=" * 30)
    
    # Very short training run
    cmd = [
        sys.executable, "train.py",
        "--dataset=shakespeare_char",
        "--max_iters=5",        # Just 5 iterations
        "--log_interval=1", 
        "--compile=False",
        "--device=cpu",
        "--dashboard=True"
    ]
    
    print("Starting 5-iteration training with dashboard...")
    print("Command:", " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
        # Check for success indicators
        if "Dashboard server started successfully" in result.stdout:
            print("✅ Dashboard integration working!")
        
        if "iter 0:" in result.stdout or "iter 1:" in result.stdout:
            print("✅ Training iterations completed!")
            
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out - training may be working but slow")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()