#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run VM client
from simulation_client import main

if __name__ == "__main__":
    main()
