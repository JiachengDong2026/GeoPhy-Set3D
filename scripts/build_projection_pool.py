#!/usr/bin/env python3
"""Build the debug or full MVP pool; source and output roots are locked."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geophy_set3d.data.generation import main

if __name__ == '__main__':
    main()
